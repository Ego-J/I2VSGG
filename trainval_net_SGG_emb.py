# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pprint
import pdb
import time
import _init_paths


import torch
from torch.autograd import Variable
import torch.nn as nn
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, sampler, calc_supp

from model.utils.parser_func import parse_args, set_dataset_args

def vis_showagent_image(im, boxes,path):
    for i in range(boxes.shape[0]):
        #print(boxes[i])
        bbox=boxes[i]
        #print("sum(bbox):",sum(bbox))
        if sum(bbox)==0:
          continue
        im_new = im[bbox[1]:bbox[3],bbox[0]:bbox[2]]

        #print("im_new:",im_new)
        save_path_new = path+os.sep+'result_'+str(i)+'.jpg'
        #print(save_path_new)
        try:
          cv2.imwrite(save_path_new,im_new)
        except:
          continue


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    # with open(args.predicate_file, 'rb') as fid:
    #     all_lines=all_file.readlines()
    #     for all_line in all_lines:
    #         args.predicate_class.append(all_line.replace('\n',''))

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = False
    cfg.USE_GPU_NMS = args.cuda
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    print('{:d} source roidb entries'.format(len(roidb)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)

    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True, path_return=True)

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size, sampler=sampler_batch, num_workers=args.num_workers)
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    from model.faster_rcnn.resnet_SGG_emb import resnet
    fasterRCNN = resnet(imdb.classes, args, 101, pretrained=True)

    else:
        print("network is not defined")
        pdb.set_trace()
    
    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    vrd_lr = args.vrd_lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'vrd' in key:
                print("key in vrd:",key)
                if 'bias' in key:
                    params += [{'params': [value], 'lr': vrd_lr* (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params': [value], 'lr': vrd_lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]   
            else:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]   

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume: # resume only for load faster rcnn model, not for other model status!!!!
        print("loading checkpoint %s" % (args.load_name))
        checkpoint = torch.load('./models/'+args.net +'/VGOR/'+args.load_name)
        state_dict = fasterRCNN.state_dict()
        for k in state_dict.keys():
            if 'vrd' not in k:
                if k not in checkpoint['model']:
                    print(k)
                    continue
                state_dict[k] = checkpoint['model'][k]
            #     print("RCNN paremeters:",k)
            #     print(state_dict[k].size())
            # else:
            #     print("vrd parameters:",k)
            #     print(state_dict[k].size())
        fasterRCNN.load_state_dict(state_dict)
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (args.load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    iters_per_epoch = int(train_size / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        log_dir = "./log_SGG_emb/"+args.tfb_path
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_name = log_dir+os.sep+"tb"

        logger = SummaryWriter(log_name)
    count_iter = 0

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        # loss_pred_cls_temp = 0
        # loss_obj_cls_temp = 0
        # loss_subj_cls_temp = 0
        start = time.time()
        # if epoch % (args.update_target) == 0:
        #     print("updating encode with backbone %d"%(epoch))
        if (epoch > 1) and ((epoch - 1) % args.lr_decay_step == 0):
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma
            vrd_lr *= args.lr_decay_gamma

        data_iter_s = iter(dataloader_s)
        for step in range(iters_per_epoch):
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)
                data_s = next(data_iter_s)
            #eta = 1.0
            count_iter += 1
            if len(data_s) <= 2:
                continue
            #put source data into variable

            im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
            im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
            gt_boxes.data.resize_(data_s[2].size()).copy_(data_s[2])
            num_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])
            im_path = data_s[4][0].split("/")[-1]

            fasterRCNN.zero_grad()  
            
            # rois=cls_prob=bbox_pred= \
            # rpn_loss_cls=rpn_loss_box= \
            # RCNN_loss_cls=RCNN_loss_bbox= \
            # rois_label = torch.tensor([0]).type(torch.FloatTensor)
            # loss_pre_prd,loss_obj_prd,loss_subj_prd = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, im_path, target=False)
            loss = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, im_path, target=False)
            # loss = loss_pre_prd+loss_obj_prd+loss_subj_prd
            
            # rois, cls_prob, bbox_pred, \
            # rpn_loss_cls, rpn_loss_box, \
            # RCNN_loss_cls, RCNN_loss_bbox, \
            # rois_label, vrd_loss = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, im_path, target=False)
            # loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
            #      + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() + vrd_loss

            
            # loss_pred_cls = loss_pre_prd.mean().item()
            # loss_obj_cls = loss_obj_prd.mean().item()
            # loss_subj_cls = loss_subj_prd.mean().item()
            # loss_rcnn_box = RCNN_loss_bbox.mean()
            vrd_loss = loss.mean().item()

            loss_temp += loss.item()
            # loss_pred_cls_temp += loss_pre_prd.item()
            # loss_obj_cls_temp += loss_obj_prd.item()
            # loss_subj_cls_temp += loss_subj_prd.item()
            # loss_vrd_temp += loss_vrd.item() 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #if vis:
            #    im = cv2.imread(imdb.image_path_at(i))
            #    #print("im.shape in vision:",im.shape)
            #    im2show = np.copy(im)
            if args.use_tfboard:
                info = {
                    # 'loss_pred_cls':  loss_pred_cls,
                    # 'loss_obj_cls':  loss_obj_cls,
                    # 'loss_subj_cls': loss_subj_cls,
                    'loss_vrd': vrd_loss
                }
                logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                   (epoch - 1) * iters_per_epoch + step)
            args.disp_interval = 100
            if (step!= 0) and (step % args.disp_interval == 0):
                end = time.time()

                loss_temp /= args.disp_interval
                # loss_pred_cls_temp /= args.disp_interval
                # loss_obj_cls_temp /= args.disp_interval
                # loss_subj_cls_temp /= args.disp_interval
           
                # fg_cnt = torch.sum(rois_label.data.ne(0))
                # bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e, vrd_lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr, vrd_lr))
                print("time cost: %f" % (end - start))
                # print(
                #     "\t\t\tpre_cls: %.4f, obj_cls: %.4f, subj_cls: %.4f" \
                #     % (loss_pred_cls_temp, loss_obj_cls_temp, loss_subj_cls_temp))


                loss_temp = 0
                # loss_pred_cls_temp = 0
                # loss_obj_cls_temp = 0
                # loss_subj_cls_temp = 0
                start = time.time()
        if args.vrd_task == "pre_det":                
            save_name = os.path.join(output_dir,
                                     'SGG_emb_p_prior_{}_{}_pre_det_session_{}_epoch_{}_step_{}_un.pth'.format(
                                        args.adaptation,
                                        args.dataset,
                                        args.session,
                                        epoch,
                                        step))
        elif args.vrd_task == "rel_det":
            save_name = os.path.join(output_dir,
                                     'SGG_emb_p_prior_{}_{}_rel_det_session_{}_epoch_{}_step_{}_un.pth'.format(
                                        args.adaptation,
                                        args.dataset,
                                        args.session,
                                        epoch,
                                        step))

        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()
