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
    adjust_learning_rate, save_checkpoint, clip_gradient, sampler

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

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
    train_size_t = len(roidb_t)

    print('{:d} source roidb entries'.format(len(roidb)))
    print('{:d} target roidb entries'.format(len(roidb_t)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)

    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True)

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=args.num_workers)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                               sampler=sampler_batch_t, num_workers=args.num_workers)
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    im_data_t = torch.FloatTensor(1)
    im_info_t = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        im_data_t = im_data_t.cuda()
        im_info_t = im_info_t.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    im_data_t = Variable(im_data_t)
    im_info_t = Variable(im_info_t)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    from model.faster_rcnn.resnet_instance_styleD_bilinear import resnet
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,ic=args.ic,gc=args.gc)
    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    
    if args.cuda:
        fasterRCNN.cuda()

    parameter_list = []
    wo_parameter = ['netD_pixel','RPN_cls_score','RPN_bbox_pred','RCNN_cls_score','RCNN_bbox_pred']
    for k in fasterRCNN.state_dict():
        flag = True
        for tag in wo_parameter:
            if tag in k:
                flag=False
        if flag:
            parameter_list.append(k)
    # print(parameter_list)

    
    # ------------------------model initialization with object detection ckpt-----------------------
    if args.resume and 'faster_rcnn' in args.load_name:
        print("loading checkpoint %s" % (args.load_name))
        checkpoint = torch.load(args.load_name,map_location='cpu')['model']
        fasterRCNN_dict = fasterRCNN.state_dict()
        # print("parameters in checkpoint")
        # for k,v in checkpoint.items():
        #     print(k)
        state_dict = {k:v for k,v in checkpoint.items() if k in parameter_list}
        fasterRCNN_dict.update(state_dict)
        fasterRCNN.load_state_dict(fasterRCNN_dict)
        #fasterRCNN.load_state_dict(checkpoint['model'])
        # args.session = checkpoint['session']
        # args.start_epoch = checkpoint['epoch']
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (args.load_name))
    
    # ------model resume with relation detection ckpt-------
    if args.resume and 'faster_rcnn' not in args.load_name:
        print("loading checkpoint %s" % (args.load_name))
        checkpoint = torch.load(args.load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (args.load_name))
    
    print("args.start_epoch:",args.start_epoch)

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    iters_per_epoch = int(train_size / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        log_dir = "./log/log_instance_pixel_styleD/"+args.tfb_path
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_name = log_dir+os.sep+"tb"

        logger = SummaryWriter(log_name)
    count_iter = 0
    consistency_loss = torch.nn.MSELoss()
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        loss_rpn_cls_temp = 0
        loss_rpn_box_temp = 0
        loss_rcnn_cls_temp = 0
        loss_rcnn_box_temp = 0
        loss_s_p_temp = 0
        loss_t_p_temp = 0
        loss_s_style_temp = 0
        loss_t_style_temp = 0
        loss_s_cst_temp = 0
        loss_t_cst_temp = 0
        loss_style_temp = 0
        start = time.time()
        # if epoch % (args.update_target) == 0:
        #     print("updating encode with backbone %d"%(epoch))
        if (epoch > 1) and ((epoch - 1) % args.lr_decay_step == 0):
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)
        for step in range(iters_per_epoch):
        #for step in range(1000):
            try:
                data_s = next(data_iter_s)
                if type(data_s) != list:
                    continue
            except:
                continue
            try:
                data_t = next(data_iter_t)
                if type(data_t) != list:
                    continue
            except Exception as e:
                if type(e) == StopIteration:
                    data_iter_t = iter(dataloader_t)
                    data_t = next(data_iter_t)
                else:
                    continue
            
            # if len(data_s) <= 2:
            #     continue
            #eta = 1.0
            count_iter += 1
            #put source data into variable
            im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
            im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
            gt_boxes.data.resize_(data_s[2].size()).copy_(data_s[2])
            num_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])

            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label,out_d_instance,out_d_style = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,target=False,eta=args.eta,eta_style=args.eta_style)
            # print("out_d_instance.shape:",out_d_instance.shape)
            # print("out_d_style.shape:",out_d_style.shape)
            #print("rpn_loss_cls:",rpn_loss_cls)
            #print("rpn_loss_cls.mean():",rpn_loss_cls.mean())
            loss_rpn_cls = rpn_loss_cls.mean()
            loss_rpn_box = rpn_loss_box.mean()
            loss_rcnn_cls = RCNN_loss_cls.mean()
            loss_rcnn_box = RCNN_loss_bbox.mean()  
            loss = loss_rpn_cls+ loss_rpn_box \
                   + loss_rcnn_cls+ loss_rcnn_box

            # local alignment loss
            dloss_s_p = 0.5 * torch.mean(out_d_instance ** 2)
            dloss_s_style = 0.5 * torch.mean(out_d_style ** 2)
    
            #put target data into variable
            im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            im_info_t.data.resize_(data_t[1].size()).copy_(data_t[1])
            #gt is empty
            gt_boxes.data.resize_(args.batch_size, 1, 5).zero_()
            num_boxes.data.resize_(args.batch_size).zero_()
            out_d_instance_t,out_d_style_t = fasterRCNN(im_data_t, im_info_t, gt_boxes, num_boxes, target=True, eta=args.eta,eta_style=args.eta_style)
            dloss_t_p = 0.5 * torch.mean((1 - out_d_instance_t) ** 2)
            dloss_t_style = 0.5 * torch.mean((1 - out_d_style_t) ** 2)
            loss_style = args.style_lambda *(dloss_s_style+dloss_t_style)


            if args.cr:
                out_d_instance_consist = torch.mean(out_d_instance,dim=3)
                out_d_instance_consist = torch.mean(out_d_instance_consist,dim=2)
                # print("out_d_instance_consist.shape",out_d_instance_consist.shape)
                consistency_prob=out_d_style.repeat(1,128).view(-1,1)

                out_d_instance_t_consist = torch.mean(out_d_instance_t,dim=3)
                out_d_instance_t_consist = torch.mean(out_d_instance_t_consist,dim=2)
                # print("out_d_instance_t_consist.shape",out_d_instance_t_consist.shape)
                consistency_prob_t=out_d_style_t.repeat(1,128).view(-1,1)
                # print("consistency_prob.shape:",consistency_prob.shape)
                source_adv_cst_loss = consistency_loss(out_d_instance_consist,consistency_prob.detach())
                target_adv_cst_loss = consistency_loss(out_d_instance_t_consist,consistency_prob_t.detach())
                loss += (dloss_t_p + dloss_s_p)+loss_style+source_adv_cst_loss+target_adv_cst_loss
            else:
                loss += (dloss_t_p + dloss_s_p)+loss_style
            '''
            if args.debug:
                print("im_data.size",im_data.shape)
                print("im_data_t.size",im_data_t.shape)
                print("g_t.size",g_t.shape)
            '''
            # domain label
            #if vis and epoch % args.disp_interval == 0
            #domain_t = Variable(torch.ones(out_d_instance.size(0)).long().cuda())
            #dloss_t = 0.5 * FL(out_d_instance, domain_t)
            # local alignment loss
            loss_temp += loss.item()
            loss_rpn_cls_temp += loss_rpn_cls.item()
            loss_rpn_box_temp += loss_rpn_box.item()
            loss_rcnn_cls_temp += loss_rcnn_cls.item()
            loss_rcnn_box_temp += loss_rcnn_box.item()
            loss_s_p_temp += dloss_s_p.item()
            loss_t_p_temp += dloss_t_p.item()
            loss_s_style_temp += dloss_s_style.item()
            loss_t_style_temp += dloss_t_style.item()
            if args.cr:
                loss_s_cst_temp += source_adv_cst_loss.item()
                loss_t_cst_temp += target_adv_cst_loss.item()
            loss_style_temp +=loss_style.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if vis:
            #    im = cv2.imread(imdb.image_path_at(i))
            #    #print("im.shape in vision:",im.shape)
            #    im2show = np.copy(im)

            start_time_tf = time.time()
            if args.use_tfboard:
                if args.cr:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls':  loss_rpn_cls.item(),
                        'loss_rpn_box':  loss_rpn_box.item(),
                        'loss_rcnn_cls': loss_rcnn_cls.item(),
                        'loss_rcnn_box': loss_rcnn_box.item(),
                        'dloss_s_p':     dloss_s_p.item(),
                        'dloss_t_p':     dloss_t_p.item(),
                        'dloss_s_style': dloss_s_style.item(),
                        'dloss_t_style': dloss_t_style.item(),
                        'loss_s_cst_temp': source_adv_cst_loss.item(),
                        'loss_t_cst_temp': target_adv_cst_loss.item(),
                        'loss_style':    loss_style.item()

                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)
                else:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls':  loss_rpn_cls.item(),
                        'loss_rpn_box':  loss_rpn_box.item(),
                        'loss_rcnn_cls': loss_rcnn_cls.item(),
                        'loss_rcnn_box': loss_rcnn_box.item(),
                        'dloss_s_p':     dloss_s_p.item(),
                        'dloss_t_p':     dloss_t_p.item(),
                        'dloss_s_style': dloss_s_style.item(),
                        'dloss_t_style': dloss_t_style.item(),
                        'loss_style':    loss_style.item()

                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)                    
            #print("[iter %4d] style_s: %.4f style_t: %.4f style: %.4f time: %.4f" \
            #    % ((epoch - 1) * iters_per_epoch + step, dloss_s_style.item(),dloss_t_style.item(),loss_style.item(),time.time()-start_time_tf))
            if step % args.disp_interval == 0:
                end = time.time()
                loss_temp /= args.disp_interval
                loss_rpn_cls_temp /= args.disp_interval
                loss_rpn_box_temp /= args.disp_interval
                loss_rcnn_cls_temp /= args.disp_interval
                loss_rcnn_box_temp /= args.disp_interval
                loss_s_p_temp /= args.disp_interval
                loss_t_p_temp /= args.disp_interval
                loss_s_style_temp /= args.disp_interval
                loss_t_style_temp /= args.disp_interval
                loss_s_cst_temp /= args.disp_interval
                loss_t_cst_temp /= args.disp_interval
                loss_style_temp /= args.disp_interval
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f dloss s: %.4f dloss t: %.4f style_s: %.4f style_t: %.4f style: %.4f eta: %.4f eta_style: %.4f" \
                    % (loss_rpn_cls_temp, loss_rpn_box_temp, loss_rcnn_cls_temp, loss_rcnn_box_temp, loss_s_p_temp, loss_t_p_temp,loss_s_style_temp,loss_t_style_temp,loss_style_temp,args.eta,args.eta_style))
                loss_temp = 0
                loss_rpn_cls_temp = 0
                loss_rpn_box_temp = 0
                loss_rcnn_box_temp = 0
                loss_rcnn_cls_temp = 0 
                loss_s_p_temp = 0
                loss_t_p_temp = 0
                loss_s_style_temp = 0
                loss_t_style_temp = 0
                loss_t_cst_temp = 0
                loss_s_cst_temp = 0
                loss_style_temp = 0
                start = time.time()
        save_name = os.path.join(output_dir,
                                 'instance_pixel_styleD_bilinear_cr_{}_source_{}_target_{}_session_{}_lr_{}_epoch_{}_bs_{}_mscoco.pth'.format(
                                     args.cr,args.dataset,args.dataset_t,
                                     args.session, args.lr,epoch,
                                     args.batch_size))
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
