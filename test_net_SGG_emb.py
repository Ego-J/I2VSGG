# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json 
import os
import sys
import numpy as np
import pprint
import time
import _init_paths
import torch
from torch.autograd import Variable
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
#from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.parser_func import parse_args,set_dataset_args
from utils import recognition_output,detection_output,alignment,evaluate_recognition,association,evaluate,generate_static_relation_feat
from evaluation import test_vrd,eval_reall_at_N
import pdb
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3



lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)
  print(args.dataset)
  args = set_dataset_args(args,test=True)
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  np.random.seed(cfg.RNG_SEED)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  # input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  # if not os.path.exists(input_dir):
  #   raise Exception('There is no input directory for loading network from ' + input_dir)
  # load_name = os.path.join(input_dir,
  #   'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  # from model.faster_rcnn.vgg16_SGG_emb import vgg16
  from model.faster_rcnn.resnet_SGG_emb_VidVRD import resnet

  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, args, pretrained=True)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, args,101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, args,50, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()


  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (args.load_name))
  checkpoint = torch.load(args.load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
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

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  thresh = 0.0


  save_name = args.load_name.split('/')[-1]
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')
  fasterRCNN.eval()
  if not os.path.exists(args.save_feat_path):
    os.makedirs(args.save_feat_path)
  fasterRCNN.vrd.save_semantic_embedding(args.save_feat_path+os.sep+'semantic_embedding.npy')
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  res_rel = {"rlp_labels_ours":[],"rlp_confs_ours":[],"sub_bboxes_ours":[],"obj_bboxes_ours":[]}
  img_vid_map = pickle.load(open("./data/VidVRD/map.pkl","rb"))
  tracking_inputs = {}
  frame_detections = {}
  frame_recognitions = {}
  frame_object_detections = {}
  nont_detected = 0
  for i in range(num_images):
      print(i)
      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])
      im_path = data[4][0][-10:]

      det_tic = time.time()
      with torch.no_grad():
        rois, cls_prob, bbox_pred, vrd_data = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, im_path)

      # collect relation feature
      vid, fstart = img_vid_map[im_path]
      if len(vrd_data["bboxes"])<=1:
        nont_detected += 1
        #print("Nothing to save!")
      else:
        # print('saving feat of %s'%(vid+'_'+str(fstart)))
        # save_dir=args.save_feat_path+os.sep+vid
        # if not os.path.exists(save_dir):
        #   os.makedirs(save_dir)
        # np.savez(save_dir+os.sep+str(fstart),pre_feat=vrd_data["pre_feat"])
        pass

      # collect object detection results in frame

      vid, fno = img_vid_map[im_path]
      # collect relation detection results in frame
      # (given image, detect triplets with bboxes)
      if args.vrd_task == "rel_det":

        rlp_labels_im, tuple_confs_im, sub_bboxes_im, obj_bboxes_im, rel_idex= detection_output(vrd_data)
        # res_rel["rlp_labels_ours"].append(rlp_labels_im)
        # res_rel["rlp_confs_ours"].append(tuple_confs_im)
        # res_rel["sub_bboxes_ours"].append(sub_bboxes_im)
        # res_rel["obj_bboxes_ours"].append(obj_bboxes_im)
        if vid not in frame_detections.keys():
          frame_detections[vid] = []
        if isinstance(tuple_confs_im,np.ndarray):
          rlp_labels_im = rlp_labels_im.tolist()
          tuple_confs_im = tuple_confs_im.tolist()
          sub_bboxes_im = sub_bboxes_im.tolist()
          obj_bboxes_im = obj_bboxes_im.tolist()
          rel_idex = rel_idex.tolist()
          # print(im_path+"  "+str(detected)+"\n")
          # print("len(tuple_confs_im):",len(tuple_confs_im))
          # print("rel_idex:",rel_idex)
          # print(vrd_data["pre_feat"].shape)
          #frameid,score,triplet,bbox
          frame_detections[vid].append([fno,[[tuple_confs_im[j],rlp_labels_im[j],[sub_bboxes_im[j],obj_bboxes_im[j]],rel_idex[j]] for j in range(len(tuple_confs_im))]])
        else:
          frame_detections[vid].append([fno,[]])

      # collect relation recognition results in frame
      # (given image and gt boxes pair, recognize subjects, objects, predicates and relationships/triplets)
      else:
        sub_scores, obj_scores, pre_scores, tids= recognition_output(vrd_data)
        if vid not in frame_recognitions.keys():
          frame_recognitions[vid] = {}
        if isinstance(pre_scores,np.ndarray):
          print("*"*30)
          print(vid,fno)
          print("sub:",np.argsort(-sub_scores)[:5])
          print("pre:",np.argsort(-pre_scores)[:10])
          print("obj:",np.argsort(-obj_scores)[:5])
          print("*"*30)
          frame_recognitions[vid][fno] = {"sub_scores":sub_scores.tolist(),"obj_scores":obj_scores.tolist(),"pre_scores":pre_scores.tolist(),"tids":tids}
        else:
          frame_recognitions[vid][fno] = {}

      # regular test output of faster-rcnn
      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]
      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= data[1][0][2].item()

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>0.05).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]

            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array


      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1] for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]
       
  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))

  # evaluation
  if args.vrd_task == "rel_det":
    # rec_det_50  = eval_reall_at_N(args.dataset, 50, res_rel)
    # rec_det_100 = eval_reall_at_N(args.dataset, 100, res_rel)
    # print("Not detected: ",nont_detected)
    # print("Relation Recall@50: ",rec_det_50)
    # print("Relation Recall@100: ",rec_det_100)

    with open("./frame_detections_results_"+args.adaptation+".json","w") as f:
      json.dump(frame_detections,f)
    #frame_detections = json.load(open("./frame_detections_results.json","r"))
    video_detections_static = association(frame_detections)
    with open("./video_association_results_"+args.adaptation+".json","w") as f:
      json.dump(video_detections_static,f)
    #video_detections_static = json.load(open("./video_association_results_"+args.adaptation+".json","r"))
    #generate_static_relation_feat(video_detections_static,args.save_videofeat_path,args.save_feat_path)
    #video_detections = dynamic_reasoning(video_detections_static,"detection")
    mean_ap, rec_at_n, mprec_at_n = evaluate(video_detections_static, "./data/VidVRD/video_annotations_static.json")
  else:
    #frame_recognitions = json.load(open("./frame_recognitions.json","r"))

    video_recognitions_aligned,video_recognitions_static = alignment(frame_recognitions)
    with open("./frame_recognitions.json","w") as f:
      json.dump(video_recognitions_static,f)
    #video_recognitions_dynamic = dynamic_reasoning(video_recognitions_aligned,"recognition")
    # video_recognitions = merge(video_recognitions_static,video_recognitions_dynamic)
    acc_at_n = evaluate_recognition(video_recognitions_static)
