import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.bbox_transform import bbox_transform_inv,clip_boxes
# from model.utils.net_utils import res_detections
import time
import pdb
import math
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta  #,grad_reverse
import cv2
import os
import copy 
import pickle

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def get_keys(d, value):
     return [k for k,v in d.items() if v == value]


class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, args):
        super(_fasterRCNN, self).__init__()
        self.args=args
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = args.class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

        # self.tracking_outputs = pickle.load(open("./data/VidOR/tracking_outputs_deepsort.pkl","rb"))
        self.map = pickle.load(open("./data/VidOR/map.pkl","rb"))

 
    def forward(self, im_data, im_info, gt_boxes, num_boxes, im_path, target=False):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # # base_feat = self.RCNN_base2(base_feat1)
        # # feed base feature map tp RPN to obtain rois'''
        # #print("target is ",target)
        # rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes,target)
        # #print("rois.shape:",rois.shape)
        # # if it is training phrase, then use ground trubut bboxes for refining
        # if self.training and not target:
        #     #print("source traning---------------------------")
        #     #print("batch_size:",batch_size)
        #     #print("gt_boxes.shape:",gt_boxes.shape)
        #     #print("num_boxes:",num_boxes.data)
        #     '''
        #     print(self.training)
        #     print(~target)
        #     print("use ground trubut bboxes for refining")'''
        #     roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
        #     rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
        #     rois_label = Variable(rois_label.view(-1).long())
        #     rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
        #     rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
        #     rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        # else:
        #     rois_label = None
        #     rois_target = None
        #     rois_inside_ws = None
        #     rois_outside_ws = None
        #     rpn_loss_cls = 0
        #     rpn_loss_bbox = 0
        #     lossQ = -1
        # rois = Variable(rois)
        # # do roi pooling based on predicted rois

        # pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        
        # # if cfg.POOLING_MODE == 'crop':
        # #     # pdb.set_trace()
        # #     # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
        # #     grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
        # #     grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
        # #     pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
        # #     if cfg.CROP_RESIZE_WITH_MAX_POOL:
        # #         pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        # # elif cfg.POOLING_MODE == 'align':
        # #     pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        # # elif cfg.POOLING_MODE == 'pool':
        # #     pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # # feed pooled features to top model
        # #print("pooled_feat before _head_to_tail:",pooled_feat.shape)
        # #print("eta in faster rcnn:",eta)
        # pooled_feat = self._head_to_tail(pooled_feat)
        # # compute bbox offset
        # bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        # bbox_pred_vrd = bbox_pred # need boxes deltas for get detection results
        # if self.training and not self.class_agnostic and not target:
        #     bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
        #     bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
        #     bbox_pred = bbox_pred_select.squeeze(1)
        # # compute object classification probability
        # #print("pooled_feat.shape in faster_rcnn_global_pixel_instance:",pooled_feat.shape)
        # cls_score = self.RCNN_cls_score(pooled_feat)
        # cls_prob = F.softmax(cls_score, 1)
        # #print("cls_prob is ",cls_prob.shape)


        # RCNN_loss_cls = 0
        # RCNN_loss_bbox = 0

        # if self.training:
        #     # classification loss
        #     RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

        #     # bounding box regression L1 loss
        #     RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        # cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        # bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        # scores = cls_prob.data
        # boxes = rois.data[:, :, 1:5]
        

        rois, cls_prob, bbox_pred, bbox_pred_vrd, scores, boxes = 0, 0, 0, 0, 0, 0
        if self.training:
            if self.args.vrd_task == "pre_det":
                base_feat = base_feat.detach().cpu().numpy()
                loss_cls_prd = self.forward_predicate(base_feat,im_info,im_path)
            if self.args.vrd_task == "rel_det":
                base_feat = base_feat.detach().cpu().numpy()
                loss_cls_prd = self.forward_relation(base_feat,scores,boxes,bbox_pred_vrd,im_info,im_path)
            return loss_cls_prd
        else:
            if self.args.vrd_task == "pre_det":
                base_feat = base_feat.detach().cpu().numpy()
                vrd_data = self.forward_predicate(base_feat,im_info,im_path)
            if self.args.vrd_task == "rel_det":
                base_feat = base_feat.detach().cpu().numpy()
                vrd_data = self.forward_relation(base_feat,scores,boxes,bbox_pred_vrd,im_info,im_path)
            return rois, cls_prob, bbox_pred, vrd_data

            
    def forward_predicate(self, fmap, im_info, im_path):

        # <----------------process gt bboxes---------------->
        #print("im_path:",im_path)
        if self.training:
            anno = self.vrd.source_gt_rels[im_path]
            gt_boxes = (np.array(anno["boxes"])*im_info[0][2].item()) # scale to x:600 or 600:y
            gt_box_classes = anno["box_classes"]
            gt_rels = anno["rels"]
            
            #print("anno:",anno)
            if len(gt_rels) < 1:
                if (len(anno["boxes"]) == 0):
                    vrd_data = {"boxes":[], "classes":[], "confs":[]}
                    return vrd_data
                elif (len(anno["boxes"]) == 1):
                    vrd_data = {"boxes":anno["boxes"], "classes":anno["box_classes"], "confs":[]}
                    return vrd_data
            pairs = []
            tmp_rels = []
            
            # find unique relation?
            for i,rel in enumerate(gt_rels):
                if [rel[0],rel[1]] not in pairs:
                    pairs.append([rel[0],rel[1]])
                    tmp_rels.append([rel[2]])
                    if not self.training:
                        tids.append(gt_tids[i])
                else:
                    tmp_rels[pairs.index([rel[0],rel[1]])].append(rel[2])
            ixs = []
            ixo = []    
            for pair in pairs:
                ixs.append(pair[0])
                ixo.append(pair[1])
            ixs = np.array(ixs)
            ixo = np.array(ixo)
            gt_rels = tmp_rels 
            n_rel_inst = len(gt_rels)
            #print("gt_rels:",gt_rels)


            # <----------------get vrd input---------------->
            ih = im_info[0][0].item()
            iw = im_info[0][1].item()
            rel_boxes = np.zeros((n_rel_inst, 5))
            #rel_so_prior = np.zeros((n_rel_inst, self.vrd.n_rel))
            rel_labels = np.zeros((n_rel_inst,self.vrd.n_rel))
            # print("rel_labels.shape:",rel_labels.shape)
            # obj_labels = np.zeros(n_rel_inst)
            # subj_labels = np.zeros(n_rel_inst)
            SpatialFea = np.zeros((n_rel_inst, 2, 32, 32))
            # SpatialFea = np.zeros((n_rel_inst, 8))
            pos_idx = 0
            for ii in range(n_rel_inst):
                sBBox = gt_boxes[ixs[ii]]
                oBBox = gt_boxes[ixo[ii]]
                rBBox = self.vrd._getUnionBBox(sBBox, oBBox, ih, iw)
                rel_boxes[ii, 1:5] = np.array(rBBox)
                SpatialFea[ii] = [self.vrd._getDualMask(ih, iw, sBBox), \
                                  self.vrd._getDualMask(ih, iw, oBBox)]                      
                # rel_so_prior[ii] = self.vrd._so_prior[gt_box_classes[ixs[ii]]-1, gt_box_classes[ixo[ii]]-1]
                # obj_labels[ii] = gt_box_classes[ixo[ii]]
                # subj_labels[ii] = gt_box_classes[ixs[ii]]
                # print("gt_rels[ii]:",gt_rels[ii])
                # rel_labels[ii] = gt_rels[ii]

                # SpatialFea[ii] = self._getRelativeLoc(sBBox, oBBox)
                # for r in gt_rels[ii]:
                #     print("r:",r)
                for r in gt_rels[ii]:
                    rel_labels[ii,r] = 1
                    # pos_idx += 1
            # print("rel_labels:",rel_labels)
            # rel_labels=np.array(rel_labels)

            boxes = np.zeros((gt_boxes.shape[0], 5))
            boxes[:, 1:5] = gt_boxes
            boxes = boxes.astype(np.float32, copy=False)
            # print("rel_boxes.shape:",rel_boxes)
            classes = np.array(gt_box_classes).astype(np.float32, copy=False)
            # print("classes:",classes)

            target = Variable(torch.from_numpy(rel_labels).type(torch.FloatTensor)).cuda()
            # print("target:",target)
            # target_obj = Variable(torch.from_numpy(obj_labels).type(torch.LongTensor)).cuda()
            # print("target_obj:",target_obj)
            # target_subj = Variable(torch.from_numpy(subj_labels).type(torch.LongTensor)).cuda()
            # print("target_subj:",target_subj)
            # rel_so_prior = -0.5*(rel_so_prior+1.0/self.vrd.n_rel)
            # rel_so_prior = Variable(torch.from_numpy(rel_so_prior).type(torch.FloatTensor)).cuda()
            rel_score,pre_feat = self.vrd(fmap, boxes, rel_boxes, SpatialFea, classes, ixs, ixo)
            # print("rel_score:",F.softmax(rel_score, dim=1)) 
            # print("sbj_cls_scores:",F.softmax(sbj_cls_scores, dim=1))
            # print("obj_cls_scores:",F.softmax(obj_cls_scores, dim=1))
            # print("target.shape:",target.shape) 
            # print("sbj_cls_scores.shape:",sbj_cls_scores.shape) 
            # print("target_obj.shape:",target_obj.shape)
            # print("target_subj.shape:",target_subj.shape) 
            # print("rel_score.shape:",rel_score.shape)        

            #obj_score, rel_score = self.vrd(fmap, boxes, rel_boxes, SpatialFea, classes, ixs, ixo)
            # print(gt_rels)
            # print(rel_score)
            #loss_cls_prd = self.vrd.criterion((rel_so_prior+rel_score), target)
            loss_cls_prd = self.vrd.criterion(rel_score, target)
            # loss_subj_prd = F.cross_entropy(sbj_cls_scores, target_subj)
            # loss_obj_prd = F.cross_entropy(obj_cls_scores, target_obj)
            #loss = self.vrd.criterion((rel_so_prior+rel_score).view(1, -1), target)
            # loss = loss_cls_prd+loss_obj_prd+loss_subj_prd
            return loss_cls_prd

        else:
            anno = self.vrd.target_gt_rels[im_path]
            
            box_num = len(anno["boxes"])
            rois = (np.array(anno["boxes"])*im_info[0][2].item()) # scale to x:600 or 600:y
            rois = np.concatenate((np.zeros((rois.shape[0],1)),rois),axis=1)
            padding = np.zeros((300-box_num,5))
            rois = np.concatenate((rois,padding),axis=0)
            rois = torch.FloatTensor(np.expand_dims(rois,axis=0)).cuda()
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            pooled_feat = self._head_to_tail(pooled_feat)
            cls_score = self.RCNN_cls_score(pooled_feat)
            cls_prob = F.softmax(cls_score, 1)[:box_num].detach().cpu().numpy()
            cls_prob[:,0]  = 0.0
            classes = cls_prob.argmax(axis=-1)
            confs = [cls_prob[i,_class] for i,_class in enumerate(classes)]


            gt_tids = anno["tids"]
            gt_rels = anno["rels"]
            gt_boxes = np.array(anno["boxes"])*im_info[0][2].item()
            
            if len(gt_rels) < 1:
                if (len(anno["boxes"]) == 0):
                    vrd_data = {"boxes":[], "classes":[], "confs":[]}
                    return vrd_data
                elif (len(anno["boxes"]) == 1):
                    vrd_data = {"boxes":anno["boxes"], "classes":classes, "confs":confs}
                    return vrd_data
            pairs = []
            tmp_rels = []
            tids = []
            sub_scores = []
            obj_scores = []
            ixs = []
            ixo = [] 
            # find unique relation?
            for i,rel in enumerate(gt_rels):
                if [rel[0],rel[1]] not in pairs:
                    pairs.append([rel[0],rel[1]])
                    tmp_rels.append([rel[2]])
                    sub_scores.append(cls_prob[rel[0]])
                    obj_scores.append(cls_prob[rel[1]])
                    ixs.append(rel[0])
                    ixo.append(rel[1])
                    if not self.training:
                        tids.append(gt_tids[i])
                else:
                    tmp_rels[pairs.index([rel[0],rel[1]])].append(rel[2])
            ixs = np.array(ixs)
            ixo = np.array(ixo)
            sub_scores = np.array(sub_scores)
            obj_scores = np.array(obj_scores)
            gt_rels = tmp_rels 
            n_rel_inst = len(gt_rels)
            #print("gt_rels:",gt_rels)


            # <----------------get vrd input---------------->
            ih = im_info[0][0].item()
            iw = im_info[0][1].item()
            rel_boxes = np.zeros((n_rel_inst, 5))
            rel_so_prior = np.zeros((n_rel_inst, self.vrd.n_rel))
            rel_labels = np.zeros((n_rel_inst,self.vrd.n_rel))
            # print("rel_labels.shape:",rel_labels.shape)
            obj_labels = np.zeros(n_rel_inst)
            subj_labels = np.zeros(n_rel_inst)
            SpatialFea = np.zeros((n_rel_inst, 2, 32, 32))
            # SpatialFea = np.zeros((n_rel_inst, 8))
            pos_idx = 0
            for ii in range(n_rel_inst):
                sBBox = gt_boxes[ixs[ii]]
                oBBox = gt_boxes[ixo[ii]]
                rBBox = self.vrd._getUnionBBox(sBBox, oBBox, ih, iw)
                rel_boxes[ii, 1:5] = np.array(rBBox)
                SpatialFea[ii] = [self.vrd._getDualMask(ih, iw, sBBox), \
                                  self.vrd._getDualMask(ih, iw, oBBox)]                      
                print(classes[ixs[ii]]-1,classes[ixo[ii]]-1)
                rel_so_prior[ii] = self.vrd._so_prior[classes[ixs[ii]]-1, classes[ixo[ii]]-1]
                # obj_labels[ii] = classes[ixo[ii]]
                # subj_labels[ii] = classes[ixs[ii]]
                # print("gt_rels[ii]:",gt_rels[ii])
                # rel_labels[ii] = gt_rels[ii]

                # SpatialFea[ii] = self._getRelativeLoc(sBBox, oBBox)
                # for r in gt_rels[ii]:
                #     print("r:",r)
                for r in gt_rels[ii]:
                    rel_labels[ii,r] = 1
                    # pos_idx += 1
            # print("rel_labels:",rel_labels)
            # rel_labels=np.array(rel_labels)

            boxes = np.zeros((gt_boxes.shape[0], 5))
            boxes[:, 1:5] = gt_boxes
            boxes = boxes.astype(np.float32, copy=False)
            # print("rel_boxes.shape:",rel_boxes)
            # print("classes:",classes)

            fmap = fmap.detach().cpu().numpy()
            rel_scores,pre_feat= self.vrd(fmap, boxes, rel_boxes, SpatialFea, ixs, ixo)
            vrd_data = {"ixs":ixs, "ixo":ixo, "boxes":anno["boxes"], "classes":classes, "confs":confs,"rel_so_prior":rel_so_prior,\
            "rel_scores":rel_scores,"sub_scores":sub_scores,"obj_scores":obj_scores,"pre_feat":pre_feat,"tids":tids}
            return vrd_data
    
    def _extract_feature(self, base_feat, bboxes):
        print(bboxes.shape, base_feat.shape)
        base_feat = copy.deepcopy(base_feat)
        base_feat = Variable(torch.from_numpy(base_feat).type(torch.FloatTensor)).cuda()
        bboxes = copy.deepcopy(bboxes)
        bboxes = np.hstack((np.zeros((bboxes.shape[0], 1)), bboxes))
        bboxes = Variable(torch.from_numpy(bboxes).type(torch.FloatTensor)).cuda()
        pooled_feat = self.RCNN_roi_align(base_feat, bboxes.view(-1, 5))
        pooled_feat = self._head_to_tail(pooled_feat)
        pooled_feat = pooled_feat.detach().cpu().numpy()
        return pooled_feat

    def forward_relation(self, fmap, scores, boxes, bbox_pred, im_info, im_path):
        
        # tracking = False
        res = {}
        res['bboxes'] = np.zeros((0,4))
        res['classes'] = []
        res['scores'] = []
        # <----------------get predicted bboxes---------------->
        # if not tracking:
        #     if cfg.TEST.BBOX_REG:
        #         # Apply bounding-box regression deltas
        #         box_deltas = bbox_pred.data
        #         if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        #         # Optionally normalize targets by a precomputed mean and stdev
        #             if self.class_agnostic:
        #                 box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
        #                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        #                 box_deltas = box_deltas.view(1, -1, 4)
        #             else:
        #                 box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
        #                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        #                 box_deltas = box_deltas.view(1, -1, 4 * len(self.classes))    
        #         pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        #         pred_boxes = clip_boxes(pred_boxes, im_info, 1)
        #     else:
        #         # Simply repeat the boxes, once for each class
        #         pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        #     # pred_boxes /= im_info[0][2].item() # resize to be same as raw pic

        #     scores = scores.squeeze()
        #     pred_boxes = pred_boxes.squeeze()

        #     thresh = 0.05 
        #     for j in range(1, self.n_classes):
                
        #         inds = torch.nonzero(scores[:,j]>thresh).view(-1)
        #         if inds.shape[0] == 0:
        #             continue
        #         # if there is det
        #         if inds.numel() > 0:
        #             cls_scores = scores[:,j][inds]
        #             _, order = torch.sort(cls_scores, 0, True)
        #         if self.class_agnostic:
        #             cls_boxes = pred_boxes[inds, :]
        #         else:  
        #             cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

        #         cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        #         # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
        #         cls_dets = cls_dets[order]
        #         keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
        #         cls_dets = cls_dets[keep.view(-1).long()]

        #         for i in range(np.minimum(10, cls_dets.shape[0])):
        #             bbox = tuple(int(np.round(x)) for x in cls_dets[i, :4])
        #             score = cls_dets[i, -1]
        #             if score > 0.7:
        #                 res['bboxes'] = np.vstack((res['bboxes'], cls_dets[i, :4]))
        #                 res['classes'].append(j)
        #                 res['scores'].append(score.item())
        #     # if len(res['bboxes']) > 0:
        #     #     res['feats'] = self._extract_feature(fmap, res['bboxes'])
        #     # else:
        #     #     res['feats'] = []
        #     detected_boxes = res['bboxes']/im_info[0][2].item()
        # else:

        # data_root = "./tracking/data_" + self.args.adaptation
        # vname, fno = self.map[im_path]
        # tracking_outputs = pickle.load(open(data_root+"/"+vname+"/seqnms_output.pkl","rb"))
        # if fno in tracking_outputs:
        #     detected_boxes = tracking_outputs[fno]['bboxes']
        #     h,w = im_info[0][0].item()/im_info[0][2].item(),im_info[0][1].item()/im_info[0][2].item()
        #     tmp_boxes = []
        #     for box_id,box in enumerate(detected_boxes):
        #         if box[0]>=w or box[1]>=h or box[2]<=0 or box[3]<=0:
        #             continue
        #         tmp_boxes.append(np.array([max(0,box[0]),max(0,box[1]),min(w,box[2]),min(h,box[3])]))
        #         res['classes'].append(tracking_outputs[fno]['classes'][box_id]) 
        #         res['scores'].append(tracking_outputs[fno]['scores'][box_id])    
        #     detected_boxes = np.array(tmp_boxes)
        #     res['bboxes'] = detected_boxes*im_info[0][2].item()

        if self.training:
            # <----------------assgin rels---------------->
            anno = self.vrd.source_gt_rels[im_path]
            gt_boxes = np.array(anno["boxes"])*im_info[0][2].item() # scale to x:600 or 600:y
            gt_box_classes = np.array(anno["box_classes"])
            gt_rels = np.array(anno["rels"])

            pred_boxes = np.array(res['box'])
            pred_class = np.array(res['cls'])
            # use confs to filter rels
            # pred_confs = np.array(res['confs'])  

            if pred_boxes.shape[0] == 0:
                return Variable(torch.zeros(1).type(torch.FloatTensor), requires_grad=True).cuda()
            ious = bbox_overlaps(pred_boxes,gt_boxes)
            is_match = (pred_class[:,None] == gt_box_classes[None]) & (ious >= 0.5)

            if is_match.sum() == 0:
                return Variable(torch.zeros(1).type(torch.FloatTensor), requires_grad=True).cuda()
            
            rels = []
            for i, (from_gtind, to_gtind, rel_id) in enumerate(gt_rels):
                rels_i = []
                scores_i = []

                for from_ind in np.where(is_match[:, from_gtind])[0]:
                    for to_ind in np.where(is_match[:, to_gtind])[0]:
                        if from_ind != to_ind:
                            rels_i.append((from_ind, to_ind, rel_id))
                            scores_i.append((ious[from_ind, from_gtind] * ious[to_ind, to_gtind]))
                if len(rels_i) == 0:
                    continue
                p = np.array(scores_i)
                p = p / p.sum()
                num_sample_per_gt = 10
                num_to_add = min(p.shape[0], num_sample_per_gt)
                for rel_to_add in np.random.choice(p.shape[0], p=p, size=num_to_add, replace=False):
                    rels.append(rels_i[rel_to_add])
            
            if len(rels) == 0:
                return Variable(torch.zeros(1).type(torch.FloatTensor), requires_grad=True).cuda()

            
            ## inst number 
            pairs = []
            tmp_rels = []
            for rel in rels:
                if [rel[0],rel[1]] not in pairs:
                    pairs.append([rel[0],rel[1]])
                    tmp_rels.append([rel[2]])
                else:
                    tmp_rels[pairs.index([rel[0],rel[1]])].append(rel[2])
            ixs = []
            ixo = []
            for pair in pairs:
                ixs.append(pair[0])
                ixo.append(pair[1])
            ixs = np.array(ixs)
            ixo = np.array(ixo)
            rels = tmp_rels 
            n_rel_inst = len(rels)

            # <----------------get vrd input---------------->
            pred_boxes = res['box']
            pred_classes = res['cls']
            ih = im_info[0][0].item()
            iw = im_info[0][1].item()
            rel_boxes = np.zeros((n_rel_inst, 5))
            rel_labels = -1*np.ones((1, n_rel_inst*self.vrd.n_rel))
            rel_so_prior = np.zeros((n_rel_inst, self.vrd.n_rel))
            SpatialFea = np.zeros((n_rel_inst, 2, 32, 32))
            # SpatialFea = np.zeros((n_rel_inst, 8))
            pos_idx = 0
            for ii in range(n_rel_inst):
                sBBox = pred_boxes[ixs[ii]]
                oBBox = pred_boxes[ixo[ii]]
                rBBox = self.vrd._getUnionBBox(sBBox, oBBox, ih, iw)
                rel_boxes[ii, 1:5] = np.array(rBBox)    
                SpatialFea[ii] = [self.vrd._getDualMask(ih, iw, sBBox), \
                                  self.vrd._getDualMask(ih, iw, oBBox)]                      
                rel_so_prior[ii] = self.vrd._so_prior[pred_classes[ixs[ii]]-1, pred_classes[ixo[ii]]-1]
                # SpatialFea[ii] = self._getRelativeLoc(sBBox, oBBox)
                for r in rels[ii]:
                    rel_labels[0, pos_idx] = ii*self.vrd.n_rel + r
                    pos_idx += 1

            boxes = np.zeros((pred_boxes.shape[0], 5))
            boxes[:, 1:5] = pred_boxes
            boxes = boxes.astype(np.float32, copy=False)
            classes = np.array(pred_classes).astype(np.float32, copy=False)

            # <----------------forward---------------->
            target = Variable(torch.from_numpy(rel_labels).type(torch.FloatTensor)).cuda()
            # rel_so_prior = -0.5*(rel_so_prior+1.0/self.vrd.n_rel)
            # rel_so_prior = Variable(torch.from_numpy(rel_so_prior).type(torch.FloatTensor)).cuda()
            rel_score,sbj_cls_scores, obj_cls_scores = self.vrd(fmap, boxes, rel_boxes, SpatialFea, ixs, ixo)
            #loss = self.vrd.criterion((rel_score).view(1, -1), target)
            loss = self.vrd.criterion(rel_score, target)
            # loss_subj_prd = F.cross_entropy(sbj_cls_scores, target_subj)
            # loss_obj_prd = F.cross_entropy(obj_cls_scores, target_obj)
            #loss = self.vrd.criterion((rel_so_prior+rel_score).view(1, -1), target)
            # loss = loss_cls_prd+loss_obj_prd+loss_subj_prd
            return loss
        
        else:
            
            # <----------- gt boxes to test ----------->
            anno = self.vrd.target_gt_rels[im_path]
            detected_boxes = anno["boxes"]
            res['bboxes'] = np.array(anno["boxes"])*im_info[0][2].item()
            res['classes'] = anno["box_classes"]
            res['scores'] = [1 for _ in range(len(anno["box_classes"]))]
            #<----------- gt boxes to test ----------->

            if (len(res['bboxes']) == 0):
                vrd_data = {"bboxes":[], "classes":[], "scores":[]} #, "feats":[]}
                return vrd_data
            elif (len(res['bboxes']) == 1):
                vrd_data = {"bboxes":detected_boxes, "classes":res['classes'], "scores":res['scores']} #, "feats":res['feats']}
                return vrd_data

            ixs = []
            ixo = []
            for i in range(len(res['bboxes'])):
                for j in range(len(res['bboxes'])):
                    if i != j:
                        ixs.append(i)
                        ixo.append(j)
            ixs = np.array(ixs)
            ixo = np.array(ixo)
            n_rel_inst = len(ixs)

            # #<----------- gt pairs to test ----------->
            # gt_rels = anno["rels"]
            # if len(gt_rels) < 1:
            #     if (len(anno["boxes"]) == 0):
            #         vrd_data = {"boxes":[], "classes":[], "confs":[]}
            #         return vrd_data
            #     elif (len(anno["boxes"]) == 1):
            #         vrd_data = {"boxes":anno["boxes"], "classes":anno["box_classes"], "confs":[]}
            #         return vrd_data
            # pairs = []
            # tmp_rels = []
            
            # # find unique relation?
            # for i,rel in enumerate(gt_rels):
            #     if [rel[0],rel[1]] not in pairs:
            #         pairs.append([rel[0],rel[1]])
            #         tmp_rels.append([rel[2]])
            #     else:
            #         tmp_rels[pairs.index([rel[0],rel[1]])].append(rel[2])
            # ixs = []
            # ixo = []    
            # for pair in pairs:
            #     ixs.append(pair[0])
            #     ixo.append(pair[1])
            # ixs = np.array(ixs)
            # ixo = np.array(ixo)
            # gt_rels = tmp_rels 
            # n_rel_inst = len(gt_rels)
            # #<----------- gt pairs to test ----------->


            # <----------------get vrd input---------------->
            pred_boxes = res['bboxes']
            pred_classes = res['classes']
            ih = im_info[0][0].item()
            iw = im_info[0][1].item()
            rel_boxes = np.zeros((n_rel_inst, 5))
            SpatialFea = np.zeros((n_rel_inst, 2, 32, 32))
            rel_so_prior = np.zeros((n_rel_inst, self.vrd.n_rel)) # debug
            # SpatialFea = np.zeros((n_rel_inst, 8))
            pos_idx = 0
            for ii in range(n_rel_inst):
                sBBox = pred_boxes[ixs[ii]]
                oBBox = pred_boxes[ixo[ii]]
                rBBox = self.vrd._getUnionBBox(sBBox, oBBox, ih, iw)
                rel_boxes[ii, 1:5] = np.array(rBBox)    
                SpatialFea[ii] = [self.vrd._getDualMask(ih, iw, sBBox), \
                                  self.vrd._getDualMask(ih, iw, oBBox)] 
                rel_so_prior[ii] = self.vrd._so_prior[pred_classes[ixs[ii]]-1, pred_classes[ixo[ii]]-1]  # debug                   
                # SpatialFea[ii] = self._getRelativeLoc(sBBox, oBBox)

            boxes = np.zeros((pred_boxes.shape[0], 5))
            boxes[:, 1:5] = pred_boxes
            boxes = boxes.astype(np.float32, copy=False)
            classes = np.array(pred_classes).astype(np.float32, copy=False)
            rel_score,pre_feat = self.vrd(fmap, boxes, rel_boxes, SpatialFea, pred_classes, ixs, ixo)
            # print("rel_boxes",rel_boxes.shape)
            # print("boxes",boxes.shape)
            # print("rel_score",rel_score.shape)
            # print("sbj_score",sbj_score.shape)
            # print("obj_score",obj_score.shape)

            # print("-"*30)
            # print(im_path)
            # anno = self.vrd.target_gt_rels[im_path]
            # print("GT rels:")
            # gt_rels = anno["rels"]
            # gt_classes = anno["box_classes"]
            # for rel in gt_rels:
            #     print(gt_classes[rel[0]],rel[2],gt_classes[rel[1]])
            
            # print("Faster RCNN class:")
            # for c in classes:
            #     print(c)

            # print("SGG prediction:")
            # sbj_score_t = sbj_score.detach().cpu().numpy()
            # obj_score_t = obj_score.detach().cpu().numpy()
            # rel_score_t = rel_score.detach().cpu().numpy()
            # for tuple_idx in range(rel_score.shape[0]):
            #     print("sub:",np.argsort(-sbj_score_t[tuple_idx])[:5])
            #     print("pre:",np.argsort(-rel_score_t[tuple_idx])[:5])
            #     print("obj:",np.argsort(-obj_score_t[tuple_idx])[:5])
            # print("-"*30)


            vrd_data = {
                "ixs":ixs, "ixo":ixo, "bboxes":detected_boxes,"classes":res['classes'],"scores":res['scores'], #,"feats":res['feats'],\
                "rel_score":rel_score,"pre_feat":pre_feat,"rel_so_prior":rel_so_prior}
            return vrd_data


        
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
