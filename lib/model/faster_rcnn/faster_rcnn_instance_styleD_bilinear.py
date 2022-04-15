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
# from model.roi_layers import ROIAlign, ROIPool
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
import math
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

import cv2
#from model.faster_rcnn.utils import calc_gramma

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic,ic,gc):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.ic = ic
        self.gc = gc
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        # self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

        # self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        # self.RCNN_roi_crop = _RoICrop()
    
    def forward(self, im_data, im_info, gt_boxes, num_boxes,target=False,eta=1.0,eta_style=1.0):
        # print("eta is:",eta)
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat,base_feat1 = self.extract_feature(im_data)
        # base_feat = self.RCNN_base(im_data)
        #print("bash_feat.shape:",base_feat.shape)
        # print("base_feat1.shape:",base_feat1.shape)
        #input_gramma = calc_gramma(base_feat1)
        #print("input_gramma.shape:",input_gramma.shape)

        if self.gc:
            d_style,_= self.netD_style(base_feat1, eta_style)
            if not target:
               _,feat_image = self.netD_style(base_feat1.detach(), eta_style) 
        else:
            d_style= self.netD_style(base_feat1, eta_style)  
        #d_logit,d_style= self.netD_style(base_feat1)
        # print("d_logit:",d_logit)
        # print("d_style:",d_style)
        # base_feat = self.RCNN_base2(base_feat1)
  
        #base_feat = self.RCNN_base2(im_data)
        # feed base feature map tp RPN to obtain rois'''
        # print("target is ",target)
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, target)
        # print("rois.shape:",rois.shape)
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training and not target:
            # print("source traning---------------------------")
            #print("batch_size:",batch_size)
            #print("gt_boxes.shape:",gt_boxes.shape)
            #print("num_boxes:",num_boxes.data)
            # print(self.training)
            # print(~target)
            # print("use ground trubut bboxes for refining")
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # print("rois.shape:",rois.shape)
        # do roi pooling based on predicted rois
        # print("cfg.POOLING_MODE:",cfg.POOLING_MODE)
        # if cfg.POOLING_MODE == 'crop':
        #     # pdb.set_trace()
        #     # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
        #     grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
        #     # print("rois.shape:",rois.shape)
        #     grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
        #     # print("rois.shape:",rois.shape)
        #     pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
        #     if cfg.CROP_RESIZE_WITH_MAX_POOL:
        #         pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        # print("pooled_feat before _head_to_tail:",pooled_feat.shape)
        if self.ic:
            d_instance, _ = self.netD_pixel(pooled_feat, eta)
            if not target:
                _,feat_instance = self.netD_pixel(pooled_feat.detach(),eta)
        else:
            d_instance = self.netD_pixel(pooled_feat, eta)

        if target:
            return d_instance, d_style

        pooled_feat = self._head_to_tail(pooled_feat)
        if self.gc:
            # print("feat_image.shape:",feat_image.shape)
            # print("pooled_feat.shape before gc:",pooled_feat.shape)
            feat_image = feat_image.unsqueeze(1)
            context_dim = feat_image.size(2)
            # print("feat_image.shape:",feat_image.shape)
            proposal_num = int(pooled_feat.size(0)/batch_size)
            # print("proposal_num:",proposal_num)
            feat_image = feat_image.repeat(1,proposal_num,1).view(-1,context_dim)
            pooled_feat = torch.cat((feat_image, pooled_feat), 1)
            # print("pooled_feat.shape:",pooled_feat.shape)
        if self.ic:
            # print("pooled_feat.shape before ic:",pooled_feat.shape)
            feat_instance = feat_instance.view(feat_instance.size(0),-1)
            pooled_feat = torch.cat((feat_instance, pooled_feat), 1)
            # print("pooled_feat.shape:",pooled_feat.shape)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic and not target:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        #print("pooled_feat.shape in faster_rcnn_global_pixel_instance:",pooled_feat.shape)
        cls_score = self.RCNN_cls_score(pooled_feat)

        cls_prob = F.softmax(cls_score, 1)
        #print("cls_prob is ",cls_prob.shape)


        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
                
        if batch_size ==1 or not self.training:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, d_instance,d_style
        else:
            return rois, cls_prob, bbox_pred, rpn_loss_cls.view(-1), rpn_loss_bbox.view(-1), RCNN_loss_cls.view(-1), RCNN_loss_bbox.view(-1), rois_label, d_instance,d_style

    def _transform_style_input(self,im_data,method):
        
        if method == "scale":
            style_inputs = np.squeeze(im_data.cpu().numpy().transpose(0,2,3,1))
            style_inputs = np.expand_dims(cv2.resize(style_inputs,self.content_size),0).transpose(0,3,1,2)
            style_inputs = torch.from_numpy(style_inputs).cuda()
        
        return style_inputs
        
        

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
