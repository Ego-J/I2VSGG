from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn_SGG_emb import _fasterRCNN
from model.faster_rcnn.utils import calc_mean_std,calc_gramma,Conv2d,FC
from model.roi_layers import ROIPool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

from .glove import GloVe
import pickle 
import json
import numpy as np

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)
def get_obj_prd_vecs(glove_path,predicate_file,object_list):
    glove = GloVe(glove_path)
    with open(predicate_file, 'rb') as fid:
        prds = json.load(fid)    
    # with open(object_file, 'rb') as fid:
    #     objs = json.load(fid)
    # represent background with the word 'unknown'
    # obj_cats.insert(0, 'unknown')
    #prds.insert('unknown',0)
    print("objects:")
    all_obj_vecs = np.zeros((len(object_list), 300), dtype=np.float32)
    for i in range(len(object_list)):
        # print(object_list[i])
        # print(glove[object_list[i]][0])
        all_obj_vecs[i] = glove[object_list[i]]
    # for r,obj in enumerate(objs):
    #     obj_words = obj.split()
    #     print("obj_words:",obj_words)
    #     all_obj_vecs[r] = glove[obj_words]
    print("predicates:")
    all_prd_vecs = np.zeros((len(prds), 300), dtype=np.float32)
    for r, prd_cat in enumerate(prds):
        # prd_words = prd_cat.split()
        # print("prd_words:",prd_words)
        all_prd_vecs[r] = glove[prd_cat]
    return all_obj_vecs, all_prd_vecs
class vrd(nn.Module):
    def __init__(self, args, all_obj_vecs=None, all_prd_vecs=None,bn=False):
        super(vrd, self).__init__()

        self.args = args
        self.n_rel = args.num_relations
        self.n_obj = args.num_classes
        self.emb_dim = args.emb_dim
        self.obj_vecs = all_obj_vecs
        self.prd_vecs = all_prd_vecs
        with open(args.source_so_prior_path, 'rb') as fid:
            self._so_prior = np.array(pickle.load(fid,encoding='bytes'))
        with open(args.source_gt_rels_path, 'rb') as fid:
            self.source_gt_rels = pickle.load(fid,encoding='bytes')
        with open(args.target_gt_rels_path, 'rb') as fid:
            self.target_gt_rels = pickle.load(fid,encoding='bytes')


        self.roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.fc6 = FC(1024 * 7 * 7, 4096)
        self.fc7 = FC(4096, 4096) 
        #the last classification layer of VGG16     
        self.so_vis_embeddings = FC(4096, self.emb_dim, relu=False)


        self.fc8 = FC(4096, 256)
        #self.criterion = nn.MultiLabelMarginLoss().cuda()
        # self.constractcls = ContrastiveLoss()
        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.BCEWithLogitsLoss()

        n_fusion = 256
        if(args.use_obj_visual):
            self.fc_so = FC(300*2, 256)
            n_fusion += 256

        if(args.spatial_type == 1):
            self.fc_lov = FC(8, 256)
            n_fusion += 256

        elif(args.spatial_type == 2):
            self.conv_lo = nn.Sequential(Conv2d(2, 96, 5, same_padding=True, stride=2, bn=bn),
                                       Conv2d(96, 128, 5, same_padding=True, stride=2, bn=bn),
                                       Conv2d(128, 64, 8, same_padding=False, bn=bn))
            self.fc_lov = FC(64, 256)
            n_fusion += 256

        # if(args.use_semantic):
        #     # self.emb = nn.Embedding(self.n_obj, 300)
        #     # self._set_trainable(self.emb, requires_grad=False)
        #     self.fc_so_emb = FC(300*2, 256)
        #     n_fusion += 256

        self.fc_fusion = FC(n_fusion,256)
        self.fc_rel = FC(256, self.emb_dim,relu=False)
        # self.sigmoid = nn.Sigmoid()
        # self.so_sem_embeddings = nn.Sequential(
        #     nn.Linear(300, 1024),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(1024, self.emb_dim))
        self.prd_sem_embeddings = nn.Sequential(
            nn.Linear(300, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, self.emb_dim))
    def forward(self, fmap, boxes, rel_boxes, SpatialFea, classes, ix1, ix2):

        fmap = self._np_to_variable(fmap, is_cuda=True)

        # fmap_so = self._np_to_variable(fmap_so, is_cuda=True)
        # fmap_rel = self._np_to_variable(fmap_rel, is_cuda=True)
        
        boxes = self._np_to_variable(boxes, is_cuda=True)
        rel_boxes = self._np_to_variable(rel_boxes, is_cuda=True)
        SpatialFea = self._np_to_variable(SpatialFea, is_cuda=True)
        classes = [int(x) for x in classes]
        emb = self.obj_vecs[classes]
        emb = self._np_to_variable(emb, is_cuda=True)
        ix1 = self._np_to_variable(ix1, is_cuda=True, dtype=torch.LongTensor)
        ix2 = self._np_to_variable(ix2, is_cuda=True, dtype=torch.LongTensor)

        x_so = self.roi_pool(fmap, boxes)
        # print('after roi pool x_so.shape',x_so.shape)
        x_so = x_so.view(x_so.size()[0], -1)
        # print('after reshape x_so.shape',x_so.shape)
        x_so = self.fc6(x_so)
        x_so = F.dropout(x_so, training=self.training)
        x_so = self.fc7(x_so)
        x_so = F.dropout(x_so, training=self.training)
        obj_feature = self.so_vis_embeddings(x_so)
        x_s = torch.index_select(obj_feature, 0, ix1)
        x_o = torch.index_select(obj_feature, 0, ix2)
        #print("sbj_vis_embeddings.shape",sbj_vis_embeddings.shape)
        # x_so = self.fc8(x_so)

        x_u = self.roi_pool(fmap, rel_boxes)
        x = x_u.view(x_u.size()[0], -1)
        x = self.fc6(x)
        x = F.dropout(x, training=self.training)
        x = self.fc7(x)
        x = F.dropout(x, training=self.training)
        x = self.fc8(x)
        
        if(self.args.use_obj_visual):
            x_so = torch.cat((x_s, x_o), 1)
            #print("x_so.shape",x_so.shape)
            x_so = self.fc_so(x_so)
            x = torch.cat((x, x_so), 1)

        if(self.args.spatial_type == 1):
            lo = self.fc_lov(SpatialFea)
            x = torch.cat((x, lo), 1)            
        elif(self.args.spatial_type == 2):
            lo = self.conv_lo(SpatialFea)
            lo = lo.view(lo.size()[0], -1)
            lo = self.fc_lov(lo)
            x = torch.cat((x, lo), 1)

        # if(self.args.use_semantic):
        #     emb = torch.squeeze(emb, 1)
        #     emb_s = torch.index_select(emb, 0, ix1)
        #     emb_o = torch.index_select(emb, 0, ix2)
        #     emb_so = torch.cat((emb_s, emb_o), 1)
        #     emb_so = self.fc_so_emb(emb_so)
        #     x = torch.cat((x, emb_so), 1)

        x = self.fc_fusion(x)
        x = self.fc_rel(x)

        # ds_obj_vecs = self.obj_vecs
        # ds_obj_vecs = Variable(torch.from_numpy(ds_obj_vecs.astype('float32'))).cuda()
        # so_sem_embeddings = self.so_sem_embeddings(ds_obj_vecs)
        # so_sem_embeddings = F.normalize(so_sem_embeddings, p=2, dim=1)  # (#prd, 1024)
        # so_sem_embeddings.t_()
        # sbj_vis_embeddings = F.normalize(x_s, p=2, dim=1)  # (#bs, 1024)
        # sbj_sim_matrix = torch.mm(sbj_vis_embeddings, so_sem_embeddings)  # (#bs, #prd)
        # #print("sbj_sim_matrix:",sbj_sim_matrix)
        # sbj_cls_scores = sbj_sim_matrix
        
        
        # obj_vis_embeddings = F.normalize(x_o, p=2, dim=1)  # (#bs, 1024)
        # obj_sim_matrix = torch.mm(obj_vis_embeddings, so_sem_embeddings)  # (#bs, #prd)
        # #print("obj_sim_matrix:",obj_sim_matrix)
        # obj_cls_scores = obj_sim_matrix

        ds_prd_vecs = self.prd_vecs
        ds_prd_vecs = Variable(torch.from_numpy(ds_prd_vecs.astype('float32'))).cuda()
        prd_sem_embeddings = self.prd_sem_embeddings(ds_prd_vecs)
        prd_sem_embeddings = F.normalize(prd_sem_embeddings, p=2, dim=1)  # (#prd, 1024)
        prd_vis_embeddings = F.normalize(x, p=2, dim=1)  # (#bs, 1024)
        prd_sim_matrix = torch.mm(prd_vis_embeddings, prd_sem_embeddings.t_())  # (#bs, #prd)
        prd_cls_scores = prd_sim_matrix
        #print("prd_cls_scores.shape:",prd_cls_scores.shape)
        if not self.training:
            # sbj_cls_scores = F.softmax(sbj_cls_scores, dim=1)
            # obj_cls_scores = F.softmax(obj_cls_scores, dim=1)
            prd_cls_scores = F.softmax(prd_cls_scores, dim=1)
        
        return prd_cls_scores,x.detach().cpu().numpy()

        #return x_s, x_o, x_p
    def save_semantic_embedding(self,save_path):
        ds_prd_vecs = self.prd_vecs
        ds_prd_vecs = Variable(torch.from_numpy(ds_prd_vecs.astype('float32'))).cuda()
        prd_sem_embeddings = self.prd_sem_embeddings(ds_prd_vecs)
        np.save(save_path,prd_sem_embeddings.detach().cpu().numpy())

    def _set_trainable(self, model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad
    
    def _np_to_variable(self, x, is_cuda=True, dtype=torch.FloatTensor):
        v = Variable(torch.from_numpy(x).type(dtype))
        if is_cuda:
            v = v.cuda()
        return v
    
    def _getUnionBBox(self, aBB, bBB, ih, iw, margin = 10):
        return [max(0, min(aBB[0], bBB[0]) - margin), \
            max(0, min(aBB[1], bBB[1]) - margin), \
            min(iw, max(aBB[2], bBB[2]) + margin), \
            min(ih, max(aBB[3], bBB[3]) + margin)]
    
    def _getDualMask(self, ih, iw, bb):
        rh = 32.0 / ih
        rw = 32.0 / iw
        x1 = max(0, int(math.floor(bb[0] * rw)))
        x2 = min(32, int(math.ceil(bb[2] * rw)))
        y1 = max(0, int(math.floor(bb[1] * rh)))
        y2 = min(32, int(math.ceil(bb[3] * rh)))
        mask = np.zeros((32, 32))
        mask[y1 : y2, x1 : x2] = 1
        assert(mask.sum() == (y2 - y1) * (x2 - x1))
        return mask    

    def _getRelativeLoc(self, aBB, bBB):
        sx1, sy1, sx2, sy2 = aBB.astype(np.float32)
        ox1, oy1, ox2, oy2 = bBB.astype(np.float32)
        sw, sh, ow, oh = sx2-sx1, sy2-sy1, ox2-ox1, oy2-oy1
        xy = np.array([(sx1-ox1)/ow, (sy1-oy1)/oh, (ox1-sx1)/sw, (oy1-sy1)/sh])
        wh = np.log(np.array([sw/ow, sh/oh, ow/sw, oh/sh]))
        return np.hstack((xy, wh))

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class resnet(_fasterRCNN):
  def __init__(self, classes, args,num_layers=101, pretrained=False, class_agnostic=False):
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.layers = num_layers
    self.args = args
    print("classes:",classes)
    self.obj_vecs, self.prd_vecs =get_obj_prd_vecs(self.args.glove_path,self.args.predicate_file,classes)
    _fasterRCNN.__init__(self, classes, args)

  def _init_modules(self):
    #resnet = resnet101()
    if self.layers == 50:
      self.model_path = cfg.RESNET_PATH50
      resnet = resnet50()
      print("backbone:ResNet50")
    elif self.layers == 101:
      self.model_path = cfg.RESNET_PATH
      resnet = resnet101()
      print("backbone:ResNet101")
    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
    self.vrd = vrd(self.args,self.obj_vecs, self.prd_vecs)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    # assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    # if cfg.RESNET.FIXED_BLOCKS >= 3:
    #   for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    # if cfg.RESNET.FIXED_BLOCKS >= 2:
    #   for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    # if cfg.RESNET.FIXED_BLOCKS >= 1:
    #   for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[5].train()
      self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7
