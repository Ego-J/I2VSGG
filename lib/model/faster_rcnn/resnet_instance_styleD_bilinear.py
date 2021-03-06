from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg

from model.faster_rcnn.faster_rcnn_instance_styleD_bilinear  import _fasterRCNN
from model.utils.net_utils import GradReverse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

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
def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)
class netD_pixel(nn.Module):
    def __init__(self,context=False):
        super(netD_pixel, self).__init__()
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        #self.conv1 = conv1x1(512, 256)
        #self.conv2 = conv1x1(256, 128)
        #self.conv3 = conv1x1(128, 1)
        self.context = context
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x, lamb=1.0):
        #default no droopout
        # print("lamb for netD_pixel:",lamb)
        x = GradReverse.apply(x,lamb)
        x = F.relu(self.conv1(x))
        #print("x.shape in conv1 in pixelD:",x.shape)
        x = F.relu(self.conv2(x))
        #print("x.shape in conv2 in pixelD:",x.shape)
        if self.context:
          feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
          #print("feat.shape",feat.shape)
          #feat = x
          x = self.conv3(x)
          #print("x.shape in conv3 in pixelD:",x.shape)
          return torch.sigmoid(x),feat#F.sigmoid(x),feat#torch.cat((feat1,feat2),1)#F
        else:
          x = self.conv3(x)
          return torch.sigmoid(x)#F.sigmoid(x)

class netD_style(nn.Module):
    def __init__(self,context=False,dim=512,rank=5):
        super(netD_style, self).__init__()
        self.dim = dim
        self.rank =rank
        self.fc_1 = nn.Linear(512,dim*rank)
        self.fc_2 = nn.Linear(512,dim*rank)
        self.fc1 = nn.Linear(dim,1)
        self.context = context
        # self.fc2 = nn.Linear(256,1)
        # self.fc3 = nn.Linear(512,256)
        # self.fc4 = nn.Linear(256,128)      
        # self.fc5 = nn.Linear(128,1)
        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      # normal_init(self.fc_1, 0, 0.01)
      # normal_init(self.fc_2, 0, 0.01)
      # normal_init(self.fc1, 0, 0.01)
      nn.init.kaiming_normal_(self.fc_1.weight, mode='fan_out', nonlinearity='relu')
      nn.init.kaiming_normal_(self.fc_2.weight, mode='fan_out', nonlinearity='relu')
      nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
      #nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
      # normal_init(self.fc2, 0, 0.01)
      # normal_init(self.fc3, 0, 0.01)
      # normal_init(self.fc4, 0, 0.01)
      # normal_init(self.fc5, 0, 0.01)
    def forward(self, x, lamb=1.0):
        # print("lamb:",lamb)
        x = GradReverse.apply(x,lamb)
        b,c,w,h = x.shape
        # print("x.shape",x.shape)
        x = x.reshape(b,c,-1)
        # print("x.reshape",x.shape)
        x = x.permute(0,2,1)
        # print("x_T.shape",x.shape)
        x1 = self.fc_1(x)
        x2 = self.fc_2(x)
        # print("x1.shape",x1.shape)
        x = x1*x2
        # print("x1*x2.shape",x.shape)
        x = torch.sum(x.reshape(b,w*h,self.dim,self.rank),dim=-1)
        # print("x.shape",x.shape)
        x = torch.sum(x,dim=1)
        # print("x.shape",x.shape)
        x = torch.sqrt(F.relu(x)) - torch.sqrt(F.relu(-x))  
        x = F.normalize(x, p=2, dim=1)
        if self.context:
          feat = x.view(x.size(0),-1)
          x = self.fc1(feat)
          return torch.sigmoid(x),feat
        else:
          x = self.fc1(x)
          return torch.sigmoid(x)


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
  print("begin resnet50")
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
  def __init__(self, classes, num_layers, pretrained=False, class_agnostic=False,ic=False,gc=False):
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.ic = ic
    self.gc = gc
    self.layers = num_layers
    #if self.layers == 50:
    #  self.model_path = '/home/grad3/keisaito/data/pretrained_model/resnet50_caffe.pth'
    _fasterRCNN.__init__(self, classes, class_agnostic,ic,gc)

  def _init_modules(self):

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
      # print("parameters in pretrained_model")
      # for k,v in state_dict.items():
      #   print(k)
      # print("parameters in target model")
      # for k in resnet.state_dict():
      #   print(k)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})
    # Build resnet.
    # self.RCNN_base1 = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
    #   resnet.maxpool,resnet.layer1)
    # self.RCNN_base2 = nn.Sequential(resnet.layer2,resnet.layer3)
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
    # print("self.RCNN_base[4]:",self.RCNN_base[4])
    # print("self.RCNN_base[4].shape:",self.RCNN_base[4].shape)
    self.netD_pixel = netD_pixel(context=self.ic)
    self.netD_style = netD_style(context=self.gc)
    self.RCNN_top = nn.Sequential(resnet.layer4)
    feat_d = 2048
    #if self.lc:
    #  feat_d += 128
    if self.gc:
     feat_d += 512
    if self.ic:
      feat_d += 128
    self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    # assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    # if cfg.RESNET.FIXED_BLOCKS >= 3:
    #   for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    # if cfg.RESNET.FIXED_BLOCKS >= 2:
    #   for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    # if cfg.RESNET.FIXED_BLOCKS >= 1:
    #   print("fix RCNN_base[4]")
    #   for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)
  def extract_feature(self,x):
    for i in range(len(self.RCNN_base)):
      x = self.RCNN_base[i](x)
      # print(i)
      # print(x.shape)
      if i==5:
        base_feat1 = x
        #print("base_feat1.shape:",base_feat1.shape)
    return x, base_feat1


  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      # self.RCNN_base[4].train()
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
