import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
from .resnet import resnet50
# from random_erasing import RandomErasing_vertical, RandomErasing_2x2
import math

__all__ = ['HPM']
######################################################################

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ConvBlock(nn.Module):
    """Basic convolutional block.
    
    convolution + batch normalization + relu.

    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class DimReduceLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)  


def pcb_block(num_ftrs, num_stripes, local_conv_out_channels, num_classes, avg=False):
    if avg:
        pooling_list = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in range(num_stripes)])
    else:
        pooling_list = nn.ModuleList([nn.AdaptiveMaxPool2d(1) for _ in range(num_stripes)])
    conv_list = nn.ModuleList([nn.Conv2d(num_ftrs, local_conv_out_channels, 1, bias=False) for _ in range(num_stripes)])
    batchnorm_list = nn.ModuleList([nn.BatchNorm2d(local_conv_out_channels) for _ in range(num_stripes)])
    relu_list = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(num_stripes)])
    fc_list = nn.ModuleList([nn.Linear(local_conv_out_channels, num_classes, bias=False) for _ in range(num_stripes)])
    for m in conv_list:
        weight_init(m)
    for m in batchnorm_list:
        weight_init(m)
    for m in fc_list:
        weight_init(m)
    return pooling_list, conv_list, batchnorm_list, relu_list, fc_list

def spp_vertical(feats, pool_list, conv_list, bn_list, relu_list, fc_list, num_strides, feat_list=[], logits_list=[]):
    for i in range(num_strides):
        pcb_feat = pool_list[i](feats[:, :, i * int(feats.size(2) / num_strides): (i+1) *  int(feats.size(2) / num_strides), :])
        pcb_feat = conv_list[i](pcb_feat)
        pcb_feat = bn_list[i](pcb_feat)
        pcb_feat = relu_list[i](pcb_feat)
        pcb_feat = pcb_feat.view(pcb_feat.size(0), -1)
        feat_list.append(pcb_feat)
        logits_list.append(fc_list[i](pcb_feat))
    return feat_list, logits_list

def global_pcb(feats, pool, conv, bn, relu, fc, feat_list=[], logits_list=[]):
    global_feat = pool(feats)
    global_feat = conv(global_feat)
    global_feat = bn(global_feat)
    global_feat = relu(global_feat)
    global_feat = global_feat.view(feats.size(0), -1)
    feat_list.append(global_feat)
    logits_list.append(fc(global_feat))
    return feat_list, logits_list


class HPM(nn.Module):
    def __init__(self, num_classes, num_stripes=6, local_conv_out_channels=256, erase=0, loss={'xent'}, avg=False, **kwargs):
        super(HPM, self).__init__()
        self.erase = erase
        self.num_stripes = num_stripes
        self.loss = loss

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        # PSP
        # self.psp_pool, self.psp_conv, self.psp_bn, self.psp_relu, self.psp_upsample, self.conv = psp_block(self.num_ftrs)

        # global
        self.global_pooling = nn.AdaptiveMaxPool2d(1)
        self.global_conv = nn.Conv2d(self.num_ftrs, local_conv_out_channels, 1, bias=False)
        self.global_bn = nn.BatchNorm2d(local_conv_out_channels)
        self.global_relu = nn.ReLU(inplace=True)
        self.global_fc = nn.Linear(local_conv_out_channels, num_classes, bias=False)

        weight_init(self.global_conv)
        weight_init(self.global_bn) 
        weight_init(self.global_fc)


        # 2x
        self.pcb2_pool_list, self.pcb2_conv_list, self.pcb2_batchnorm_list, self.pcb2_relu_list, self.pcb2_fc_list = pcb_block(self.num_ftrs, 2, local_conv_out_channels, num_classes, avg)
        # 4x
        self.pcb4_pool_list, self.pcb4_conv_list, self.pcb4_batchnorm_list, self.pcb4_relu_list, self.pcb4_fc_list = pcb_block(self.num_ftrs, 4, local_conv_out_channels, num_classes, avg)
        # 8x
        self.pcb8_pool_list, self.pcb8_conv_list, self.pcb8_batchnorm_list, self.pcb8_relu_list, self.pcb8_fc_list = pcb_block(self.num_ftrs, 8, local_conv_out_channels, num_classes, avg)

        self._init_params()

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

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    

    def forward(self, x):
        feat_list = []
        logits_list = []
        feats = self.features(x) # N, C, H, W
        assert feats.size(2) == 24
        assert feats.size(-1) == 8
        assert feats.size(2) % self.num_stripes == 0
        
        if self.erase>0:
        #    print('Random Erasing')
            erasing = RandomErasing_vertical(probability=self.erase)
            feats = erasing(feats)
        
        feat_list, logits_list = global_pcb(feats, self.global_pooling, self.global_conv, self.global_bn, 
                    self.global_relu, self.global_fc, [], [])
        feat_list, logits_list = spp_vertical(feats, self.pcb2_pool_list, self.pcb2_conv_list, 
                    self.pcb2_batchnorm_list, self.pcb2_relu_list, self.pcb2_fc_list, 2, feat_list, logits_list)
        feat_list, logits_list = spp_vertical(feats, self.pcb4_pool_list, self.pcb4_conv_list, 
                    self.pcb4_batchnorm_list, self.pcb4_relu_list, self.pcb4_fc_list, 4, feat_list, logits_list)

        feat_list, logits_list = spp_vertical(feats, self.pcb8_pool_list, self.pcb8_conv_list, 
                    self.pcb8_batchnorm_list, self.pcb8_relu_list, self.pcb8_fc_list, 8, feat_list, logits_list)
    
        if not self.training:
            return torch.cat(feat_list, dim=1)

        if self.loss == {'softmax'}:
            return logits_list
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))





def hpm(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = HPM(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        parts=6,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model 

    

def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
