"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import division, absolute_import
import torch.utils.model_zoo as model_zoo
from torch import nn
import torchvision
from torchvision.models.resnet import resnet50, Bottleneck
import random
import torch
import math
import copy
import numpy as np
from torch.nn import functional as F


__all__ = ['bfe']


def NoBiasBatchNorm1d(in_features):
    bn_layer = nn.BatchNorm1d(in_features)
    bn_layer.bias.requires_grad_(False)
    return bn_layer


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class CAM_Module(nn.Module):
    """ Channel attention module"""
    
    def __init__(self, channels, reduction=16):
        super(CAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SAM_Module(nn.Module):
    """ Position attention module"""

    def __init__(self, channels):
        super(SAM_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_after_concat = nn.Conv2d(1, 1, kernel_size = 3, stride=1, padding = 1)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        module_input = x
        avg = torch.mean(x, 1, True)
        x = self.conv_after_concat(avg)
        # x = self.relu(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x


class CBAM_Module(nn.Module):

    def __init__(self, channels, reduction=16):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid_spatial = nn.Sigmoid()
        self.k = 64

    def forward(self, x):
        # channel attention
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        x = module_input * x
        # spatial attention
        module_input = x
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        spatial_att_map = self.sigmoid_spatial(x)
        x = module_input * spatial_att_map
        # print(spatial_att_map[0])
        return x


class Batch_DropBlock(nn.Module):
    def __init__(self):
        super(Batch_DropBlock, self).__init__()
        self.h_ratio = 0.3
        self.w_ratio = 1.0

    def forward(self, x):
        h, w = x.size()[-2:]
        rh = round(self.h_ratio * h)
        rw = round(self.w_ratio * w)
        sx = random.randint(0, h - rh)
        sy = random.randint(0, w - rw)
        mask = x.new_ones(x.size())
        mask[:, :, sx:sx + rh, sy:sy + rw] = 0
        x = x * mask
        return x, mask


class SlowDropBlock(nn.Module):
    def __init__(self, h_ratio=0.3, w_ratio=1, Threshold=1):
        super(SlowDropBlock, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
        self.it = 0
        self.Threshold = Threshold
        self.sx = None
        self.sy = None

    def forward(self, x):
        h, w = x.size()[-2:]
        rh = round(self.h_ratio * h)
        rw = round(self.w_ratio * w)
        if self.it % self.Threshold == 0:
            self.sx = random.randint(0, h - rh)
            self.sy = random.randint(0, w - rw)
        self.it += 1
        mask = x.new_ones(x.size())
        mask[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw] = 0
        x = x * mask
        return x, mask


class BatchRandomErasing(nn.Module):
    def __init__(self, sl=0.25, sh=0.5, r1=0.25, mean=[0.4914, 0.4822, 0.4465], Threshold=1):
        super(BatchRandomErasing, self).__init__()

        self.it = 0
        self.Threshold = Threshold
        self.sx = None
        self.sy = None
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def forward(self, x):
        for attempt in range(100):
            h, w = x.size()[-2:]
            area = h * w
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            rh = int(round(math.sqrt(target_area * aspect_ratio)))
            rw = int(round(math.sqrt(target_area / aspect_ratio)))
            if rw < w and rh < h:
                if self.it % self.Threshold == 0:
                    self.sx = random.randint(0, h - rh)
                    self.sy = random.randint(0, w - rw)
                self.it += 1
                mask = x.new_ones([x.size()[0], 1, h, w])
                mask[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw] = 0
                x[:, 0, self.sx:self.sx + rh, self.sy:self.sy + rw] = self.mean[0]
                x[:, 1, self.sx:self.sx + rh, self.sy:self.sy + rw] = self.mean[1]
                x[:, 2, self.sx:self.sx + rh, self.sy:self.sy + rw] = self.mean[2]
                return x, mask


class RandomErasing(nn.Module):
    def __init__(self, sl=0.25, sh=0.5, r1=0.5, mean=[0.4914, 0.4822, 0.4465]):
        super(RandomErasing, self).__init__()
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    # img 32,3,384,128
    def forward(self, img, epoch=-1):
        for i in range(img.size(0)):
            for attempt in range(100):
                area = img.size()[2] * img.size()[3]
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.size()[3] and h < img.size()[2]:
                    x1 = random.randint(0, img.size()[2] - h)
                    y1 = random.randint(0, img.size()[3] - w)
                    if img.size()[1] == 3:
                        img[i, 0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                        img[i, 1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                        img[i, 2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                        # print(img[i, 0, x1:x1 + h, y1:y1 + w])
                        break
        mask = img.new_ones(img.shape)
        return img, mask


class RandomDrop(nn.Module):
    def __init__(self, sl=0.25, sh=0.5, r1=0.5, mean=[0.4914, 0.4822, 0.4465]):
        super(RandomDrop, self).__init__()
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    # img 32,3,384,128
    def forward(self, img, epoch=-1):
        for i in range(img.size(0)):
            for attempt in range(100):
                area = img.size()[2] * img.size()[3]
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.size()[3] and h < img.size()[2]:
                    x1 = random.randint(0, img.size()[2] - h)
                    y1 = random.randint(0, img.size()[3] - w)
                    if img.size()[1] == 3:
                        img[i, :, x1:x1 + h, y1:y1 + w] = 0
                        break
        mask = img.new_ones(img.shape)
        return img, mask 


class BatchDrop(nn.Module):
    def __init__(self, sl=0.25, sh=0.5, r1=0.5, Threshold=1):
        super(BatchDrop, self).__init__()
        self.it = 0
        self.Threshold = Threshold
        self.sx = None
        self.sy = None
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def forward(self, x, epoch=-1):
        if self.training:
            h, w = x.size()[-2:]
            area = h * w
            if epoch <= 55:
                target_area = area * (0.1 + 0.4 * epoch / 90)
            else:
                target_area = area / 3
            for attempt in range(10000000000):
                if math.sqrt(target_area) <= w:
                    aspect_ratio = random.uniform(self.r1, 1)
                    rh = int(round(math.sqrt(target_area * aspect_ratio)))
                    rw = int(round(math.sqrt(target_area / aspect_ratio)))
                else:
                    rw = w
                    rh = int(round(target_area/rw))

                if rw <= w and rh <= h:
                    if self.it % self.Threshold == 0:
                        self.sx = random.randint(0, h - rh)
                        self.sy = random.randint(0, w - rw)
                    self.it += 1
                    mask = x.new_ones(x.size())
                    mask[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw] = 0
                    x = x * mask
                    return x, mask
        if not self.training:
            return x


class BatchPatch(nn.Module):
    def __init__(self, sl=0.25, sh=0.5, r1=0.5, mean=[0.4914, 0.4822, 0.4465], Threshold=1):
        super(BatchPatch, self).__init__()
        self.it = 0
        self.Threshold = Threshold
        self.sx = None
        self.sy = None
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.epochm = 55

    def forward(self, x, epoch=-1, training=False):
        if training:
            h, w = x.size()[-2:]
            area = h * w
            if epoch <= self.epochm:
                target_area = area * (0.1 + 0.4 * epoch / 90)
            else:
                target_area = area / 3

            for attempt in range(10000000000):
                if math.sqrt(target_area) <= w:
                    aspect_ratio = random.uniform(self.r1, 1)
                    rh = int(round(math.sqrt(target_area * aspect_ratio)))
                    rw = int(round(math.sqrt(target_area / aspect_ratio)))
                else:
                    rw = w
                    rh = int(round(target_area/rw))

                if rw <= w and rh <= h:
                    if self.it % self.Threshold == 0:
                        if epoch <= self.epochm:
                            self.sx = random.randint(0, h - rh)
                            self.sy = random.randint(0, w - rw)
                        else:
                            self.sx = random.sample([0, 64, 128, 192, 256, 320], 1)[0]
                            self.sy = 0
                    self.it += 1
                    index = torch.randperm(64).cuda()
                    mask = x.new_ones([x.size()[0], 1, h, w])
                    mask[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw] = 0
                    x[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw] = \
                        x[index, :, self.sx:self.sx + rh, self.sy:self.sy + rw]
                    break
            return x, mask, (self.sx, self.sx+rh, self.sy, self.sy+rw)
        if not training:
            return x


class BatchCrop(nn.Module):
    def __init__(self, sl=0.25, sh=0.75, r1=0.25, Threshold=1):
        super(BatchCrop, self).__init__()
        self.sl = sl
        self.sh = sh
        self.r1 = 0.5
        self.r2 = 10
        self.it = 0
        self.Threshold = 2
        self.sx = None
        self.sy = None
        self.epoch_m = 55
        self.p = 3

    def forward(self, x, epoch=-1):
        if self.training:
            h, w = x.size()[-2:]
            area = h * w
            if epoch <= self.epoch_m:
                target_area = area * (0.5 - 0.3 * epoch / 90)
            else:
                target_area = area / self.p
            for attempt in range(1000):
                # target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, self.r2)
                rh = int(round(math.sqrt(target_area * aspect_ratio)))
                rw = int(round(math.sqrt(target_area / aspect_ratio)))
                if rw <= w and rh <= h:
                    self.sx = random.randint(0, h - rh)
                    if self.it % self.Threshold == 0:
                        self.sy = random.randint(0, w - rw)
                    else:
                        self.sy = int(w/2)-int(rw/2)
                    self.it += 1

                    x_crop = x[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw]
                    # x_crop = x[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw]
                    # print(self.sx, self.sx + rh, self.sy-rw, self.sy + rw)
                    x = F.interpolate(x_crop, (384, 128), mode='bilinear', align_corners=True)
                    return x


class BatchNewCrop(nn.Module):
    def __init__(self, sl=0.25, sh=0.75, r1=0.25, Threshold=1):
        super(BatchNewCrop, self).__init__()
        self.sl = sl
        self.sh = sh
        self.r1 = 0.5
        self.r2 = 10
        self.it = 0
        self.Threshold = 2
        self.sx = None
        self.sy = None
        self.epoch_m = 55
        self.p = 3

    def forward(self, x, epoch=-1):
        h, w = x.size()[-2:]
        area = h * w
        if epoch <= self.epoch_m:
            target_area = area * (0.5 - 0.3 * epoch / 90)
        else:
            target_area = area / self.p
        for attempt in range(1000):
            # target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, self.r2)
            rh = int(round(math.sqrt(target_area * aspect_ratio)))
            rw = int(round(math.sqrt(target_area / aspect_ratio)))
            if rw <= w and rh <= h:
                self.sx = random.randint(0, h - rh)
                # self.sy = random.randint(0, w - rw)
                self.sy = int(w/2)-int(rw/2)
                self.it += 1

                x_crop = x.new_ones(x.shape)
                x_crop[:, 0, :, :] = random.uniform(0, 1)
                x_crop[:, 1, :, :] = random.uniform(0, 1)
                x_crop[:, 2, :, :] = random.uniform(0, 1)

                mask = x.new_zeros([x.shape[0], 1, h, w])
                mask[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw] = 1
                x_crop[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw] = \
                    x[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw]
                # print(self.sx, self.sx + rh, self.sy-rw, self.sy + rw)
                return x_crop, mask, (self.sx, self.sx + rh, self.sy, self.sy + rw)


class BatchIncErasing(nn.Module):
    def __init__(self, sl=0.25, sh=0.333, Threshold=1):
        super(BatchIncErasing, self).__init__()
        self.it = 0
        self.Threshold = Threshold
        self.sx = None
        self.sy = None
        self.sl = sl
        self.sh = sh
        self.r1 = 0.5
        self.r2 = 1
        self.epochm = 55
        self.p = 1/3  # Oculusionreid p=1/3

    def forward(self, x, epoch=-1):
        h, w = x.size()[-2:]
        area = h * w

        if epoch <= self.epochm:
            target_area = area * (0.1 + 0.4 * epoch / 90)  # easy-to-hard
            # target_area = area * (0.5 - 0.4 * epoch / 90)
        else:
            target_area = area * self.p

        # target_area = random.uniform(self.sl, self.sh) * area
        # target_area = 0

        for attempt in range(1000000):
            if math.sqrt(target_area) <= w:
                aspect_ratio = random.uniform(self.r1, self.r2)
                rh = int(round(math.sqrt(target_area * aspect_ratio)))
                rw = int(round(math.sqrt(target_area / aspect_ratio)))
            else:
                rw = w
                rh = int(round(target_area/rw))

            if rw <= w and rh <= h:
                if self.it % self.Threshold == 0:
                    self.sx = random.randint(0, h - rh)
                    self.sy = random.randint(0, w - rw)
                self.it += 1
                mask = x.new_ones([x.size()[0], 1, h, w])
                mask[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw] = 0
                x[:, 0, self.sx:self.sx + rh, self.sy:self.sy + rw] = random.uniform(0, 1)
                x[:, 1, self.sx:self.sx + rh, self.sy:self.sy + rw] = random.uniform(0, 1)
                x[:, 2, self.sx:self.sx + rh, self.sy:self.sy + rw] = random.uniform(0, 1)
                # print(self.sx, self.sx+rh, self.sy, self.sy+rw)
                break
        return x, mask


class BatchNewErasing(nn.Module):
    def __init__(self, sl=0.25, sh=0.5, Threshold=1):
        super(BatchNewErasing, self).__init__()
        self.it = 0
        self.Threshold = Threshold
        self.sx = None
        self.sy = None
        self.sl = sl
        self.sh = sh
        self.epochm = 55
        self.p = 1/2  # Oculusionreid p=1/3

    def forward(self, x, epoch=-1):
        h, w = x.size()[-2:]
        area = h * w

        if epoch <= self.epochm:
            target_area = area * (0.2 + 0.3 * epoch / 55)  # occulded-reid
            # target_area = area * (0.1 + 0.4 * epoch / 90)  # occluded_duke
        else:
            target_area = area * self.p

        # 上下左右遮挡法
        mode = random.randint(0, 3)

        # 上
        if mode == 0:
            rw = int(round(math.sqrt(target_area / 1.5)))
            rh = int(round(target_area / rw))
            self.sx = random.randint(0, h / 2 - rh)
            self.sy = random.randint(0, w - rw)
        # 下
        if mode == 1:
            rw = int(round(math.sqrt(target_area / 1.5)))
            rh = int(round(target_area / rw))
            self.sx = random.randint(h / 2, h - rh)
            self.sy = random.randint(0, w - rw)
        # 左
        if mode == 2:
            rw = int(round(math.sqrt(target_area / 6)))
            rh = int(round(target_area / rw))
            self.sx = random.randint(0, h - rh)
            self.sy = random.randint(0, w / 2 - rw)
        # 右
        if mode == 3:
            rw = int(round(math.sqrt(target_area / 6)))
            rh = int(round(target_area / rw))
            self.sx = random.randint(0, h - rh)
            self.sy = random.randint(w / 2, w - rw)

        mask = x.new_ones([x.size()[0], 1, h, w])
        mask[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw] = 0
        x[:, 0, self.sx:self.sx + rh, self.sy:self.sy + rw] = random.uniform(0, 1)
        x[:, 1, self.sx:self.sx + rh, self.sy:self.sy + rw] = random.uniform(0, 1)
        x[:, 2, self.sx:self.sx + rh, self.sy:self.sy + rw] = random.uniform(0, 1)
        # print(self.sx, self.sx+rh, self.sy, self.sy+rw)
        return x, mask, (self.sx, self.sx+rh, self.sy, self.sy+rw)


class ResNet(nn.Module):

    def __init__(self, num_classes, fc_dims=None, loss=None, dropout_p=None,  **kwargs):
        super(ResNet, self).__init__()
        resnet_ = resnet50(pretrained=True)
        self.loss = loss
        self.layer0 = nn.Sequential(
            resnet_.conv1,
            resnet_.bn1,
            resnet_.relu,
            resnet_.maxpool)
        self.layer1 = resnet_.layer1
        self.layer2 = resnet_.layer2
        self.layer3 = resnet_.layer3

        layer4 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        layer4.load_state_dict(resnet_.layer4.state_dict())

        self.layer41 = nn.Sequential(copy.deepcopy(layer4))
        self.layer42 = nn.Sequential(copy.deepcopy(layer4))

        self.res_part1 = Bottleneck(2048, 512)
        self.res_part2 = Bottleneck(2048, 512)

        self.att1 = CBAM_Module(1024)
        self.att_module2 = CBAM_Module(2048)

        self.batch_drop = BatchDrop()
        self.batch_increrase = BatchIncErasing()
        self.batch_new_erase = BatchNewErasing()
        self.batch_crop = BatchCrop()
        self.batch_new_crop = BatchNewCrop()
        self.batch_patch = BatchPatch()
        self.batch_dropblock = Batch_DropBlock()
        self.slow_dropblock = SlowDropBlock()
        self.batch_erase = BatchRandomErasing()

        self.random_erase = RandomErasing()
        self.random_drop = RandomDrop()

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.75)

        self.reduction1 = nn.Sequential(
            nn.Linear(2048, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.reduction1.apply(weights_init_kaiming)

        self.reduction2 = nn.Sequential(
            nn.Linear(2048, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.reduction2.apply(weights_init_kaiming)

        self.classifier1 = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)
        self.classifier3 = nn.Linear(1024, num_classes)
        self.classifier4 = nn.Linear(512, num_classes)

        nn.init.normal_(self.classifier1.weight, 0, 0.01)
        if self.classifier1.bias is not None:
            nn.init.constant_(self.classifier1.bias, 0)
        nn.init.normal_(self.classifier2.weight, 0, 0.01)
        if self.classifier2.bias is not None:
            nn.init.constant_(self.classifier2.bias, 0)
        nn.init.normal_(self.classifier3.weight, 0, 0.01)
        if self.classifier3.bias is not None:
            nn.init.constant_(self.classifier3.bias, 0)

    def featuremaps(self, x, epoch=-1, batch_idx=-1):
        if self.training:
            b = x.size(0)
            # x1 = x[:b//2, :, :, :]
            # x2 = x[b//2:, :, :, :]
            # x2 = self.batch_crop(x2)
            # x = torch.cat([x1, x2], 0)
            x_1 = x.clone()
            final_x_1, mask = self.batch_increrase(x_1)
            x = torch.cat([x, final_x_1], 0)

            # final_x_1, mask = self.batch_erase(x_1)
            # x = torch.cat([final_x_1, x], 0)

        x = self.layer0(x)   # 64, 96, 32
        x = self.layer1(x)   # 256, 96, 32
        x = self.layer2(x)   # 512, 48, 16
        x = self.layer3(x)   # 1024, 24, 8

        if self.training:
            x_1 = x[:b, :, :, :]
            x_2 = x[b:, :, :, :]
            x_1 = self.layer41(x_1)
            x_2 = self.att1(x_2)
            x_2 = self.layer42(x_2)
        else:
            x_1 = self.layer41(x)
            x_2 = self.att1(x)
            x_2 = self.layer42(x_2)
        
        if self.training:
            return x_1, x_2, mask  # , coord

        return x_1, x_2

    def forward(self, x, epoch=-1, batch_idx=-1, return_featuremaps=False):
        w1, w2 = 1, 1
        if self.training:

            # f1, f2, mask, coord = self.featuremaps(x, epoch, batch_idx)
            f1, f2, mask = self.featuremaps(x, epoch, batch_idx)

            f1 = self.res_part1(f1)
            f2 = self.res_part2(f2)
            f2 = self.att_module2(f2)

            # supervised att
            ture_mask = nn.functional.interpolate(
                mask, (f2.size(2), f2.size(3)),
                mode='bilinear',
                align_corners=True
            )

            pred_f1 = torch.mean(f1, 1, True)
            pred_f2 = torch.mean(f2, 1, True)
            ture_f1 = pred_f1*ture_mask
            ture_f2 = pred_f2*ture_mask

            if return_featuremaps:
                return f2

            v1 = self.global_avgpool(f1)
            v1 = v1.view(v1.size(0), -1)
            v1_1 = self.reduction1(v1)  # 512

            v2 = self.global_maxpool(f2)
            v2 = v2.view(v2.size(0), -1)
            v2_1 = self.reduction2(v2)
            
            v3_1 = torch.cat([v1_1, v2_1], 1)

            y1 = self.classifier1(v1_1)
            y2 = self.classifier2(v2_1)
            y3 = self.classifier3(v3_1)

            if self.loss == 'softmax':
                return y1, y2, y3, pred_f1, ture_f1, pred_f2, ture_f2
            elif self.loss == 'triplet':
                return y1, y2, y3, fea
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))

        if not self.training:
            f1, f2 = self.featuremaps(x)

            f1 = self.res_part1(f1)
            f2 = self.res_part2(f2)
            f2 = self.att_module2(f2)

            v1 = self.global_avgpool(f1)
            v1 = v1.view(v1.size(0), -1)
            v1_1 = self.reduction1(v1)  # 512

            if return_featuremaps:
                return f2

            v2 = self.global_maxpool(f2)
            v2 = v2.view(v2.size(0), -1)
            v2_1 = self.reduction2(v2)

            v1_1 = F.normalize(v1_1, p=2, dim=1)
            v2_1 = F.normalize(v2_1, p=2, dim=1)

            return torch.cat([v1_1, v2_1], 1)


# ResNet
def bfe(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        fc_dims=None,
        loss=loss,
        dropout_p=None,
        **kwargs
    )
    return model


