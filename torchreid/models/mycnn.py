from __future__ import division, absolute_import
import torch.utils.model_zoo as model_zoo
import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F

__all__ = ['mycnn']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


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
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
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
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, 1, stride=1, padding=0, bias=False
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))

        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class STN(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""

    def __init__(self):
        super(STN, self).__init__()
        # localization net
        # self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False) # size : [1x3x32x32]
        self.fc_stn = nn.Linear(1024 * 24 * 8, 3 * 2)
        self.init_params()

    # last fc weight and bias
    def init_params(self):
        self.fc_stn.weight.data.zero_()
        self.fc_stn.bias.data.copy_(
            torch.tensor(
                [0.8, 0, 0, 0, 0.8, 0], dtype=torch.float
            )
        )

    def forward(self, x4, x2):
        # squeeze operation (global average pooling)
        x4 = x4.view(x4.size(0), -1)

        # predict transformation parameters
        # theta = self.fc(x)
        theta = torch.tanh(self.fc_stn(x4))
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x2.size())
        x = F.grid_sample(x2, grid)
        return x


class PCB(nn.Module):
    """Part-based Convolutional Baseline.
    Reference:
        Sun et al. Beyond Part Models: Person Retrieval with Refined
        Part Pooling (and A Strong Convolutional Baseline). ECCV 2018.
    Public keys:
        - ``pcb_p4``: PCB with 4-part strips.
        - ``pcb_p6``: PCB with 6-part strips.
    """

    def __init__(
            self,
            num_classes,
            loss,
            block,
            layers,
            parts=6,
            reduced_dim=256,
            fc_dims=None,
            dropout_p=None,
            nonlinear='relu',
            **kwargs
    ):
        self.inplanes = 64
        super(PCB, self).__init__()
        self.loss = loss
        self.parts = parts
        self.feature_dim = 512 * block.expansion

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # base branch network
        self.layer1 = self._make_layer(block, 64, layers[0])  # x2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # x4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self._construct_fc_layer(fc_dims, 512 * block.expansion, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)



        # align branch network
        self.stn = STN()
        alignet = models.resnet50(pretrained=True)
        # Modifiy the stride of last conv layer
        alignet.layer4[0].conv2 = nn.Conv2d(
            512, 512, kernel_size=3, bias=False, stride=1, padding=1)
        alignet.layer4[0].downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048))

        # Remove layer of resnet
        modules = list(alignet.children())[5:-2]
        self.align_net = nn.Sequential(*modules)

        # pcb layers
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.conv5 = DimReduceLayer(
            512 * block.expansion, reduced_dim, nonlinear=nonlinear
        )

        self.feature_dim = reduced_dim

        self.classifier_align = nn.ModuleList(
            [
                nn.Linear(self.feature_dim, num_classes)
                for _ in range(self.parts)
            ]
        )

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    
    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        
        self.feature_dim = fc_dims[-1]
        
        return nn.Sequential(*layers)


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
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
        x2 = self.layer1(x)  # ([1, 256, 96, 32])
        x = self.layer2(x2)
        x4 = self.layer3(x)
        x = self.layer4(x4)
        return x, x4, x2


    def forward(self, x):
        f, x4, x2 = self.featuremaps(x)
        x_align = self.stn(x4, x2)  # ([1, 256, 96, 32])
        x_align = F.upsample(
            x_align, (x2.size(2), x2.size(3)), mode='bilinear', align_corners=True
        )
        x_align = self.align_net(x_align)
        v_g = self.global_avgpool(f)
        v_g = v_g.view(v_g.size(0), -1)

        if self.fc is not None:
            v_g = self.fc(v_g)  # 32,512

        v_align = self.parts_avgpool(x_align)
        v_align = self.dropout(v_align)
        v_align = self.conv5(v_align)

        y = []
        v_align_list = []

        y_g = self.classifier(v_g)
        y.append(y_g)

        for i in range(self.parts):
            v_align_i = v_align[:, :, i, :]
            v_align_i = v_align_i.view(v_align_i.size(0), -1)
            v_align_list.append(v_align_i)
            y_i = self.classifier_align[i](v_align_i)
            y.append(y_i)

        if not self.training:
            v_g = F.normalize(v_g, p=2, dim=1) # 1,512
            v_align = torch.cat(v_align_list,1)
            v_align = F.normalize(v_align, p=2, dim=1)
            return torch.cat([v_g, v_align], 1)  # daigai
            # return v_align

        if self.loss == 'softmax':
            return y
        # elif self.loss == 'triplet':
        #     v_g = F.normalize(v_g, p=2, dim=1)
        #     return y, v_g.view(v_g.size(0), -1)
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def mycnn(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PCB(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        parts=6,
        reduced_dim=256,
        fc_dims=[256],
        dropout_p=0.5,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model
