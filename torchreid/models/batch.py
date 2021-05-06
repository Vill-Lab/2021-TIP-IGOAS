import random
from torch import nn
import math

class BatchCrop(nn.Module):
    def __init__(self, p = 0.5, sl=0.25, sh=0.75, r1=0.25, Threshold=1):
        super(BatchCrop, self).__init__()
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.it = 0
        self.Threshold = Threshold
        self.sx = None
        self.sy = None

    def forward(self, x):       

        if self.training:
            # if random.uniform(0, 1) > self.p:
            #     return x
            for attempt in range(100):
                h, w = x.size()[-2:]
                area = h * w
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
                rh = int(round(math.sqrt(target_area * aspect_ratio)))
                rw = int(round(math.sqrt(target_area / aspect_ratio)))
                if rw < w and rh < h:
                    if self.it % self.Threshold == 0:
                        self.sx = random.randint(0, h - rh)
                        self.sy = random.randint(0, w - rw)
                    self.it += 1
                    x_crop = x[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw]
                    x = F.interpolate(x_crop, (384,128), mode='bilinear',align_corners= True)
                    return x
        return x


# class BatchDrop(nn.Module):
#     def __init__(self, h_ratio=0.3, w_ratio=1, Threshold=1):
#         super(BatchDrop, self).__init__()
#         self.h_ratio = h_ratio
#         self.w_ratio = w_ratio
#         self.it = 0
#         self.Threshold = Threshold
#         self.sx = None
#         self.sy = None

#     def forward(self, x):
#         if self.training:
#             h, w = x.size()[-2:]
#             rh = round(self.h_ratio * h)
#             rw = round(self.w_ratio * w)
#             if self.it % self.Threshold == 0:
#                 self.sx = random.randint(0, h - rh)
#                 self.sy = random.randint(0, w - rw)
#             self.it += 1
#             mask = x.new_ones(x.size())
#             mask[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw] = 0
#             x = x * mask
#             return x
#         return x


class BatchDrop(nn.Module):
    def __init__(self, sl=0.2, sh=0.5, r1=0.25, Threshold=1):
        super(BatchDrop, self).__init__()
        self.it = 0
        self.Threshold = Threshold
        self.sx = None
        self.sy = None
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def forward(self, x):
        if self.training:
            for attempt in range(100):
                h, w = x.size()[-2:]
                area = h * w
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)              
                
                rh = int(round(math.sqrt(target_area * aspect_ratio)))
                rw = int(round(math.sqrt(target_area / aspect_ratio)))
                if rw < w and rh < h:
                    if self.it % self.Threshold == 0:
                        self.sx = random.randint(0, h - rh)
                        self.sy = random.randint(0, w - rw)
                    self.it += 1
                    mask = x.new_ones(x.size())
                    mask[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw] = 0
                    x = x * mask
                    return x
        return x


class BatchErasing(nn.Module):
    def __init__(self, sl=0.2, sh=0.5, r1=0.25, mean=[0.4914, 0.4822, 0.4465], Threshold=1):
        super(BatchErasing, self).__init__()
        
        self.it = 0
        self.Threshold = Threshold
        self.sx = None
        self.sy = None
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def forward(self, x):
        if self.training:
            
            for attempt in range(100):
                h, w = x.size()[-2:]
                area = h * w
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)              
                
                rh = int(round(math.sqrt(target_area * aspect_ratio)))
                rw = int(round(math.sqrt(target_area / aspect_ratio)))
                if rw < w and rh < h:
                    if self.it % self.Threshold == 0:
                        self.sx = random.randint(0, h - rh)
                        self.sy = random.randint(0, w - rw)
                    self.it += 1
                    x[:, 0, self.sx:self.sx + rh, self.sy:self.sy + rw] = self.mean[0]
                    x[:, 1, self.sx:self.sx + rh, self.sy:self.sy + rw] = self.mean[1]
                    x[:, 2, self.sx:self.sx + rh, self.sy:self.sy + rw] = self.mean[2]
                    return x
        return x


# class BatchChange(nn.Module):
#     def __init__(self, choose, sl=0.25, sh=0.75, r1=0.25, mean=[0.4914, 0.4822, 0.4465], Threshold=1):
#         super(BatchChange, self).__init__()
#         self.it = 0
#         self.Threshold = Threshold
#         self.sx = None
#         self.sy = None
#         self.mean = mean
#         self.sl = sl
#         self.sh = sh
#         self.r1 = r1
#         self.choose = choose

#     def forward(self, x):
#         if self.training:
#             h, w = x.size()[-2:]
#             area = h * w
#             target_area = random.uniform(self.sl, self.sh) * area
#             aspect_ratio = random.uniform(self.r1, 1/self.r1)            
#             for attempt in range(100):              
#                 rh = int(round(math.sqrt(target_area * aspect_ratio)))
#                 rw = int(round(math.sqrt(target_area / aspect_ratio)))
#                 if rw < w and rh < h:
#                     if self.it % self.Threshold == 0:
#                         self.sx = random.randint(0, h - rh)
#                         self.sy = random.randint(0, w - rw)
#                     self.it += 1

#                     if self.choose == 0:
#                         # print(self.choose)
#                         mask = x.new_zeros(x.size())
#                         mask[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw] = 1
#                         x = x * mask
#                         return x

#                     if self.choose == 1:
#                         # print(self.choose)
#                         x[:, 0, 0:self.sx , :] = self.mean[0]
#                         x[:, 0, self.sx + rh:h , :] = self.mean[0]
#                         x[:, 0, self.sx :self.sx+rh , 0:self.sy] = self.mean[0]
#                         x[:, 0, self.sx :self.sx+rh , self.sy+rw:w] = self.mean[0]

#                         x[:, 1, 0:self.sx , :] = self.mean[1]
#                         x[:, 1, self.sx + rh:h , :] = self.mean[1]
#                         x[:, 1, self.sx :self.sx+rh , 0:self.sy] = self.mean[1]
#                         x[:, 1, self.sx :self.sx+rh , self.sy+rw:w] = self.mean[1]

#                         x[:, 2, 0:self.sx , :] = self.mean[2]
#                         x[:, 2, self.sx + rh:h , :] = self.mean[2]
#                         x[:, 2, self.sx :self.sx+rh , 0:self.sy] = self.mean[2]
#                         x[:, 2, self.sx :self.sx+rh , self.sy+rw:w] = self.mean[2]
#                         return x              
#         return x


class RandomErasing(nn.Module):
    def __init__(self, sl=0.1, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        super(RandomErasing, self).__init__()
        # self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    # img 32,3,384,128
    def forward(self, img):
        # if random.uniform(0, 1) > self.probability:
        #     return img
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
        return img

class RandomDrop(nn.Module):
    def __init__(self, sl=0.1, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        super(RandomDrop, self).__init__()
        # self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    # img 32,3,384,128
    def forward(self, img):
        # if random.uniform(0, 1) > self.probability:
        #     return img
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
        return img