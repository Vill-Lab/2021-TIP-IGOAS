from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime
import torch
import torch.nn.functional as F

from torchreid.engine import Engine
from torchreid.losses import CrossEntropyLoss
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid import metrics


class ImageSoftmaxEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(self, datamanager, model, optimizer, scheduler=None, use_gpu=True,
                 label_smooth=True):
        super(ImageSoftmaxEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)
        
        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion_mask = torch.nn.MSELoss(reduction='mean')
        self.criterion_L1 = torch.nn.L1Loss(reduction='mean')
        
    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        losses = AverageMeter()
        losses_x1 = AverageMeter()
        losses_x2 = AverageMeter()
        losses_x3 = AverageMeter()
        losses_mask = AverageMeter()
        losses_max = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_w1 = AverageMeter()
        loss_w2 = AverageMeter()
        loss_w3 = AverageMeter()

        self.model.train()
        if (epoch+1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch+1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(trainloader)
        end = time.time()
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
            
            self.optimizer.zero_grad()
            # print(batch_idx)
            output1, output2, output3, pred_f1, ture_f1, pred_f2, ture_f2 = self.model(imgs, epoch, batch_idx)  #

            loss_x1 = self._compute_loss(self.criterion, output1, pids)
            loss_x2 = self._compute_loss(self.criterion, output2, pids)
            loss_x3 = self._compute_loss(self.criterion, output3, pids)

            w_mask = 1
            w_max = 0
            loss_mask = self._compute_loss(self.criterion_mask, pred_f1, ture_f1) \
                        + self._compute_loss(self.criterion_mask, pred_f2, ture_f2)

            loss_max = loss_mask

            loss_x = torch.stack([loss_x1, loss_x2, loss_x3], dim=0)
            loss_w = F.softmax(loss_x, dim=-1)
            loss = torch.sum(loss_w * loss_x) + w_mask*loss_mask + w_max*loss_max
            # loss = (loss_x1 + loss_x2 + loss_x3)/3 + w_mask*loss_mask + w_max*loss_max
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            losses.update(loss.item(), pids.size(0))
            losses_x1.update(loss_x1.item(), pids.size(0))
            losses_x2.update(loss_x2.item(), pids.size(0))
            losses_x3.update(loss_x3.item(), pids.size(0))
            losses_mask.update(loss_mask.item(), pids.size(0))
            losses_max.update(loss_max.item(), pids.size(0))
            loss_w1.update(loss_w[0].item(), pids.size(0))
            loss_w2.update(loss_w[1].item(), pids.size(0))
            loss_w3.update(loss_w[2].item(), pids.size(0))

            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (num_batches - (batch_idx + 1) + (max_epoch - (epoch + 1)) * num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss_x1 {loss_x1.val:.4f} ({loss_x1.avg:.4f})\t'
                      'Loss_x2 {loss_x2.val:.4f} ({loss_x2.avg:.4f})\t'
                      'Loss_x3 {loss_x3.val:.4f} ({loss_x3.avg:.4f})\t'
                      'Loss_mask {loss_mask.val:.4f} ({loss_mask.avg:.4f})\t'
                      'Loss_max {loss_max.val:.4f} ({loss_max.avg:.4f})\t'
                      'Loss_weight {loss_w1.avg:.4f} {loss_w2.avg:.4f} {loss_w3.avg:.4f}\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                    epoch + 1, max_epoch, batch_idx + 1, num_batches,
                    loss=losses,
                    loss_x1=losses_x1,
                    loss_x2=losses_x2,
                    loss_x3=losses_x3,
                    loss_mask=losses_mask,
                    loss_max=losses_max,
                    loss_w1=loss_w1,
                    loss_w2=loss_w2,
                    loss_w3=loss_w3,
                    lr=self.optimizer.param_groups[0]['lr'],
                    eta=eta_str
                )
                )

            if self.writer is not None:
                n_iter = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Loss', losses.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x1', losses_x1.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x2', losses_x2.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x3', losses_x3.avg, n_iter)
                self.writer.add_scalar('Train/Loss_mask', losses_mask.avg, n_iter)
                self.writer.add_scalar('Train/Loss_max', losses_max.avg, n_iter)
                self.writer.add_scalar('Train/Loss_w1', loss_w1.avg, n_iter)
                self.writer.add_scalar('Train/Loss_w2', loss_w2.avg, n_iter)
                self.writer.add_scalar('Train/Loss_w3', loss_w3.avg, n_iter)
                self.writer.add_scalar('Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter)

                # self.writer.add_scalars('Train',{'loss':losses.avg,'Loss_x1':losses_x1.avg, 'Loss_x2':losses_x2.avg},n_iter)
            
            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()
