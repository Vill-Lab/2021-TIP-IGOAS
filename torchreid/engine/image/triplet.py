from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

import torch

import torchreid
from torchreid.engine import engine
from torchreid.losses import CrossEntropyLoss, TripletLoss, HctLoss
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid import metrics


class ImageTripletEngine(engine.Engine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::

        import torch
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
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
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(self, datamanager, model, optimizer, margin=0.3,
                 weight_t=0.0001, weight_x=1.0, scheduler=None, use_gpu=True,
                 label_smooth=True):
        super(ImageTripletEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.weight_t = weight_t
        self.weight_x = weight_x

        # self.criterion_m = torch.nn.MSELoss()
        self.criterion_t = TripletLoss(margin=margin)
        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )


    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        
        losses = AverageMeter()
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses3 = AverageMeter()
        accs1 = AverageMeter()
        accs2 = AverageMeter()
        accs3 = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch + 1, fixbase_epoch))
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
            output1, output2, output3, fea1, fea2, fea3 = self.model(imgs)

            loss_x1 = self._compute_loss(self.criterion, output1, pids)
            loss_x2 = self._compute_loss(self.criterion, output2, pids)
            loss_x3 = self._compute_loss(self.criterion, output3, pids)

            loss_t1 = self._compute_loss(self.criterion_t, fea1, pids)
            loss_t2 = self._compute_loss(self.criterion_t, fea2, pids)
            loss_t3 = self._compute_loss(self.criterion_t, fea3, pids)

            loss1 = loss_x1 + loss_t1
            loss2 = loss_x2 + loss_t2
            loss3 = loss_x3 + loss_t3

            loss = 1.0 * loss1 + 1.0 * loss2 + 1.0 * loss3



            # loss_m1 = self._compute_loss(self.criterion_m, fea1[0], fea2[0])
            # loss_m2 = self._compute_loss(self.criterion_m, fea1[1], fea2[1])
            # loss_m3 = self._compute_loss(self.criterion_m, fea1[2], fea2[2])
            # loss_m4 = self._compute_loss(self.criterion_m, fea1[3], fea2[3])
            # loss_m = (loss_m1 + loss_m2 + loss_m3 + loss_m4) / 4

            # loss = loss_x + loss_t + loss_m

            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            losses.update(loss.item(), pids.size(0))
            losses1.update(loss1.item(), pids.size(0))
            losses2.update(loss2.item(), pids.size(0))
            losses3.update(loss3.item(), pids.size(0))

            if (batch_idx + 1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (num_batches - (batch_idx + 1) + (max_epoch - (epoch + 1)) * num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                      'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                      'Loss3 {loss3.val:.4f} ({loss3.avg:.4f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                    epoch + 1, max_epoch, batch_idx + 1, num_batches,
                    loss=losses,
                    loss1=losses1,
                    loss2=losses2,
                    loss3=losses3,
                    lr=self.optimizer.param_groups[0]['lr'],
                    eta=eta_str
                )
                )

            if self.writer is not None:
                n_iter = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Loss', losses.avg, n_iter)
                self.writer.add_scalar('Train/Loss1', losses1.avg, n_iter)
                self.writer.add_scalar('Train/Loss2', losses2.avg, n_iter)
                self.writer.add_scalar('Train/Loss3', losses3.avg, n_iter)
                self.writer.add_scalar('Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter)

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()
