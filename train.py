import os
import fire
import time
import collections
import numpy as np
import os.path as osp
from tqdm import tqdm

from configs.csm_dronecc import opt

from models import CSMNet as Model
from models.losses import build_loss
from dataloaders import make_data_loader
from models.utils import Evaluator

from utils import (Saver, Timer, TensorboardSummary,
                   calculate_weigths_labels)
import torch
import torch.optim as optim
# import multiprocessing
# multiprocessing.set_start_method('spawn', True)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)


class Trainer(object):
    def __init__(self, mode):
        # Define Saver
        self.saver = Saver(opt, mode)
        self.logger = self.saver.logger

        # visualize
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Dataset dataloader
        self.train_dataset, self.train_loader = make_data_loader(opt)
        self.nbatch_train = len(self.train_loader)
        self.val_dataset, self.val_loader = make_data_loader(opt, mode="val")
        self.nbatch_val = len(self.val_loader)

        # model
        model = Model(opt)
        self.model = model.to(opt.device)

        # Loss
        if opt.use_balanced_weights:
            classes_weights_file = osp.join(opt.root_dir, 'train_classes_weights.npy')
            if os.path.isfile(classes_weights_file):
                weight = np.load(classes_weights_file)
            else:
                weight = calculate_weigths_labels(
                    self.train_loader, opt.root_dir)
            print(weight)
            opt.loss_region['weight'] = weight
        self.loss_region = build_loss(opt.loss_region)
        self.loss_density = build_loss(opt.loss_density)

        # Define Evaluator
        self.evaluator = Evaluator(dataset=opt.dataset)  # use region to eval: class_num is 2

        # Resuming Checkpoint
        self.best_pred = float('inf')
        self.start_epoch = 0
        if opt.resume:
            if os.path.isfile(opt.pre):
                print("=> loading checkpoint '{}'".format(opt.pre))
                checkpoint = torch.load(opt.pre)
                self.start_epoch = checkpoint['epoch']
                self.best_pred = checkpoint['best_pred']
                self.model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(opt.pre, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(opt.pre))

        if len(opt.gpu_id) > 1:
            self.logger.info("Using multiple gpu")
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=opt.gpu_id)

        # Define Optimizer and Lr Scheduler
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=opt.lr,
                                         momentum=opt.momentum,
                                         weight_decay=opt.decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[round(opt.epochs * x) for x in opt.steps],
            gamma=opt.gamma)

        # Time
        self.loss_hist = collections.deque(maxlen=500)
        self.timer = Timer(opt.epochs, self.nbatch_train, self.nbatch_val)
        self.step_time = collections.deque(maxlen=opt.print_freq)

    def train(self, epoch):
        self.model.train()
        if opt.freeze_bn:
            self.model.module.freeze_bn() if len(opt.gpu_id) > 1 \
                else self.model.freeze_bn()
        last_time = time.time()
        epoch_loss = []
        for iter_num, sample in enumerate(self.train_loader):
            # if iter_num >= 0: break
            try:
                imgs = sample["image"].to(opt.device)
                density_gt = sample["label"].to(opt.device)
                region_gt = (sample["label"] > 0).float().to(opt.device)

                region_pred, density_pred = self.model(imgs)

                region_loss = self.loss_region(region_pred, region_gt)
                density_loss = self.loss_density(density_pred, density_gt)
                loss = region_loss + density_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.loss_hist.append(float(loss))
                epoch_loss.append(float(loss.cpu().item()))

                self.optimizer.step()
                self.optimizer.zero_grad()
                # self.scheduler(self.optimizer, iter_num, epoch)

                # Visualize
                global_step = iter_num + self.nbatch_train * epoch + 1
                self.writer.add_scalar('train/loss', loss.cpu().item(), global_step) 
                batch_time = time.time() - last_time
                last_time = time.time()
                eta = self.timer.eta(global_step, batch_time)
                self.step_time.append(batch_time)
                if global_step % opt.print_freq == 0:
                    printline = ('Epoch: [{}][{}/{}] '
                                 'lr: {:1.5f}, '  # 10x:{:1.5f}), '
                                 'eta: {}, time: {:1.1f}, '
                                 'region loss: {:1.4f}, '
                                 'density loss: {:1.4f}, '
                                 'loss: {:1.4f}').format(
                                    epoch, iter_num+1, self.nbatch_train,
                                    self.optimizer.param_groups[0]['lr'],
                                    # self.optimizer.param_groups[1]['lr'],
                                    eta, np.sum(self.step_time),
                                    region_loss, density_loss,
                                    np.mean(self.loss_hist))
                    self.logger.info(printline)

                del loss, region_loss, density_loss

            except Exception as e:
                print(e)
                continue

        self.scheduler.step()

    def validate(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        SMAE = 0
        with torch.no_grad():
            tbar = tqdm(self.val_loader, desc='\r')
            for i, sample in enumerate(tbar):
                # if i > 3: break
                imgs = sample['image'].to(opt.device)
                density_gt = sample["label"].to(opt.device)
                region_gt = (sample["label"] > 0).float()
                path = sample["path"]

                region_pred, density_pred = self.model(imgs)

                # Visualize
                global_step = i + self.nbatch_val * epoch + 1
                if global_step % opt.plot_every == 0:
                    # pred = output.data.cpu().numpy()
                    pred = torch.argmax(region_pred, dim=1)
                    self.summary.visualize_image(self.writer,
                                                 opt.dataset,
                                                 imgs,
                                                 density_gt,
                                                 pred,
                                                 global_step)

                # metrics
                target = region_gt.numpy()
                pred = region_pred.data.cpu().numpy()
                pred = np.argmax(pred, axis=1).reshape(target.shape)
                self.evaluator.add_batch(target, pred, path)
                # density_pred = density_pred.clamp(min=0.0) * region_pred.argmax(1, keepdim=True)
                density_pred = density_pred.clamp(min=0.0)
                SMAE += (density_gt.sum() - density_pred.sum()).abs().item()

            # Fast test during the training
            MAE = SMAE / (len(self.val_dataset) * opt.norm_cfg['para'])
            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            result = MAE
            titles = ["mIoU", "MAE", "Acc", "Acc_class", "fwIoU", "Result"]
            values = [mIoU, MAE, Acc, Acc_class, FWIoU, result]
            for title, value in zip(titles, values):
                self.writer.add_scalar('val/'+title, value, epoch)

            printline = ("Val: mIoU: {:.4f}, MAE: {:.4f}, Acc: {:.4f}, "
                         "Acc_class: {:.4f}, fwIoU: {:.4f}, Result: {:.4f}]").format(*values)
            self.logger.info(printline)

        return result


def train(**kwargs):
    start_time = time.time()
    opt._parse(kwargs)
    trainer = Trainer(mode="train")

    trainer.logger.info('Num training images: {}'.format(len(trainer.train_dataset)))

    for epoch in range(trainer.start_epoch, opt.epochs):
        # train
        trainer.train(epoch)

        # val
        val_time = time.time()
        pred = trainer.validate(epoch)
        trainer.timer.set_val_eta(epoch, time.time() - val_time)

        trainer.logger.info("Val[New pred: {:1.4f}, previous best: {:1.4f}]".format(
            pred, trainer.best_pred
        ))
        is_best = pred < trainer.best_pred
        trainer.best_pred = min(pred, trainer.best_pred)
        if (epoch % 20 == 0 and epoch != 0) or is_best:
            trainer.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': trainer.model.module.state_dict() if len(opt.gpu_id) > 1
                else trainer.model.state_dict(),
                'best_pred': trainer.best_pred,
                'optimizer': trainer.optimizer.state_dict(),
            }, is_best)

    all_time = trainer.timer.second2hour(time.time() - start_time)
    trainer.logger.info("Train done!, Sum time: {}, Best result: {}".format(all_time, trainer.best_pred))

    # cache result
    print("Backup result...")
    trainer.saver.backup_result()
    print("Done!")


if __name__ == '__main__':
    # train()
    fire.Fire(train)
