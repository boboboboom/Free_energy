from __future__ import print_function
import argparse
import os.path
import os
import logging
import time
import datetime
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from core.datasets.image_list import ImageList
from core.models.network import ResNetFc
from core.active.active import EADA_active, RAND_active
from core.utils.utils import set_random_seed, mkdir, momentum_update
from core.datasets.transforms import build_transform
from core.active.loss import NLLLoss, FreeEnergyAlignmentLoss
from core.utils.metric_logger import MetricLogger
from core.utils.logger import setup_logger
from core.config import cfg

trainloss_txt = ".../DFAR.txt"


def test(model, test_loader):
    start_test = True
    model.eval()
    with torch.no_grad():
        all_output = torch.tensor([0., 0.])
        for batch_idx, test_data in enumerate(test_loader):
            img, labels = test_data['img0'], test_data['label']
            img = img.cuda()
            feas,outputs = model(img, return_feat=True)

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
        # print(all_output.shape)
    _, predict = torch.min(all_output, 1)
    acc = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) * 100

    return acc


def train(cfg, task):
    logger = logging.getLogger("EADA.trainer")

    use_cuda = True if torch.cuda.is_available() else False

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    source_transform = build_transform(cfg, is_train=True, choices=cfg.INPUT.SOURCE_TRANSFORMS)
    target_transform = build_transform(cfg, is_train=True, choices=cfg.INPUT.TARGET_TRANSFORMS)
    test_transform = build_transform(cfg, is_train=False, choices=cfg.INPUT.TEST_TRANSFORMS)

    src_train_ds = ImageList(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, cfg.DATASET.SOURCE_TRAIN_DOMAIN),
                             transform=source_transform)
    src_train_loader = DataLoader(src_train_ds, batch_size=cfg.DATALOADER.SOURCE.BATCH_SIZE, shuffle=True,
                                  drop_last=True, **kwargs)

    tgt_unlabeled_ds = ImageList(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, cfg.DATASET.TARGET_TRAIN_DOMAIN),
                                 transform=target_transform)
    tgt_unlabeled_loader = DataLoader(tgt_unlabeled_ds, batch_size=cfg.DATALOADER.TARGET.BATCH_SIZE, shuffle=True,
                                      drop_last=True, **kwargs)
    tgt_unlabeled_loader_full = DataLoader(tgt_unlabeled_ds, batch_size=cfg.DATALOADER.TARGET.BATCH_SIZE,
                                              shuffle=True, drop_last=False, **kwargs)

    tgt_test_ds = ImageList(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, cfg.DATASET.TARGET_VAL_DOMAIN),
                            transform=test_transform)
    tgt_test_loader = DataLoader(tgt_test_ds, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, shuffle=False, **kwargs)


    tgt_selected_ds = ImageList(empty=True, transform=source_transform)
    tgt_selected_loader = DataLoader(tgt_selected_ds, batch_size=cfg.DATALOADER.SOURCE.BATCH_SIZE,
                                     shuffle=True, drop_last=False, **kwargs)


    model = ResNetFc(class_num=cfg.DATASET.NUM_CLASS, cfg=cfg).cuda()


    optimizer = optim.Adam(model.parameters_list(cfg.OPTIM.LR), lr=cfg.OPTIM.LR)

    nll_criterion = NLLLoss(cfg)


    uns_criterion = FreeEnergyAlignmentLoss(cfg)


    totality = tgt_unlabeled_ds.__len__()

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    start_training_time = time.time()
    end = time.time()

    final_acc = 0.
    final_model = None
    all_epoch_result = []
    all_selected_images = None
    loss = []
    for epoch in range(1, cfg.TRAINER.MAX_EPOCHS + 1):

        model.train()

        iter_per_epoch = max(len(src_train_loader), len(tgt_unlabeled_loader))
        for batch_idx in range(iter_per_epoch):
            data_time = time.time() - end

            if batch_idx % len(src_train_loader) == 0:
                src_iter = iter(src_train_loader)
            if batch_idx % len(tgt_unlabeled_loader) == 0:
                tgt_unlabeled_iter = iter(tgt_unlabeled_loader)
            if not tgt_selected_ds.empty:
                if batch_idx % len(tgt_selected_loader) == 0:
                    tgt_selected_iter = iter(tgt_selected_loader)

            src_data = src_iter.next()
            tgt_unlabeled_data = tgt_unlabeled_iter.next()

            src_img, src_lbl = src_data['img0'], src_data['label']
            src_img, src_lbl = src_img.cuda(), src_lbl.cuda()

            tgt_unlabeled_img = tgt_unlabeled_data['img']
            tgt_unlabeled_img = tgt_unlabeled_img.cuda()

            optimizer.zero_grad()
            total_loss = 0

            # supervised loss on label source data
            src_out = model(src_img, return_feat=False)
            nll_loss = nll_criterion(src_out, src_lbl)
            total_loss += nll_loss
            meters.update(nll_loss=nll_loss.item())

            if cfg.TRAINER.ENERGY_ALIGN_WEIGHT > 0:

                tgt_unlabeled_out = model(tgt_unlabeled_img, return_feat=False)
                with torch.no_grad():
                    # free energy of samples
                    output_div_t = -1.0 * cfg.TRAINER.ENERGY_BETA * src_out
                    output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
                    free_energy = -1.0 * output_logsumexp / cfg.TRAINER.ENERGY_BETA


                    src_batch_free_energy = free_energy.mean().detach()


                    if epoch == 1 and batch_idx == 0:
                        global_mean = src_batch_free_energy

                    global_mean = momentum_update(global_mean, src_batch_free_energy)

                fea_loss = uns_criterion(inputs=tgt_unlabeled_out, bound=global_mean)

                total_loss += cfg.TRAINER.ENERGY_ALIGN_WEIGHT * fea_loss
                meters.update(fea_loss=(cfg.TRAINER.ENERGY_ALIGN_WEIGHT * fea_loss).item())


            if not tgt_selected_ds.empty:
                tgt_selected_data = tgt_selected_iter.next()
                tgt_selected_img, tgt_selected_lbl = tgt_selected_data['img0'], tgt_selected_data['label']
                tgt_selected_img, tgt_selected_lbl = tgt_selected_img.cuda(), tgt_selected_lbl.cuda()

                if tgt_selected_img.size(0) == 1:

                    tgt_selected_img = torch.cat((tgt_selected_img, tgt_selected_img), dim=0)
                    tgt_selected_lbl = torch.cat((tgt_selected_lbl, tgt_selected_lbl), dim=0)

                tgt_selected_out = model(tgt_selected_img, return_feat=False)
                selected_nll_loss = nll_criterion(tgt_selected_out, tgt_selected_lbl)

                total_loss += selected_nll_loss
                meters.update(selected_nll_loss=selected_nll_loss.item())


            total_loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (iter_per_epoch * cfg.TRAINER.MAX_EPOCHS - batch_idx * epoch)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))


            if batch_idx % cfg.TRAIN.PRINT_FREQ == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "task: {task}",
                            "epoch: {epoch}",
                            f"[iter: {batch_idx}/{iter_per_epoch}]",
                            "{meters}",
                            "max mem: {memory:.2f} GB",
                        ]
                    ).format(
                        task=task,
                        eta=eta_string,
                        epoch=epoch,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                    )
                )\




        if epoch % 5 == 0:
            testacc = test(model, tgt_test_loader)
            logger.info('Task: {} Test Epoch: {} testacc: {:.2f}'.format(task, epoch, testacc))
            all_epoch_result.append({'epoch': epoch, 'acc': testacc})
            if epoch == cfg.TRAINER.MAX_EPOCHS:
                final_model = model.state_dict()
                final_acc = testacc

        if epoch in cfg.TRAINER.ACTIVE_ROUND:
            logger.info('Task: {} Active Epoch: {}'.format(task, epoch))
            if cfg.TRAINER.NAME == 'RAND':
                active_samples = RAND_active(tgt_unlabeled_ds=tgt_unlabeled_ds,
                                             tgt_selected_ds=tgt_selected_ds,
                                             active_ratio=0.01,
                                             totality=totality)
            elif cfg.TRAINER.NAME == 'EADA':
                active_samples = EADA_active(tgt_unlabeled_loader_full=tgt_unlabeled_loader_full,
                                             tgt_unlabeled_ds=tgt_unlabeled_ds,
                                             tgt_selected_ds=tgt_selected_ds,
                                             active_ratio=0.01,
                                             totality=totality,
                                             model=model,
                                             cfg=cfg)


            if all_selected_images is None:
                all_selected_images = active_samples
            else:
                all_selected_images = np.concatenate((all_selected_images, active_samples), axis=0)
                print("all_selected_images shape:",all_selected_images.shape)



    ckt_path = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME, task)
    mkdir(ckt_path)
    torch.save(all_selected_images, os.path.join(ckt_path, "all_selected_images.pth"))
    torch.save(final_model, os.path.join(ckt_path, "final_model_{}.pth".format(task)))  #


    with open(os.path.join(ckt_path, 'all_epoch_result.csv'), 'w') as handle:
        for i, rec in enumerate(all_epoch_result):
            if i == 0:
                handle.write(','.join(list(rec.keys())) + '\n')
            line = [str(rec[key]) for key in rec.keys()]
            handle.write(','.join(line) + '\n')

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / ep)".format(
            total_time_str, total_training_time / cfg.TRAINER.MAX_EPOCHS
        )
    )

    return task, final_acc


def main():
    parser = argparse.ArgumentParser(description='PyTorch Free_energy Adaptation')
    parser.add_argument('--cfg',
                        default='.../DFWS.yaml',
                        metavar='FILE',
                        help='path to config file',
                        type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME)
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("EADA", output_dir, 0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)

    cudnn.deterministic = True

    all_task_result = []
    for source in cfg.DATASET.SOURCE_DOMAINS:
        for target in cfg.DATASET.TARGET_DOMAINS:
            if source != target:
                cfg.DATASET.SOURCE_TRAIN_DOMAIN = os.path.join(source + '_train.txt')
                cfg.DATASET.TARGET_TRAIN_DOMAIN = os.path.join(target + '_train.txt')
                cfg.DATASET.TARGET_VAL_DOMAIN = os.path.join(target + '_test.txt')

                cfg.freeze()
                task, final_acc = train(cfg, task=source + '2' + target)
                all_task_result.append({'task': task, 'final_acc': final_acc})
                cfg.defrost()

    # record all results for all tasks
    with open(os.path.join(output_dir, 'all_task_result.csv'), 'w') as handle:
        for i, rec in enumerate(all_task_result):
            if i == 0:
                handle.write(','.join(list(rec.keys())) + '\n')
            line = [str(rec[key]) for key in rec.keys()]
            handle.write(','.join(line) + '\n')


if __name__ == '__main__':
    main()
