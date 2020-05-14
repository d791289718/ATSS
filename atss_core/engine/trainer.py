# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
from collections import defaultdict
 
import torch
import os
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from atss_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from atss_core.utils.metric_logger import MetricLogger
from atss_core.config import cfg
from atss_core.data.datasets.evaluation import evaluate
from atss_core.utils.miscellaneous import mkdir
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from .bbox_aug import im_detect_bbox_aug
from .bbox_aug_vote import im_detect_bbox_aug_vote


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_validation(model, data_loader_val, device, is_rotated):
    val_loss = 0.0
    start_val_time = time.time()
    val_loss_dict_reduced = defaultdict(float)
    with torch.no_grad():
        for _, (images, targets, rtargets, _) in enumerate(data_loader_val[0], 0):
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            rtargets = [target.to(device) for target in rtargets]

            loss_dict = model(images, targets=targets, rtargets=rtargets, is_rotated=is_rotated)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            val_loss += losses_reduced
            for itm, loss in loss_dict_reduced.items():
                val_loss_dict_reduced[itm] += loss

        val_loss /= len(data_loader_val[0])
        val_loss_dict_reduced = {itm: loss / len(data_loader_val[0]) for itm, loss in val_loss_dict_reduced.items()}
        total_val_time = time.time() - start_val_time
    return val_loss, val_loss_dict_reduced, total_val_time


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(data_loader):
        images, tatgets, rtargets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            # bbox_aug
            if cfg.TEST.BBOX_AUG.ENABLED:
                if cfg.TEST.BBOX_AUG.VOTE:
                    output = im_detect_bbox_aug_vote(model, images, device)
                else:
                    output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("atss_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def get_APs(model, data_loader_val, device, is_rotated, cfg, iteration):
    iou_types = ("segm",)
    dataset = data_loader_val[0].dataset

    predictions = compute_on_dataset(model, data_loader_val[0], device)
    synchronize()

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    if not is_main_process():
        return
    output_folder = [None]
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST[0], str(iteration))
        mkdir(output_folder)
    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=False if cfg.MODEL.ATSS_ON or cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
        iou_types=iou_types,
        expected_results=cfg.TEST.EXPECTED_RESULTS,
        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
    )
    # 到这里是得到了预测的RotatedBoxList的结果，下面要和gt作比较了
    results, coco_results = evaluate(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        is_rotated=is_rotated,
        **extra_args)
    return results.results[iou_types]


def do_train(
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    is_rotated,
    cfg
):
    logger = logging.getLogger("atss_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    # 设置模型为训练模式
    model.train()
    # 计时
    start_training_time = time.time()
    end = time.time()
    # pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    pytorch_1_1_0_or_later = True
    # tensorboard
    writer = SummaryWriter(os.path.join('runs', cfg.OUTPUT_DIR,))

    for iteration, (images, targets, rtargets, _) in enumerate(data_loader, start_iter):  # dim=0上遍历
        # images, targets 是每个batch的 Tensor
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        rtargets = [target.to(device) for target in rtargets]

        # loss
        loss_dict = model(images, targets=targets, rtargets=rtargets, is_rotated=is_rotated)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if pytorch_1_1_0_or_later:
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if iteration % 200 == 0 or iteration == max_iter:
            val_loss_reduced, val_loss_dict_reduced, total_val_time = do_validation(
                model, data_loader_val, device, is_rotated)

            total_time_str = str(datetime.timedelta(seconds=total_val_time))
            val_logger_list = ["{}: {:.4f} ".format(itm, loss) for itm, loss in val_loss_dict_reduced.items()]
            logger.info(
                "validation total time: {} ({:.4f} s / it)".format(
                    total_time_str, total_val_time)
            )
            logger.info("validation: loss {} {}".format(val_loss_reduced, str("".join(val_logger_list))))
            loss_dict_reduced_logger = {itm+"_train": loss for itm, loss in loss_dict_reduced.items()}
            val_loss_dict_reduced_logger = {itm+"_val": loss for itm, loss in val_loss_dict_reduced.items()}
            val_loss_dict_reduced_logger.update(loss_dict_reduced_logger)
            writer.add_scalars(
                'Loss_dict', val_loss_dict_reduced_logger, iteration
            )
            writer.add_scalars(
                "Loss_sum", {'train': losses_reduced, 'val': val_loss_reduced}, iteration
            )

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            coco_redict = get_APs(model, data_loader_val, device, is_rotated, cfg, iteration)
            writer.add_scalars(
                "APs", coco_redict, iteration
            )
               
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    writer.close()
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

