# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from atss_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from atss_core.config import cfg
from atss_core.data import make_data_loader
from atss_core.solver import make_lr_scheduler
from atss_core.solver import make_optimizer
from atss_core.engine.inference import inference
from atss_core.engine.trainer import do_train
from atss_core.modeling.detector import build_detection_model
from atss_core.utils.checkpoint import DetectronCheckpointer
from atss_core.utils.collect_env import collect_env_info
from atss_core.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
# from atss_core.utils.imports import import_file
from atss_core.utils.logger import setup_logger
from atss_core.utils.miscellaneous import mkdir


def train(cfg, local_rank, distributed):
    # net建立
    model = build_detection_model(cfg)
    # 设置device，模型转到device上
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # ? about multi gpu
    # SynBatch
    if cfg.MODEL.USE_SYNCBN:
        assert is_pytorch_1_1_0_or_later(), \
            "SyncBatchNorm is only available in pytorch >= 1.1.0"
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 优化器
    optimizer = make_optimizer(cfg, model)
    # lr更新策略
    scheduler = make_lr_scheduler(cfg, optimizer)

    # ? about multi gpu
    # distributed
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    # 保存的地址
    output_dir = cfg.OUTPUT_DIR

    # 仅在主机保存
    save_to_disk = get_rank() == 0

    # 预训练模型（MODEL.WEIGHT）参数加载
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    # dataloader
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    data_loader_val = make_data_loader(cfg, is_train=True, is_distributed=distributed, start_iter=0)

    # 每间隔多少个循环保存一次模型参数
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    # 循环迭代
    do_train(
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        cfg.INPUT.ROTATED,
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.ATSS_ON or cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # num_gpus = 1
    args.distributed = num_gpus > 1

    # ? about multiple gpu
    if args.distributed:
        # 设定使用的GPU
        torch.cuda.set_device(args.local_rank)
        # multiprocess初始化
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    # 读取config信息，.freeze()防止后续修改参数
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 创建输出dir
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    # 打logger
    logger = setup_logger("atss_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # 训练，返回模型
    model = train(cfg, args.local_rank, args.distributed)

    # 测试
    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
