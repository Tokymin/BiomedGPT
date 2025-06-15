#!/usr/bin/env python3
# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

"""
Checkpoint utility functions for model training and evaluation.
"""

import os
import math
import logging
import torch
from typing import Dict, Optional, Any, Tuple, Union
from fairseq import utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.trainer import Trainer
from fairseq.data import iterators
from omegaconf import DictConfig
from typing import Optional
logger = logging.getLogger(__name__)

# 用于跟踪最佳检查点的全局变量
save_checkpoint = type('SaveCheckpoint', (), {'best': float('inf')})()


def verify_checkpoint_directory(save_dir: str) -> None:
    """
    Verify and create the checkpoint directory if it doesn't exist.

    Args:
        save_dir: Directory path to save checkpoints.
    """
    if not os.path.exists(save_dir):
        logger.info(f"Creating checkpoint directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
    elif not os.path.isdir(save_dir):
        raise NotADirectoryError(f"{save_dir} exists but is not a directory")


def get_latest_checkpoint(save_dir: str) -> Optional[str]:
    """
    Get the latest checkpoint file from the save directory.

    Args:
        save_dir: Directory path containing checkpoints.

    Returns:
        Path to the latest checkpoint file, or None if no checkpoints found.
    """
    checkpoint_files = [
        f for f in os.listdir(save_dir)
        if f.startswith("checkpoint") and f.endswith(".pt")
    ]

    if not checkpoint_files:
        return None

    # 按修改时间排序，获取最新的检查点
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(save_dir, x)))
    return os.path.join(save_dir, checkpoint_files[-1])


def load_checkpoint(
        cfg: FairseqConfig,
        trainer: Trainer,
        disable_iterator_cache: bool = False
) -> Tuple[Optional[Dict[str, Any]], iterators.EpochBatchIterator]:
    """
    Load the latest checkpoint and restore training state.

    Args:
        cfg: Configuration object.
        trainer: Trainer instance.
        disable_iterator_cache: Whether to disable iterator caching.

    Returns:
        A tuple containing extra state and the epoch iterator.
    """
    save_dir = cfg.checkpoint.save_dir
    verify_checkpoint_directory(save_dir)

    extra_state = None
    last_epoch = 1
    best = save_checkpoint.best

    # 获取最新的检查点文件
    checkpoint_path = get_latest_checkpoint(save_dir)

    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=lambda storage, loc: storage
        )

        # 恢复训练状态
        extra_state = checkpoint.get('extra_state', {})
        last_epoch = extra_state.get('epoch', 1)
        save_checkpoint.best = extra_state.get('best', save_checkpoint.best)

        # 加载模型和优化器状态
        trainer.load_checkpoint(checkpoint)

        logger.info(f"Loaded checkpoint from epoch {last_epoch}")
    else:
        logger.info("No existing checkpoint found, starting fresh training")

    # 获取训练数据迭代器
    epoch_itr = trainer.get_train_iterator(
        epoch=last_epoch,
        load_dataset=True,
        disable_iterator_cache=disable_iterator_cache
    )

    return extra_state, epoch_itr


def save_checkpoint(
        cfg: DictConfig,
        trainer: Trainer,
        epoch_itr: iterators.EpochBatchIterator,
        val_loss: Optional[float] = None
) -> None:
    """
    Save model checkpoint to the specified directory.

    Args:
        cfg: Configuration object.
        trainer: Trainer instance.
        epoch_itr: Epoch iterator.
        val_loss: Validation loss for this checkpoint.
    """
    save_dir = cfg.save_dir
    verify_checkpoint_directory(save_dir)

    epoch = epoch_itr.epoch
    num_updates = trainer.get_num_updates()

    # 构建检查点文件名
    if num_updates > 0:
        checkpoint_name = f"checkpoint_{num_updates}.pt"
    else:
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"

    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    last_checkpoint_path = os.path.join(save_dir, "checkpoint_last.pt")
    best_checkpoint_path = os.path.join(save_dir, "checkpoint_best.pt")

    logger.info(f"Saving checkpoint to: {checkpoint_path}")

    # 准备检查点数据
    checkpoint = {
        'model': trainer.get_model().state_dict(),
        'optimizer': trainer.get_optimizer().state_dict(),
        'lr_scheduler': trainer.get_lr_scheduler().state_dict(),
        'extra_state': {
            'epoch': epoch,
            'num_updates': num_updates,
            'best': save_checkpoint.best,
        },
        'cfg': cfg
    }

    # 保存检查点
    utils.save_checkpoint(checkpoint, checkpoint_path)
    utils.save_checkpoint(checkpoint, last_checkpoint_path)

    # 更新最佳检查点
    maximize_best = cfg.get('maximize_best_checkpoint_metric', False)
    if val_loss is not None:
        if (maximize_best and val_loss > save_checkpoint.best) or \
                (not maximize_best and val_loss < save_checkpoint.best):
            save_checkpoint.best = val_loss
            logger.info(f"New best checkpoint found with loss: {val_loss}")
            utils.save_checkpoint(checkpoint, best_checkpoint_path)
        else:
            logger.info(f"Current loss: {val_loss}, best loss: {save_checkpoint.best}")