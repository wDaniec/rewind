# -*- coding: utf-8 -*-
"""
Basic callbacks available in the project
"""

import datetime
import json
import logging
import os
import sys
import time
import torch

import gin

logger = logging.getLogger(__name__)

# from src.training_loop import training_loop
from src.utils import parse_gin_config

from pytorch_lightning.callbacks import Callback


class EarlyStop(Callback):
    def __init__(self, path, patience, delta=0.0):
        self.monitor = "valid_acc"
        self.path = path
        self.patience = patience
        self.current_patience = patience
        self.delta = delta
        self.finished = False
        self.best_acc = 0.0
        super(Callback, self).__init__()


    def on_train_start(self, trainer, pl_module):
        model_path = os.path.join(self.path, "initial.pth")
        torch.save(pl_module.model.state_dict(), model_path)
    

    def on_validation_end(self, trainer, pl_module):
        if trainer.running_sanity_check:
            return

        current_acc = trainer.logger_connector.callback_metrics[self.monitor]

        if not self.finished:
            if current_acc > self.best_acc + self.delta:
                self.best_acc = current_acc
                self.current_patience = self.patience
                model_path = os.path.join(self.path, "best_valid_acc.pth")
                torch.save(pl_module.model.state_dict(), model_path)
            else:
                self.current_patience -= 1
                if not self.current_patience:
                    self.finished = True

@gin.configurable
class MetaSaver(Callback):
    def __init__(self):
        super(MetaSaver, self).__init__()

    def on_train_start(self, trainer, pl_module):
        logger.info("Saving meta data information from the beginning of training")

        assert os.system(
            "cp {} {}".format(sys.argv[0], trainer.default_root_dir)) == 0, "Failed to execute cp of source script"

        utc_date = datetime.datetime.utcnow().strftime("%Y_%m_%d")

        time_start = time.time()
        cmd = "python " + " ".join(sys.argv)
        self.meta = {"cmd": cmd,
            "save_path": trainer.default_root_dir,
            "most_recent_train_start_date": utc_date,
            "execution_time": -time_start}

        json.dump(self.meta, open(os.path.join(trainer.default_root_dir, "meta.json"), "w"), indent=4)

    def on_train_end(self, trainer, pl_module):
        self.meta['execution_time'] += time.time()
        json.dump(self.meta, open(os.path.join(trainer.default_root_dir, "meta.json"), "w"), indent=4)
        os.system("touch " + os.path.join(trainer.default_root_dir, "FINISHED"))


class Heartbeat(Callback):
    def __init__(self, interval=10):
        self.last_time = time.time()
        self.interval = interval

    def on_train_start(self, trainer, pl_module):
        logger.info("HEARTBEAT - train begin")
        os.system("touch " + os.path.join(trainer.default_root_dir, "HEARTBEAT"))

    def on_batch_start(self, trainer, pl_module):
        if time.time() - self.last_time > self.interval:
            logger.info("HEARTBEAT")
            os.system("touch " + os.path.join(trainer.default_root_dir, "HEARTBEAT"))
            self.last_time = time.time()



@gin.configurable
class LRSchedule(Callback):
    def __init__(self, base_lr, schedule):
        self.schedule = schedule
        self.base_lr = base_lr
        super(LRSchedule, self).__init__()

    def on_epoch_start(self, trainer, pl_module):
        # Epochs starts from 0
        for e, v in self.schedule:
            if trainer.current_epoch < e:
                break
        for group in trainer.optimizers[0].param_groups:
            group['lr'] = v * self.base_lr
        logger.info("Set learning rate to {}".format(v * self.base_lr))

