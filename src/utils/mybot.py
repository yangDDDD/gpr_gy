import os
import random
import logging
from pathlib import Path
from collections import deque
import argparse

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from .logger import Logger
from src.utils.tools import set_seed
from myparser.gpr_parser import add_feature_args

AVERAGING_WINDOW = 300

parser = argparse.ArgumentParser(description='gpr')

parser = add_feature_args(parser)

args = parser.parse_args()

set_seed(args.seed)

class MyBot:

    name = "basebot"

    def __init__(
            self, model, train_loader, val_loader, *, optimizer, clip_grad=0,
            avg_window=AVERAGING_WINDOW, logger=None,
            checkpoint_dir="./data/cache/model_cache/", batch_idx=0,
            device="cuda:0"):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.avg_window = avg_window
        self.clip_grad = clip_grad
        self.optimizer = optimizer
        self.model = model
        self.batch_idx = batch_idx
        self.logger = logger

        self.logger.info("SEED: %s", args.seed)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        self.device = device
        self.best_performers = []
        self.step = 0
        self.train_losses = None
        self.train_weights = None
        # Should be overriden when necessary:
        self.criterion = torch.nn.MSELoss()
        self.loss_format = "%.8f"

        self.count_model_parameters()

    def count_model_parameters(self):
        self.logger.info(
            "# of paramters: {:,d}".format(
                np.sum(p.numel() for p in self.model.parameters())))
        self.logger.info(
            "# of trainable paramters: {:,d}".format(
                np.sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    @staticmethod
    def extract_prediction(output):
        """Assumes single output"""
        return output[:, 0]

    @staticmethod
    def transform_prediction(prediction):
        return prediction

    def train_one_step(self, input_tensors, target):
        self.model.train()
        assert self.model.training
        self.optimizer.zero_grad()
        output = self.model(*input_tensors)
        batch_loss = self.criterion(self.extract_prediction(output), target)
        batch_loss.backward()
        self.train_losses.append(batch_loss.data.cpu().numpy())
        self.train_weights.append(target.size(self.batch_idx))
        if self.clip_grad > 0:
            clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

    def log_progress(self):
        train_loss_avg = np.average(
            self.train_losses, weights=self.train_weights)
        self.logger.info(
            "Step %s: train %.6f lr: %.3e",
            self.step, train_loss_avg, self.optimizer.param_groups[-1]['lr'])
        self.logger.tb_scalars(
            "lr", self.optimizer.param_groups[0]['lr'], self.step)
        self.logger.tb_scalars(
            "losses", {"train": train_loss_avg}, self.step)

    def snapshot(self):
        loss = self.eval(self.val_loader)
        loss_str = self.loss_format % loss
        self.logger.info("Snapshot loss %s", loss_str)
        self.logger.tb_scalars(
            "losses", {"val": loss},  self.step)

        target_path = (
                self.checkpoint_dir / "best.pth")
        self.best_performers.append((loss, target_path, self.step))
        self.logger.info("Saving checkpoint %s...", target_path)

        torch.save(self.model.state_dict(), target_path)
        assert Path(target_path).exists()
        return loss

    def eval(self, loader):
        self.model.eval()
        losses, weights = [], []
        with torch.set_grad_enabled(False):
            for *input_tensors, y_local in tqdm(loader):
                input_tensors = [x.to(self.device) for x in input_tensors]
                output = self.model(*input_tensors)
                batch_loss = self.criterion(
                    self.extract_prediction(output), y_local.to(self.device))
                losses.append(batch_loss.data.cpu().numpy())
                weights.append(y_local.size(self.batch_idx))
        loss = np.average(losses, weights=weights)
        return loss

    def train(
            self, n_steps, *, log_interval=50,
            early_stopping_cnt=0, min_improv=1e-4,
            scheduler=None, snapshot_interval=2500):
        self.train_losses = deque(maxlen=self.avg_window)
        self.train_weights = deque(maxlen=self.avg_window)
        if self.val_loader is not None:
            best_val_loss = np.inf
        epoch = 0
        wo_improvement = 0
        self.best_performers = []
        self.logger.info(
            "Optimizer {}".format(str(self.optimizer)))
        self.logger.info("Batches per epoch: {}".format(
            len(self.train_loader)))
        try:
            while self.step < n_steps:
                epoch += 1
                self.logger.info(
                    "=" * 20 + "Epoch %d" + "=" * 20, epoch)
                for *input_tensors, target in self.train_loader:
                    input_tensors = [x.to(self.device) for x in input_tensors]
                    self.train_one_step(input_tensors, target.to(self.device))
                    self.step += 1
                    if self.step % log_interval == 0:
                        self.log_progress()
                    if self.step % snapshot_interval == 0:
                        loss = self.snapshot()
                        if best_val_loss > loss + min_improv:
                            self.logger.info("New low\n")
                            best_val_loss = loss
                            wo_improvement = 0
                        else:
                            wo_improvement += 1
                    if scheduler:
                        scheduler.batch_step()
                    if early_stopping_cnt and wo_improvement > early_stopping_cnt:
                        return
                    if self.step >= n_steps:
                        break
        except KeyboardInterrupt:
            pass
        self.best_performers = sorted(self.best_performers, key=lambda x: x[0])

    def load_model(self, target_path):
        self.model.load_state_dict(torch.load(target_path))

    def predict(self, loader, *, return_y=False):
        self.model.eval()
        outputs, y_global = [], []
        with torch.set_grad_enabled(False):
            for *input_tensors, y_local in tqdm(loader):
                input_tensors = [x.to(self.device) for x in input_tensors]
                outputs.append(self.predict_batch(input_tensors).cpu())
                y_global.append(y_local.cpu())
            outputs = torch.cat(outputs, dim=0)
            y_global = torch.cat(y_global, dim=0)
        if return_y:
            return outputs, y_global
        return outputs

    def predict_batch(self, input_tensors):
        self.model.eval()
        tmp = self.model(*input_tensors)
        return self.extract_prediction(tmp)