from argparse import Namespace

from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from torchnlp.metrics import get_accuracy, get_token_accuracy

from pytorch_lightning import Trainer

from .dataset.dialogue_dataset import DialogueDataset
from .model.models import DialogueTransformer

import os, sys
import multiprocessing
import dill
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl

class ResponseInteractiveGenerator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.model = DialogueTransformer(vocab_size=self.hparams.vocab_size)

        self.batch_size = 1
        self.optimizer = self.hparams.optimizer
        self.optimizer_lr = self.hparams.optimizer_lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        file_list = []
        for root, dir, files in os.walk(self.hparams.file_path):
            for each_file in files:
                file_list.append(root + os.sep + each_file)
        file_list = sorted(file_list, key=lambda t:os.stat(t).st_mtime)

        print ('preparing train dataset')
        self.train_dataset = DialogueDataset(file_path=file_list[self.current_epoch], session_col='ho_idnt_num', text_col='text', tokenize_fn=self.hparams.tokenize_fn)
        print ('preparing val dataset')
        self.val_dataset = DialogueDataset(file_path=file_list[self.current_epoch+1], session_col='ho_idnt_num', text_col='text', tokenize_fn=self.hparams.tokenize_fn)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
        )
        return val_loader

    def configure_optimizers(self):
        optimizer = eval(
            f"{self.optimizer}(self.parameters(), lr={self.optimizer_lr})"
        )

        return optimizer, ReduceLROnPlateau(optimizer, patience=1)

    def training_step(self, batch, batch_idx):
        self.model.train()

        source, target = batch

        pred = self.forward(tokens)

        acc = get_token_accuracy(
            target.cpu(),
            pred.max(2)[1].cpu(),
        )[0]

        tensorboard_logs = {
            "train/acc": acc,
        }

        loss = self.loss_fn(pred.transpose(1, 2), target.long())
        tensorboard_logs["train/loss"] = _loss
        return {
            "loss": loss,
            "log": tensorboard_logs,
        }

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        source, target = batch

        pred = self.forward(source)

        acc = get_token_accuracy(
            target.cpu(),
            pred.max(2)[1].cpu(),
        )[0]

        loss = self.loss_fn(pred.transpose(1, 2), target.long())

        return {
            "val_acc": acc,
            "val_loss": loss,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = np.mean([x["val_loss"] for x in outputs])
        avg_acc = np.mean([x["val_acc"] for x in outputs])

        tensorboard_logs = {
            "val/loss": avg_loss,
            "val/acc": avg_acc,
        }

        print ('### preparing next dataset ###')
        self.prepare_data()

        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

