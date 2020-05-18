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

        self.model = DialogueTransformer(
            vocab_size=self.hparams.vocab_size,
            seq_len=self.hparams.seq_len,
            pad_token_id=self.hparams.pad_token_id,
        )

        self.batch_size = self.hparams.batch_size
        self.optimizer_lr = self.hparams.optimizer_lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        file_list = []
        for root, dir, files in os.walk(self.hparams.file_path):
            for each_file in files:
                file_list.append(root + os.sep + each_file)
        self.file_list = sorted(file_list, key=lambda t: os.stat(t).st_mtime)

        print(f"parameter setting: {self.hparams}")

        print("preparing train dataset")
        self.train_dataset = DialogueDataset(
            file_path=self.file_list[self.current_epoch],
            session_col="ho_idnt_num",
            text_col="text",
            seq_len=self.hparams.seq_len,
            tokenize_fn=self.hparams.tokenizer.encode,
            sep_token_id=self.hparams.tokenizer.sep_token_id,
            pad_token_id=self.hparams.tokenizer.pad_token_id,
        )
        print("preparing val dataset")
        self.val_dataset = DialogueDataset(
            file_path=self.file_list[self.current_epoch + 1],
            session_col="ho_idnt_num",
            text_col="text",
            seq_len=self.hparams.seq_len,
            tokenize_fn=self.hparams.tokenizer.encode,
            sep_token_id=self.hparams.tokenizer.sep_token_id,
            pad_token_id=self.hparams.tokenizer.pad_token_id,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
        )

    def prepare_next_data(self, epoch: int):
        print (f'current epoch: {epoch}')

        if epoch > 1:
            print("### preparing next epoch dataset ###")

        print("preparing train dataset")
        self.train_dataset = DialogueDataset(
            file_path=self.file_list[self.current_epoch],
            session_col="ho_idnt_num",
            text_col="text",
            seq_len=self.hparams.seq_len,
            tokenize_fn=self.hparams.tokenizer.encode,
            sep_token_id=self.hparams.tokenizer.sep_token_id,
            pad_token_id=self.hparams.tokenizer.pad_token_id,
        )
        print("preparing val dataset")
        self.val_dataset = DialogueDataset(
            file_path=self.file_list[self.current_epoch + 1],
            session_col="ho_idnt_num",
            text_col="text",
            seq_len=self.hparams.seq_len,
            tokenize_fn=self.hparams.tokenizer.encode,
            sep_token_id=self.hparams.tokenizer.sep_token_id,
            pad_token_id=self.hparams.tokenizer.pad_token_id,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_lr)

        return [optimizer], [ReduceLROnPlateau(optimizer, patience=1)]

    def training_step(self, batch, batch_idx):
        self.model.train()

        source, target = batch

        pred = self.forward(source)

        acc = get_token_accuracy(target.cpu(), pred.max(2)[1].cpu(),)[0]

        tensorboard_logs = {
            "train/acc": acc,
        }

        loss = self.loss_fn(pred.transpose(1, 2), target.long())
        tensorboard_logs["train/loss"] = loss
        return {
            "loss": loss,
            "log": tensorboard_logs,
        }

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        source, target = batch

        pred = self.forward(source)

        acc = get_token_accuracy(target.cpu(), pred.max(2)[1].cpu(),)[0]

        loss = self.loss_fn(pred.transpose(1, 2), target.long())

        return {
            "val_loss": loss,
            "val_acc": torch.Tensor([acc]),
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        tensorboard_logs = {
            "val/loss": avg_loss,
            "val/acc": avg_acc,
        }

        self.prepare_next_data(self.current_epoch)

        return {
            "val_loss": avg_loss,
            "val_acc": avg_acc,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }
