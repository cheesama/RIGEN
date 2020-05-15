from pytorch_lightning import Trainer
from transformers import ElectraTokenizer
from argparse import Namespace

from .RIGEN_lightning_model import ResponseInteractiveGenerator

import os, sys
import torch


def train(
    file_path,
    seq_len=512,
    batch_size=128,
    optimizer_lr=1e-5,  # default optimizer -> adam
    tokenizer=ElectraTokenizer.from_pretrained(
        "monologg/koelectra-small-discriminator"
    ),
    checkpoint_path=os.getcwd(),
    **kwargs
):
    """
    file_path: folder containing dialogue session data files are located
    """
    file_list = []
    for root, dir, files in os.walk(file_path):
        for each_file in files:
            file_list.append(root + os.sep + each_file)
    max_epochs = len(file_list) - 2

    gpu_num = torch.cuda.device_count()
    distributed_backend = None
    if gpu_num > 1:
        # distributed_backend='ddp'
        # distributed_backend='ddp2'
        distributed_backend = "horovod"

    if distributed_backend == "horovod":
        gpu_num = 1

    trainer = Trainer(
        default_root_dir=checkpoint_path,
        max_epochs=max_epochs,
        gpus=gpu_num,
        distributed_backend=distributed_backend,
    )

    model_args = {}

    # training args
    model_args["max_epochs"] = max_epochs
    model_args["file_path"] = file_path
    model_args["seq_len"] = seq_len
    model_args["batch_size"] = batch_size
    model_args["optimizer_lr"] = optimizer_lr

    # currently, only support KoELECTRA tokenizer
    model_args["tokenizer"] = tokenizer
    model_args["vocab_size"] = len(tokenizer.get_vocab())
    model_args["pad_token_id"] = tokenizer.pad_token_id

    for key, value in kwargs.items():
        model_args[key] = value

    hparams = Namespace(**model_args)

    model = ResponseInteractiveGenerator(hparams)

    trainer.fit(model)
