from pytorch_lightning import Trainer
from transformers import ElectraTokenizer
from argparse import Namespace

from .RIGEN_lightning_model import ResponseInteractiveGenerator

import os, sys
import torch

def train(
    file_path,
    optimizer_lr=1e-5, #default optimizer -> adam
    checkpoint_path=os.getcwd(),
    tokenizer=ElectraTokenizer.from_pretrained("monologg/koelectra-small-discriminator"),
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

    trainer = Trainer(
        default_root_dir=checkpoint_path, max_epochs=max_epochs, gpus=gpu_num
    )

    model_args = {}

    # training args
    model_args["max_epochs"] = max_epochs
    model_args["file_path"] = file_path
    model_args["optimizer_lr"] = optimizer_lr

    #currently, only support KoELECTRA tokenizer
    model_args["tokenize_fn"] = tokenizer.encode
    model_args["vocab_size"] = len(tokenizer.get_vocab())

    for key, value in kwargs.items():
        model_args[key] = value

    hparams = Namespace(**model_args)

    model = ResponseInteractiveGenerator(hparams)

    trainer.fit(model)
