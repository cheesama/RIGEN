from .DIET_lightning_model import ResponseInteractiveGenerator

import torch
import torch.nn as nn

import logging

model = None
intent_dict = {}
entity_dict = {}

class Inferencer:
    def __init__(self, checkpoint_path: str):
        self.model = ResponseInteractiveGenerator.load_from_checkpoint(checkpoint_path)
        self.model.model.eval()

    def inference(self, text: str):
        if self.model is None:
            raise ValueError(
                "model is not loaded, first call load_model(checkpoint_path)"
            )

        tokens = self.model.hparams.tokenize_fn(text)
        result = self.model.forward(tokens.unsqueeze(0))

