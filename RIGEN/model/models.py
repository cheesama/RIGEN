from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

import torch
import torch.nn as nn
import torch.nn.functional as F

class DialogueTransformer(nn.Module):
    """
    Dialogue Response Seq2Seq model based on One batch using transformer(no use padding mask!)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super(DialogueTransformer, self).__init__()

        self.vocab_embedding = nn.Embedding(vocab_size, d_model)
        self.feature_embedding = nn.Linear(d_model, vocab_size)
        self.encoder = nn.TransformerEncoder(
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation
            ),
            num_encoder_layers,
            LayerNorm(d_model),
        )

    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        embedding = self.vocab_embedding(x)
        src_mask = self._generate_square_subsequent_mask(x.size(1)).type_as(embedding)

        feature = self.encoder(embedding.transpose(1, 0), mask=src_mask)  #(N,S,E) -> (S,N,E)

        return self.feature_embedding(feature.transpose(0,1)) #(S,N,E) -> (N,S,E) -> (N,S,C)
