from tqdm import tqdm
from transformers import ElectraTokenizer

import torch
import pandas as pd


class DialogueDataset(torch.utils.data.Dataset):
    """
    Dialogue dataset sorted by session_id & sequence id

    EX)
    Input:
        session_id  seq text
            A       0   안녕하세요
            A       1   네 반갑습니다
            B       0   오늘 날씨 좋네요
            B       1   비가 오는데 어디가 좋아요?
        ....

    Output:
        source_tensors: Tensor([[tokens of '안녕하세요', SEP, ... ],]) 
        target_tensors: One windows left shifted Tensor([[..., EOS(CLS)],])
    """

    def __init__(
        self,
        file_path,
        tokenize_fn,
        session_col: str,
        text_col: str,
        seq_len: int,
        sep_token_id: int,
        pad_token_id: int,
        sep="\t",
    ):
        """
        file_path       : dialogue file path
        tokenizer_fn    : tokenizer func
        session_col     : session indicated column name
        text_col        : text indicated column name
        sep_token_id    : SEP token id which seperate talker
        pad_token_id    : PAD token id which set padding
        sep             : delimiter of data file(default: ',')
        """
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.seq_len = seq_len

        self.source = []
        self.target = []

        self.tokenize_fn = tokenize_fn

        df = pd.read_csv(
            file_path, delimiter=sep, usecols=[session_col, text_col], encoding="utf-8"
        )

        for name, group in tqdm(
            df.groupby(session_col),
            desc=f"generating session token dataset -> {file_path}",
        ):
            dialog = list(group[text_col])
            dialog_tokens = []

            for utterance in dialog:
                #based on KoELETRA tokenizer, ignore [CLS] token
                dialog_tokens += self.tokenize_fn(str(utterance))[1:]

            if len(dialog_tokens) < self.seq_len + 1:
                self.source.append(
                    dialog_tokens
                    + [self.pad_token_id] * (self.seq_len - len(dialog_tokens))
                )
                self.target.append(
                    dialog_tokens[1:]
                    + [self.pad_token_id] * (self.seq_len - len(dialog_tokens[1:]))
                )

            else:
                self.source.append(dialog_tokens[: self.seq_len])
                self.target.append(dialog_tokens[1 : self.seq_len + 1])

            #if len(self.source) > 10:
            #    break

    def __getitem__(self, idx):
        return torch.tensor(self.source[idx]), torch.tensor(self.target[idx])

    def __len__(self):
        return len(self.source)
