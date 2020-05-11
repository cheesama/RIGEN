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
        target_tensors: One windows left shifted Tensor([[..., CLS],])
    """
    def __init__(self, file_path, session_col, text_col, sep_token_id=3, eos_token_id=2, sep=',', tokenize_fn=None):
        """
        file_path       : dialogue file path
        session_col     : session indicated column name
        text_col        : text indicated column name
        sep_token_id    : SEP token indicated id(default: 3, KoELECTRA based)
        eos_token_id    : EOS token indicated id(default: 2, KoELECTRA based(it does not contain BOS
        token basically, replace CLS token instead))
        sep             : delimiter of data file(default: ',')
        tokenizer_fn    : tokenizer func(default: None(using KoELECTRA tokenizer))
        """
        df = pd.read_csv(file_path, delimiter=sep, encoding='utf-8')

        self.sep_token_id = sep_token_id
        self.eos_token_id = eos_token_id

        self.source = []
        self.target = []
        

        if tokenize_fn is None:
            tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-small-discriminator')
            self.tokenize_fn = tokernizer.convert_tokens_to_ids
        else:
            self.tokenize_fn = tokenize_fn

        for name, group in tqdm(df.groupby(session_col), desc='generating session token dataset ...'):
            dialog = list(group[text_col])
            dialog_tokens = []

            for utterance in dialog:
                dialog_tokens += self.tokenize_fn(utterance)
                dialog_tokens += [self.sep_token_id]

            self.source.append(dialog_tokens[:-1])
            self.target.append(dialog_tokens[1:-1] + [self.eos_token_id])

                
                
        



        


