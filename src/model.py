import re
import datetime
from tqdm import tqdm
from typing import List, Tuple

import numpy as np
from transformers import MarianMTModel, MarianTokenizer

from src.config import BATCH_SIZE


class Translator():
    def __init__(self, name: str, split_regex: str) -> None:
        self.split_regex = split_regex
        self.tokenizer = MarianTokenizer.from_pretrained(name)
        self.model = MarianMTModel.from_pretrained(name)

    def _prepare_text(self, text: str) -> List[str]:
        text_filtered = text.replace('\n', ' ').strip()
        return re.split(self.split_regex, text_filtered)

    def _translate(self, text: str) -> str:
        sentences = self._prepare_text(text)
        data = self.tokenizer.prepare_seq2seq_batch(sentences, return_tensors="pt")

        tokens = data['input_ids']
        atts = data['attention_mask']

        sentences_translated = list()
        for i in range(0, tokens.shape[0], BATCH_SIZE):
            batch_tokens = tokens[i : i + BATCH_SIZE, :]
            batch_atts = atts[i : i + BATCH_SIZE, :]

            batch_translated = self.model.generate(input_ids=batch_tokens, attention_mask=batch_atts)
            sentences_translated += [self.tokenizer.decode(s, skip_special_tokens=True) for s in batch_translated]

        return " ".join(sentences_translated)

    def translate_dataset(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        texts_trans, times = [], []
        for text in tqdm(texts):
            start_time = datetime.datetime.now()
            text_trans = self._translate(text)
            end_time = datetime.datetime.now()

            time = (end_time - start_time).total_seconds()*1000

            texts_trans.append(text_trans)
            times.append(time)
        return texts_trans, times
