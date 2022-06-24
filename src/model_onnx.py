import re
import datetime
from tqdm import tqdm
from typing import Tuple, Dict, List

import numpy as np
from transformers import MarianTokenizer, MarianConfig, PretrainedConfig
from onnxruntime import InferenceSession # type: ignore

from src.config import BATCH_SIZE


class TranslationEncoderOnnx:
    def __init__(self, config: Dict):
        super().__init__()
        self.encoder_path = config.get("encoder_path")
        self.processing_unit = config.get("processing_unit")
        self.encoder_session = InferenceSession(
            self.encoder_path, providers=[self.processing_unit])

    def __call__(self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray
        ):

        onnx_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        hidden = self.encoder_session.run(None, onnx_inputs)
        return hidden[0]


class TranslationDecoderOnnx:
    def __init__(self, config: Dict):
        super().__init__()
        self.decoder_path = config.get("decoder_path")
        self.decoder_pkv_path = config.get("decoder_pkv_path")
        self.processing_unit = config.get("processing_unit")
        self.decoder_session = InferenceSession(
            self.decoder_path,
            providers=[self.processing_unit]
            )
        self.decoder_pkv_session = InferenceSession(
            self.decoder_pkv_path,
            providers=[self.processing_unit]
            )
        self.pkv_names: List = [
            input.name for input in self.decoder_pkv_session.get_inputs()][2:]

    def __call__(
        self,
        input_ids: np.ndarray,
        encoder_outputs: np.ndarray,
        encoder_attention_mask: np.ndarray,
        past_key_values: List[np.ndarray]=None
        ):

        if past_key_values:
            decoder_names = ['input_ids', 'encoder_attention_mask']
            decoder_inputs = [input_ids, encoder_attention_mask]
            decoder_onnx_inputs = dict(
                zip(decoder_names + self.pkv_names, decoder_inputs + past_key_values))

            output = self.decoder_pkv_session.run(None, decoder_onnx_inputs)
            logits = output[0]
            pkv = output[1:]
        else:
            decoder_onnx_inputs = {
                'input_ids': input_ids,
                'encoder_hidden_states': encoder_outputs,
                'encoder_attention_mask': encoder_attention_mask
            }
            output = self.decoder_session.run(None, decoder_onnx_inputs)
            logits = output[0]
            pkv = output[1:]

        return logits, pkv


class TranslationModelOnnx:
    def __init__(self, marian_config: PretrainedConfig, onnx_config: Dict):

        self.encoder = TranslationEncoderOnnx(onnx_config)
        self.decoder = TranslationDecoderOnnx(onnx_config)
        self.config = marian_config
        self.max_length: int = onnx_config['max_length']

    def generate(self, tokens: Dict):
        enc_inputs = tokens['input_ids']
        enc_att_mask = tokens['attention_mask']
        hidden = self.encoder(enc_inputs, enc_att_mask)

        bsz = enc_inputs.shape[0]
        indices_active = np.arange(bsz)
        dec_inputs = np.ones((bsz, self.max_length), dtype=int) * self.config.pad_token_id
        dec_inputs[:, 0] = self.config.decoder_start_token_id
        past_key_values = None

        for idx in range(self.max_length - 1):
            if past_key_values:
                logits, past_key_values = self.decoder(
                    dec_inputs[indices_active, idx].reshape(-1, 1),
                    hidden[indices_active, :],
                    enc_att_mask[indices_active, :],
                    past_key_values
                )
            else:
                logits, past_key_values = self.decoder(
                    dec_inputs[indices_active, idx].reshape(-1, 1),
                    hidden[indices_active, :],
                    enc_att_mask[indices_active, :],
                )
            logits[:, 0, self.config.pad_token_id] = float("-inf")
            token_ids = logits.argmax(axis=2).flatten()
            dec_inputs[indices_active, idx + 1] = token_ids

            indices_non_end = np.where(token_ids != self.config.eos_token_id)[0]
            indices_active = indices_active[indices_non_end]

            if indices_active.size == 0:
                break

            past_key_values = [pkv[indices_non_end, :, :, :] for pkv in past_key_values]

        return dec_inputs[:, :idx + 2]


class Translator():
    def __init__(self, name: str, regex: str, config: Dict) -> None:
        self.split_regex = regex
        self.tokenizer = MarianTokenizer.from_pretrained(name)
        marian_config = MarianConfig.from_pretrained(name)
        self.model = TranslationModelOnnx(marian_config, config)

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

            tokens = {
                'input_ids': batch_tokens,
                'attention_mask': batch_atts
            }

            batch_translated = self.model.generate(tokens)
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