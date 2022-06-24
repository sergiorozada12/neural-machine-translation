import sys

import pandas as pd

from src.model_huggingface import Translator as TransHug
from src.model_onnx import Translator as TransOnx

from src.utils import save_dataset
from src.config import REGEX, NAME, MODEL_CONFIG_HUGGINGFACE, MODEL_CONFIG_ONNX

PATH_INPUT = sys.argv[1]
PATH_OUTPUT = sys.argv[2]
MODEL = sys.argv[3]


if __name__ == "__main__":
    df = pd.read_csv(PATH_INPUT, sep=';')

    texts = df['text'].values
    labels = df['label'].values

    if MODEL == 'onnx':
        translator = TransOnx(NAME, REGEX, MODEL_CONFIG_ONNX)
    else:
        translator = TransHug(NAME, REGEX, MODEL_CONFIG_HUGGINGFACE)

    texts_trans, times = translator.translate_dataset(texts)

    save_dataset(PATH_OUTPUT, texts, texts_trans, labels, times)
