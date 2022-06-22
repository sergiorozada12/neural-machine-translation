import sys

import pandas as pd

from src.model import Translator
from src.utils import save_dataset
from src.config import REGEX, NAME, MODEL_CONFIG

PATH_INPUT = sys.argv[1]
PATH_OUTPUT = sys.argv[2]


if __name__ == "__main__":
    df = pd.read_csv(PATH_INPUT, sep=';')

    texts = df['text'].values
    labels = df['label'].values

    translator = Translator(NAME, REGEX, MODEL_CONFIG)
    texts_trans, times = translator.translate_dataset(texts)

    save_dataset(PATH_OUTPUT, texts, texts_trans, labels, times)
