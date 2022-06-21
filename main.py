import sys

import pandas as pd

from src.model import Translator


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], sep=';')

    text = df['text'].values()
    labels = df['label'].values()

    print(df)
