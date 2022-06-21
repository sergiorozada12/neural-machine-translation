from typing import List
from typing import List, Union
import pandas as pd


def save_dataset(
    name: str,
    text: List[str],
    text_trans: List[str],
    label: List[Union[str, bool]],
    time: List[float]
):
    pd.DataFrame({
        'text': text,
        'text_translated': text_trans,
        'time': time,
        'label': label
    }).to_csv(name, sep=';')