import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn

# 'datasets' is a huggiface library with datasets such as squad/glue
from datasets import load_dataset
from numpy import ndarray
from pandas import DataFrame, Series

from constants import DATA, DATA_LIMIT


class Phase(Enum):
    # huggingface datasets don't have a seperate test split
    Train = 'train'
    Val = 'validation'
    # Test = 'test'

def ensure_dir(path: Path) -> bool:
    if path.exists():
        return True
    return False

def download_data(phase: Phase) -> None:
    print('Getting data..')
    df = load_dataset('squad', split=f'{phase.value}')
    df = DataFrame(data=df)
    df.to_csv(f'data/{phase.value}_df.csv', index=False)

def load_data(phase: Phase) -> DataFrame:
    dir = DATA / f"{phase.value}_df.csv"
    data_exists = ensure_dir(dir)
    if data_exists:
        df = pd.read_csv(dir)
    else:
        download_data(phase=phase)
        df = load_data(phase=phase)
    return df

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = text.strip()

    return text

def normalize_data(df: DataFrame, phase: Phase) -> None:
    print('Normalizing data..')
    df = df[["question", "answers"]]
    df["question"] = df["question"].apply(normalize_text)
    df["answers"] = df["answers"].apply(normalize_text)
    df = df.iloc[:DATA_LIMIT]
    df.to_csv(DATA/f'normalized_{phase.value}_df.csv')
    print('Saved normalized data.')
