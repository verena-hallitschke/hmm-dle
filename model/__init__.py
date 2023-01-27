from typing import Callable

from .dictionary.dictionary import Dictionary
from .environment import Game
from model.lstm.lstm import WordleLSTM
from model.hmm.hmm import HMMdle
from model.abstract_model import AbstractModel


def get_model(name: str) -> AbstractModel:
    if name.lower() == "lstm":
        return WordleLSTM
    if name.lower() == "hmm":
        return HMMdle

    raise ValueError(f'Unknown model name "{name}"!')
