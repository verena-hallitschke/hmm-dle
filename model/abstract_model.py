from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Callable

from model import Dictionary


class AbstractModel(ABC):
    """
    Abstract Model interface

    :param dictionary: Dictionary containing legal words
    """

    def __init__(self, dictionary: Dictionary):
        self.dictionary = dictionary

    @abstractmethod
    def predict(self, X):
        """
        Performs a prediction on X
        :param X: Input sequence (length, num_maximum_guesses, 2 * word_length).
            Each entry consists of [dictionary representation (size word_length) score (size word_length).
            The sequences are padded with [-1, -1, ..., -1, Code.UNKNOWN.value, ..., Code.UNKNOWN.value] for each guess
            that has not yet been made.
        :return: Array/Tensor containing the predicted dictionary indices (shape (length))
        """
        raise NotImplementedError("ABC function not implemented!")

    @abstractmethod
    def load_from_directory(self, path: str):
        """
        Loads the model from the given directory
        :param path: Path to the directory containing the model.
        """
        raise NotImplementedError("ABC function not implemented!")

    @abstractmethod
    def load_best(self):
        """
        Loads the best model configuration
        """
        raise NotImplementedError("ABC function not implemented!")

    @staticmethod
    @abstractmethod
    def get_train_function() -> Callable[[Dictionary], AbstractModel]:
        """
        Abstract interface for the training function.
        :return: Training function of the class (callable)
        """
        raise NotImplementedError("ABC function not implemented!")
