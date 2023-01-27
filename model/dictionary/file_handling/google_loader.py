import os
import json
from typing import List
import pandas as pd


import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model.util import load_settings
from model.util.constants import (
    MAX_WORD_LENGTH,
    GOOGLE_DICTIONARY_FILE_PATH,
    GOOGLE_DICTIONARY_BASE_PATH,
)
from model.dictionary.file_handling.abstract_loader import AbstractLoader


class GoogleLoader(AbstractLoader):
    """
    Loader for the words from the following Kaggle Dataset:
    https://www.kaggle.com/datasets/rtatman/english-word-frequency

    Expects unigram_freq.csv or google_<word_length>_<threshold>.json to be located in model/dictionary/ressources/ or
    an explicit file path to a file created by create_dictionary_file
    """

    def __init__(self, word_length: int = MAX_WORD_LENGTH, path: str = None):
        """
        Initializes a new loader
        :param word_length: Word length of the words in the dictionary
        :param path: (optional) Path to a specific resource file. If not given the default file is used/created.
        """
        super().__init__(word_length=word_length, path=path)
        self.threshold = int(
            load_settings().get("DICTIONARY", "threshold", fallback=400000000)
        )

        if path is None:
            self._update_file_path()

    def _update_file_path(self):
        """
        Updates the default file path based on the selected threshold and word length
        """
        self.file_path = GOOGLE_DICTIONARY_FILE_PATH.format(
            self.word_length, self.threshold
        )

    def load(self) -> List[str]:
        """
        Loads the word list from self.file_path. If the file does not exist a new one is created.
        :return: List of loaded words
        """
        if self.file_path is None:
            self._update_file_path()
        return super().load()

    def create_dictionary_file(self):
        """
        Creates a new dictionary file by filtering the unigram_freq.csv based on the word length and threshold.
        Saves the new file to model/dictionary/ressources/
        """
        df = pd.read_csv(GOOGLE_DICTIONARY_BASE_PATH)

        # Only keep words of certain length
        df["word"] = df["word"].astype("str")
        df = df[df["word"].str.len() == self.word_length]

        # Only keep words above certain usage threshold
        df = df[df["count"] > self.threshold]

        # Convert to lower case
        df["word"] = df["word"].str.lower()

        word_list = df["word"].tolist()

        # Save file
        modifier = "wt" if os.path.exists(self.file_path) else "xt"

        with open(self.file_path, modifier) as output_file:
            # save as json
            json.dump({AbstractLoader.WORD_LIST_KEY: word_list}, output_file)
