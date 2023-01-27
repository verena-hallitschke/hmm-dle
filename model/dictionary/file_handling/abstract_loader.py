import os
import json
from abc import ABC, abstractmethod
from typing import List

from model.util.constants import MAX_WORD_LENGTH


class AbstractLoader(ABC):
    """
    Abstract interface for dictionary resource file loading
    """

    WORD_LIST_KEY = "word_list"

    def __init__(self, word_length: int = MAX_WORD_LENGTH, path: str = None):
        """
        Initializes a new loader
        :param word_length: Word length of the words in the dictionary
        :param path: (optional) Path to a specific resource file. If not given the default file is used/created.
        """
        self.file_path = path
        self.word_length = word_length

    def load(self) -> List[str]:
        """
        Loads the word list from self.file_path. If the file does not exist a new one is created.
        :return: List of loaded words
        """
        if not self.dictionary_file_exists():
            print(f'Could not find a dictionary file at "{self.file_path}".')
            print("Creating new one.")
            self.create_dictionary_file()

        with open(self.file_path, "rt") as output_file:
            # load json
            dictionary = json.load(output_file)

        return dictionary[AbstractLoader.WORD_LIST_KEY]

    @abstractmethod
    def create_dictionary_file(self):
        """
        Interface for the creation of a new dictionary file
        """
        raise NotImplementedError("Abstract Base Class.")

    def dictionary_file_exists(self):
        """
        Checks whether the default file of the Loader exists.
        :return: True when the file exists, False else.
        """
        return os.path.exists(self.file_path)
