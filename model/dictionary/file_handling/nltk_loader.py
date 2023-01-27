import os
import json
import nltk
from typing import List


import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model.util.constants import MAX_WORD_LENGTH, NLTK_DICTIONARY_FILE_PATH
from model.dictionary.file_handling.abstract_loader import AbstractLoader


def get_word_list():
    """
    Getter for the NLTK word list
    :return: Words as list
    """
    nltk.download("words")
    return nltk.corpus.words.words()


def filter_words_by_length(word_list: List[str], word_length: int) -> List[str]:
    """
    Filters the given list to only contain words of the given length
    :param word_list: List that will be filtered
    :param word_length: Length of the words
    :return: List only containing the words from word_list that have the length word_length
    """
    return list(filter(lambda x: len(x) == word_length, word_list))


class NLTKLoader(AbstractLoader):
    def __init__(self, word_length: int = MAX_WORD_LENGTH, path: str = None):
        """
        Initializes a new loader
        :param word_length: Word length of the words in the dictionary
        :param path: (optional) Path to a specific resource file. If not given the default file is used/created.
        """
        super().__init__(word_length=word_length, path=path)

        if path is None:
            self.file_path = NLTK_DICTIONARY_FILE_PATH.format(self.word_length)

    def create_dictionary_file(self):
        """
        Creates a new dictionary file by filtering the NLTK word bank on the word length.
        Saves the new file to model/dictionary/ressources/
        """
        # Read corpus
        word_list = get_word_list()

        # filter list by word length
        word_list = filter_words_by_length(word_list, self.word_length)

        # make all words lower case
        word_list = list(map(lambda x: x.lower(), word_list))

        modifier = "wt" if os.path.exists(self.file_path) else "xt"

        with open(self.file_path, modifier) as output_file:

            # save as json
            json.dump({AbstractLoader.WORD_LIST_KEY: word_list}, output_file)
