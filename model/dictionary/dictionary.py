import random

from typing import Tuple, List

from model.dictionary.file_handling.abstract_loader import AbstractLoader


class Dictionary:
    """
    Represents the dictionary used in the wordle games. Used as mapping of the words from int index (one int for
    the whole word) to string to int letter embedding (one integer for each letter)
    """

    def __init__(self, loader: AbstractLoader):
        """
        Creates a dictionary using a file. The file is loaded by loader
        :param loader: AbstractLoader that loads the dictionary file
        """
        self.words = loader.load()
        self.word_length = loader.word_length

    def contains(self, word: str):
        """
        Checks whether a word is in the dictionary
        :param word: Word that will be looked up
        :return: True if word is in the dictionary, False else
        """
        return word.lower() in self.words

    def get_random_word(self) -> Tuple[str, int]:
        """
        Samples a random word and its index from the dictionary
        :return: Tuple(word, word index)
        """
        index = random.randint(0, len(self.words) - 1)
        return self.words[index], index

    def get_index(self, word: str) -> int:
        """
        Getter for the index of a word
        :param word: Word that will be looked up
        :raises ValueError: If word is not in dictionary
        """
        try:
            return self.words.index(word)
        except ValueError as e:
            raise ValueError(f'Unknown word "{word}"!') from e

    @staticmethod
    def word_to_int(word: str) -> List[int]:
        """
        Converts a word into its letter embedding
        :param word: Word that will be converted
        :return: List of integers of the same length as the input word
        """
        return [ord(letter) - ord("a") for letter in word.lower()]

    @staticmethod
    def int_to_word(code: List[int]):
        """
        Converts a letter embedding back to a string word
        :param code: Letter embedding that will be converted
        :return: Word as string
        """
        output = ""

        for letter in code:
            output += chr(letter + ord("a"))

        return output

    def load_int_code(self, code: List[int]):
        """
        Converts a letter embedding back to a string word and checks if the word is legal
        :param code: Letter embedding that will be converted
        :return: Word as string
        :raises ValueError: If the code is illegal or the word is unknown
        """
        if len(code) != self.word_length:
            raise ValueError(f"Input size mismatch! {len(code)} != {self.word_length}")

        output = self.int_to_word(code)

        if not self.contains(output):
            raise ValueError(f"Unknown word {output}!")

        return output
