from enum import Enum
from typing import List
from colorama import Back


class Code(Enum):
    """
    Enum representing the possible colors in a Wordle game
    """

    UNKNOWN = 0
    GREY = 1
    YELLOW = 2
    GREEN = 3

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_color(self):
        """
        Getter for the matching colorama color of the enum value
        :return: Colorama color
        """
        colormap = {1: Back.LIGHTWHITE_EX, 2: Back.YELLOW, 3: Back.GREEN}

        return colormap.get(self.value, Back.RED)


MAX_CODE_VALUE = max(
    [Code.GREEN.value, Code.YELLOW.value, Code.GREY.value, Code.UNKNOWN.value]
)

MIN_CODE_VALUE = min(
    [Code.GREEN.value, Code.YELLOW.value, Code.GREY.value, Code.UNKNOWN.value]
)


def code_to_int_list(code_list: List[Code]) -> List[int]:
    """
    Converts the given code list to a list of the matching code integer values
    :param code_list: List of Code enums
    :return: List of their integer values
    """
    return [c.value for c in code_list]


def int_list_to_score(int_list: List[int]) -> int:
    """
    Converts a list of integers (between 0 <= value <= 3) into a integer number
    :param int_list: List that will be turned into a number
    :return: Integer score of the list
    """
    base = 1

    output = 0

    for i in int_list:
        output += base * (MAX_CODE_VALUE - i)
        base *= 4

    return output


def code_to_int(code_list: List[Code]) -> int:
    """
    Converts a list Code enums into a integer number
    :param int_list: List that will be turned into a number
    :return: Integer score of the list
    """
    return int_list_to_score(code_to_int_list(code_list))


def int_to_code(int_code: int, word_length: int) -> List[Code]:
    """
    Converts an integer number back into a List of Code enums
    :param int_code: Integer score
    :param word_length: Length of the origial list
    :return: List of Code enums
    """
    code_str = format(int_code, "b").zfill(2 * word_length)

    code_list: List[Code] = []

    for index in range(len(code_str) - 1, 0, -2):
        current_segment = code_str[index - 1 : index + 1]
        code_list.append(Code(MAX_CODE_VALUE - int(current_segment, 2)))

    return code_list


class Guess:
    """
    Represents the score of a Wordle guess

    :param code: List of word length containing the color of the guess at the position as Code enum
    """

    def __init__(self, code_list: List[Code] = None):
        self.code = code_list
        if self.code is None:
            self.code = []

    def to_int(self) -> int:
        """
        Converts the code into an integer
        :return: Integer score
        """
        return code_to_int(self.code)

    def from_int(self, code: int, word_length: int):
        """
        Loads the code from an integer
        :param code: Integer score
        :param word_length: Length of the origial list
        """
        self.code = int_to_code(code, word_length)

    def get_score(self):
        """
        Converts the code into an integer score without positional information
        :return: Integer score without positional information
        """
        global MAX_CODE_VALUE
        score = len(self.code) * MAX_CODE_VALUE

        for letter in self.code:
            score -= letter.value

        return score

    def get_colors(self):
        """
        Converts the code into colorama colors
        :return: List of colorama colors of the same size as code
        """
        return [c.get_color() for c in self.code]

    def __str__(self):
        return str(self.get_score())

    def __repr__(self):
        return str(self.code)
