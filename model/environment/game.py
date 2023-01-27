import sys
import os
from typing import List, Union

from colorama import Fore, Style

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model.environment.guess import Code, Guess
from model.environment.score import create_wordle_score_matrix
from model import Dictionary
from model.util.constants import MAX_GUESS_NUMBER


class Game:
    """
    Represents a game of Wordle with a (random) solution

    :param previous_guesses: List of the previous guesses in the game
    :param previous_replies: List of the previous replies to the guesses
    """

    def __init__(self, input_dict: Dictionary, solution: str = None, score_matrix=None):
        """
        Initializes a new Game
        :param input_dict: Dictionary containing the legal words
        :param solution: (optional) Solution of the game. If not given a random word from the dictionary is chosen
        :param score_matrix: (optional) Matrix mapping the solution and guess to a wordle score.
            If not given it is calculated after initialization
        """
        self.dictionary = input_dict

        self.solution = solution
        if self.solution is None or not self.dictionary.contains(self.solution):
            self.solution, self.solution_index = self.dictionary.get_random_word()
        else:
            self.solution_index = self.dictionary.get_index(self.solution)
        self.number_of_guesses = 0
        self.previous_guesses: List[str] = []
        self.previous_indices: List[int] = []
        self.previous_replies: List[Guess] = []
        self.solved = False

        self.score_matrix = score_matrix

        if self.score_matrix is None:
            create_wordle_score_matrix(self.dictionary)

    def guess(self, word: Union[str, int]) -> Guess:
        """
        Guess a solution for the current game
        :param word: Word that will be guessed either as a string or the index of the word in the dictionary.
        :return: Score of the guess
        """
        if self.is_lost():
            raise ValueError(
                f"Game is over. Maximum number of guesses is: {MAX_GUESS_NUMBER}"
            )

        guess_word = word
        guess_ind = word

        if isinstance(word, int):
            if 0 > word or word >= len(self.dictionary.words):
                raise ValueError(f'Unknown guess word "{word}"!')

            guess_word = self.dictionary.words[word]
        else:
            guess_ind = self.dictionary.get_index(word)

        if len(guess_word) > self.dictionary.word_length:
            raise ValueError(
                f"Word length mismatch ({len(guess_word)} != {self.dictionary.word_length})"
            )
        if self.solved:
            return None

        self.previous_guesses.append(guess_word)
        self.previous_indices.append(guess_ind)

        score = self.score_matrix[self.solution_index, guess_ind]
        guess = Guess()
        guess.from_int(score, self.dictionary.word_length)

        self.previous_replies.append(guess)
        self.number_of_guesses += 1

        # Check if game was solved
        self.solved = score == 0

        return guess

    def is_solved(self):
        """
        :return: Whether the game is solved
        """
        return self.solved

    def is_lost(self):
        """
        :return: Whether the game is lost
        """
        return not self.is_solved() and self.number_of_guesses >= MAX_GUESS_NUMBER

    def is_over(self):
        """
        :return: Whether the game is over
        """
        return self.is_solved() or self.is_lost()

    def print_game(self):
        """
        Prints the current game to the console using colorama
        """
        for word, code in zip(self.previous_guesses, self.previous_replies):
            output_str = ""
            for background_color, letter in zip(code.get_colors(), word):
                output_str += (
                    background_color
                    + Fore.BLACK
                    + Style.BRIGHT
                    + letter
                    + Style.RESET_ALL
                )

            print(output_str)

        if self.is_lost():
            print("-" * (self.dictionary.word_length + 10))
            output_str = ""
            for letter in self.solution:
                output_str += (
                    Code.GREEN.get_color()
                    + Fore.BLACK
                    + Style.BRIGHT
                    + letter
                    + Style.RESET_ALL
                )

            print("solution: " + output_str)


if __name__ == "__main__":
    from model.dictionary.file_handling import GoogleLoader
    from model.util.constants import MAX_WORD_LENGTH

    # Test the game
    loader = GoogleLoader(MAX_WORD_LENGTH)
    dictionary = Dictionary(loader)
    my_game = Game(dictionary, solution="there")

    my_game.guess("these")
    my_game.guess("about")

    my_game.print_game()
