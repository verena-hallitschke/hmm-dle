from collections import defaultdict

import numpy as np

from model import Dictionary
from model.environment.guess import Code, Guess


def score(guess: str, solution: str) -> Guess:
    """
    Calculates the Wordle score for the given words
    :param guess: Word that is being guessed
    :param solution: The solution word of the game
    :return: The game score encapsulated by the Guess class.
    """
    guess_letter_count = defaultdict(lambda: 0)
    solution_letter_count = defaultdict(lambda: 0)

    reply_code = []
    num_green = 0
    for index, letter in enumerate(guess):
        guess_letter_count[letter] += 1
        solution_letter_count[solution[index]] += 1

        code = Code.GREY

        if letter == solution[index]:
            code = Code.GREEN
            num_green += 1
        elif letter in solution:
            code = Code.YELLOW

        reply_code.append(code)

    for letter in solution_letter_count.keys():
        if guess_letter_count[letter] > 0:
            guess_letter_count[letter] -= solution_letter_count[letter]

    # Parse second time
    for index, letter in enumerate(guess):

        if solution_letter_count[letter] == 0:
            continue
        # Letter is contained in solution, check number of times
        # Example: solution: paste, guess: sense. The color code should be:
        # yellow, grey, grey, grey, green
        if reply_code[index] == Code.GREEN:
            continue
        if guess_letter_count[letter] > 0:
            reply_code[index] = Code.GREY
        guess_letter_count[letter] -= 1

    return Guess(reply_code)


def create_wordle_score_matrix(dictionary: Dictionary):
    """
    Creates a matrix of shape (dictionary size, dictionary size) that maps a solution word and a guess to
    their wordle score (using their indices in the dictionary)
    i.e. score_matrix[dictionary.get_index(solution), dictionary.get_index(guess)]
    :param dictionary: Dictionary containing all legal words
    :return: Matrix of shape (dictionary size, dictionary size)
    """
    wordle_score_matrix = np.empty(
        (len(dictionary.words), len(dictionary.words)), dtype=np.int64
    )

    for s_ind, solution in enumerate(dictionary.words):
        for g_ind, guess in enumerate(dictionary.words):
            reply = score(guess, solution)
            wordle_score_matrix[s_ind, g_ind] = reply.to_int()

    return wordle_score_matrix
