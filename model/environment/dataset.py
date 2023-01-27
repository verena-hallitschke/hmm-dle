from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

from model import Dictionary, Game
from model.environment.guess import Code
from model.environment.score import create_wordle_score_matrix
from model.util.constants import MAX_GUESS_NUMBER


def create_empty_row(word_length: int):
    """
    Creates an empty row for the dataset (row of a sequence without any guesses)
    :param word_length: Length of the words
    :return: Row for the dataset
    """
    game_row = []
    for guess_index in range(MAX_GUESS_NUMBER):
        game_row.extend([-1 for _ in range(word_length)])
        game_row.extend([Code.UNKNOWN.value for _ in range(word_length)])
    return game_row


def create_dataset(
    dictionary: Dictionary, num_games: int = 5, add_first=True, hmm_conform=False
):
    """
    Simulates num_games for each possible solution word in the dictionary and creates a dataset
    :param dictionary: Dictionary containing all legal words
    :param num_games: (optional, defaults to 5) Number of games that will be simulated for each solution
    :param add_first: (optional, defaults to True) Whether an empty row should be added to the dataset for each game
    :param hmm_conform: (optional, defaults to False) Whether the sequences should be extended to contain the true
        guess in the end (needed for the HMM training)
    :return: Tuple of a numpy array of shape (length, num guesses, 2 * word length) containing the sequences and
        a numpy array of shape (length) containing the solution words as indices in the dictionary
    """
    game_results = []

    solutions = []

    word_usage = np.zeros((len(dictionary.words), len(dictionary.words)), dtype=np.int)
    score_matrix = create_wordle_score_matrix(dictionary)
    with tqdm(dictionary.words, desc=f"Generating games") as progress_bar:
        for s_index, word in enumerate(progress_bar):

            solution_code = dictionary.word_to_int(word)

            for current_game_num in range(num_games):
                progress_bar.set_postfix(
                    game_number=f"{str(current_game_num + 1).zfill(len(str(num_games)))}/{num_games}",
                    refresh=True,
                )

                c_game = Game(dictionary, solution=word, score_matrix=score_matrix)

                game_row = create_empty_row(dictionary.word_length)

                # Add initial
                if add_first:
                    if hmm_conform:
                        new_row = game_row.copy()
                        new_row[0 : dictionary.word_length] = solution_code
                        new_row[dictionary.word_length : 2 * dictionary.word_length] = [
                            Code.GREEN.value for _ in range(dictionary.word_length)
                        ]
                        game_results.append(new_row)
                    else:
                        game_results.append(game_row.copy())
                    solutions.append(s_index)

                game_index = 0
                while not c_game.is_over():
                    # Guess word
                    r_word = None

                    while r_word is None:
                        # Do not use same word twice in one game
                        r_word, r_index = dictionary.get_random_word()

                        if r_word in c_game.previous_guesses:
                            r_word = None

                    word_usage[s_index][r_index] += 1
                    # r_code = convert_word_to_code(r_word)
                    reply = c_game.guess(r_index)

                    score = reply.code
                    word_repr = dictionary.word_to_int(r_word)

                    for letter_ind in range(dictionary.word_length):
                        c_ind = 2 * game_index * dictionary.word_length + letter_ind
                        game_row[c_ind] = word_repr[letter_ind]
                        game_row[dictionary.word_length + c_ind] = score[
                            letter_ind
                        ].value

                    if not c_game.is_solved() or not add_first:
                        # Don't add solved games since the model won't be able to continue guessing
                        if hmm_conform and game_index + 1 < MAX_GUESS_NUMBER:
                            new_row = game_row.copy()
                            current_index = (
                                (game_index + 1) * dictionary.word_length * 2
                            )
                            new_row[
                                current_index : current_index + dictionary.word_length
                            ] = solution_code
                            new_row[
                                current_index
                                + dictionary.word_length : current_index
                                + 2 * dictionary.word_length
                            ] = [
                                Code.GREEN.value for _ in range(dictionary.word_length)
                            ]
                            game_results.append(new_row)
                        else:
                            game_results.append(game_row.copy())
                        solutions.append(s_index)
                    elif (
                        c_game.is_solved()
                        and hmm_conform
                        and game_index + 1 < MAX_GUESS_NUMBER
                    ):
                        new_row = game_row.copy()
                        current_index = (game_index + 1) * dictionary.word_length * 2
                        new_row[
                            current_index : current_index + dictionary.word_length
                        ] = solution_code
                        new_row[
                            current_index
                            + dictionary.word_length : current_index
                            + 2 * dictionary.word_length
                        ] = [Code.GREEN.value for _ in range(dictionary.word_length)]
                        game_results.append(new_row)
                        solutions.append(s_index)

                    game_index += 1

    game_results = np.array(game_results)
    game_results = np.reshape(
        game_results,
        (game_results.shape[0], MAX_GUESS_NUMBER, dictionary.word_length * 2),
    )
    return game_results, np.array(solutions)


def create_lstm_tensors(
    dictionary: Dictionary, num_games: int = 5,
):
    """
        Simulates num_games for each possible solution word in the dictionary and creates a dataset
        :param dictionary: Dictionary containing all legal words
        :param num_games: (optional, defaults to 5) Number of games that will be simulated for each solution)
        :return: Tuple of a float tensor of shape (length, num guesses, 2 * word length) containing the sequences and
            a long tensor array of shape (length) containing the solution words as indices in the dictionary
        """
    game_results, solutions = create_dataset(dictionary, num_games=num_games,)

    game_tensor = torch.from_numpy(game_results).float()
    solutions_tensor = torch.from_numpy(solutions).long()

    return game_tensor, solutions_tensor
