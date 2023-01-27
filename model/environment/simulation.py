import random
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np

from model import Dictionary
from model.abstract_model import AbstractModel
from model.environment.score import create_wordle_score_matrix
from model.util.constants import MAX_GUESS_NUMBER
from model.environment.guess import Code
from model.environment.game import Game


class Simulation:
    """
    Simulates games using the given targets as solution and keeps the final stats

    :param avg_tries: Average tries needed in each run of the simulation. Lost games count as 7 tries
    :param avg_tries_won: Average tries needed in each run of the simulation. Lost games are ignores (all lost = -1).
    :param percentage_won: Winrate for each run
    :param percentage_repeated_guesses: Percentage over all guesses that were repeated words for each run.
    :param percentage_repeated_games: Percentage of all games that contained repeated guesses for each run.
    :param won_games_per_word: Dictionary for all words. List of 0 for lost games and 1 for won games.
    :param word_confusion: Last guess in lost games for each solution word and each game.
    :param word_scores: Numerical score for each word.
    :param first_guesses: List of the first guess in each run.
    """

    def __init__(
        self, model: AbstractModel, dictionary: Dictionary, targets: List[str]
    ):
        self.model = model
        self.dictionary = dictionary
        self.targets = targets
        self._num_games = len(targets)

        self.avg_tries = []
        self.avg_tries_won = []
        self.percentage_won = []
        self.percentage_repeated_guesses = []
        self.percentage_repeated_games = []
        self.won_games_per_word = defaultdict(lambda: [])
        self.word_confusion = defaultdict(lambda: [])
        self.word_scores = defaultdict(lambda: [])
        self.first_guesses = []

        self.score_matrix = create_wordle_score_matrix(dictionary)

    def update_model(self, new_model: AbstractModel):
        """
        Upates the model that is evaluated
        :param new_model: New model
        """
        self.model = new_model

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialization of the score to a JSON conform dictionary
        :return: Dictionary containing the scores
        """
        scaled_won = {
            key: sum(value) / len(self.percentage_won)
            for key, value in self.won_games_per_word.items()
        }
        return {
            "win_rate": self.percentage_won,
            "avg_tries": self.avg_tries,
            "avg_tries_won": self.avg_tries_won,
            "percentage_repeated_guesses": self.percentage_repeated_guesses,
            "percentage_repeated_games": self.percentage_repeated_games,
            "won_games_per_word": self.won_games_per_word,
            "win_rate_per_word": scaled_won,
            "word_confusions": self.word_confusion,
            "last_won_per_game": {
                key: value[-1] for key, value in self.won_games_per_word.items()
            },
            "last_confusion": {
                key: value[-1] if key != value[-1] else ""
                for key, value in self.word_confusion.items()
            },
            "word_scores": self.word_scores,
            "last_word_scores": {
                key: value[-1] for key, value in self.word_scores.items()
            },
            "first_guesses": self.first_guesses,
        }

    def run(self, verbose: bool = False):
        """
        Evaluates the given model and saves the results
        :param verbose: (optional) Determines whether game stats should be printed to the console
        :return: Tuple(current win rate, current avg number of tries, current average number of tries w/o lost games)
        """
        number_of_tries = np.empty(self._num_games)
        scores = np.empty(self._num_games)
        number_won = 0
        number_repeated = 0
        number_games_w_repeated = 0

        first_guess_done = False

        shuffled_targets = random.sample(self.targets, self._num_games)

        for g_ind, solution_word in enumerate(shuffled_targets):
            current_game = Game(
                self.dictionary, solution=solution_word, score_matrix=self.score_matrix
            )

            game_row = []
            for guess_index in range(MAX_GUESS_NUMBER):
                game_row.extend([-1 for _ in range(self.dictionary.word_length)])
                game_row.extend(
                    [Code.UNKNOWN.value for _ in range(self.dictionary.word_length)]
                )

            guess_arr = np.array([game_row])
            guess_arr = np.reshape(
                guess_arr,
                (guess_arr.shape[0], MAX_GUESS_NUMBER, self.dictionary.word_length * 2),
            )

            guess_index = 0
            score_list = []
            repeated = 0
            while not current_game.is_over():
                word_ind = self.model.predict(guess_arr)[0]
                word = self.dictionary.words[word_ind]

                if not first_guess_done:
                    self.first_guesses.append(word)
                    first_guess_done = True

                score = current_game.guess(word_ind)
                score_list.append(score.get_score())

                guess_arr[
                    0, guess_index, 0 : self.dictionary.word_length
                ] = self.dictionary.word_to_int(word)

                for score_index in range(self.dictionary.word_length):
                    guess_arr[
                        0, guess_index, self.dictionary.word_length + score_index
                    ] = score.code[score_index].value

                if word in current_game.previous_guesses[:-1]:
                    repeated += 1

                guess_index += 1

            number_repeated += repeated
            number_games_w_repeated += min(1, repeated)
            number_of_tries[g_ind] = current_game.number_of_guesses
            scores[g_ind] = current_game.previous_replies[-1].to_int()

            self.word_confusion[solution_word].append(word)
            if current_game.is_lost():
                number_of_tries[g_ind] += 1

                self.won_games_per_word[solution_word].append(0)
            else:
                number_won += 1
                self.won_games_per_word[solution_word].append(1)

            self.word_scores[solution_word].append(score.get_score())

            if verbose:
                current_game.print_game()
                print(
                    "------------------------------------------------------------------------------"
                )

        avg_number_of_tries = np.average(number_of_tries)
        avg_without_lost = (
            (avg_number_of_tries * self._num_games - 7 * (self._num_games - number_won))
            / number_won
            if number_won != 0
            else -1
        )

        self.avg_tries.append(avg_number_of_tries)
        self.avg_tries_won.append(avg_without_lost)
        self.percentage_won.append(number_won / self._num_games)
        self.percentage_repeated_games.append(number_games_w_repeated / self._num_games)
        self.percentage_repeated_guesses.append(
            number_repeated / np.sum(np.minimum(number_of_tries, 6))
        )

        if verbose:
            print(
                f"Won {number_won}/{self._num_games} ({avg_number_of_tries:.2f} / {avg_without_lost:.2f}) tries, "
                + f"{self.percentage_repeated_games[-1]:.2%} of all games contain repeated guesses"
            )

        return self.percentage_won[-1], avg_number_of_tries, avg_without_lost
