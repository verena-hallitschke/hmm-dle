from __future__ import annotations

import json
import random
import time
from datetime import datetime
from typing import Callable, List

import os
from warnings import warn

import numpy as np
import pickle

from hmmlearn import hmm
from tqdm.auto import tqdm

from model.abstract_model import AbstractModel
from model import Dictionary
from model.util import load_settings

from model.environment.dataset import create_dataset
from model.environment import Simulation
from model.util.constants import MAX_WORD_LENGTH, PROJECT_ROOT_DIR
from model.environment.guess import int_list_to_score, MIN_CODE_VALUE


def check_if_converged(model: hmm.MultinomialHMM):
    """
    Helper function to check if a HMM model has converged
    :param model: HMM model from hmmlearn
    :return: True if the model converged else False
    """
    return (
        model is not None
        and model.monitor_.converged
        and not np.isnan(model.monitor_.history[-1])
    )


class HMMdle(AbstractModel):
    """
    HMM model for solving Wordle. Uses 2 component MultinominalHMMs for each word in the dictionary.
    The sub-models use the score as integer as observations. The word matching the submodel with the highest probability
    to emmit 0 is used as prediction.

    :param hmms: List of sub HMMs
    """

    def __init__(self, dictionary: Dictionary):
        super().__init__(dictionary)

        self.hmms: List[hmm.MultinomialHMM] = [
            None for _ in range(len(dictionary.words))
        ]
        self.max_reply_code = int_list_to_score(
            [MIN_CODE_VALUE for _ in range(MAX_WORD_LENGTH)]
        )

        self.best = None

    def get_unconverged(self) -> List[int]:
        """
        Getter for a list of sub-model indices that have not converged yet
        :return: List of sub-model indices that have not converged yet
        """
        num_not_converged = []
        for m_ind, w_model in enumerate(self.hmms):
            if not check_if_converged(w_model):
                num_not_converged.append(m_ind)
        return num_not_converged

    def load_from_directory(self, path: str):
        """
        Loads the model from the given directory
        :param path: Path to the directory containing the model.
        """
        self.load(os.path.join(path, "model.pkl"))

    def load_best(self):
        """
        Loads the best model configuration
        """
        self.load_from_directory(self.best)

    def save(self, path: str):
        """
        Saves the model parameters and a metadata file to the given directory
        :param path: Directory path the files will be saved to
        """
        meta_dict = {
            "dictionary_size": len(self.dictionary.words),
            "word_length": self.dictionary.word_length,
            "training_complete": None not in self.hmms,
            "type": type(self).__name__,
        }
        self.best = path
        with open(os.path.join(path, "model.pkl",), "wb") as model_file:
            pickle.dump(self, model_file, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "metadata.json"), "xt") as meta_file:
            json.dump(meta_dict, meta_file, indent=2)

    def update(self, new_data: HMMdle):
        """
        Updates the internal parameters with the ones of new_data
        :param new_data: Other HMMdle instance. Used for updating own parameters
        """
        self.hmms = new_data.hmms
        self.dictionary = new_data.dictionary
        self.max_reply_code = new_data.max_reply_code

    def load(self, path: str):
        """
        Loads model parameters from a "model.pkl" file.
        :param path: Path to the "model.pkl" file that will be loaded
        """
        with open(path, "rb") as model_file:
            new_data = pickle.load(model_file)

        self.update(new_data)

    def predict(self, X):
        """
        Predicts the Wordle game solution. Only supports one single sequence!
        :param X: Input sequences of shape (batch_size, num_maximum_guesses, 2 * word_length).
            Each entry consists of [dictionary representation (size word_length) score (size word_length)
        :return: List of predicted indices in the dictionary
        """
        # Only supports single prediction
        # Convert to sequence
        helper = np.argwhere(np.all(X != -1, axis=2))

        guess_start = np.zeros(X.shape[0], dtype=np.int64)
        guess_start[helper[:, 0]] = helper[:, 1]

        guess_start += 1

        if len(helper) > 0:
            words = [
                self.dictionary.int_to_word(code)
                for code in X[0, 0 : guess_start[0], : self.dictionary.word_length]
            ]
            replies = [
                int_list_to_score(score)
                for score in X[0, 0 : guess_start[0], self.dictionary.word_length :]
            ]
            replies.append(0)  # Get probability for observing 0
            replies = np.array(replies)
        else:
            words = []
            replies = np.array([0])

        one_hot_sequence = np.zeros(
            (replies.shape[0], self.max_reply_code + 1), dtype=np.int64
        )
        one_hot_sequence[np.arange(0, replies.shape[0]), replies] = 1

        best_prob = None
        best_seq = None

        alternative = None
        alt_prob = None

        for w_ind, model in enumerate(self.hmms):
            log_prob, state_sequence = model.decode(
                one_hot_sequence, lengths=np.array([replies.shape[0]])
            )

            if not np.isinf(log_prob) and (best_prob is None or log_prob > best_prob):

                # Avoid words that have already been guessed
                if self.dictionary.words[w_ind] in words:
                    if alt_prob is None or log_prob > alt_prob:
                        alternative = w_ind
                        alt_prob = log_prob
                else:
                    best_prob = log_prob
                    best_seq = w_ind

        if best_prob is None:
            best_prob = alt_prob
            best_seq = alternative

        if best_prob is None:
            # Select randomly
            print("Could not find a guess, choosing randomly!")
            best_seq = random.randint(0, len(self.hmms) - 1)

        return [best_seq]

    def is_converged(self):
        return len(self.get_unconverged()) == 0

    @staticmethod
    def get_train_function() -> Callable[[Dictionary], AbstractModel]:
        """
        Returns the train function for this class
        :return: The train function (callable)
        """
        return train


def train(dictionary: Dictionary) -> AbstractModel:
    """
    Creates a new model and trains it using the given dictionary and the settings in config.ini
    :param dictionary: Dictionary containing legal words
    :return: Trained model
    """
    model = HMMdle(dictionary)
    percentage = float(load_settings().get("MODELS", "game_percentage", fallback=0.8))
    num_games = round(percentage * len(dictionary.words))
    check_point_path = os.path.join(
        PROJECT_ROOT_DIR,
        load_settings().get("MODELS", "trained_model_path", fallback="trained"),
        datetime.now().strftime("%Y%m%d%H%M%S"),
        "checkpoints",
    )
    if not os.path.exists(check_point_path):
        os.makedirs(check_point_path)

    print(f'Saving checkpoints to "{check_point_path}".')

    start_time = time.time()
    dataset_iteration = 0
    max_dataset_it = int(
        load_settings().get("MODELS", "max_num_dataset_regen", fallback=5)
    )

    number_iterations = int(
        load_settings().get("MODELS", "number_iterations", fallback=50)
    )
    while not model.is_converged() and dataset_iteration < max_dataset_it:
        unconverged = model.get_unconverged()
        print(
            f"Running iteration {dataset_iteration + 1}, {len(unconverged)}/{len(model.hmms)} not converged"
        )
        print(f"Currently unconverged: {unconverged}")
        X, y = create_dataset(
            dictionary, num_games=num_games, add_first=True, hmm_conform=True
        )

        # Get ends of the sequences
        helper = np.argwhere(np.all(X != -1, axis=2))
        guess_start = np.zeros(X.shape[0], dtype=np.int64)
        guess_start[helper[:, 0]] = helper[:, 1]

        guess_start += 1

        sequences = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))
        sequences = sequences[np.argwhere(np.all(sequences != -1, axis=1)).flatten()]
        starts = np.zeros(guess_start.shape[0] + 1, dtype=np.int64)
        starts[1:] = np.cumsum(guess_start)

        encoded_sequences = np.empty((sequences.shape[0], 2), dtype=np.int64)

        def map_to_ind(x):
            return dictionary.get_index(dictionary.int_to_word(x))

        def map_to_code(x):
            return int_list_to_score(x)

        encoded_sequences[:, 0] = np.apply_along_axis(
            map_to_ind, axis=1, arr=sequences[:, :MAX_WORD_LENGTH]
        )
        encoded_sequences[:, 1] = np.apply_along_axis(
            map_to_code, axis=1, arr=sequences[:, MAX_WORD_LENGTH:]
        )

        max_reply_code = int_list_to_score(
            [MIN_CODE_VALUE for _ in range(MAX_WORD_LENGTH)]
        )
        with tqdm(
            enumerate(dictionary.words), desc="Models", total=len(dictionary.words)
        ) as progress_bar:
            for w_index, word in progress_bar:
                w_model = model.hmms[w_index]

                if w_index not in unconverged:
                    continue

                progress_bar.set_postfix(word=word, refresh=True)

                index_subset = np.argwhere(y == w_index).flatten()
                lengths = guess_start[index_subset]

                first = starts[index_subset[0]]
                last = starts[index_subset[-1] + 1]

                current_sequences = encoded_sequences[first:last]

                one_hot_sequence = np.zeros(
                    (current_sequences.shape[0], max_reply_code + 1), dtype=np.int64
                )
                one_hot_sequence[
                    np.arange(0, current_sequences.shape[0]), current_sequences[:, 1]
                ] = 1

                # create one state for each possible output code in the dictionary
                n_components = 2

                emission_prob = np.zeros((n_components, max_reply_code + 1))
                emission_prob[1, 0] = 1.0
                emission_prob[0, 1:] = 1.0 / max_reply_code

                # Adjust to 2
                emission_prob[0, 1] += n_components - np.sum(emission_prob)

                # Define initial transition matrix since there are no transitions from w_ind out in the dataset
                transmat = np.full((n_components, n_components), 1.0 / n_components)
                transmat[1] = np.zeros(n_components)
                transmat[1, 1] = 1.0

                w_model = hmm.MultinomialHMM(
                    n_components=n_components,
                    n_iter=number_iterations,
                    init_params="st",
                    params="ste",
                )

                w_model.n_features = max_reply_code + 1
                w_model.emissionprob_ = emission_prob

                w_model = w_model.fit(
                    one_hot_sequence, lengths
                )
                w_model.init_params = ""

                if not check_if_converged(w_model):
                    w_model = None
                model.hmms[w_index] = w_model
                weights_path = os.path.join(
                    check_point_path,
                    datetime.now().strftime("%Y%m%d%H%M%S") + f"_{w_index}",
                )

                if not os.path.exists(weights_path):
                    os.mkdir(weights_path)
                model.save(weights_path)
        dataset_iteration += 1

    model.save(os.path.dirname(check_point_path))

    if not model.is_converged():
        num_not_converged = model.get_unconverged()
        warn(f"{len(num_not_converged)} out of {len(model.hmms)} have not converged!")
        warn(f"List of unconverged submodels: {num_not_converged}")

    end_time = time.time()

    print(f"Finished training in {end_time - start_time} s")

    return model


if __name__ == "__main__":
    from model.dictionary.file_handling import GoogleLoader

    dictionary_file_path = load_settings().get(
        "DICTIONARY", "filtered_dictionary_path", fallback=None
    )
    if (
        load_settings()
        .get("DICTIONARY", "use_filtered_dictionary", fallback="false")
        .lower()
        in ["t", "true"]
        and dictionary_file_path is not None
    ):
        loader = GoogleLoader(MAX_WORD_LENGTH, dictionary_file_path)
    else:
        loader = GoogleLoader(MAX_WORD_LENGTH)

    google_dict = Dictionary(loader)

    model = HMMdle(google_dict)
    model.load_from_directory(
        r"C:\Users\veren\erasmus_projekte\hmm-dle\trained\20230108183615"
    )

    model = train(google_dict)

    sim = Simulation(model, google_dict, targets=google_dict.words)

    sim.run(verbose=True)
    a = 1
