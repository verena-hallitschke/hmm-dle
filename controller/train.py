"""
Implements the click interface for the train function

Get more information using python cli.py train --help
"""

import click
import random

from model import get_model, Dictionary
from model.util import load_settings
from model.util.constants import NUM_GAMES_DEFAULT, MAX_WORD_LENGTH
from model.dictionary.file_handling import get_handler
from model.environment.simulation import Simulation


@click.command()
@click.option(
    "--num-games",
    "-n",
    "num_games",
    default=NUM_GAMES_DEFAULT,
    help=f"Number of games that will be simulated. Simulation is skipped if <= 0. Defaults to {NUM_GAMES_DEFAULT}.",
)
@click.option(
    "--num-folds",
    "-f",
    "num_folds",
    default=1,
    help="Number of models that will be trained. Defaults to 1.",
)
@click.argument("model_name")
def train(model_name, num_games, num_folds):
    """
    Train a model of type MODEL_NAME to solve Wordle. Training uses the settings in config.ini.

    MODEL_NAME is the name of the model. Has to be either "lstm" or "hmm"
    """
    # Init dicionary
    file_handler = get_handler(
        load_settings().get("DICTIONARY", "dictionary_type", fallback="")
    )

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
        file_handler = file_handler(MAX_WORD_LENGTH, dictionary_file_path)
    else:
        file_handler = file_handler(MAX_WORD_LENGTH)

    dictionary = Dictionary(file_handler)

    model_class = get_model(model_name)
    train_func = model_class.get_train_function()

    for _ in range(num_folds):
        model = train_func(dictionary)

        if num_games <= 0:
            return

        model.load_best()

        test_indices = random.sample(
            range(len(dictionary.words)), min(num_games, len(dictionary.words))
        )
        test_targets = [dictionary.words[ind] for ind in test_indices]
        simulation = Simulation(model, dictionary, test_targets)

        simulation.run(verbose=True)
