"""
Implements the click interface for the predict function

Get more information using python cli.py predict --help
"""

import click
import random
import os
import glob
import json

from model import get_model, Dictionary
from model.util import load_settings
from model.util.constants import PROJECT_ROOT_DIR, NUM_GAMES_DEFAULT, MAX_WORD_LENGTH
from model.dictionary.file_handling import get_handler
from model.environment.simulation import Simulation


@click.command()
@click.option(
    "--num-games",
    "-n",
    "num_games",
    default=NUM_GAMES_DEFAULT,
    help=f"Number of games that will be simulated. Defaults to {NUM_GAMES_DEFAULT}.",
)
@click.option(
    "--model-path",
    "-m",
    "model_path",
    default=None,
    help="Path to the model weights, if not given it will use the most recent one in the directory defined in the config file.",
)
@click.argument("model_name")
def predict(model_name, num_games, model_path):
    """
        Simulate Wordle games using a model of type MODEL_NAME. Prediction uses the settings in config.ini.

        MODEL_NAME is the name of the model. Has to be either "lstm" or "hmm"
    """
    if num_games <= 0:
        print(
            f"Illegal number of games ({num_games}), setting to default: {NUM_GAMES_DEFAULT}."
        )
        num_games = NUM_GAMES_DEFAULT

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

    model = get_model(model_name)(dictionary)

    # Load model
    if model_path is None:
        folder_path = os.path.join(
            PROJECT_ROOT_DIR,
            load_settings().get("MODELS", "trained_model_path", fallback="trained"),
        )
        directories = glob.glob(os.path.join(folder_path, "*"))
        directories = filter(os.path.isdir, directories)
        directories = list(reversed(sorted(directories, key=os.path.getctime)))

        for d in directories:
            # Check for metadata path
            metadata_path = os.path.join(d, "metadata.json")
            if not os.path.exists(metadata_path):
                print(f'Could not find metadata file in "{metadata_path}"')
                continue

            with open(metadata_path, "rt") as meta_file:
                meta_dict = json.load(meta_file)

            if meta_dict.get("type") is None:
                print(f'Malformed meta file "{metadata_path}"')
                continue
            if meta_dict["type"].lower() != type(model).__name__.lower():
                continue

            model_path = d
            break

        if model_path is None:
            raise ValueError(f'Could not find matching model file in "{folder_path}"!')
    else:
        metadata_path = os.path.join(model_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise ValueError(
                f'Could not find metadata file in model directory! "{metadata_path}"'
            )

        with open(metadata_path, "rt") as meta_file:
            meta_dict = json.load(meta_file)

        if meta_dict.get("type") is None:
            raise ValueError("Missing type field in metadata file!")
        if meta_dict["type"].lower() != type(model).__name__.lower():
            raise ValueError(
                f"Model type and type declared in metadata file are not matching! {type(model).__name__.lower()} != {meta_dict['type'].lower()}"
            )

    model.load_from_directory(model_path)

    test_indices = random.sample(
        range(len(dictionary.words)), min(num_games, len(dictionary.words))
    )
    test_targets = [dictionary.words[ind] for ind in test_indices]
    simulation = Simulation(model, dictionary, test_targets)

    simulation.run(verbose=True)
