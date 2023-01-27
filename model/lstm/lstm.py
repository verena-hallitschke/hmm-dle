import random
import os

import json
import glob
from math import log2, ceil
from typing import Callable

import torch
import numpy as np

from shutil import copyfile
from datetime import datetime
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from model import Dictionary
from model.abstract_model import AbstractModel
from model.environment.dataset import create_lstm_tensors
from model.dictionary.file_handling import GoogleLoader
from model.environment import Simulation
from model.util import load_settings
from model.util.constants import MAX_GUESS_NUMBER, MAX_WORD_LENGTH, PROJECT_ROOT_DIR


def repetition_penalty(X, target, prediction, embedding):
    """
    Penalty for guessing the same word multiple times
    :param X: Input sequences
    :param target: Target values (Index in dictionary)
    :param prediction: Model predictions (softmax no argmax)
    :param embedding: Embedding to transform the prediction and target indices into dictionary embedding
    :return: Scalar penalty value
    """
    selected_predictions = torch.argmax(prediction, dim=1)
    embeddings = embedding(selected_predictions)
    word_length = embeddings.shape[1]
    guess_number = X.shape[1]

    embedding_diff = torch.zeros_like(target).float()
    for index in range(guess_number):
        embedding_diff += torch.ones_like(target).float() - torch.minimum(
            torch.sum(torch.abs(X[:, index, :word_length] - embeddings), dim=1),
            torch.ones_like(target).float(),
        )

    embedding_diff = torch.minimum(embedding_diff, torch.ones_like(target).float())
    return torch.sum(embedding_diff)


class WordleLSTM(torch.nn.Module, AbstractModel):
    """
    LSTM model for solving Wordle. Most of the parameters are set using config.ini.

    :param model: PyTorch LSTM
    :param mapping: Linear Layer mapping from hidden dim -> dictionary size
    :param embedding: Mapping from dictionary index to word vector representation (see Dictionary.word_to_int)
    """

    def __init__(
        self, dictionary: Dictionary,
    ):
        """
        Creates a WordleLSTM instance as Pytorch Module
        :param dictionary: Dictionary containing legal words
        :type dictionary: Dictionary
        """
        torch.nn.Module.__init__(self)
        AbstractModel.__init__(self, dictionary)

        self.dictionary_size = len(dictionary.words)
        self.max_word_length = MAX_WORD_LENGTH
        self.max_guesses = MAX_GUESS_NUMBER
        self.hidden_dim = int(load_settings().get("MODELS", "hidden_dim", fallback=60))
        self.dropout = float(load_settings().get("MODELS", "dropout", fallback=0.3))
        self.num_layers = int(load_settings().get("MODELS", "num_layers", fallback=3))
        self.model = torch.nn.LSTM(
            2 * self.max_word_length,
            self.hidden_dim,
            batch_first=True,
            dropout=self.dropout,
            num_layers=self.num_layers,
        )
        self.mapping = torch.nn.Linear(
            in_features=self.hidden_dim, out_features=self.dictionary_size
        )
        self.device = "cpu"

        weight = torch.zeros((len(dictionary.words), dictionary.word_length))

        for index, word in enumerate(dictionary.words):
            c_embedding = dictionary.word_to_int(word)
            weight[index, :] = torch.Tensor(c_embedding).long()

        self.embedding = torch.nn.Embedding.from_pretrained(
            weight, freeze=True
        )

        self.current_best = None
        self.best_meta = {}
        self.check_point_path = None

    def forward(self, X):
        """

        :param X: Input sequences of shape (batch_size, num_maximum_guesses, 2 * word_length).
            Each entry consists of [dictionary representation (size word_length) score (size word_length)
        :return: Prediction after softmax but before argmax
        """
        batch_size = X.shape[0]
        _, (hn, _) = self.model(X)
        outputs = self.mapping(hn[0])

        outputs = torch.log_softmax(outputs, dim=1)
        return outputs

    def predict(self, X):
        """
        Predicts the Wordle game solution
        :param X: Input sequences of shape (batch_size, num_maximum_guesses, 2 * word_length).
            Each entry consists of [dictionary representation (size word_length) score (size word_length)
        :return: List of predicted indices in the dictionary
        """
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                predict_X = torch.from_numpy(X).float().to(self.device)
            elif isinstance(X, torch.Tensor):
                predict_X = X.float().to(self.device)
            else:
                raise TypeError(f'Unknown input type "{type(X).__name__}"!')

            predictions = self(predict_X)
            indices = torch.argmax(predictions, dim=1).tolist()

        return indices

    def save(self, path: str):
        """
        Saves the model weights to the given path
        :param path: Folder the weights will be saved to
        :return: Model weights file path
        """
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        saving_path = os.path.join(path, timestamp)

        if not os.path.exists(saving_path):
            os.mkdir(saving_path)

        file_path = os.path.join(saving_path, "model")
        torch.save(self.state_dict(), file_path)

        return file_path

    def load(self, path: str):
        """
        Loads model weights from a model weight file
        :param path: Path to model weight file
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_from_directory(self, path: str):
        """
        Loads model weights from a given folder. Crawls the folder for "model" file, "checkpoints/model" file or
        "checkpoints/*/model" file
        :param path: Path to folder
        """
        # check if model file is already in parent folder

        model_file_path = os.path.join(path, "model")

        if not os.path.exists(model_file_path):
            # try checkpoint folder
            model_file_path = os.path.join(path, "checkpoints")
            if not os.path.exists(model_file_path):
                raise ValueError(
                    f'Could not find "model" file in "{path}" and "{model_file_path}"!'
                )

            # try loading best folder
            model_file_path = os.path.join(model_file_path, "best")

            if not os.path.exists(model_file_path) or not os.path.exists(
                os.path.join(model_file_path, "model")
            ):
                # Load newest
                model_file_path = os.path.join(path, "checkpoints")
                directories = glob.glob(os.path.join(model_file_path, "*"))
                directories = list(
                    reversed(
                        sorted(filter(os.path.isdir, directories), key=os.path.getctime)
                    )
                )

                if len(directories) == 0:
                    raise ValueError(
                        f'Could not find "model" file in "{model_file_path}"!'
                    )

                model_file_path = None

                while model_file_path is None:
                    model_file_path = os.path.join(directories[0], "model")
                    if not os.path.exists(model_file_path):
                        model_file_path = None

                if model_file_path is None:
                    raise ValueError(f'Could not find "model" file in "{path}"!')
            else:
                model_file_path = os.path.join(model_file_path, "model")

        self.load(model_file_path)

    def load_best(self):
        """
        Loads best model weights.
        """
        self.load(self.current_best)

    def train_loop(
        self,
        X,
        y,
        optimizer,
        loss_function=torch.nn.NLLLoss(),
        epochs: int = 30,
        shuffle=True,
        batch_size: int = 256,
        testing_simulation: Simulation = None,
        validation_set: tuple = None,
    ):
        """

        :param X: Input sequences of shape (batch_size, num_maximum_guesses, 2 * word_length).
            Each entry consists of [dictionary representation (size word_length) score (size word_length)
        :param y: Training targets (shape (batch_size))
        :param optimizer: Optimizer used for training
        :param loss_function: Loss function used for training
        :param epochs: Number of epochs the model will be trained
        :param shuffle: Whether the data should be shuffled each epoch
        :param batch_size: Batch size the training and validation data will be split into
        :param testing_simulation: Simulation used for testing the model
        :param validation_set: Tuple(validation_X, validation_y) of dataset used for validation (optional)
        :return: mean training losses, std traning losses, mean validation loss
        """
        assert (
            X.shape[0] == y.shape[0]
        ), f"Number of sequences is not matching the number of targets! {X.shape[0]} != {y.shape[0]}"
        # assert y.shape[1] == model.max_word_length, f"Illegal target word length! {y.shape[1]} != {model.max_word_length}"

        checkpoint_duration = max(int(round(0.1 * epochs)), 5)
        X_train = X
        y_train = y

        X_val = None
        y_val = None

        use_gpu = load_settings().get("MODELS", "use_gpu", fallback="f") in [
            "t",
            "true",
        ]
        if use_gpu and not torch.cuda.is_available():
            print('Could not detect a CUDA device, ignoring "use_gpu" setting')
            use_gpu = False
        device = torch.device("cuda:0" if use_gpu else "cpu")

        final_repetition_penalty_scale = float(
            load_settings().get("MODELS", "repetition_penalty_scale", fallback=0.5)
        )

        penalty_scale = final_repetition_penalty_scale
        penalty_steps_size = epochs
        if load_settings().get(
            "MODELS", "penalty_warm_up", fallback="True"
        ).lower() in ["true", "t"]:
            penalty_scale = 0.01
            penalty_steps_size = log2(final_repetition_penalty_scale / penalty_scale)
            penalty_steps_size = ceil(0.5 * epochs / penalty_steps_size)

        self.to(device)
        self.device = device

        if (
            validation_set is not None
            and validation_set[0] is not None
            and validation_set[1] is not None
        ):
            X_val, y_val = validation_set
            X_val = [entry.to(device) for entry in torch.split(X_val, batch_size)]
            y_val = [entry.to(device) for entry in torch.split(y_val, batch_size)]

        mean_losses = np.zeros(epochs)
        std_losses = np.zeros(epochs)
        validation_loss = np.zeros(epochs)

        best_winrate = None
        best_model = None

        with tqdm(range(epochs), desc="Epochs") as progress_bar:
            loss_print = -1
            game_stat = ""
            val_stat = None

            for i in progress_bar:
                if shuffle:
                    indices = torch.randperm(X.shape[0])
                    X_train = X[indices]
                    y_train = y[indices]

                # Split into batches
                batches_X = torch.split(X_train, batch_size)
                batches_y = torch.split(y_train, batch_size)
                total_batch_number = len(batches_X)

                losses = []
                for batch_ind in range(total_batch_number):
                    progress_bar.set_postfix(
                        batch=f"{str(batch_ind).zfill(len(str(total_batch_number)))}/{total_batch_number}",
                        game_stats=game_stat,
                        validation_loss=val_stat,
                        loss=loss_print,
                        refresh=True,
                    )
                    current_X = batches_X[batch_ind].to(device)
                    current_y = batches_y[batch_ind].to(device)

                    self.train()

                    output = self(current_X)

                    loss = loss_function(
                        output, current_y
                    ) + penalty_scale * repetition_penalty(
                        current_X, current_y, output, self.embedding
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses.append(float(loss))

                mean_losses[i] = np.mean(np.array(losses))
                std_losses[i] = np.std(np.array(losses))
                loss_print = f"{mean_losses[i]:.2f}+-{std_losses[i]:.3f}"

                win_rate = None
                avg_tries = None
                avg_tries_won = None
                percentage_repeated_games = None
                percentage_repeated_guesses = None
                if testing_simulation is not None:
                    testing_simulation.update_model(self)
                    win_rate, avg_tries, avg_tries_won = testing_simulation.run()
                    percentage_repeated_games = testing_simulation.percentage_repeated_games[
                        -1
                    ]
                    percentage_repeated_guesses = testing_simulation.percentage_repeated_guesses[
                        -1
                    ]

                    game_stat = f"{win_rate:.2%} - {avg_tries_won:.2f}"

                val_loss = -1
                if X_val is not None:
                    self.eval()
                    with torch.no_grad():
                        val_loss = 0
                        for val_ind in range(len(X_val)):
                            output = self(X_val[val_ind])

                            val_loss += float(
                                loss_function(output, y_val[val_ind])
                                + penalty_scale
                                * repetition_penalty(
                                    X_val[val_ind],
                                    y_val[val_ind],
                                    output,
                                    self.embedding,
                                )
                            )
                        val_loss = val_loss / len(X_val)
                        val_stat = f"{val_loss:.3f}"
                validation_loss[i] = val_loss

                if (
                    penalty_scale < final_repetition_penalty_scale
                    and i % penalty_steps_size == 0
                    and i > 0
                ):
                    penalty_scale = min(
                        penalty_scale * 2, final_repetition_penalty_scale
                    )

                if (
                    win_rate is not None
                    and (best_winrate is None or win_rate >= best_winrate)
                ) or (i % checkpoint_duration == 0 and i > 0):
                    # Save model periodically
                    save_path = self.save(self.check_point_path)

                    # Save some metadata
                    meta_dict = {
                        "loss": mean_losses[i],
                        "loss_std": std_losses[i],
                        "win_rate": win_rate,
                        "avg_tries": avg_tries,
                        "avg_tries_won": avg_tries_won,
                        "dictionary_size": self.dictionary_size,
                        "word_length": self.max_word_length,
                        "num_guesses": self.max_guesses,
                        "name": save_path,
                        "penalty_scale": penalty_scale,
                        "percentage_repeated_guesses": percentage_repeated_guesses,
                        "percentage_repeated_games": percentage_repeated_games,
                        "validation_loss": val_loss,
                        "type": type(self).__name__,
                        "sim_len": len(testing_simulation.avg_tries_won)
                        if testing_simulation is not None
                        else 0,
                    }
                    with open(
                        os.path.join(os.path.dirname(save_path), "metadata.json"), "xt"
                    ) as meta_file:
                        json.dump(meta_dict, meta_file, indent=2)

                    if win_rate is not None and (
                        best_winrate is None or win_rate >= best_winrate
                    ):
                        best_winrate = win_rate
                        best_model = save_path
                        self.best_meta = meta_dict

                    # Add plots to folder
                    _plot_losses(
                        os.path.dirname(save_path),
                        testing_simulation,
                        mean_losses[: i + 1],
                        std_losses[: i + 1],
                        validation_loss[: i + 1],
                        best_ind=self.best_meta["sim_len"] - 1,
                    )

                # Update progress bar
                progress_bar.set_postfix(
                    batch=f"{total_batch_number}/{total_batch_number}",
                    game_stats=game_stat,
                    validation_loss=val_stat,
                    loss=loss_print,
                    refresh=True,
                )

        print(f'Best model at "{best_model}" with {best_winrate:.2%}')

        best_model_dir = os.path.join(self.check_point_path, "best")

        if not os.path.exists(best_model_dir):
            os.mkdir(best_model_dir)

        copyfile(best_model, os.path.join(best_model_dir, "model"))
        copyfile(
            os.path.join(os.path.dirname(best_model), "metadata.json"),
            os.path.join(best_model_dir, "metadata.json"),
        )

        self.current_best = best_model

        return mean_losses, std_losses, validation_loss

    @staticmethod
    def get_train_function() -> Callable[[Dictionary], AbstractModel]:
        """
        Returns the train function for this class
        :return: The train function (callable)
        """
        return train


def train(dictionary: Dictionary):
    """
    Creates a new model and trains it using the given dictionary and the settings in config.ini
    :param dictionary: Dictionary containing legal words
    :return: Trained model
    """

    percentage = float(load_settings().get("MODELS", "game_percentage", fallback=0.8))
    num_games = round(percentage * len(dictionary.words))
    epochs = int(load_settings().get("MODELS", "epochs", fallback=150))
    max_games_per_iteration = int(
        load_settings().get("MODELS", "max_games_per_iteration", fallback=-1)
    )

    # Process in one chunk
    if max_games_per_iteration <= 0:
        max_games_per_iteration = num_games + 1

    batch_size = int(load_settings().get("MODELS", "batch_size", fallback=1024))
    learning_rate = float(load_settings().get("MODELS", "learning_rate", fallback=0.02))
    validation_split_size = float(
        load_settings().get("MODELS", "validation_split_size", fallback=0.2)
    )

    check_point_path = os.path.join(
        PROJECT_ROOT_DIR,
        load_settings().get("MODELS", "trained_model_path", fallback="trained"),
        datetime.now().strftime("%Y%m%d%H%M%S"),
        "checkpoints",
    )
    if not os.path.exists(check_point_path):
        os.makedirs(check_point_path)

    print(f'Saving checkpoints to "{check_point_path}".')

    simulation_target_size = round(1.0 * len(dictionary.words))
    simulation_targets = random.sample(dictionary.words, simulation_target_size)

    my_model = WordleLSTM(dictionary)
    my_model.check_point_path = check_point_path
    sim = Simulation(my_model, dictionary, simulation_targets)

    current_games_num = 0
    mean_losses = []
    std_losses = []
    validation_losses = []

    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

    # Split into smaller datasets in case of hardware constraints
    while current_games_num < num_games:
        current_step = min(max_games_per_iteration, num_games - current_games_num)
        lstm_dataset, lstm_targets = create_lstm_tensors(
            dictionary, num_games=current_step,
        )
        print(f"Created {int(lstm_dataset.shape[0])} data rows.")
        validation_set = None

        if 0.999999 > validation_split_size > 0.001:
            val_split = ceil(validation_split_size * int(lstm_dataset.shape[0]))

            perm = torch.randperm(lstm_dataset.shape[0])

            validation_set = (
                lstm_dataset[perm[:val_split]],
                lstm_targets[perm[:val_split]],
            )
            lstm_dataset = lstm_dataset[perm[val_split:]]
            lstm_targets = lstm_targets[perm[val_split:]]

        c_means, c_stds, c_validation = my_model.train_loop(
            lstm_dataset,
            lstm_targets,
            optimizer,
            epochs=epochs,
            batch_size=batch_size,
            testing_simulation=sim,
            validation_set=validation_set,
        )

        mean_losses.append(c_means)
        std_losses.append(c_stds)
        validation_losses.append(c_validation)
        current_games_num += current_step
        print(f"Processed {current_games_num}/{num_games} games")

    mean_losses = np.concatenate(mean_losses)
    std_losses = np.concatenate(std_losses)
    validation_losses = np.concatenate(validation_losses)

    # Save model
    saving_path = os.path.dirname(check_point_path)

    # Plot
    _plot_losses(
        saving_path,
        sim,
        mean_losses,
        std_losses,
        validation_losses,
        best_ind=my_model.best_meta["sim_len"] - 1,
    )

    # Save metadata
    meta_dict = {
        "loss": mean_losses.tolist(),
        "loss_std": std_losses.tolist(),
        "validation_loss": validation_losses.tolist(),
        "dictionary_size": my_model.dictionary_size,
        "word_length": my_model.max_word_length,
        "num_guesses": my_model.max_guesses,
        "best": my_model.current_best,
        "best_meta": my_model.best_meta,
        "simulation": sim.to_dict(),
        "type": type(my_model).__name__,
    }
    with open(os.path.join(saving_path, "metadata.json"), "xt") as meta_file:
        json.dump(meta_dict, meta_file, indent=2)

    return my_model


def _plot_losses(
    saving_path: str,
    sim: Simulation,
    mean_losses: np.ndarray,
    std_losses: np.ndarray,
    validation_losses: np.ndarray,
    best_ind=None,
):
    """
    Plots training losses and stats to png files in given path
    :param saving_path: Folder the images will be saved to
    :param sim: Simulation containing the current model's performance
    :param mean_losses: Array containing loss means (shape (epochs))
    :param std_losses: Array containing loss std (shape (epochs))
    :param validation_losses: Array containing validation loss (shape (epochs))
    :param best_ind: epoch of the best model (optional)
    """
    # Plot sim
    plt.figure()
    ax = plt.subplot(211)

    plt.plot(sim.avg_tries_won, label="Tries")

    if best_ind:
        ax.axvline(x=best_ind, alpha=0.4, label="best")

    ax.set_ylabel("Tries")
    ax.grid()

    ax = plt.subplot(212)

    plt.plot(sim.percentage_won, label="Percentage won")

    if best_ind:
        ax.axvline(x=best_ind, alpha=0.4, label="best")

    ax.set_xlabel("Epochs")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Percentage")
    ax.grid()

    plt.savefig(os.path.join(saving_path, "training.png"), dpi=300)

    x_axis = np.arange(len(mean_losses))
    # Plot losses
    fig, ax = plt.subplots(1)
    ax.plot(x_axis, mean_losses, lw=2, label="training", color="blue")
    if best_ind:
        ax.axvline(x=best_ind, alpha=0.4, label="best")

    ax.fill_between(
        x_axis,
        mean_losses + std_losses,
        mean_losses - std_losses,
        facecolor="blue",
        alpha=0.5,
    )
    ax.set_title("Training Losses")
    ax.legend(loc="upper left")
    ax.set_xlabel("num steps")
    ax.set_ylabel("loss")
    ax.grid()

    fig.savefig(os.path.join(saving_path, "training_loss.png"), dpi=300)

    # Plot validation loss
    fig, ax = plt.subplots(1)

    ax.plot(x_axis, validation_losses, lw=2, label="validation", color="red")
    if best_ind:
        ax.axvline(x=best_ind, alpha=0.4, label="best")

    ax.set_title("Validation Losses")
    ax.legend(loc="upper left")
    ax.set_xlabel("num steps")
    ax.set_ylabel("loss")
    ax.grid()

    fig.savefig(os.path.join(saving_path, "validation_loss.png"), dpi=300)

    # Plot both losses
    fig, ax = plt.subplots(1)
    ax.plot(x_axis, mean_losses, lw=2, label="training", color="blue")
    ax.plot(x_axis, validation_losses, lw=2, label="validation", color="red")

    if best_ind:
        ax.axvline(x=best_ind, alpha=0.4, label="best")

    ax.set_title("All Losses")
    ax.legend(loc="upper left")
    ax.set_xlabel("num steps")
    ax.set_ylabel("loss")
    ax.grid()

    fig.savefig(os.path.join(saving_path, "total_loss.png"), dpi=300)

    plt.close("all")
