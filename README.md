# hmm-dle: Solving Wordle using HMMs

The aim of this repository is to solve the popular NY Times game [Wordle](https://www.nytimes.com/games/wordle/) using HMMS or an LSTM. This work is the result of a university project.

## Set-up
1. Install the dependencies in [`requirements.txt`](./requirements.txt) using 
```bash
pip install -r ./requirements.txt
```
2. There are two possible backend dictionaries that can be used. One of them is using the NLTK corpus, the other one the tokens from the Google Web Trillion Word Corpus. In order to use the Google Corpus, the file `unigram_freq.csv` has to be copied from [Kaggle](https://www.kaggle.com/datasets/rtatman/english-word-frequency) into [`./model/dictionary/ressources/unigram_freq.csv`](./model/dictionary/ressources/unigram_freq.csv). A custom dataset type can be added using the `AbstractLoader` class defined in [`model/dictionary/file_handling/abstract_loader.py`](./model/dictionary/file_handling/abstract_loader.py).
3. Set up the [configuration file](./config.ini).

**Please note that the Google dataset is not screened and may contain harmful language. A filtered dictionary file can be created and then linked in the [configuration file](./config.ini)!**

## CLI

The CLI supports two different commands: `train` and `predict`. Please call `python cli.py --help` for more details.

## Model

### HMM

The "HMM" model consists of one HMM for each word in the dictionary. Each HMM has 2 different states: 0 that emmits scores from 1 to 1023 and state 1 that only emmits the score 0. The score is a single number representation of the reponse of the game. It encodes the number of grey, yellow and green letters. The best score is 0, meaning all letters are green. Therefore during inference, the word corresponding to the HMM that has the highest chance to emmit 0 is chosen as next guess.

### LSTM

The LSTM model contains the LSTM, a linear layer that maps from the latent space to the dictionary space and an embedding tensor that encodes the index in the dictionary as a vector. Each entry in the vector is a number that represents the letter at the same position in the word.
The model performs a classification task, where it tries to predict the solution based on the dictionary index. An NLL Loss and the Adam optimizer are used.

## Results

| | Setting 1 | Setting 2 |
| -| -------- | --------- |
| Dictionary size | 15 | 100 |
| Win rates | 100% (both) | ~70% (HMM), ~85% (LSTM) |
| Number of guesses | 2 - 3 (both) | 3 - 4 (both) |
| Training time* | 10 s (HMM), 1 m (LSTM) | 20 m (HMM), 1 h (LSTM) |

\* Please note: The training of the LSTM in Setting 2 was performed on Google Colab. All other models were trained on my private notebook (i5 8th Gen, GeForce MX150). The HMM does not support training on the GPU.

## Configuration

The configuration is kept in [`config.ini`](./config.ini). Here are short descriptions of the possible settings.

### General
| Name | Default | Description |
|------|---------|-------------|
|dictionary_type | Google | Type of dictionary to use. Can be either "google" or "nltk" |
| threshold | 365410017 | (for Google dictionary type) Number of times a word has to be mentioned in the corpus to be considered for the dictionary |
|filtered_dictionary_path | - | (for Google dictionary type) Path to a custom dictionary file |
| use_filtered_dictionary | False | (for Google dictionary type) Decides whether the custom file path should be used |
| game_percentage | 1.5 | Number of games that should be generated per word in the dictionary during training. The resulting number of games per word is: game_percentage * dictionary length |
| trained_model_path | trained | Path to the folder where the trained models are located/will be saved to |

### LSTM specific

| Name | Default | Description |
|------|---------|-------------|
| use_gpu | True | Whether the GPU should be used during training |
| epochs | 350 | Number of epochs the LSTM will train |
| batch_size | 8192 | Batch size |
| learning_rate | 0.02 | Learning rate applied to the LSTM |
| hidden_dim | 60 | Dimension of the LSTM hidden dimension |
| repetition_penalty_scale | 0.0 | (Not recommended) Weight of the repitition penalty during loss calculation |
| penalty_warm_up | False | Whether warm up should be used on repetition_penalty_scale |
| max_games_per_iteration | -1 | Decides whether the number of games should be split into smaller batches of size max_games_per_iteration |
| validation_split_size | 0.25 | Size of the validation split (in percent) |
| dropout | 0.3 | Drop-out rate |
| num_layers | 3 | Number of hidden layers |

### HMM specific

| Name | Default | Description |
|------|---------|-------------|
| number_iterations | 5 | Number of iterations of the HMM training algorithm |
| max_num_dataset_regen | 50 | Since sometimes individual HMMs don't converge, their training is attempted again. This is the maximum number of retries |


## Developer

verena-hallitschke