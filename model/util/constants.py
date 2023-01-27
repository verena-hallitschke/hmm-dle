"""
Contains constants for the programm
"""
import os

MAX_GUESS_NUMBER = 6
MAX_WORD_LENGTH = 5
NUM_GAMES_DEFAULT = 5

GOOGLE_DICTIONARY_BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dictionary",
    "ressources",
    "unigram_freq.csv",
)

NLTK_DICTIONARY_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dictionary",
    "ressources",
    "nltk_{}.json",
)
GOOGLE_DICTIONARY_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dictionary",
    "ressources",
    "google_{}_{}.json",
)

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
