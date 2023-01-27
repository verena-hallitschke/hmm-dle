from model.dictionary.file_handling.google_loader import GoogleLoader
from model.dictionary.file_handling.nltk_loader import NLTKLoader


def get_handler(name: str):

    loader_map = {
        "google": GoogleLoader,
        "nltk": NLTKLoader
    }
    
    mapped_class = loader_map.get(name.lower())

    if mapped_class is not None:
        return mapped_class
    
    possibilities = ''.join(f'\"{t}\" ' for t in loader_map.keys()).strip()
    raise ValueError(f"Unknown dictionary type: \"{name}\". Has to be one of {possibilities}!")
