"""
This module contains the settings handling (reading the config.ini)
"""
import configparser
import os

SETTINGS_FILE_NAME = "config.ini"
SETTINGS_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), SETTINGS_FILE_NAME
)

CURRENT_SETTINGS = None


def load_settings() -> configparser.ConfigParser:
    """
    Loads the settings from the config file once. Every subsequent call returns the buffered settings
    :return: configparser.ConfigParser containing the settings
    """
    global CURRENT_SETTINGS

    if CURRENT_SETTINGS is not None:
        return CURRENT_SETTINGS
    config = configparser.ConfigParser()
    config.read(SETTINGS_FILE_PATH)
    CURRENT_SETTINGS = config

    return CURRENT_SETTINGS
