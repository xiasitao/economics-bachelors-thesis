import pytask
import re
import pickle
import pandas as pd
import numpy as np

from pathlib import Path
from pathlib import PosixPath
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

def letter_shift(string: str, shift: int = 1, common_replacements=True) -> str:
    """Shift all letters by a number of places in the alphabet

    Args:
        string (str): substrate string
        shift (int, optional): number of alphabet places to shift. Defaults to 1.
    """
    def apply_common_char_replacements(char: str) -> str:
        if char == 'å':
            return 'ä'
        if char == '÷':
            return 'ö'
        if char == 'ý':
            return 'ü'
        if char == 'à':
            return 'ß'
        return char
    def replace_char(char):
        if (ord(char) >= 33 - shift and ord(char) <= ord('z') - shift):
            return chr(ord(char) + shift)
        if common_replacements:
            return apply_common_char_replacements(char)
        return char

    new_chars = [replace_char(char) for char in string]
    return ''.join(new_chars)


def clean(string: str) -> str:
    """Remove unnecessary morphemes

    Args:
        string (str): substrate

    Returns:
        str: cleaned substrate
    """
    str = re.sub(r'\n', '', str)
    return str


hint_words = [letter_shift(word) for word in ('und', 'auch', 'ein', 'eine', 'der', 'die', 'das', 'for', 'and', 'with')]


def remove_URLs(string: str) -> str:
    regex = r'[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&\/\/=]*)'
    return re.sub(regex, '', string)


def remove_non_latin(string: str) -> str:
    regex = r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]'
    return re.sub(regex, '', string)


@pytask.mark.produces(f'{BUILD_PATH}/data.csv')
def task_preparation(produces: PosixPath):
    produces.write_text('')
    articles = None
    source_files = [f'{ASSET_PATH}/role_model_articles_de.pkl', f'{ASSET_PATH}/role_model_articles_en.pkl']
    for source_file in source_files:
        with open(source_file, 'rb') as file:
            these_articles = pickle.load(file)
            articles = these_articles if articles is None else pd.concat([articles, these_articles])
    articles = articles.rename(columns={'addedAt': 'added_at'})
    
    # Process in chunks in order to avoid memory leak
    for chunk in np.array_split(articles, len(articles) // 10000 + 1):
        chunk['content_raw'] = chunk['content']
        chunk['content'] = chunk['content_raw'].apply(lambda text: remove_non_latin(remove_URLs(text)))
        chunk['obfuscated'] = chunk['content'].str.contains('|'.join([rf'\b{word}\b' for word in hint_words]))
        chunk.to_csv(produces, mode='a')

if __name__ == '__main__':
    raise Exception('Can only be executed by pytask')