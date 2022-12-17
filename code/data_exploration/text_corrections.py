# %%
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk.corpus
# %%
with open('../../assets/role_model_articles_de.pkl', 'rb') as file:
    articles_de = pickle.load(file)
with open('../../assets/role_model_articles_en.pkl', 'rb') as file:
    articles_en = pickle.load(file)
articles = pd.concat([articles_en, articles_de])


# %%
# Letter shifting
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
        #if (ord(char) >= ord('A') - shift and ord(char) <= ord('Z') - shift) or (ord(char) >= ord('a') - shift and ord(char) <= ord('z') - shift):
        #if (ord(char) >= 33 and ord(char) <= ord('Z')) or (ord(char) >= ord('a') and ord(char) <= ord('z')):
        if (ord(char) >= 33 - shift and ord(char) <= ord('z') - shift):
            return chr(ord(char) + shift)
        if common_replacements:
            return apply_common_char_replacements(char)
        return char

    new_chars = [replace_char(char) for char in string]
    return ''.join(new_chars)


def clean(string: str) -> str:
    str = re.sub(r'\n', '', str)
    return str


def remove_URLs(string: str) -> str:
    regex = r'[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&\/\/=]*)'
    return re.sub(regex, '', string)


def remove_non_latin(string: str) -> str:
    regex = r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]'
    return re.sub(regex, '', string)


hint_words = [letter_shift(word) for word in ('und', 'auch', 'ein', 'eine', 'der', 'die', 'das', 'for', 'and', 'with')]


#%%
articles['content_raw'] = articles['content']
articles['content'] = articles['content_raw'].apply(lambda text: remove_non_latin(remove_URLs(text)))
obfuscated_articles = articles[articles['content'].str.contains('|'.join([rf'\b{word}\b' for word in hint_words]))]


# %%
