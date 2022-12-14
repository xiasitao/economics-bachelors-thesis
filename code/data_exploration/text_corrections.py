# %%
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import nltk.corpus
# %%
with open('../../assets/role_model_articles_de.pkl', 'rb') as file:
    articles = pickle.load(file)
# %%
# Letter shifting
def apply_common_char_replacements(char):
    if char == 'å':
        return 'ä'
    if char == '÷':
        return 'ö'
    if char == 'ý':
        return 'ü'
    if char == 'à':
        return 'ß'
    return char

def letter_shift(string: str, shift: int = 1, common_replacements=True) -> str:
    """Shift all letters by a number of places in the alphabet

    Args:
        string (str): substrate string
        shift (int, optional): number of alphabet places to shift. Defaults to 1.
    """
    def replace_char(char):
        if ord(char) >= ord('A') - 1 and ord(char) <= ord('z') + 1:
            return chr(ord(char) + shift)
        if common_replacements:
            return apply_common_char_replacements(char)
        return char

    new_chars = [replace_char(char) for char in string]
    return ''.join(new_chars)
hint_words = [letter_shift(word) for word in ('und', 'auch', 'ein', 'eine', 'der', 'die', 'das')]

# %%
letter_shift(articles[articles['content'].str.contains('|'.join([rf'\b{word}\B' for word in hint_words]))].content.iloc[3], shift=-1)
# %%
hint_words
# %%
