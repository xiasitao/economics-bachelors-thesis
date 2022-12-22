# %%
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

import pickle
import regex as re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from nltk.text import Text
from nltk.corpus import stopwords
from nltk.probability import FreqDist
# %%
# Load data
articles = pd.read_pickle(f'{BUILD_PATH}/data.pkl')
articles_en = articles[articles.language_ml == 'en']
with open(BUILD_PATH / 'corpora.pkl', 'rb') as file:
    corpora = pickle.load(file)
with open(BUILD_PATH / 'sentence_tokens.pkl', 'rb') as file:
    sentence_tokens = pickle.load(file)
with open(BUILD_PATH / 'word_statistic.pkl', 'rb') as file:
    word_statistics = pickle.load(file)
# %%
# Histograms
# plt.title('Sentence count')
# plt.hist(articles[articles.sentences < 200].sentences)
# plt.show()

# plt.title('Word count')
# plt.hist(articles[articles.terms < 2000].terms)
# plt.show()
# %%
freq_dist = FreqDist()
# freq_dist.most_common(1000)
# %%
sentence_tokens
sentence_tokens_frequency_en = FreqDist(sentence_tokens['en'])
sentence_tokens_frequency_en.most_common(200)
# %%
