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
from nltk.tokenize import sent_tokenize, word_tokenize
# %%
articles = pd.read_pickle(f'{BUILD_PATH}/data.pkl')
articles_en = articles[articles.language_ml == 'en']
with open(BUILD_PATH / 'corpora.pkl', 'rb') as file:
    corpora = pickle.load(file)
# %%
# Histograms
# plt.title('Sentence count')
# plt.hist(articles[articles.sentences < 200].sentences)
# plt.show()

# plt.title('Word count')
# plt.hist(articles[articles.terms < 2000].terms)
# plt.show()
# %%
sentence_tokens_en = sent_tokenize(corpora['en'])
# freq_dist = FreqDist(word_tokenize('. '.join(articles_en.content)))
# freq_dist.most_common(1000)
# %%
sentence_tokens_en
# %%
words_tokens_en = []
for i, sentence in enumerate(sentence_tokens_en):
    if i % 10000 == 0:
        print(f'at sentence {i} of {len(sentence_tokens_en)}')
    words_tokens_en += word_tokenize(sentence)
# %%
