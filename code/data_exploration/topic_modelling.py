# %%
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from gensim.corpora import Dictionary
from gensim.models import LdaModel


 # %%
articles = pd.read_pickle(BUILD_PATH / 'data_balanced_50.pkl')
ses_scores = pd.read_pickle(BUILD_PATH / 'ses_scores.pkl')
articles_en = articles[articles.language_ml == 'en']
articles_en = articles_en.join(ses_scores[['average_ses', 'rank_weighted_ses', 'significance_weighted_ses']], on='role_model')


# %%
def filter_tokens(doc: list) -> list:
    return [token for token in doc
        if (
            len(token) > 1
        )
    ]


# %%
articles_tokenized = articles_en['content_slim'].str.split(' ')
dictionary = Dictionary(articles_tokenized)
dictionary.filter_extremes(no_below=5, no_above=0.3)
dictionary[0]
corpus = [dictionary.doc2bow(filter_tokens(article)) for article in articles_tokenized]
# %%
model = LdaModel(corpus=corpus, id2word=dictionary.id2token, iterations=400, num_topics=5)
top_topics = list(model.top_topics(corpus))
# %%
pprint([topic[0][0:10] for topic in top_topics])
# %%
model[]
# %%