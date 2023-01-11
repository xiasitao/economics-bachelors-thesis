# %%
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

import pickle
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize()
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


# %%
def filter_tokens(doc: list) -> list:
    nltk_tokens = word_tokenize(' '.join(doc))
    return [token[0] for token in pos_tag(nltk_tokens)
        if len(token[0]) > 1
        and token[1] in ('FW', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')  # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    ]


def equilibrate_role_models(articles: pd.DataFrame, ses_data: pd.DataFrame) -> pd.DataFrame:
    """Equilibrate role models such that for each prevalent_ses the same amount of role models are present

    Args:
        articles (pd.DataFrame): all articles relevant for these role models
        ses_data (pd.DataFrame): substrate

    Returns:
        pd.DataFrame: substrate with equal amounts of role models for each prevalent ses
    """    
    ses_data = ses_data[ses_data.index.isin(articles['role_model'])]
    minimum_count = ses_data.groupby('prevalent_ses').count()['count'].min()
    ses_data = ses_data.sample(frac=1.0, random_state=42)
    ses_data['_group_index'] = ses_data.groupby('prevalent_ses').cumcount()
    ses_data = ses_data[ses_data['_group_index'] < minimum_count]
    ses_data = ses_data.drop('_group_index', axis=1)
    return ses_data


 # %%
articles = pd.read_pickle(BUILD_PATH / 'data_balanced_50.pkl')
ses_scores = pd.read_pickle(BUILD_PATH / 'ses_scores_equilibrated.pkl')
articles_en = articles[articles.language_ml == 'en']
ses_scores = equilibrate_role_models(articles_en, ses_scores)
articles_en = articles_en.join(ses_scores[['average_ses', 'rank_weighted_ses', 'significance_weighted_ses', 'prevalent_ses']], on='role_model', how='inner')


# %%
articles_tokenized = articles_en['content_slim'].str.split(' ')
dictionary = Dictionary(articles_tokenized)
dictionary.filter_extremes(no_below=5, no_above=0.3)
dictionary[0]
corpus = [dictionary.doc2bow(filter_tokens(article)) for article in articles_tokenized]
# %%
model = LdaModel(corpus=corpus, id2word=dictionary.id2token, iterations=100, num_topics=10)
topics = [[topic_entry[1] for topic_entry in topic[0][0:10]] for topic in model.top_topics(corpus)]


# %%
def find_topic_and_entropy(model: LdaModel, doc: list):
    topic_probabilities = np.array(model[dictionary.doc2bow(filter_tokens(doc))])
    topics = topic_probabilities[:, 0]
    probabilities = topic_probabilities[:, 1]
    topic = topics[probabilities.argmax()]
    entropy = - probabilities.dot(np.log(probabilities))
    return topic, entropy
articles_en[['topic', 'topic_entropy']] = articles_tokenized.parallel_apply(lambda doc: pd.Series(find_topic_and_entropy(model, doc)))


# %%
articles_en[['topic', 'prevalent_ses', 'role_model', 'content']].groupby(['topic', 'prevalent_ses']).nunique()


# %%
print(topics)
# %%
articles_en.groupby(['prevalent_ses']).count()
# %%
