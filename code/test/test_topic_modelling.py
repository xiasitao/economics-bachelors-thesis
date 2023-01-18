# %%
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import chisquare
import re

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

from pprint import pprint
from scipy.stats import chisquare, chi2_contingency


# %%
articles = pd.read_pickle(BUILD_PATH / 'articles_balanced_50.pkl')
articles = articles[articles['language_ml'] == 'en']
ses = pd.read_pickle(BUILD_PATH / 'ses_scores_filtered.pkl')
articles = articles.join(ses, how='inner', on='role_model')
with open(BUILD_PATH / 'topic_modelling.pkl', 'rb') as file:
    topic_words, article_topics = pickle.load(file)
articles = articles.join(article_topics, how='inner', on='article_id')
articles_per_SES = articles[articles['average_ses']==-1.0].count()['content'], articles[articles['average_ses']==+1.0].count()['content']
topic_columns = [column for column in article_topics.columns if not column.endswith('_entropy')]


# %%
# Sanity checks
assert(articles.groupby('role_model').count()['content'].unique() == np.array([50]))


# %%
def find_SES_distribution_per_topic(articles: pd.DataFrame, n_topics: int) -> tuple:
    topics_distribution_articles = articles[[f'topic_{n_topics}', 'prevalent_ses', 'content']].groupby([f'topic_{n_topics}', 'prevalent_ses']).count()
    topics_distribution_role_models = articles[[f'topic_{n_topics}', 'prevalent_ses', 'role_model']].groupby([f'topic_{n_topics}', 'prevalent_ses']).nunique()
    return topics_distribution_articles, topics_distribution_role_models


def plot_SES_distribution_per_topic(articles: pd.DataFrame, n_topics: int, mode='articles', topic_names=None):
    topics_distribution_articles, topics_distribution_role_models = find_SES_distribution_per_topic(articles, n_topics)
    plt.figure(figsize=(10, 2*n_topics))
    plt.title(f'SES distribution within topics')
    for i, topic_index in enumerate(topics_distribution_articles.index.get_level_values(0).unique()):
        ax = plt.subplot(len(topics_distribution_articles)//3+1, 3, i+1)
        ax.set_title(f'topic {int(topic_index)}' if topic_names is None or n_topics not in topic_names else topic_names[n_topics][i])
        if mode == 'articles':
            this_topic_distribution = topics_distribution_articles.loc[topic_index]
        elif mode == 'role_models':
            this_topic_distribution = topics_distribution_role_models.loc[topic_index]
        else:
            raise Exception(f'Mode {mode} unknown, use articles or role_models.')
        ax.bar(['low', 'high'], this_topic_distribution['content'], label=mode.replace('_', ' '))
    plt.legend()
    plt.tight_layout()
    plt.show()


def find_topic_distributions(articles: pd.DataFrame, columns: list) -> dict:   
    """Find the distribution of topics for low and high SES for number of topics available.

    Args:
        articles (pd.DataFrame): Article data.
        category_columns (list): List of column in the data corresponding to topics.

    Returns:
        dict: dict of category distribution data frames for each number of topics.
    """
    topic_distributions = {}
    for n_topics_column in columns:
        topic_distribution = pd.DataFrame(data=None, index=articles[n_topics_column].unique(), columns=['low', 'high'])
        topic_distribution['low'] = articles[articles['average_ses'] == -1.0].groupby(n_topics_column).count()['content']
        topic_distribution['high'] = articles[articles['average_ses'] == +1.0].groupby(n_topics_column).count()['content']
        topic_distributions[n_topics_column] = topic_distribution
    return topic_distributions


def chi2_per_label_test(distribution: pd.DataFrame, articles_per_SES: tuple) -> pd.DataFrame:
    """Perform a chi2 test on the absolute frequencies articles in each category.

    Args:
        category_distribution (pd.DataFrame): Distributions of SES (columns) in the cateogories (index)
        articles_per_SES (tuple): Number of overall articles per SES (low, high)

    Raises:
        ValueError: If relative frequencies are supplied.

    Returns:
        pd.DataFrame: chi2 and p per category
    """    
    if not (distribution == distribution.astype(int)).all().all():
        raise ValueError('Cannot accept relative frequencies.')

    results = pd.DataFrame(None, columns=['chi2', 'p'], index=distribution.index)
    for category in distribution.index:
        frequencies = distribution.loc[category]
        expected_frequencies = np.array(articles_per_SES)/np.sum(np.array(articles_per_SES)) * np.sum(frequencies)
        result = chisquare(distribution.loc[category], expected_frequencies)
        results.loc[category] = [result.statistic, result.pvalue]
    return results


def chi2_contingency_test(distribution: pd.DataFrame) -> tuple:
    """Perform a chi2 test checking whether the labels of a category are differently distributed for low and the high SES.

    Args:
        distribution (pd.DataFrame): Low and high SES distribution of labels in a category.

    Returns:
        tuple: chi2, p values
    """    
    result = chi2_contingency(np.array(distribution.T))
    return result.statistic, result.pvalue


def plot_topic_distribution(topic_distribution: pd.DataFrame, relative=True):
    """Plot the distribution of articles over the categories for low and high SES.

    Args:
        category_distribution (pd.DataFrame): Distribution matrix with categories (index) and SES (columns)
        category_name (str): Name of the category
        relative (bool, optional): Whether to normalize frequencies for each SES level. Defaults to True.
    """    
    fig, ax = plt.gcf(), plt.gca()
    if relative:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        topic_distribution = topic_distribution.copy()
        topic_distribution = topic_distribution.apply(lambda col: col/col.sum())
    topic_distribution.plot(kind='bar', ax=ax)
    fig.show()


def print_topic_words(topic_words: dict, n_topics: int):
    """Print topic words more readably.

    Args:
        topic_words (dict): Topic word lists per n_topics.
        n_topics (int): Display word lists for n_topics topics.
    """    
    topics = topic_words[n_topics]
    print(f'{n_topics} topics:')
    for i, topic in enumerate(topics):
        print(f'\t{i}\t{" ".join(topic)}')


def find_hypertopics(articles: pd.DataFrame, hypertopic_table: dict, columns: list) -> pd.DataFrame:
    articles = articles[['article_id', 'average_ses', 'content'] + columns].drop_duplicates().set_index('article_id', drop=True)
    hypertopics = pd.DataFrame(data=None, columns=['average_ses', 'content']+columns, index=articles.index)
    hypertopics[['average_ses', 'content']] = articles[['average_ses', 'content']]
    for column in columns:
        n_topics = int(re.match(r'topic_(\d+)', column).groups()[0])
        hypertopics[f'topic_{n_topics}'] = articles[f'topic_{n_topics}'].apply(lambda topic: hypertopic_table[n_topics][int(topic)])
    return hypertopics


def plot_hypertopic_distribution_by_n(hypertopic_distributions: dict, hypertopics: list):
    ns = [re.match(r'topic_(\d+)', column).groups()[0] for column in hypertopic_distributions]
    low_ses_hypertopic_frequencies = pd.DataFrame(data=0, index=hypertopics, columns=ns)
    high_ses_hypertopic_frequencies = pd.DataFrame(data=0, index=hypertopics, columns=ns)
    for n in ns:
        low_ses_hypertopic_frequencies[n] = hypertopic_distributions[f'topic_{n}']['low']
        high_ses_hypertopic_frequencies[n] = hypertopic_distributions[f'topic_{n}']['high']
    
    plt.title('Hypertopic distributions')
    plt.ylabel('percentage of low SES articles')
    plt.xlabel('number of topics')
    for hypertopic in hypertopics:
        plt.plot(ns, low_ses_hypertopic_frequencies.loc[hypertopic]/(low_ses_hypertopic_frequencies.loc[hypertopic]+high_ses_hypertopic_frequencies.loc[hypertopic]), label=hypertopic)
    plt.legend()
    plt.show()


# %%
topic_distributions = find_topic_distributions(articles, topic_columns)
plot_topic_distribution(topic_distributions['topic_15'])


# %%
chi2_contingency_test(topic_distributions['topic_15'])
# %%
HT_MOVIE, HT_SPORT, HT_MUSIC, HT_LIFE = 'movie', 'sport', 'music', 'life'
hypertopics = [HT_MOVIE, HT_SPORT, HT_MUSIC, HT_LIFE]
print_topic_words(topic_words, 15)
hypertopic_table = {
    2: [HT_MOVIE, HT_SPORT],
    3: [HT_MUSIC, HT_SPORT, HT_MOVIE],
    4: [HT_MUSIC, HT_MOVIE, HT_SPORT, HT_LIFE],
    5: [HT_MUSIC, HT_LIFE, HT_SPORT, HT_MOVIE, HT_LIFE],
    6: [HT_MUSIC, HT_LIFE, HT_MOVIE, HT_SPORT, HT_LIFE, HT_LIFE],
    7: [HT_MUSIC, HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE],
    8: [HT_MUSIC, HT_MOVIE, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE],
    9: [HT_LIFE, HT_MUSIC, HT_MOVIE, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE],
    10: [HT_LIFE, HT_MOVIE, HT_MUSIC, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE],
    11: [HT_LIFE, HT_MOVIE, HT_MUSIC, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE],
    12: [HT_LIFE, HT_MOVIE, HT_SPORT, HT_MUSIC, HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE],
    13: [HT_SPORT, HT_LIFE, HT_MOVIE, HT_MUSIC, HT_LIFE, HT_LIFE, HT_LIFE, HT_MOVIE, HT_LIFE, HT_MOVIE, HT_LIFE, HT_SPORT, HT_LIFE],
    14: [HT_SPORT, HT_LIFE, HT_MOVIE, HT_MUSIC, HT_LIFE, HT_MOVIE, HT_MOVIE, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE],
    15: [HT_SPORT, HT_MOVIE, HT_LIFE, HT_MUSIC, HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE, HT_MOVIE, HT_SPORT, HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE],
}
assert(all([len(hypertopic_table[n]) == n for n in hypertopic_table]))
"""Observations:
    - Most important are movies and sports
    - Life becomes ever more fine-grained the larger n_topics (police, beauty etc)
"""


# %%
article_hypertopics = find_hypertopics(articles, columns=topic_columns, hypertopic_table=hypertopic_table)
hypertopics_distributions = find_topic_distributions(article_hypertopics, topic_columns)
plot_hypertopic_distribution_by_n(hypertopics_distributions, hypertopics)
# %%
