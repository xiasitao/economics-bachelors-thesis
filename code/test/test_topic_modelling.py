# %%
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import chisquare
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
articles = pd.read_pickle(BUILD_PATH / 'articles_balanced_50.pkl')
ses = pd.read_pickle(BUILD_PATH / 'ses_scores_filtered.pkl')
articles = articles.join(ses, how='inner', on='role_model')
with open(BUILD_PATH / 'topic_modelling.pkl', 'rb') as file:
    topic_words, article_topics = pickle.load(file)
articles = articles.join(article_topics, how='inner', on='article_id')
articles_per_SES = articles[articles['average_ses']==-1.0].count()['content'], articles[articles['average_ses']==+1.0].count()['content']
topic_columns = [column for column in article_topics.columns if not column.endswith('_entropy')]


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


def chi2_test(distribution: pd.DataFrame, articles_per_SES: tuple) -> pd.DataFrame:
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


# %%
topic_distributions = find_topic_distributions(articles, topic_columns)
plot_topic_distribution(topic_distributions['topic_15'])


# %%
