# %%
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import chisquare
import re
import random

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

from scipy.stats import chisquare, chi2_contingency
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# %%
articles_raw = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
articles_raw = articles_raw[articles_raw['language_ml']=='en']
ses = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores.pkl')
ses_distinct = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_distinct.pkl')
human_annotated = pd.read_pickle(BUILD_PATH / 'articles/articles_human_annotated.pkl')
with open(BUILD_PATH / 'topic_modelling/topic_modelling.pkl', 'rb') as file:
    topic_words, article_topics = pickle.load(file)
topic_columns = [column for column in article_topics.columns if not column.endswith('_entropy')]

def load_prepare_articles(articles: pd.DataFrame, ses: pd.DataFrame, article_topics: pd.DataFrame):
    articles = articles.join(ses, how='inner', on='role_model')
    articles = articles.join(article_topics, how='inner', on='article_id')
    articles_per_SES = articles[articles['low_ses']].count()['content'], articles[articles['high_ses']].count()['content']
    return articles, articles_per_SES
articles, articles_per_SES = load_prepare_articles(articles_raw, ses, article_topics)
articles_distinct, articles_per_SES_distinct = load_prepare_articles(articles_raw, ses_distinct, article_topics)


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
        topic_distribution['low'] = articles[articles['low_ses']].groupby(n_topics_column).count()['content']
        topic_distribution['high'] = articles[articles['high_ses']].groupby(n_topics_column).count()['content']
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
    return results.sort_index()


def chi2_contingency_test(distribution: pd.DataFrame) -> tuple:
    """Perform a chi2 test checking whether the labels of a category are differently distributed for low and the high SES.

    Args:
        distribution (pd.DataFrame): Low and high SES distribution of labels in a category.

    Returns:
        tuple: chi2, p values
    """    
    result = chi2_contingency(np.array(distribution.T))
    return result.statistic, result.pvalue


def plot_topic_distribution(topic_distribution: pd.DataFrame, relative=True, additional_title_text: str=None):
    """Plot the distribution of articles over the topics for low and high SES.

    Args:
        topic_distribution (pd.DataFrame): Distribution matrix with categories (index) and SES (columns)
        category_name (str): Name of the category
        relative (bool, optional): Whether to normalize frequencies for each SES level. Defaults to True.
    """    
    topic_distribution = topic_distribution.copy().sort_index()

    fig, ax = plt.gcf(), plt.gca()
    ax.set_xlabel('topic')
    ax.set_ylabel('topic article count')
    if additional_title_text is not None:
        ax.set_title(additional_title_text)
    if relative:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.set_ylabel('topic article percentage')
        topic_distribution = topic_distribution.apply(lambda col: col/col.sum())
    topic_distribution.index = topic_distribution.index.astype(int)
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
    articles = articles[['article_id', 'average_ses', 'low_ses', 'high_ses', 'content'] + columns].drop_duplicates().set_index('article_id', drop=True)
    hypertopics = pd.DataFrame(data=None, columns=['average_ses', 'low_ses', 'high_ses', 'content']+columns, index=articles.index)
    hypertopics[['average_ses', 'low_ses', 'high_ses', 'content']] = articles[['average_ses', 'low_ses', 'high_ses', 'content']]
    for column in columns:
        n_topics = int(re.match(r'topic_(\d+)', column).groups()[0])
        hypertopics[f'topic_{n_topics}'] = articles[f'topic_{n_topics}'].apply(lambda topic: hypertopic_table[n_topics][int(topic)])
    return hypertopics


def plot_hypertopic_distribution_by_n(hypertopic_distributions: dict, hypertopics: list, articles_per_SES: tuple=None):
    ns = [re.match(r'topic_(\d+)', column).groups()[0] for column in hypertopic_distributions]
    low_ses_hypertopic_frequencies = pd.DataFrame(data=0, index=hypertopics, columns=ns)
    high_ses_hypertopic_frequencies = pd.DataFrame(data=0, index=hypertopics, columns=ns)
    for n in ns:
        low_ses_hypertopic_frequencies[n] = hypertopic_distributions[f'topic_{n}']['low']
        high_ses_hypertopic_frequencies[n] = hypertopic_distributions[f'topic_{n}']['high']
    
    fig, ax = plt.gcf(), plt.gca()
    ax.set_title('Hypertopic distributions')
    ax.set_ylabel('percentage of low SES articles')
    ax.set_xlabel('number of topics')
    for hypertopic in hypertopics:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.plot(ns, low_ses_hypertopic_frequencies.loc[hypertopic]/(low_ses_hypertopic_frequencies.loc[hypertopic]+high_ses_hypertopic_frequencies.loc[hypertopic]), label=hypertopic)
    if articles_per_SES is not None:
        overall_ratio = articles_per_SES[0]/(articles_per_SES[0]+articles_per_SES[1])
        ax.plot(ns, len(ns)*[overall_ratio], '--', label='all articles', color='grey', )
    ax.legend()
    ax.grid()
    fig.show()


def plot_human_annotation_confusion_matrix(hypertopic_data: pd.DataFrame, human_annotated: pd.DataFrame, n_topics: int):
    annotation_category = 'topic'
    if annotation_category not in human_annotated.columns:
        raise Exception('No topics in human annotations.')
    hypertopic_column = f'topic_{n_topics}'
    if hypertopic_column not in hypertopic_data.columns:
        raise Exception(f'No {hypertopic_column} in articles with hypertopics.')
    
    human_annotated_topic = human_annotated[~human_annotated[annotation_category].isna()]
    if len(human_annotated_topic) == 0:
        return
    articles_with_annotation = hypertopic_data.join(human_annotated_topic[[annotation_category]], on='article_id', how='inner')[['content', hypertopic_column, annotation_category]]
    topic_labels = np.unique(articles_with_annotation[[hypertopic_column, annotation_category]].values.ravel())
    topic_confusion_matrix = confusion_matrix(y_true=articles_with_annotation[annotation_category], y_pred=articles_with_annotation[hypertopic_column], labels=topic_labels)
    fig, ax = plt.gcf(), plt.gca()
    ax.set_title(f'Confusion matrix for {n_topics} topics')
    ConfusionMatrixDisplay(topic_confusion_matrix, display_labels=topic_labels).plot(ax=ax)


def evaluate_topic(
        articles: pd.DataFrame,
        n_topics: int,
        articles_per_SES: tuple,
        human_annotated: pd.DataFrame=None,
        relative_dist_plot: bool=True,
        is_distinct: bool=None
    ):
    column_name = f'topic_{n_topics}'
    topic_distributions = find_topic_distributions(articles, topic_columns)
    
    distinct_text = None
    if is_distinct is not None:
        distinct_text = 'distinct' if is_distinct else 'general'
    plot_topic_distribution(topic_distributions[column_name], relative=relative_dist_plot, additional_title_text=distinct_text)

    contingency_chi2, contingency_p = chi2_contingency_test(topic_distributions[column_name])
    print(f'Distribution chi2 test:\nchi2={contingency_chi2:.1f}, p={contingency_p:.3e}\n')

    print('Per-label chi2 test:')
    print(chi2_per_label_test(topic_distributions[column_name], articles_per_SES))
    
    if human_annotated is not None:
        plot_human_annotation_confusion_matrix(articles, human_annotated, column_name)


# %%
evaluate_topic(articles, 5, articles_per_SES=articles_per_SES, is_distinct=False)


# %%
evaluate_topic(articles_distinct, 5, articles_per_SES=articles_per_SES_distinct, is_distinct=True)


# %%
HT_MOVIE, HT_SPORT, HT_MUSIC, HT_LIFE = 'movie', 'sport', 'music', 'life'
hypertopics = [HT_MOVIE, HT_SPORT, HT_MUSIC, HT_LIFE]
print_topic_words(topic_words, 15)
hypertopic_table = {
    2: [HT_LIFE, HT_MOVIE],
    3: [HT_SPORT, HT_MUSIC, HT_MOVIE],
    4: [HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE],
    5: [HT_LIFE, HT_SPORT, HT_MOVIE, HT_LIFE, HT_MUSIC],
    6: [HT_MUSIC, HT_SPORT, HT_LIFE, HT_LIFE, HT_MOVIE, HT_LIFE],
    7: [HT_MUSIC, HT_SPORT, HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE],
    8: [HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_MUSIC],
    9: [HT_SPORT, HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_MUSIC, HT_SPORT],
    10: [HT_SPORT, HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE, HT_MUSIC, HT_LIFE],
    11: [HT_SPORT, HT_MOVIE, HT_LIFE, HT_LIFE, HT_MUSIC, HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE, HT_SPORT, HT_LIFE],
    12: [HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE, HT_MUSIC, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT, HT_SPORT],
    13: [HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE, HT_MUSIC, HT_LIFE, HT_LIFE, HT_SPORT, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE],
    14: [HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE, HT_MUSIC, HT_LIFE, HT_LIFE, HT_SPORT, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE],
    15: [HT_LIFE, HT_SPORT, HT_MOVIE, HT_LIFE, HT_MUSIC, HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE],
}
#generator = np.random.Generator(np.random.PCG64(42))
#hypertopic_table = {n: sorted(hypertopic_table[n], key=lambda k: generator.random()) for n in hypertopic_table}
assert(all([len(hypertopic_table[n]) == n for n in hypertopic_table]))


# %%
def hypertopic_crosscheck(hypertopic_table: "dict[int, list[str]]", topic_words, n_topics):
    for hypertopic in hypertopics:
        indices = [n for n, val in enumerate(hypertopic_table[n_topics]) if val==hypertopic]
        words = [topic_words[n_topics][i] for i in indices]
        print(hypertopic.upper())
        print(f'  {words}')
        print()
hypertopic_crosscheck(hypertopic_table, topic_words, 7)


# %%
article_hypertopics = find_hypertopics(articles, columns=topic_columns, hypertopic_table=hypertopic_table)
hypertopics_distributions = find_topic_distributions(article_hypertopics, topic_columns)
plot_hypertopic_distribution_by_n(hypertopics_distributions, hypertopics, articles_per_SES=articles_per_SES)


# %%
plot_human_annotation_confusion_matrix(article_hypertopics, human_annotated, 5)


# %%
