# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import chisquare, chi2_contingency
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
articles = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
ses = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_distinct.pkl')
articles = articles.join(ses, how='inner', on='role_model')
human_annotated = pd.read_pickle(BUILD_PATH / 'articles/articles_human_annotated.pkl')

def collect_zero_shot_data():
    zero_shot_data = None
    for category in ['article_type', 'crime', 'crime_type', 'emotion', 'prosociality', 'relatability', 'sentiment', 'sentiment_n', 'topic', 'writing_style']:
        filename = BUILD_PATH / f'zero_shot_classification/zero_shot_classification_{category}.pkl'
        if filename.exists:
            category_data = pd.read_pickle(filename)
            zero_shot_data = pd.concat([zero_shot_data, category_data], axis=1) if zero_shot_data is not None else category_data
    return zero_shot_data
zero_shot_data = collect_zero_shot_data()
articles = articles.join(zero_shot_data, how='inner', on='article_id')

articles_per_SES = articles[articles['average_ses']==-1.0].count()['content'], articles[articles['average_ses']==+1.0].count()['content']
category_columns = [column for column in zero_shot_data.columns if not column.endswith('_entropy') and not column.endswith('_p')]


# %%
def find_low_entropy_articles(articles: pd.DataFrame, category_column: str, percentile=50) -> pd.DataFrame:
    """Find all articles having less entropy for a category than a certain percentile.

    Args:
        articles (pd.DataFrame): articles
        category_column (str): column name to filter for
        percentile (int, optional): Percentile, from 0 to 100. Defaults to 50.

    Returns:
        list: Article ids, their category values and their entropies that have entropy lower than the percentile.
    """    
    articles = articles[[category_column, f'{category_column}_entropy']]
    percentile_boundary = np.percentile(articles[f'{category_column}_entropy'], percentile)
    articles = articles[articles[f'{category_column}_entropy'] >= percentile_boundary]
    return articles.index.values.tolist()


def find_category_distributions(articles: pd.DataFrame, category_columns: list) -> dict:   
    """Find the distribution of category expressions for low and high SES for all categories available.

    Args:
        articles (pd.DataFrame): Article data.
        category_columns (list): List of columns in the data corresponding to attributes.

    Returns:
        dict: dict of category distribution data frames for each category.
    """    
    category_distributions = {}
    for category in category_columns:
        category_articles = articles[['average_ses', 'content', category]]
        category_articles = category_articles[~category_articles[category].isna()]
        category_distribution = pd.DataFrame(data=None, index=category_articles[category].unique(), columns=['low', 'high'])
        category_distribution['low'] = category_articles[category_articles['average_ses'] == -1.0].groupby(category).count()['content']
        category_distribution['high'] = category_articles[category_articles['average_ses'] == +1.0].groupby(category).count()['content']
        category_distributions[category] = category_distribution
    return category_distributions


def chi2_per_label_test(distribution: pd.DataFrame, articles_per_SES: tuple) -> pd.DataFrame:
    """Perform a chi2 test on the absolute frequencies articles
    for each category label independently, e.g. for "movies" and for "sport".

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


def plot_category_distribution(category_distributions: dict, category: str, relative=True):
    """Plot the distribution of articles over the categories for low and high SES.

    Args:
        category_distribution (pd.DataFrame): Distribution matrix with categories (index) and SES (columns)
        category_name (str): Name of the category
        relative (bool, optional): Whether to normalize frequencies for each SES level. Defaults to True.
    """
    category_distribution = category_distributions[category].copy().sort_index()

    fig, ax = plt.gcf(), plt.gca()
    ax.set_title(f'{category.capitalize()} distribution by SES')
    if relative:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        category_distribution = category_distribution.apply(lambda col: col/col.sum())
    category_distribution.plot(kind='bar', ax=ax)
    fig.show()


def plot_human_annotation_confusion_matrix(articles: pd.DataFrame, human_annotated: pd.DataFrame, category: str):
    if category not in human_annotated.columns:
        return
    human_annotated_category = human_annotated[~human_annotated[category].isna()]
    if len(human_annotated_category) == 0:
        return
    articles_with_annotation = articles.join(human_annotated_category[[category]], rsuffix='_annotated', on='article_id', how='inner')[['content', category, f'{category}_annotated']]
    category_labels = np.unique(articles_with_annotation[[category, f'{category}_annotated']].values.ravel())
    category_confusion_matrix = confusion_matrix(y_true=articles_with_annotation[f'{category}_annotated'], y_pred=articles_with_annotation[category], labels=category_labels)
    ConfusionMatrixDisplay(category_confusion_matrix, display_labels=category_labels).plot()


def evaluate_category(category_distributions: dict, category, articles_per_SES: tuple, relative_dist_plot: bool=True):
    plot_category_distribution(category_distributions, category, relative=relative_dist_plot)

    contingency_chi2, contingency_p = chi2_contingency_test(category_distributions[category])
    print(f'Distribution chi2 test:\nchi2={contingency_chi2:.1f}, p={contingency_p:.3e}\n')

    print('Per-label chi2 test:')
    print(chi2_per_label_test(category_distributions[category], articles_per_SES))

    plot_human_annotation_confusion_matrix(articles, human_annotated, category)


# %%
category_distributions = find_category_distributions(articles, category_columns)
evaluate_category(category_distributions, 'prosociality', articles_per_SES=articles_per_SES)


# %%
articles_per_SES
# %%
