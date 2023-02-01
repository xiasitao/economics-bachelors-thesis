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
articles_raw = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
ses = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores.pkl')
ses_distinct = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_distinct.pkl')
human_annotated = pd.read_pickle(BUILD_PATH / 'articles/articles_human_annotated.pkl')
human_annotated = pd.concat([human_annotated, pd.read_pickle(BUILD_PATH / 'articles/articles_human_annotated_distinct.pkl')])

def collect_zero_shot_data():
    """Collect all zero-shot data from the different pickle files.

    Returns:
        pd.DataFrame: Zero-shot data for each article.
    """    
    zero_shot_data = None
    for category in ['article_type', 'crime', 'crime_type', 'emotion', 'prosociality', 'relatability', 'sentiment', 'sentiment_n', 'topic', 'topic_l', 'writing_style']:
        filename = BUILD_PATH / f'zero_shot_classification/zero_shot_classification_{category}.pkl'
        if filename.exists:
            category_data = pd.read_pickle(filename)
            zero_shot_data = pd.concat([zero_shot_data, category_data], axis=1) if zero_shot_data is not None else category_data
    return zero_shot_data
zero_shot_data = collect_zero_shot_data()
category_columns = [column for column in zero_shot_data.columns if not column.endswith('_entropy') and not column.endswith('_p')]


# %%
def load_prepare_article_data(articles_raw: pd.DataFrame, ses: pd.DataFrame, zero_shot_data: pd.DataFrame):
    """Combine article data, ses, and topic data.

    Args:
        articles (pd.DataFrame): Articles
        ses (pd.DataFrame): SES scores
        zero_shot_data (pd.DataFrame): Category associations of the articles.

    Returns:
        tuple: articles [pd.DataFrame], articles_per_SES [tuple]
    """
    articles = articles_raw.join(ses, how='inner', on='role_model')
    articles = articles.join(zero_shot_data, how='inner', on='article_id')
    articles_per_SES = articles[articles['low_ses']].count()['content'], articles[articles['high_ses']].count()['content']
    return articles, articles_per_SES

articles, articles_per_SES = load_prepare_article_data(articles_raw, ses, zero_shot_data)
articles_distinct, articles_per_SES_distinct = load_prepare_article_data(articles_raw, ses_distinct, zero_shot_data)



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
        category_articles = articles[['low_ses', 'high_ses', 'content', category]]
        category_articles = category_articles[~category_articles[category].isna()]
        category_distribution = pd.DataFrame(data=None, index=category_articles[category].unique(), columns=['low', 'high'])
        category_distribution['low'] = category_articles[category_articles['low_ses']].groupby(category).count()['content']
        category_distribution['high'] = category_articles[category_articles['high_ses']].groupby(category).count()['content']
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


def plot_category_distribution(category_distributions: dict, category: str, relative=True, show_title: bool=True, additional_title_text=None):
    """Plot the distribution of articles over the categories for low and high SES.

    Args:
        category_distribution (pd.DataFrame): Distribution matrix with categories (index) and SES (columns)
        category_name (str): Name of the category
        relative (bool, optional): Whether to normalize frequencies for each SES level. Defaults to True.
        show_title (bool, optional): Whether to show a title indicating the category. Defaults to False.
        additional_title_text (str, optional): Additional string to be shown in the title.
    """
    category_distribution = category_distributions[category].copy().sort_index()

    fig, ax = plt.gcf(), plt.gca()
    title = f'{category.capitalize()} distribution by SES' if show_title else ''
    if additional_title_text is not None:
        title = (' - ' if title != '' else '').join([title, additional_title_text])
    if title != '':
        ax.set_title(title)
    if relative:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        category_distribution = category_distribution.apply(lambda col: col/col.sum())
    category_distribution.plot(kind='bar', ax=ax)
    fig.show()


def plot_human_annotation_confusion_matrix(articles: pd.DataFrame, human_annotated: pd.DataFrame, category: str):
    """Plot the confusion matrix for a certain category.

    Args:
        articles (pd.DataFrame): Article data with category associations.
        human_annotated (pd.DataFrame): Human annotated data.
        category (str): Category column
    """    
    if category not in human_annotated.columns:
        return
    human_annotated_category = human_annotated[~human_annotated[category].isna()]
    if len(human_annotated_category) == 0:
        return
    articles_with_annotation = articles.join(human_annotated_category[[category]], rsuffix='_annotated', on='article_id', how='inner')[['content', category, f'{category}_annotated']]
    category_labels = np.unique(articles_with_annotation[[category, f'{category}_annotated']].values.ravel())
    category_confusion_matrix = confusion_matrix(y_true=articles_with_annotation[f'{category}_annotated'], y_pred=articles_with_annotation[category], labels=category_labels)
    ConfusionMatrixDisplay(category_confusion_matrix, display_labels=category_labels).plot()


def evaluate_category(
        articles: pd.DataFrame,
        category: str,
        articles_per_SES: tuple,
        human_annotated: pd.DataFrame=None,
        relative_dist_plot: bool=True,
        is_distinct: bool=None,
        show_title: bool=True
    ):
    """Plot the category distributions, show the chi2 statistics, and
    if possible, show the confusion matrix.

    Args:
        articles (pd.DataFrame): Article data with category associations.
        category (str): category column
        articles_per_SES (tuple): Reference articles number per SES
        human_annotated (pd.DataFrame, optional): Human annotation data for confusion matrix. Defaults to None.
        relative_dist_plot (bool, optional): Plot the distribution with relative frequencies. Defaults to True.
        is_distinct (bool, optional): Whether to indicate the distinct-SES dataset. Defaults to None.
        show_title (bool, optional): Show a title in the distribution plot. Defaults to True.
    """    
    category_distributions = find_category_distributions(articles, category_columns)
    
    distinct_text = None
    if is_distinct is not None:
        distinct_text = 'distinct' if is_distinct else 'general'
    plot_category_distribution(category_distributions, category, relative=relative_dist_plot, additional_title_text=distinct_text, show_title=show_title)

    contingency_chi2, contingency_p = chi2_contingency_test(category_distributions[category])
    print(f'Distribution chi2 test:\nchi2={contingency_chi2:.1f}, p={contingency_p:.3e}\n')

    print('Per-label chi2 test:')
    print(chi2_per_label_test(category_distributions[category], articles_per_SES))
    
    if human_annotated is not None:
        plot_human_annotation_confusion_matrix(articles, human_annotated, category)


# %%
evaluate_category(articles, 'prosociality', articles_per_SES=articles_per_SES, human_annotated=human_annotated, is_distinct=False)


# %%
evaluate_category(articles_distinct, 'prosociality', articles_per_SES=articles_per_SES_distinct, human_annotated=human_annotated, is_distinct=False)



# %%
articles_per_SES


# %%
