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
ses = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_filtered.pkl')
articles = articles.join(ses, how='inner', on='role_model')
zero_shot_data = pd.read_pickle(BUILD_PATH / 'zero_shot_classification/zero_shot_classification.pkl')
articles = articles.join(zero_shot_data, how='inner', on='article_id')
articles_per_SES = articles[articles['average_ses']==-1.0].count()['content'], articles[articles['average_ses']==+1.0].count()['content']
category_columns = [column for column in zero_shot_data.columns if not column.endswith('_entropy')]
human_annotated = pd.read_pickle(BUILD_PATH / 'articles/articles_human_annotated.pkl')


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
        category_distribution = pd.DataFrame(data=None, index=articles[category].unique(), columns=['low', 'high'])
        category_distribution['low'] = articles[articles['average_ses'] == -1.0].groupby(category).count()['content']
        category_distribution['high'] = articles[articles['average_ses'] == +1.0].groupby(category).count()['content']
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


def plot_category_distribution(category_distribution: pd.DataFrame, category_name: str, relative=True):
    """Plot the distribution of articles over the categories for low and high SES.

    Args:
        category_distribution (pd.DataFrame): Distribution matrix with categories (index) and SES (columns)
        category_name (str): Name of the category
        relative (bool, optional): Whether to normalize frequencies for each SES level. Defaults to True.
    """    
    fig, ax = plt.gcf(), plt.gca()
    ax.set_title(f'{category_name.capitalize()} distribution by SES')
    if relative:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        category_distribution = category_distribution.copy()
        category_distribution = category_distribution.apply(lambda col: col/col.sum())
    category_distribution.plot(kind='bar', ax=ax)
    fig.show()


# %%
category_distributions = find_category_distributions(articles, category_columns)
plot_category_distribution(category_distributions['difficulty'], 'topic', relative=True)


# %%
chi2_contingency_test(category_distributions['difficulty'])


# %%
chi2_per_label_test(category_distributions['difficulty'], articles_per_SES)


# %%
articles[
    (articles.index.isin(find_low_entropy_articles(articles, 'topic', 50))) 
    & (articles['topic'] == 'movie')
].iloc[0].content
# %%
np.percentile(articles[articles['topic'] == 'movie'].topic_entropy, 75)
# %%
human_annotated_topic = human_annotated[~human_annotated['topic'].isna()]
articles_with_annotation = articles.join(human_annotated_topic[['topic']], rsuffix='_annotated', how='inner')[['content', 'topic', 'topic_annotated']]
topic_labels = np.unique(articles_with_annotation[['topic', 'topic_annotated']].values.ravel())
topic_confusion_matrix = confusion_matrix(y_true=articles_with_annotation['topic_annotated'], y_pred=articles_with_annotation['topic'], labels=topic_labels)
ConfusionMatrixDisplay(topic_confusion_matrix, display_labels=topic_labels).plot()
# %%

# %%
