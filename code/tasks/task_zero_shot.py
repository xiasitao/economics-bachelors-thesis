""" This script is best executed with CUDA / on colab with CUDA.
    You need to adapt the paths to your Colab directory structure.
"""
import pytask
import pandas as pd
import numpy as np
from transformers import pipeline

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


def zs_classify_articles(model: pipeline, articles: list, candidate_labels: list) -> tuple:
    """Classify a batch of articles using a pipeline against a list of candidate topics.

    Args:
        model (pipeline): Huggingface zero-shot classification pipeline
        articles (list): List of articles to classify
        candidate_labels (list): List of possible labels

    Returns:
        tuple: List of labels, list of entropies
    """    
    results = model(articles, candidate_labels)
    assigned_labels = [result['labels'][0] for result in results]
    score_list = np.array([result['scores'] for result in results])
    entropies = -np.sum(score_list * np.log(score_list), axis=1)
    return assigned_labels, entropies


TOPIC_CATEGORIES = {
    'topic': ['movie', 'music', 'sport', 'life'],
    'difficulty': ['easy', 'difficult'],
    'emotion': ['sadness', 'happiness', 'fear', 'anger', 'surprise', 'disgust'],  # https://online.uwa.edu/infographics/basic-emotions/
}


ZERO_SHOT_BUILD_PATH  = BUILD_PATH / 'zero_shot_classification.pkl'
@pytask.mark.skip()  # Execute with CUDA on Colab
@pytask.mark.depends_on(BUILD_PATH / 'articles_balanced_50.pkl')
@pytask.mark.produces(ZERO_SHOT_BUILD_PATH)
def task_zero_shot_classification(produces: Path, n_articles=10, incremental=True, device=None):
    """Perform zero-shot classification on the 50-articles-per-role-model data set.
    Perform with CUDA or on Colab (with CUDA) by setting the device to "cuda:0".

    Args:
        produces (Path): Output file path
        n_articles (int, optional): Number of articles to process in this call. Defaults to 10.
        incremental (bool, optional): Whether to only process articles that have not been processed yet. Defaults to False.
    """    
    incremental = incremental and produces.exists()
    existing_data = None if not incremental else pd.read_pickle(produces)

    articles = pd.read_pickle(BUILD_PATH / 'articles_balanced_50.pkl')
    articles_en = articles[articles['language_ml'] == 'en']
    zs_classifier = pipeline(
        'zero-shot-classification',
        model='facebook/bart-large-mnli',
        device=device
    )
    
    article_data = articles_en[['article_id', 'content']].drop_duplicates()
    if incremental:
        print(f'Incremental classification, excluding {len(existing_data)} existing records.')
        article_data = article_data[~(article_data['article_id'].isin(existing_data.index))]
    if n_articles is not None and n_articles > 0:
        article_data = article_data.sample(n=min(n_articles, len(article_data)), random_state=42)
    
    classification_data = pd.DataFrame(
        data=None,
        index=article_data['article_id'],
        columns=[format_str.format(topic_category) for topic_category in TOPIC_CATEGORIES for format_str in ('{}', '{}_entropy')]
    )

    for topic_cateogory in TOPIC_CATEGORIES:
        labels, entropies = zs_classify_articles(
            model=zs_classifier,
            articles=article_data['content'].to_list(),
            candidate_labels=TOPIC_CATEGORIES[topic_cateogory]
        )
        classification_data[topic_cateogory] = labels
        classification_data[f'{topic_cateogory}_entropy'] = entropies

    if incremental:
        classification_data = pd.concat([existing_data, classification_data])
    classification_data.to_pickle(produces)

if __name__ == '__main__':
    task_zero_shot_classification(ZERO_SHOT_BUILD_PATH, n_articles=10, incremental=True, device='cpu')