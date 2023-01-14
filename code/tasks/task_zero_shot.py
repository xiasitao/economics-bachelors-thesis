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
@pytask.mark.persist()
@pytask.mark.depends_on(BUILD_PATH / 'data_balanced_50.pkl')
@pytask.mark.produces(ZERO_SHOT_BUILD_PATH)
def task_zero_shot_classification(produces: Path):
    articles = pd.read_pickle(BUILD_PATH / 'data_balanced_50.pkl')
    articles_en = articles[articles['language_ml'] == 'en']
    zs_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    
    article_data = articles_en[['article_id', 'content']].drop_duplicates()
    article_data = article_data.sample(n=10, random_state=42)
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
    
    classification_data.to_pickle(produces)

if __name__ == '__main__':
    task_zero_shot_classification(ZERO_SHOT_BUILD_PATH)