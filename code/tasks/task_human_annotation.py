import pytask
import pandas as pd
import numpy as np

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

@pytask.mark.depends_on(BUILD_PATH / 'articles_balanced_50.pkl')
@pytask.mark.produces(BUILD_PATH / 'articles_for_human_annotation.pkl')
def task_human_annotation_preparation(produces: Path):
    """Save the human annotation data.

    Args:
        produces (Path): Destination file path
    """    
    articles = pd.read_pickle(BUILD_PATH / 'articles_balanced_50.pkl')
    articles_en = articles[articles['language_ml'] == 'en']
    sample_articles = articles_en.sample(n=200, random_state=42)
    sample_articles = sample_articles[['content']]
    sample_articles.to_pickle(produces)