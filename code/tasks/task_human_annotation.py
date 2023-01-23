import pytask
import pandas as pd

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

@pytask.mark.persist()
@pytask.mark.depends_on(BUILD_PATH / 'articles/articles_balanced_50.pkl')
@pytask.mark.depends_on(BUILD_PATH / 'role_models/ses_scores_filtered.pkl')
@pytask.mark.produces(BUILD_PATH / 'articles/articles_for_human_annotation.pkl')
def task_human_annotation_preparation(produces: Path):
    """Select 200 articles from the set of English SES-filtered articles.

    Args:
        produces (Path): Destination file path
    """    
    articles = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
    ses = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_filtered.pkl')
    articles = articles.join(ses, how='inner', on='role_model')
    articles_en = articles[articles['language_ml'] == 'en']
    sample_articles = articles_en[['content', 'article_id']].drop_duplicates().sample(n=200, random_state=42)
    sample_articles = sample_articles.set_index('article_id', drop=True)
    sample_articles.to_pickle(produces)


if __name__ == '__main__':
    task_human_annotation_preparation(BUILD_PATH / 'articles/articles_for_human_annotation.pkl')