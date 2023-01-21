# %%
import pandas as pd
import numpy as np

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

HUMAN_ANNOTATED_PATH = (BUILD_PATH / 'articles/articles_human_annotated.pkl')


# %%
def ask_for_annotations(sample_articles: pd.DataFrame, category: str, file_path: Path=HUMAN_ANNOTATED_PATH) -> pd.Series:
    if category not in sample_articles.columns:
        raise Exception(f'Invalid column: {category}')
    
    if category not in sample_articles.columns:
        sample_articles[category] = None

    for article_id in sample_articles[sample_articles[category].isna()].index:
        article = sample_articles.loc[article_id]['content']
        chosen = input(f'''{article}

        Choose: {" ".join([f"'{option}" for option in annotation_categories[category]])} or 'END'
        ''')

        chosen = chosen.lower().strip()

        if chosen == 'end':
            break
        elif chosen in annotation_categories[category]:
            sample_articles.loc[article_id][category] = chosen
        else:
            print(f"Invalid choice: {chosen}. Stopping")
            break
    
    sample_articles.to_pickle(file_path)

# %%
sample_articles = pd.read_pickle(BUILD_PATH / 'articles/articles_for_human_annotation.pkl')
if HUMAN_ANNOTATED_PATH.exists():
    sample_articles = pd.read_pickle(HUMAN_ANNOTATED_PATH)

annotation_categories = {
    'topic': ['movie', 'music', 'sport', 'life'],
    'difficulty': ['easy', 'difficult'],
    'emotion': ['sadness', 'happiness', 'fear', 'anger', 'surprise', 'disgust'],
}

for category in annotation_categories:
    if category in sample_articles.columns:
        print(f'{category}: annotated {sample_articles[category].count()} / {len(sample_articles[category])}')
    else:
        print(f'{category}: not started annotating yet')

ask_for_annotations(sample_articles, 'topic')


# %%
