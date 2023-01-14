# %%
import pandas as pd

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
articles = pd.read_pickle(BUILD_PATH / 'articles_balanced_50.pkl')
zero_shot_data = pd.read_pickle(BUILD_PATH / 'zero_shot_classification.pkl')


# %%
zero_shot_data
# %%
articles.join(zero_shot_data, how='inner', on='article_id')
# %%
