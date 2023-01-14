# %%
import pandas as pd
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
balanced_100 = pd.read_pickle(f'{BUILD_PATH}/articles_balanced_100.pkl')


# %%
balanced_100.groupby(['language_ml', 'role_model']).count()


# %%
balanced_100


# %%
balanced_100.dtypes


# %%
