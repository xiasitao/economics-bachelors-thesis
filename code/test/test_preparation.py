# %%
import pandas as pd
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()
data = pd.read_pickle(f'{BUILD_PATH}/data.pkl')
# %% 
data
# %%
data.iloc[3]['content_slim']
# %%
