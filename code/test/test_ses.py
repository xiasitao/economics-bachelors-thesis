# %%
import pandas as pd
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
ses_data = pd.read_pickle(BUILD_PATH / 'ses.pkl')
mention_data = pd.read_pickle(BUILD_PATH / 'ses_mentions.pkl')
score_data = pd.read_pickle(BUILD_PATH / 'ses_scores.pkl')
score_data_equilibrated = pd.read_pickle(BUILD_PATH / 'ses_scores_equilibrated.pkl')
# %%
ses_data
# %%
mention_data
# %%
score_data
# %%
score_data_equilibrated.groupby('average_ses').count()
# %%
