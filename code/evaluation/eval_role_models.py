# %%
import pandas as pd
import numpy as np

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
ses = pd.read_pickle(BUILD_PATH / 'role_models/ses.pkl')
print('Number of children with sensible role models:', len(ses))
mentions = pd.read_pickle(BUILD_PATH / 'role_models/ses_mentions.pkl')
print('Number of mentions: ', len(mentions))
print('Mentions by rank:', mentions.groupby('rank').count()['id'])


# %%
role_model_data = pd.read_pickle(BUILD_PATH / 'role_models/role_model_data.pkl')
print('Number of role models:', len(role_model_data))
print('Role models by gender:', [f'{gender}: {number}' for (gender, number) in zip(*np.unique(role_model_data['sex'].fillna(2), return_counts=True))])


# %%
scores = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores.pkl')
scores


# %%
scores_distinct = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_distinct.pkl')
scores_distinct


# %%
# %%
