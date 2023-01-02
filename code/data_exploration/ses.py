# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
ses_data = pd.read_excel(ASSET_PATH / 'Role_models_by_SES_precleaned.xlsx')
role_model_data = pd.read_pickle(BUILD_PATH / 'role_model_data.pkl')
# %%
# invalid_role_models = ses_data.join(role_model_data, how='left', on='Role_model_1')
# invalid_role_models = invalid_role_models[(invalid_role_models['profession'].isna()) & ~(invalid_role_models['Role_model_1'].isna())]
# invalid_role_models


# %%
def clean_ses_data(data: pd.DataFrame, role_model_data: pd.DataFrame) -> pd.DataFrame:
    """Clean up the SES data table.
    Unifies column names. Removes rows where the first role model is not known.
    Unifies SES notations. Counts role models.

    Args:
        data (pd.DataFrame): Substrate
        role_model_data (pd.DataFrame): Reference cleaned role model data

    Returns:
        pd.DataFrame: Cleaned copy of substrate
    """    
    data = data.copy()
    data = data.rename({
        'Unnamed: 0': 'id',
        'Low_SES': 'low_ses',
        'High_SES': 'high_ses',
        'Role_model_1': 'role_model_1',
        'Role_model_2': 'role_model_2',
        'Role_model_3': 'role_model_3',
        'Role_model_4': 'role_model_4',
        'Role_model_5': 'role_model_5',
    }, axis=1)
    data = data[~(data['id'].isna())]
    data = data.astype({'id': pd.Int64Dtype()})
    data = data.set_index('id')
    data = data[data['role_model_1'].isin(role_model_data.index)]
    data['low_ses'] = data['low_ses'] == 1.0
    data['high_ses'] = data['high_ses'] == 1.0
    data['ses'] = data['high_ses'].apply(lambda high_ses: 1.0 if high_ses else 0.0)
    data['role_model_count'] = 5 - data[[f'role_model_{n}' for n in range(1, 6)]].isna().sum(axis=1)
    return data
ses_data_clean = clean_ses_data(ses_data, role_model_data)

# %%
def produce_role_model_mention_table(ses_data: pd.DataFrame) -> pd.DataFrame:
    """Creates a table of all mentions of role models in the SES data.
    Each time a role model is mentioned in one of the 1..5th rank corresponds to one row.
    This rank and a role model significance, being the inverse amount of role models the same study participant mentioned.

    Args:
        ses_data (pd.DataFrame): Substrate (cleaned) ses data

    Returns:
        pd.DataFrame: A dataframe with a row for every time a role model is mentioned by a study participant.
    """    
    mention_data = pd.DataFrame(columns=['id', 'role_model', 'ses', 'rank'])
    for rank, column in [(n, f'role_model_{n}') for n in range(1, 6)]:
        rank_mention_data = ses_data[~(ses_data[column].isna())]
        rank_mention_data = rank_mention_data.reset_index()
        rank_mention_data = rank_mention_data.rename({column: 'role_model'}, axis=1)
        rank_mention_data['rank'] = rank
        rank_mention_data['significance'] = 1 / rank_mention_data['role_model_count']
        rank_mention_data = rank_mention_data[['id', 'role_model', 'ses', 'rank', 'significance']]
        mention_data = pd.concat([mention_data, rank_mention_data]).reset_index(drop=True)
    return mention_data
role_model_mentions = produce_role_model_mention_table(ses_data_clean)


# %%
def produce_role_model_scores(mention_data: pd.DataFrame, rank_weights = [1, 1/2, 1/3, 1/4, 1/5]) -> pd.DataFrame:
    """Aggregates the mention data to the role model level,
    calculating counts, rank-weighted counts, and significance-weighted counts,
    as well as zero-centered average SES scores, average rank-weighted SES scores,
    and average significance-weighted SES scores.

    Args:
        mention_data (pd.DataFrame): List of role model mentions.
        rank_weights (list, optional): How to weigh a mention in every rank from 1 to 5. Defaults to [1, 1/2, 1/3, 1/4, 1/5].

    Returns:
        pd.DataFrame: _description_
    """    
    mention_data = mention_data.copy()
    mention_data['zero_centered_ses'] = mention_data['ses'].map({1.0: 1.0, 0.0: -1.0})
    mention_data['weighted_rank'] = mention_data['rank'].apply(lambda rank: rank_weights[rank-1])
    mention_data['rank_weighted_ses'] = mention_data['zero_centered_ses'] * mention_data['weighted_rank']
    mention_data['significance_weighted_ses'] = mention_data['zero_centered_ses'] * mention_data['significance']

    role_models = mention_data['role_model'].unique().sort()
    score_data = pd.DataFrame(index=role_models)
    score_data['count'] = mention_data.groupby('role_model').count()['id']
    score_data['rank_weighted_count'] = mention_data.groupby('role_model')['weighted_rank'].sum()
    score_data['significance_weighted_count'] = mention_data.groupby('role_model')['significance'].sum()

    score_data['average_ses'] = mention_data.groupby('role_model')['zero_centered_ses'].mean()
    score_data['rank_weighted_ses'] = mention_data.groupby('role_model')['rank_weighted_ses'].mean()
    score_data['significance_weighted_ses'] = mention_data.groupby('role_model')['significance_weighted_ses'].mean()

    return score_data
role_model_scores = produce_role_model_scores(role_model_mentions)
role_model_scores

# %%
role_model_count, count = np.unique(role_model_scores['count'], return_counts=True)
plt.title('Role model mention counts')
plt.bar(role_model_count, count)
plt.show()

role_model_ranked_count, count = np.unique(role_model_scores['rank_weighted_count'], return_counts=True)
plt.title('Rank-weighted role model mention counts')
plt.bar(role_model_ranked_count, count, width=1/5)
plt.show()

role_model_significance_count, count = np.unique(role_model_scores['significance_weighted_count'], return_counts=True)
plt.title('Significance-weighted role model mention counts')
plt.bar(role_model_significance_count, count, width=1/10)
plt.show()
# %%

plt.hist(role_model_scores['average_ses'])

# %%
