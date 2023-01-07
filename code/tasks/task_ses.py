import pytask
import pickle
import pandas as pd
import numpy as np

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


def clean_ses_data(data: pd.DataFrame, role_model_data: pd.DataFrame) -> pd.DataFrame:
    """Clean up the SES data table.

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

    score_data['prevalent_ses'] = mention_data.groupby('role_model')['zero_centered_ses'].agg(pd.Series.mode).apply(lambda val: val if type(val) != np.ndarray else np.mean(val)).astype(float)
    score_data['average_ses'] = mention_data.groupby('role_model')['zero_centered_ses'].mean()
    score_data['rank_weighted_ses'] = mention_data.groupby('role_model')['rank_weighted_ses'].mean()
    score_data['significance_weighted_ses'] = mention_data.groupby('role_model')['significance_weighted_ses'].mean()

    return score_data


@pytask.mark.depends_on(BUILD_PATH / 'role_model_data.pkl')
@pytask.mark.produces(BUILD_PATH / 'ses.pkl')
def task_ses(produces: Path):
    ses_data = pd.read_excel(ASSET_PATH / 'Role_models_by_SES_precleaned.xlsx')
    role_model_data = pd.read_pickle(BUILD_PATH / 'role_model_data.pkl')
    ses_data = clean_ses_data(ses_data, role_model_data)
    ses_data.to_pickle(produces)


@pytask.mark.depends_on(BUILD_PATH / 'ses.pkl')
@pytask.mark.produces(BUILD_PATH / 'ses_mentions.pkl')
def task_ses_mentions(produces: Path):
    ses_data = pd.read_pickle(BUILD_PATH / 'ses.pkl')
    mention_data = produce_role_model_mention_table(ses_data)
    mention_data.to_pickle(produces)


@pytask.mark.depends_on(BUILD_PATH / 'ses_mentions.pkl')
@pytask.mark.produces(BUILD_PATH / 'ses_scores.pkl')
def task_ses_model_scores(produces: Path):
    mention_data = pd.read_pickle(BUILD_PATH / 'ses_mentions.pkl')
    score_data = produce_role_model_scores(
        mention_data=mention_data,
        rank_weights=[1/n for n in range(1, 6)]
    )
    score_data.to_pickle(produces)