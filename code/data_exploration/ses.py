# %%
import pandas as pd

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
ses_data = pd.read_excel(ASSET_PATH / 'Role_models_by_SES_precleaned.xlsx')
role_model_data = pd.read_pickle(BUILD_PATH / 'role_model_data.pkl')
# %%
invalid_role_models = ses_data.join(role_model_data, how='left', on='Role_model_1')
invalid_role_models = invalid_role_models[(invalid_role_models['profession'].isna()) & ~(invalid_role_models['Role_model_1'].isna())]
invalid_role_models


# %%
def clean_ses_data(data: pd.DataFrame, role_model_data: pd.DataFrame) -> pd.DataFrame:
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
    return data
ses_data_clean = clean_ses_data(ses_data, role_model_data)

# %%
def produce_role_model_mention_table(ses_data: pd.DataFrame) -> pd.DataFrame:
    mention_data = pd.DataFrame(columns=['id', 'role_model', 'ses', 'rank'])
    for rank, column in [(n, f'role_model_{n}') for n in range(1, 6)]:
        rank_mention_data = ses_data[~(ses_data[column].isna())]
        rank_mention_data = rank_mention_data.reset_index()
        rank_mention_data = rank_mention_data.rename({column: 'role_model'}, axis=1)
        rank_mention_data['rank'] = rank
        rank_mention_data = rank_mention_data[['id', 'role_model', 'ses', 'rank']]
        mention_data = pd.concat([mention_data, rank_mention_data]).reset_index(drop=True)
    return mention_data
produce_role_model_mention_table(ses_data_clean)
# %%
