import pytask
import pickle
import pandas as pd

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
    return data


@pytask.mark.depends_on(BUILD_PATH / 'role_model_data.pkl')
@pytask.mark.produces(BUILD_PATH / 'ses.pkl')
def task_ses(produces: Path):
    ses_data = pd.read_excel(ASSET_PATH / 'Role_models_by_SES_precleaned.xlsx')
    role_model_data = pd.read_pickle(BUILD_PATH / 'role_model_data.pkl')

    ses_data = clean_ses_data(ses_data, role_model_data)

    ses_data.to_pickle(produces)