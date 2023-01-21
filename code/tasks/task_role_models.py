import pytask
import pandas as pd

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


@pytask.mark.produces(BUILD_PATH / 'role_models/role_model_data.pkl')  
def task_role_model_data(produces: Path):
    """Bring role model information into a good shape for joining with article data.

    Args:
        produces (Path): Output path
    """    
    role_model_data = pd.read_excel(ASSET_PATH / 'role_model_data.xlsx')[['Star', 'Sex', 'Birth_year', 'Nationality', 'Profession_1']]
    role_model_data = role_model_data.rename({
        'Star': 'role_model',
        'Sex': 'sex',
        'Birth_year': 'birth_year',
        'Nationality': 'nationality',
        'Profession_1': 'profession',
    }, axis=1)

    role_model_data = role_model_data[~(role_model_data['role_model'].isna())]
    role_model_data = role_model_data.drop_duplicates()

    role_model_data['role_model'] = role_model_data['role_model'].astype(str).str.strip()
    role_model_data['sex'] = role_model_data['sex'].astype(pd.Int64Dtype())
    role_model_data['birth_year'] = role_model_data['birth_year'].astype(pd.Int64Dtype())
    role_model_data['nationality'] = role_model_data['nationality'].astype(str)
    role_model_data['profession'] = role_model_data['profession'].astype(str)
    
    role_model_data = role_model_data.set_index('role_model')
    role_model_data.to_pickle(produces)