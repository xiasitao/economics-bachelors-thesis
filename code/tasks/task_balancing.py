import pytask
import pickle
import pandas as pd
from sklearn.utils import shuffle

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


def balance_role_models(data, n_target = 50, downsample=True, upsample=True):
    """Balance the number of articles for all role models in a data set.
    Does not distinguish between languages.

    Args:
        data (pd.DataFrame): Articles to balance
        n_target (int, optional): Number of articles per role model. Defaults to 50.
        downsample (bool, optional): Whether to downsample if too many articles are present for a role model. Defaults to True.
        upsample (bool, optional): Whether to upsample if too few articles are present for a role model. Defaults to True.

    Returns:
        pd.DataFrame: Data frame with balanced numbers of articles per role model
    """
    new_data = pd.DataFrame(data=None, columns=data.columns)
    new_data = new_data.astype(data.dtypes)
    for role_model in data['role_model'].unique():
        role_model_data = shuffle(data[data['role_model'] == role_model])
        if downsample and len(role_model_data) > n_target:
            role_model_data = role_model_data.iloc[0:n_target]
        if upsample and len(role_model_data) < n_target:
            full_repetitions = n_target // len(role_model_data)
            additional = role_model_data.loc[[*role_model_data.index]*full_repetitions].reset_index(drop=True)
            additional.index = 1000000000 + role_model_data.index.min() + additional.index
            role_model_data = pd.concat([role_model_data, additional]).iloc[0:n_target]
        new_data = pd.concat([new_data, role_model_data])
    return new_data


@pytask.mark.depends_on(BUILD_PATH / 'data.pkl')
@pytask.mark.parametrize(
    "produces, n",
    [(BUILD_PATH / f'data_balanced_{n}.pkl', n) for n in (50, 100, 200, 500)]
)
def task_balancing(n: int, produces: Path):
    """This task balances the number of article per role model and language by downsampling and upsampling.
    It provides data sets with 50, 100, 200, and 500 articles per role model.

    Args:
        n (int): Number of articles per role model and language.
        produces (Path): Path to respective target file.
    """    
    data = pd.read_pickle(BUILD_PATH / 'data.pkl')
    balanced_data = pd.DataFrame(data=None, columns=data.columns)
    balanced_data = balanced_data.astype(data.dtypes)
    for language in data['language_ml'].unique():
        language_data = data[data['language_ml'] == language]
        balanced_language_data = balance_role_models(data=language_data, n_target=n, downsample=True, upsample=True)
        balanced_data = pd.concat([balanced_data, balanced_language_data])
    balanced_data.to_pickle(produces)

