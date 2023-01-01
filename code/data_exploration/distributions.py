# %%
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.utils import shuffle
# %%
# Load data
articles = pd.read_pickle(f'{BUILD_PATH}/data.pkl')
articles_en = articles[articles.language_ml == 'en']
with open(BUILD_PATH / 'corpora.pkl', 'rb') as file:
    corpora = pickle.load(file)
with open(BUILD_PATH / 'sentence_tokens.pkl', 'rb') as file:
    sentence_tokens = pickle.load(file)
with open(BUILD_PATH / 'word_statistic.pkl', 'rb') as file:
    word_statistics = pickle.load(file)

# %%
def balance_role_models(data, n_target = 50, downsample=True, upsample=True):
    new_data = pd.DataFrame(data=None, columns=data.columns)
    max_index = data.index.max()
    for role_model in data['role_model'].unique():
        role_model_data = shuffle(data[data['role_model'] == role_model])
        if downsample and len(role_model_data) > n_target:
            role_model_data = role_model_data.iloc[0:n_target]
        if upsample and len(role_model_data) < n_target:
            full_repetitions = n_target // len(role_model_data)
            additional = role_model_data.loc[[*role_model_data.index]*full_repetitions].reset_index()
            additional.index = 1000000000 + role_model_data.index.min() + additional.index
            role_model_data = pd.concat([role_model_data, additional]).iloc[0:n_target]

        new_data = pd.concat([new_data, role_model_data])
    return new_data
balanced = balance_role_models(articles_en)


# %%
# Article per role model distribution
role_models, counts = np.unique(articles['role_model'], return_counts=True)
role_models_balanced, counts_balanced = np.unique(balanced['role_model'], return_counts=True)
sorted_counts, sorted_counts_balanced = np.sort(counts), np.sort(counts_balanced)
counts_cumulative, counts_cumulative_balanced = sorted_counts.cumsum(), sorted_counts_balanced.cumsum()
plt.title('Articles per role model')
plt.xlabel('percentile')
plt.ylabel('Cumulative relative amount of articles')
plt.plot(np.arange(counts_cumulative.size)/(counts_cumulative.size)*100, counts_cumulative/np.max(counts_cumulative)*100, label='data')
plt.plot(np.arange(counts_cumulative_balanced.size)/(counts_cumulative_balanced.size)*100, counts_cumulative_balanced/np.max(counts_cumulative_balanced)*100, label='balanced data')
plt.gca().xaxis.set_major_formatter(PercentFormatter())
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.margins(x=0, y=0)
plt.grid()
plt.legend()
plt.show()


# %%
# Language distribution
role_models, counts = np.unique(articles['language_ml'], return_counts=True)
plt.title('Language distribution')
plt.bar(role_models, counts/np.sum(counts)*100, align='center')
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.show()


# %%
# Article and role model gender
article_genders, counts = np.unique(articles['sex'].dropna(), return_counts=True)
plt.title('Article gender distribution')
plt.bar(['male', 'female'], counts/sum(counts)*100, align='center')
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.show()

role_model_genders, counts = np.unique(articles[['role_model', 'sex']].drop_duplicates()['sex'].dropna(), return_counts=True)
plt.title('Role model gender distribution')
plt.bar(['male', 'female'], counts/sum(counts)*100, align='center')
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.show()



# %%
# %%
