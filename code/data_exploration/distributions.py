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
# Language distribution
role_models, counts = np.unique(articles['language_ml'], return_counts=True)
plt.title('Language distribution')
plt.bar(role_models, counts/np.sum(counts)*100, align='center')
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.show()
# %%
# Article per role model distribution
role_models, counts = np.unique(articles['role_model'], return_counts=True)
sorted_counts = np.sort(counts)
count_cumulative = sorted_counts.cumsum()
plt.title('Articles per role model')
plt.xlabel('percentile')
plt.ylabel('Cumulative relative amount of articles')
plt.plot(np.arange(count_cumulative.size)/(count_cumulative.size)*100, count_cumulative/np.max(count_cumulative)*100)
plt.gca().xaxis.set_major_formatter(PercentFormatter())
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.margins(x=0, y=0)
plt.grid()
plt.show()
# %%
# Role model gender
# %%
