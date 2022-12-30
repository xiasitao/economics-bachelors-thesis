# %%
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

import sentence_transformers as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
# %%
articles = pd.read_pickle(f'{BUILD_PATH}/data.pkl')
articles_en = articles[articles.language_ml == 'en']
subset_en = articles_en.iloc[0:5000]
# %%
sentence_bert_encoder = st.SentenceTransformer('all-MiniLM-L6-v2')
# %%
# embeddings = sentence_bert_encoder.encode(subset_en['content'].to_list())
embeddings = sentence_bert_encoder.encode(subset_en['content_slim'].to_list())
# %%
# %%
embeddings_50d = PCA(n_components=10).fit_transform(embeddings)
embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings_50d)
# %%
plt.title('Embeddings')
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=6)
plt.show()
# %%
kmedoids = KMedoids(n_clusters=2).fit(embeddings_2d)
clusters = kmedoids.predict(embeddings_2d)
medoids = kmedoids.medoid_indices_
# %%
plt.title('KMedoids')
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=6, c=clusters)
plt.scatter(embeddings_2d[medoids][:, 0], embeddings_2d[medoids][:, 1], marker='*', color='red')
# %%
subset_en['profession'].hist(by=clusters)
# %%
subset_en[clusters == 1]['content'].iloc[6]
# %%
# %%
