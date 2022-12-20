# %%
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sklearn.manifold import TSNE
# %%
articles = pd.read_pickle(f'{BUILD_PATH}/data.pkl')
articles_en = articles[articles.language_ml == 'en']
articles_en['contains_term'] = articles_en['content'].str.contains(r'\bpolitics\b')
subset_en = articles_en.iloc[0:5000]
# %%
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
embedding_BERT = tokenizer.batch_encode_plus(subset_en.content.to_list(), truncation=True, padding=True)
embedding_BERT_values = np.array(embedding_BERT['input_ids'])

# %%
content_encodings_TSNE = TSNE(n_components=2).fit_transform(embedding_BERT_values)
plt.scatter(content_encodings_TSNE[:,0], content_encodings_TSNE[:,1], c=subset_en.contains_term)
# %%
