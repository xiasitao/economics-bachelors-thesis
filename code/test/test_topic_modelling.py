# %%
import pandas as pd
import pickle
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
articles = pd.read_pickle(BUILD_PATH / 'articles_balanced_50.pkl')
ses = pd.read_pickle(BUILD_PATH / 'ses_scores_filtered.pkl')
articles = articles.join(ses, how='inner', on='role_model')
with open(BUILD_PATH / 'topic_modelling.pkl', 'rb') as file:
    topic_words, article_topics = pickle.load(file)
articles = articles.join(article_topics, how='inner', on='article_id')
articles_per_SES = articles[articles['average_ses']==-1.0].count()['content'], articles[articles['average_ses']==+1.0].count()['content']
topic_columns = [column for column in article_topics.columns if not column.endswith('_entropy')]


# %%
article_topics
# %%
topic_words
# %%
