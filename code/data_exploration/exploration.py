# %%
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import spacy_langdetect
from nltk.text import Text
from nltk.tokenize import sent_tokenize, word_tokenize
# %%
with open('../../assets/role_model_articles_de.pkl', 'rb') as file:
    articles_de = pickle.load(file)
with open('../../assets/role_model_articles_en.pkl', 'rb') as file:
    articles_en = pickle.load(file)
articles = pd.concat([articles_en, articles_de])
# %%
# Histograms
plt.title('Sentence count')
plt.hist(articles[articles.sentences < 200].sentences)
plt.show()

plt.title('Word count')
plt.hist(articles[articles.terms < 2000].terms)
plt.show()
# %%
# Language detection
pipeline = spacy.load('en_core_web_trf')
spacy.language.Language.factory('language_detector', func=lambda nlp, name: spacy_langdetect.LanguageDetector())
pipeline.add_pipe('language_detector', last=True)
# %%
pipeline(articles.iloc[200000].content)._.language

# %%
