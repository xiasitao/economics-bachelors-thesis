# %%
import pickle
import pandas as pd
import matplotlib.pyplot as plt
with open('../../assets/role_model_articles_de.pkl', 'rb') as file:
    articles = pickle.load(file)
# %%
plt.hist(articles.sentences)
# %%

articles[articles.sentences > 1000].content

# %%
