# %%
import matplotlib.pyplot as plt
import nltk
nltk.download('book')
#%%

from urllib import request
#urllib is a built-in python library for handling web interactions such as accessing a web-page, reading its contents etc.
myurl = "https://www.gutenberg.org/files/2446/2446-0.txt"
response = request.urlopen(myurl)
mytext = ""
for line in response:
    mytext += line.decode("utf-8").strip() + "\n"
    
print(mytext)
# %%
from nltk.text import Text
from nltk.tokenize import sent_tokenize, word_tokenize

text_nltk = Text(word_tokenize(mytext.lower()))
# %%
text_nltk
# %%
text_nltk.concordance("town")
# %%
text_nltk.similar("people")
text_nltk.similar("stockmann")
# %%
text_nltk.dispersion_plot(["peter", "citizens", "town", "enemy", "morten"])

# %%
len(text_nltk) 
# %%
len(set(text_nltk))
# %%
sorted(set(text_nltk))
# %%
all_sens = sent_tokenize(mytext) #i am taking the raw mytext variable directly, not text_nltk.
len(all_sens)
# %%
all_sens[233] #looking at a random sentence.

# %%
words_in_a_sentence = word_tokenize(all_sens[233])
print(words_in_a_sentence)
# %%
from nltk import FreqDist
fdist1 = FreqDist(text_nltk)
fdist1.most_common(10)
# %%
fdist1["petra"]
# %%
# Selecting words longer than 15 characters.
myvocab = set(text_nltk)
long_words = [w for w in myvocab if len(w) > 15]
long_words
# %%
from nltk import bigrams
bigrams_list = list(bigrams(text_nltk))
# %%
len(bigrams_list)
# %%
bigrams_list[22]
# %%
text_nltk.collocations()
# %%
text_nltk.collocation_list()
# %%
dist = FreqDist([len(w) for w in text_nltk]) 
# %%
dist
# %%
len(dist)
# %%
dist[21] #seeing how many 21 character tokens exist
# %%
dist.max() #What is this doing? 
# %%
100*dist.freq(5)
# %%
100*dist.freq(19)
# %%
dist.plot()
# %%
