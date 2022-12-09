# %%
#Download the required Spacy model. 
import spacy
import spacy.cli
# spacy.cli.download("en_core_web_trf")
nlp = spacy.load('en_core_web_trf')
# %%
text = "Ludwig Maximilian University of Munich (also referred to as LMU or simply as the University of Munich; German: Ludwig-Maximilians-Universität München) is a public research university located in Munich, Germany, and is the country's sixth-oldest university in continuous operation."
doc = nlp(text)
# %%
for token in doc:
    print(token)
# %%
#Getting the part of speech tags for individual tokens
for token in doc:
    # Print the token and the POS tags
    print(token, token.pos_, token.tag_)
# %%
# Print the token and the results of morphological analysis
for token in doc:
    print(token, token.morph)
# %%
#Get the per token morphological information
#doc[7] is the word "referred" in our text
print(doc[7])
print(doc[7].morph.to_dict())
# %%
#View the syntactic parse tree of the sentence to see relations between words
from spacy import displacy
displacy.render(doc, style='dep', options={'compact': True})
# %%
# Loop over sentences in the Doc object and count them using enumerate()
# We have only one sentence in our doc, though. 
for number, sent in enumerate(doc.sents):    
    print(number, sent)
# %%
# Print the token and its lemma
for token in doc:
    print(token, token.lemma_)
# %%
# Loop over the named entities in the Doc object 
for ent in doc.ents:
    # Print the named entity and its label
    print(ent.text, ent.label_)
# %%
displacy.render(doc, style='ent')
# %%
# Get the noun chunks in the doc.
for item in doc.noun_chunks:
    print(item)
# %%



import textacy
text2 = """
Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.[2]

Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks, convolutional neural networks and Transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance
"""
# %%
#we still have to convert to a spacy doc, even when using textacy.
# so we have to load a spacy model first and then use it to convert.
en = textacy.load_spacy_lang("en_core_web_trf", disable=("parser",))
tdoc = textacy.make_spacy_doc(text2, en)
# %%
list(textacy.extract.ngrams(tdoc, 3, filter_stops=True, filter_punct=True, filter_nums=False))
# %%
from textacy.extract import keyterms as kt
kt.textrank(tdoc, normalize="lemma", topn=10)
# %%
