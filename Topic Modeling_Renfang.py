# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 20:27:13 2022

@author: alva
"""
import pandas as pd
import os
import sys
import glob
from hw_utils import *


the_data_out = 'C:/Users/alva/Documents/GitHub/NLP_Elon/output_Elon/'
the_data = open_pickle(the_data_out, "data_label_cnt.pkl")


# topic modeling 

lda_fun(the_data_out, the_data.tweet_cleaned)   # coherence score & num of topics. 6 topics


# lda
import gensim
def sent_to_words(sentences):
    for sentence in sentences:
        # lowercases, tokenizes, removes punctuations(deacc=True)
        # output tokens
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

data = the_data.tweet_cleaned.values.tolist()
data_words = list(sent_to_words(data))


dictionary = gensim.corpora.Dictionary(data_words)
# Term Document Frequency     a list of tuples (word_id, word_frequency)
bow_corpus = [dictionary.doc2bow(doc) for doc in data_words]

# Build LDA model

lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                    num_topics=6, 
                                    id2word = dictionary,                                    
                                    passes = 10,
                                    workers = 2)

# Print the Keyword in the 6 topics
from pprint import pprint
pprint(lda_model.print_topics())
doc_lda = lda_model[bow_corpus]



# visualize lda
import pyLDAvis.gensim_models
import pickle 
import pyLDAvis
# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_data_filepath = f'{the_data_out}lda_{str(num_topics)}'
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dictionary)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, f'{LDAvis_data_filepath}.html')
LDAvis_prepared



















