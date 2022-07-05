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

#---------------vectorize
# my_vec_data = my_bow(df_in=the_data.tweet_cleaned, path_in=the_data_out, gram_m=1, gram_n=1, sw="tf-idf", name_in="data_vec.pkl")

# my_dim_data = my_pca(my_vec_data, the_data_out, "pca.pkl", 0.95)


# #---------------topic

# lda_fun(the_data_out, the_data.tweet_cleaned)   # coherence score & num of topics. 6 topics


#-------------------------lda
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

# # # Build LDA model

num_topics = 5
lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                    num_topics=5, 
                                    id2word = dictionary,                                    
                                    passes = 10,
                                    workers = 2)

# # Print the Keyword in the 6 topics
from pprint import pprint
pprint(lda_model.print_topics())
doc_lda = lda_model[bow_corpus]

# result————5
# positive mood
[(0,
  '0.040*"yes" + 0.034*"floor" + 0.029*"model" + 0.023*"rolling" + '
  '0.020*"laughing" + 0.016*"soon" + 0.014*"coming" + 0.014*"heart" + '
  '0.013*"exactly" + 0.011*"great"'),
 # launching rocket
 (1,
  '0.013*"year" + 0.013*"launch" + 0.012*"next" + 0.011*"space" + '
  '0.011*"rocket" + 0.011*"time" + 0.009*"true" + 0.009*"many" + '
  '0.008*"starship" + 0.008*"dragon"'),
 # future life on mars
 (2,
  '0.014*"face" + 0.011*"earth" + 0.010*"would" + 0.009*"mars" + 0.007*"high" '
  '+ 0.007*"life" + 0.006*"future" + 0.006*"cool" + 0.005*"need" + '
  '0.005*"good"'),
 # production of autolipot car
 (3,
  '0.017*"good" + 0.012*"car" + 0.010*"yeah" + 0.008*"team" + '
  '0.008*"autopilot" + 0.007*"one" + 0.007*"right" + 0.007*"great" + '
  '0.007*"work" + 0.006*"production"'),
 # not much information, seems to state some facts
 (4,
  '0.016*"people" + 0.014*"thanks" + 0.013*"make" + 0.012*"sure" + '
  '0.011*"like" + 0.010*"think" + 0.009*"much" + 0.008*"ok" + 0.008*"know" + '
  '0.007*"real"')]


# result————6
# positice words
[(0,
  '0.049*"yes" + 0.019*"exactly" + 0.013*"like" + 0.013*"good" + 0.008*"would" '
  '+ 0.008*"point" + 0.007*"definitely" + 0.006*"cool" + 0.006*"way" + '
  '0.006*"time"'),
  (1,
  '0.038*"model" + 0.032*"floor" + 0.022*"rolling" + 0.020*"laughing" + '
  '0.017*"car" + 0.016*"soon" + 0.016*"yeah" + 0.014*"coming" + 0.010*"sure" + '
  '0.009*"also"'),
  (2,
  '0.015*"face" + 0.010*"new" + 0.010*"energy" + 0.009*"solar" + '
  '0.008*"actually" + 0.008*"power" + 0.008*"earth" + 0.008*"one" + '
  '0.007*"many" + 0.007*"day"'),
  (3,
  '0.023*"rocket" + 0.019*"launch" + 0.016*"falcon" + 0.014*"next" + '
  '0.014*"space" + 0.012*"starship" + 0.012*"dragon" + 0.010*"flight" + '
  '0.010*"true" + 0.010*"landing"'),
  (4,
  '0.024*"great" + 0.015*"thanks" + 0.014*"heart" + 0.011*"people" + '
  '0.009*"suit" + 0.008*"team" + 0.008*"work" + 0.008*"know" + 0.008*"working" '
  '+ 0.007*"like"'),
  (5,
  '0.011*"right" + 0.011*"would" + 0.008*"good" + 0.008*"ok" + '
  '0.008*"probably" + 0.006*"better" + 0.006*"one" + 0.006*"high" + '
  '0.006*"need" + 0.006*"back"')]


#---------------------------------visualize lda
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



















