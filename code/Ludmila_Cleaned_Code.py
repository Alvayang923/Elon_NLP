# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 20:35:55 2022

@author: alana
"""

from hw_utils import *
import glob
import os
import pandas as pd
import re
from tqdm.auto import tqdm
import nltk
global dictionary
dictionary = nltk.corpus.words.words("en")
dictionary = [word.lower() for word in dictionary]

path = "C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data/raw_data/"
the_data_out = "C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/output_Elon/"


all_files = glob.glob(os.path.join(path, "*.csv"))

df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df   = pd.concat(df_from_each_file, ignore_index=False)

the_data_col = concatenated_df[['date', 'timezone', 'hashtags', 'mentions', 'replies_count', 'retweets_count', 'likes_count', 'tweet']]

the_data = the_data_col.fillna(0).replace('[]',0)[['date', 'timezone', 'hashtags', 'mentions', 'replies_count', 'retweets_count', 'likes_count', 'tweet']]


the_data["hashtags_own"] = the_data.tweet.str.findall(r'#.*?(?=\s|$)')
the_data["mentions_own"] = the_data.tweet.str.findall(r'@.*?(?=\s|$)')

the_data['tweet'] = the_data['tweet'].str.replace('[!@#$]','')

try:
    import cPickle as pickle
except ImportError: 
    import pickle
import re


with open('C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/Emoji_Dict.p', 'rb') as fp:
    Emoji_Dict = pickle.load(fp)
Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}

def convert_emojis_to_word(text):
    for emot in Emoji_Dict:
        text = re.sub(r'('+emot+')', " ".join(Emoji_Dict[emot].replace(",","").replace(":","").split()), text)
    return text

the_data["tweet_cleaned"] = the_data.apply(lambda x: convert_emojis_to_word(x["tweet"]), axis = 1)

the_data["tweet_cleaned"] = the_data.apply(lambda x: pre_process_text_new(x["tweet_cleaned"]), axis = 1)

the_data["tweet_cleaned"] = the_data.apply(lambda x: my_stop_words(x["tweet_cleaned"]), axis = 1)

write_pickle(the_data_out, "data_cleaned.pkl", the_data)
