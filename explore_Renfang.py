
import pandas as pd
import os
import sys
import glob
from hw_utils import *

# use relative path 
root_path = os.path.join(os.path.dirname(__file__))
the_file_path =  f'{root_path}\\data\\'.replace("\\","/")
the_data_out = f'{root_path}\\output_Elon\\'.replace("\\","/")
path = r'./data/'


the_data = open_pickle(the_data_out, "data_cleaned.pkl")

# sentiment analysis using Vader
def vader_senti(sentence):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    s = sentiment_dict['compound'] 
    return s

the_data["vader"] = the_data.tweet_cleaned.apply(vader_senti)

write_pickle(the_data_out, "data_cleaned_senti.pkl", the_data)


# try NER with spacy  
the_data = open_pickle(the_data_out, "data_cleaned_senti.pkl") 

import spacy
nlp = spacy.load('en_core_web_sm')  

def show_ent(txt):
    # show the entity, its location, label, and the description of the label
    if txt.ents:
        for ent in txt.ents:
            print(ent.text+'-'+str(ent.start_char)+'-'+str(ent.end_char)+'-'+
                  str(ent.label_)+'-'+str(spacy.explain(ent.label_)))
    else:
        print('no named entities found')


label_list = ['CARDINAL',
 'DATE',
 'EVENT',
 'FAC',
 'GPE',
 'LANGUAGE',
 'LAW',
 'LOC',
 'MONEY',
 'NORP',
 'ORDINAL',
 'ORG',
 'PERCENT',
 'PERSON',
 'PRODUCT',
 'QUANTITY',
 'TIME',
 'WORK_OF_ART']

# show all labels and descriptions
# for i in label_list:
#     print(i+', '+spacy.explain(i))
        
def label_cnt(doc,label):   
    # for specific label, count the number of  entities
    test_txt=nlp(doc)
    org_cnt = len(tuple(ent for ent in test_txt.ents if ent.label_==label.upper()))
    return org_cnt


# for i in label_list:
#     the_data[f'{i}_cnt'] = the_data.apply(lambda x: label_cnt(x['tweet_cleaned'], i), axis=1)
    
write_pickle(the_data_out, "data_label_cnt.pkl", the_data)


# add columns: year month day hour
the_data = open_pickle(the_data_out, "data_label_cnt.pkl") 
the_data['year'] = the_data['date'].astype(str).str[0:4:1]
the_data['month'] = the_data['date'].astype(str).str[5:7:1]
the_data['day'] = the_data['date'].astype(str).str[8:10:1]
the_data['hour'] = the_data['date'].astype(str).str[11:13:1]



# vectorize
my_vec_data = my_bow(df_in=the_data.tweet_cleaned, path_in=the_data_out, gram_m=1, gram_n=1, sw="tf-idf", name_in="data_vec.pkl")

my_dim_data = my_pca(my_vec_data, the_data_out, "pca.pkl", 0.95)














