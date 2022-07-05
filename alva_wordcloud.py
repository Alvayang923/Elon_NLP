# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 21:50:25 2022

@author: alva
"""
import pandas as pd
import os
import sys
import glob
from hw_utils import *


# use relative path 
root_path = os.path.join(os.path.dirname(__file__))
the_file_path =  f'{root_path}\\data\\'.replace("\\","/")
the_data_out = f'{root_path}\\output_Elon\\'.replace("\\","/")


the_data = open_pickle(the_data_out, "data_cleaned.pkl")

# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(the_data['tweet_cleaned'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()