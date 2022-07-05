# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 23:26:19 2022

@author: alana
"""

from hw_utils import *
the_file_path = "C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data"
the_data_out = "C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/output_Elon/"

import pandas as pd
# data_2010 = pd.read_csv("C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data/raw_data/2010.csv")
# data_2011 = pd.read_csv("C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data/raw_data/2011.csv")
# data_2012 = pd.read_csv("C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data/raw_data/2012.csv")
# data_2013 = pd.read_csv("C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data/raw_data/2013.csv")
# data_2014 = pd.read_csv("C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data/raw_data/2014.csv")
# data_2015 = pd.read_csv("C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data/raw_data/2015.csv")
# data_2016 = pd.read_csv("C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data/raw_data/2016.csv")
# data_2017 = pd.read_csv("C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data/raw_data/2017.csv")
# data_2018 = pd.read_csv("C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data/raw_data/2018.csv")
# data_2019 = pd.read_csv("C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data/raw_data/2019.csv")
# data_2020 = pd.read_csv("C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data/raw_data/2020.csv")
# data_2021 = pd.read_csv("C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data/raw_data/2021.csv")
# data_2022 = pd.read_csv("C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data/raw_data/2022.csv")

# basic cleaning: 2010
# data_2010 = data_2010.dropna(axis=1)
# data_2010["tweet_cleaned"] = data_2010.tweet.apply(clean_txt)
# data_2010["tweet_cleaned_stem"] = data_2010.tweet_cleaned.apply(my_stem)
# data_2010["tweet_cleaned_stem_sw"] = data_2010.tweet_cleaned_stem.apply(my_stop_words)
# data_2010["tweet_unique_stem_sw"] = data_2010.tweet_cleaned_stem_sw.apply(word_unique_count)
# write_pickle(the_data_out, "data_2010.pkl", data_2010)
data_2010 = open_pickle(the_data_out, "data_2010.pkl")

# basic cleaning: 2011
# data_2011 = data_2011.dropna(axis=1)
# data_2011["tweet_cleaned"] = data_2011.tweet.apply(clean_txt)
# data_2011["tweet_cleaned_stem"] = data_2011.tweet_cleaned.apply(my_stem)
# data_2011["tweet_cleaned_stem_sw"] = data_2011.tweet_cleaned_stem.apply(my_stop_words)
# data_2011["tweet_unique_stem_sw"] = data_2011.tweet_cleaned_stem_sw.apply(word_unique_count)
# data_2011["year"] = pd.DatetimeIndex(data_2011["date"]).year
# data_2011 = data_2011.drop(data_2011[data_2011["year"] == 2010].index)
# write_pickle(the_data_out, "data_2011.pkl", data_2011)
data_2011 = open_pickle(the_data_out, "data_2011.pkl")

# basic cleaning: 2012
# data_2012 = data_2012.dropna(axis=1)
# data_2012["tweet_cleaned"] = data_2012.tweet.apply(clean_txt)
# data_2012["tweet_cleaned_stem"] = data_2012.tweet_cleaned.apply(my_stem)
# data_2012["tweet_cleaned_stem_sw"] = data_2012.tweet_cleaned_stem.apply(my_stop_words)
# data_2012["tweet_unique_stem_sw"] = data_2012.tweet_cleaned_stem_sw.apply(word_unique_count)
# data_2012["year"] = pd.DatetimeIndex(data_2012["date"]).year
# data_2012 = data_2012.drop(data_2012[data_2012["year"] <= 2011].index)
# write_pickle(the_data_out, "data_2012.pkl", data_2012)
data_2012 = open_pickle(the_data_out, "data_2012.pkl")

# basic cleaning: 2013
# data_2013 = data_2013.dropna(axis=1)
# data_2013["tweet_cleaned"] = data_2013.tweet.apply(clean_txt)
# data_2013["tweet_cleaned_stem"] = data_2013.tweet_cleaned.apply(my_stem)
# data_2013["tweet_cleaned_stem_sw"] = data_2013.tweet_cleaned_stem.apply(my_stop_words)
# data_2013["tweet_unique_stem_sw"] = data_2013.tweet_cleaned_stem_sw.apply(word_unique_count)
# data_2013["year"] = pd.DatetimeIndex(data_2013["date"]).year
# data_2013 = data_2013.drop(data_2013[data_2013["year"] <= 2012].index)
# write_pickle(the_data_out, "data_2013.pkl", data_2013)
data_2013 = open_pickle(the_data_out, "data_2013.pkl")


# basic cleaning: 2014
# data_2014 = data_2014.dropna(axis=1)
# data_2014["tweet_cleaned"] = data_2014.tweet.apply(clean_txt)
# data_2014["tweet_cleaned_stem"] = data_2014.tweet_cleaned.apply(my_stem)
# data_2014["tweet_cleaned_stem_sw"] = data_2014.tweet_cleaned_stem.apply(my_stop_words)
# data_2014["tweet_unique_stem_sw"] = data_2014.tweet_cleaned_stem_sw.apply(word_unique_count)
# data_2014["year"] = pd.DatetimeIndex(data_2014["date"]).year
# data_2014 = data_2014.drop(data_2014[data_2014["year"] <= 2013].index)
# write_pickle(the_data_out, "data_2014.pkl", data_2014)
data_2014 = open_pickle(the_data_out, "data_2014.pkl")


# basic cleaning: 2015
# data_2015 = data_2015.dropna(axis=1)
# data_2015["tweet_cleaned"] = data_2015.tweet.apply(clean_txt)
# data_2015["tweet_cleaned_stem"] = data_2015.tweet_cleaned.apply(my_stem)
# data_2015["tweet_cleaned_stem_sw"] = data_2015.tweet_cleaned_stem.apply(my_stop_words)
# data_2015["tweet_unique_stem_sw"] = data_2015.tweet_cleaned_stem_sw.apply(word_unique_count)
# data_2015["year"] = pd.DatetimeIndex(data_2015["date"]).year
# data_2015 = data_2015.drop(data_2015[data_2015["year"] <= 2014].index)
# write_pickle(the_data_out, "data_2015.pkl", data_2015)
data_2015 = open_pickle(the_data_out, "data_2015.pkl")


# basic cleaning: 2016
# data_2016 = data_2016.dropna(axis=1)
# data_2016["tweet_cleaned"] = data_2016.tweet.apply(clean_txt)
# data_2016["tweet_cleaned_stem"] = data_2016.tweet_cleaned.apply(my_stem)
# data_2016["tweet_cleaned_stem_sw"] = data_2016.tweet_cleaned_stem.apply(my_stop_words)
# data_2016["tweet_unique_stem_sw"] = data_2016.tweet_cleaned_stem_sw.apply(word_unique_count)
# data_2016["year"] = pd.DatetimeIndex(data_2016["date"]).year
# data_2016 = data_2016.drop(data_2016[data_2016["year"] <= 2015].index)
# write_pickle(the_data_out, "data_2016.pkl", data_2016)
data_2016 = open_pickle(the_data_out, "data_2016.pkl")


# basic cleaning: 2017
# data_2017 = data_2017.dropna(axis=1)
# data_2017["tweet_cleaned"] = data_2017.tweet.apply(clean_txt)
# data_2017["tweet_cleaned_stem"] = data_2017.tweet_cleaned.apply(my_stem)
# data_2017["tweet_cleaned_stem_sw"] = data_2017.tweet_cleaned_stem.apply(my_stop_words)
# data_2017["tweet_unique_stem_sw"] = data_2017.tweet_cleaned_stem_sw.apply(word_unique_count)
# data_2017["year"] = pd.DatetimeIndex(data_2017["date"]).year
# data_2017 = data_2017.drop(data_2017[data_2017["year"] <= 2016].index)
# write_pickle(the_data_out, "data_2017.pkl", data_2017)
data_2017 = open_pickle(the_data_out, "data_2017.pkl")


# basic cleaning: 2018
# data_2018 = data_2018.dropna(axis=1)
# data_2018["tweet_cleaned"] = data_2018.tweet.apply(clean_txt)
# data_2018["tweet_cleaned_stem"] = data_2018.tweet_cleaned.apply(my_stem)
# data_2018["tweet_cleaned_stem_sw"] = data_2018.tweet_cleaned_stem.apply(my_stop_words)
# data_2018["tweet_unique_stem_sw"] = data_2018.tweet_cleaned_stem_sw.apply(word_unique_count)
# data_2018["year"] = pd.DatetimeIndex(data_2018["date"]).year
# data_2018 = data_2018.drop(data_2018[data_2018["year"] <= 2017].index)
# write_pickle(the_data_out, "data_2018.pkl", data_2018)
data_2018 = open_pickle(the_data_out, "data_2018.pkl")


# basic cleaning: 2019
# data_2019 = data_2019.dropna(axis=1)
# data_2019["tweet_cleaned"] = data_2019.tweet.apply(clean_txt)
# data_2019["tweet_cleaned_stem"] = data_2019.tweet_cleaned.apply(my_stem)
# data_2019["tweet_cleaned_stem_sw"] = data_2019.tweet_cleaned_stem.apply(my_stop_words)
# data_2019["tweet_unique_stem_sw"] = data_2019.tweet_cleaned_stem_sw.apply(word_unique_count)
# data_2019["year"] = pd.DatetimeIndex(data_2019["date"]).year
# data_2019 = data_2019.drop(data_2019[data_2019["year"] <= 2018].index)
# write_pickle(the_data_out, "data_2019.pkl", data_2019)
data_2019 = open_pickle(the_data_out, "data_2019.pkl")


# basic cleaning: 2020
# data_2020 = data_2020.dropna(axis=1)
# data_2020["tweet_cleaned"] = data_2020.tweet.apply(clean_txt)
# data_2020["tweet_cleaned_stem"] = data_2020.tweet_cleaned.apply(my_stem)
# data_2020["tweet_cleaned_stem_sw"] = data_2020.tweet_cleaned_stem.apply(my_stop_words)
# data_2020["tweet_unique_stem_sw"] = data_2020.tweet_cleaned_stem_sw.apply(word_unique_count)
# data_2020["year"] = pd.DatetimeIndex(data_2020["date"]).year
# data_2020 = data_2020.drop(data_2020[data_2020["year"] <= 2019].index)
# write_pickle(the_data_out, "data_2020.pkl", data_2020)
data_2020 = open_pickle(the_data_out, "data_2020.pkl")


# basic cleaning: 2021
# data_2021 = data_2021.dropna(axis=1)
# data_2021["tweet_cleaned"] = data_2021.tweet.apply(clean_txt)
# data_2021["tweet_cleaned_stem"] = data_2021.tweet_cleaned.apply(my_stem)
# data_2021["tweet_cleaned_stem_sw"] = data_2021.tweet_cleaned_stem.apply(my_stop_words)
# data_2021["tweet_unique_stem_sw"] = data_2021.tweet_cleaned_stem_sw.apply(word_unique_count)
# data_2021["year"] = pd.DatetimeIndex(data_2021["date"]).year
# write_pickle(the_data_out, "data_2021.pkl", data_2021)
data_2021 = open_pickle(the_data_out, "data_2021.pkl")

# basic cleaning: 2022
# data_2022 = data_2022.dropna(axis=1)
# data_2022["tweet_cleaned"] = data_2022.tweet.apply(clean_txt)
# data_2022["tweet_cleaned_stem"] = data_2022.tweet_cleaned.apply(my_stem)
# data_2022["tweet_cleaned_stem_sw"] = data_2022.tweet_cleaned_stem.apply(my_stop_words)
# data_2022["tweet_unique_stem_sw"] = data_2022.tweet_cleaned_stem_sw.apply(word_unique_count)
# data_2022["year"] = pd.DatetimeIndex(data_2022["date"]).year
# write_pickle(the_data_out, "data_2022.pkl", data_2022)
data_2022 = open_pickle(the_data_out, "data_2022.pkl")

