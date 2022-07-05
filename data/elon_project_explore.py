# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 14:21:44 2022

@author: alana
"""

from hw_utils import *
import pandas as pd
the_file_path = "C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/data"
the_data_out = "C:/Users/alana/OneDrive/Desktop/NLP/NLP_Elon/output_Elon/"

data_2010 = open_pickle(the_data_out, "data_2010.pkl")
data_2011 = open_pickle(the_data_out, "data_2011.pkl")
data_2012 = open_pickle(the_data_out, "data_2012.pkl")
data_2013 = open_pickle(the_data_out, "data_2013.pkl")
data_2014 = open_pickle(the_data_out, "data_2014.pkl")
data_2015 = open_pickle(the_data_out, "data_2015.pkl")
data_2016 = open_pickle(the_data_out, "data_2016.pkl")
data_2017 = open_pickle(the_data_out, "data_2017.pkl")
data_2018 = open_pickle(the_data_out, "data_2018.pkl")
data_2019 = open_pickle(the_data_out, "data_2019.pkl")
data_2020 = open_pickle(the_data_out, "data_2020.pkl")
data_2021 = open_pickle(the_data_out, "data_2021.pkl")
data_2022 = open_pickle(the_data_out, "data_2022.pkl")

frames = [data_2010, data_2011, data_2012, data_2013, data_2014, data_2015, 
          data_2016, data_2017, data_2018, data_2019, data_2020, data_2021, 
          data_2022]
data_all = pd.concat(frames)

