
"""
"""


import os
import numpy
import pickle
import pandas as pd
from sklearn.utils import shuffle

def get_captions_dictionary (csv_path):
    df = pd.read_csv(csv_path, sep=',')
    df_0 = df.loc[:,'file_name']
    number_of_files = len(df_0)
    all_captions_dictionaries = {}
    for index_row in range(0,number_of_files):
        df_i = df.iloc[index_row]
        
        
        captions = []
        captions.append(df_i['caption_1'])
        captions.append(df_i['caption_2'])
        captions.append(df_i['caption_3'])
        captions.append(df_i['caption_4'])
        captions.append(df_i['caption_5'])
        all_captions_dictionaries [df_i['file_name']] = captions
    return all_captions_dictionaries
    
    
def load_data (filepath):
    infile = open(filepath ,'rb')
    data = pickle.load(infile)
    infile.close()
    return data
    
    
def preparX (Xdata_initial, len_of_longest_sequence):
    number_of_audios = numpy.shape(Xdata_initial)[0]
    number_of_audio_features = numpy.shape(Xdata_initial[0])[1]
    X = numpy.zeros((number_of_audios ,len_of_longest_sequence, number_of_audio_features),dtype ='float32')
    for k in numpy.arange(number_of_audios):
       item = Xdata_initial[k]
       item = item[0:len_of_longest_sequence]
       X[k,len_of_longest_sequence-len(item):, :] = item
    return X


def preparY (dict_vgg):
    Y = numpy.array(dict_vgg)    
    return Y


def prepare_XY (Ydata_initial, Xdata_initial, len_of_longest_sequence):   
    
    Ydata = preparY(Ydata_initial)
    Xdata = preparX(Xdata_initial, len_of_longest_sequence)
    return Ydata, Xdata 
