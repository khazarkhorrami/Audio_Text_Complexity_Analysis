
"""
"""

import numpy
import pandas as pd


csv_path = '/worktmp2/hxkhkh/current/Dcase/data/clotho/captions/clotho_captions_validation.csv'

    
def get_all_captions (csv_path):
    df = pd.read_csv(csv_path, sep=',')
    df_0 = df.loc[:,'file_name']
    all_filenames = df_0.tolist()

    all_captions = []
    for index_col in range(1,6,1):
        df_i = df.iloc[:,index_col]
        all_captions.extend(df_i.tolist())
    
    return all_filenames, all_captions

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

def get_metadata (csv_path):
    pass


all_captions_dictionaries = get_captions_dictionary (csv_path)