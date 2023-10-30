import numpy as np
import os
import pandas as pd
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import csv
import json
from scipy import stats
#%%
audio_path = '/worktmp2/hxkhkh/current/Dcase/data/clotho/audio/' 
root = "/worktmp2/hxkhkh/current/Dcase/retrieval-relevance-crowdsourcing/data"
p_audio_meta = os.path.join(root, "audio_metadata")
p_r = os.path.join(root, "relevance_data")
p_q_cosine = os.path.join(root, "qdata_with_cosine_sims")

file_json_Aentropy = "/worktmp2/hxkhkh/current/Dcase/data/entropy/Aentropy.json"
file_json_Tentropy = "/worktmp2/hxkhkh/current/Dcase/data/entropy/Tentropy.json"
with open(file_json_Aentropy, "r") as fp:
    dict_fid_to_h = json.load(fp)
with open(file_json_Tentropy, "r") as fp:
    dict_tid_to_p = json.load(fp)
#%% reading audio_meta
path = p_audio_meta
files = os.listdir(path)

data_audio = {}
data_audio ['all'] = []

dic_audio_file_id_to_name = {}

for f in files:
    file = os.path.join(path,f)
    df = pd.read_csv(file, sep=',') 
    print (len (df))
    for row_ind in range(len(df)):
        row = df.iloc[row_ind]
        dic_audio_file_id_to_name [row['fid']] = row['file_name']
        
    split = df.loc[:,'split'][0]
    fID = df.loc[:,'fid']
    fID_list = fID.tolist()
    fname = df.loc[:,'file_name']
    fname_list = fname.tolist()
    
    data_audio ['all'].extend(fID_list) 
    data_audio [split] = fID_list

#%% reading ratings
splits = ['development', 'evaluation', 'validation']
suf = '_relevances.csv'
path = p_r

dict_unique_audio_fids_counts = {}
dict_unique_audio_fids_split = {}
dict_unique_text_tids = {}
dict_pairs_to_ratings = {}
dict_pairs_to_ratings_TP = {}
dict_pairs_to_ratings_TN = {}
dict_pairs_to_ratings_C15 = {}
for split in splits:
    metafile = split + suf
    file = os.path.join(path,metafile)
    df = pd.read_csv(file, sep=',') 
    print (len (df))
    for row_ind in range(len(df)):
        row = df.iloc[row_ind]
        fid_row = row['fid']
        tid_row = row['tid']
        g_row = row['fid_group']
        print(g_row)
        rating_row = row['rating']
        dict_pairs_to_ratings[(tid_row, fid_row)] = rating_row
        if g_row == 'TP':
            dict_pairs_to_ratings_TP[(tid_row, fid_row)] = rating_row
        elif g_row == 'TN':
            dict_pairs_to_ratings_TN[(tid_row, fid_row)] = rating_row
        elif g_row == 'C15':
            dict_pairs_to_ratings_C15[(tid_row, fid_row)] = rating_row
        # preparing unique lists    
        if fid_row in dict_unique_audio_fids_counts:
            dict_unique_audio_fids_counts[fid_row] += 1
        else:
            dict_unique_audio_fids_counts[fid_row] = 1
            dict_unique_audio_fids_split[fid_row] = split
        
        if tid_row in dict_unique_text_tids:
            dict_unique_text_tids[tid_row] += 1
        else:
            dict_unique_text_tids[tid_row] = 1
#%% reading q cosine
from ast import literal_eval
splits = ['development', 'evaluation', 'validation']
suf = '_queries.csv'
path = p_q_cosine

dict_pairs_to_cosines = {}
dict_pairs_to_cosines_TP = {}
dict_pairs_to_cosines_TN = {}
dict_pairs_to_cosines_C15 = {}
for split in splits:
    metafile = split + suf
    file = os.path.join(path,metafile)
    df = pd.read_csv(file, encoding="utf-8", converters={"fid_C15": literal_eval, "fid_scores": literal_eval}) 
    
    for row_ind in range(len(df)):
        row = df.iloc[row_ind]
        #scores = row.iloc[4]
        tid = row['tid']
        TP_fid = row["fid_TP"]
        TN_fid = row["fid_TN"]
        C15_fids = row["fid_C15"]
        cosines = row["fid_scores"]
        
        for k,v in cosines.items():
            dict_pairs_to_cosines[(tid, k)] = v
            if k == TP_fid:
                dict_pairs_to_cosines_TP[(tid, k)] = v
            elif k == TN_fid:
                dict_pairs_to_cosines_TN[(tid, k)] = v
            else:
                dict_pairs_to_cosines_C15[(tid, k)] = v
                
        print(TP_fid, cosines[TP_fid])
        print(TN_fid, cosines[TN_fid])       
        for fid in C15_fids:     
            print(fid, cosines[fid])

#%% AUDIO COMPLEXITY CORRELATIONS


    
# mapping entropy to human ratings    
h_all = []
r_all = []
for afid, vdict in dict_fid_to_h.items():
    h_classes = dict_fid_to_h [afid]['h_class']
    h_time = dict_fid_to_h [afid]['h_time']
    
    for p , r in dict_pairs_to_ratings_TP.items():
        (tid, fid) = p
        if afid == fid:
            h_all.append(h_classes)
            r_all.append(r)

# correlations   
corr_h_rat = stats.pearsonr(h_all, r_all)

# mapping entropy to cosine distances
    
h_all = []
cos_all = []
for afid, vdict in dict_fid_to_h.items():
    h_classes = dict_fid_to_h [afid]['h_class']
    h_time = dict_fid_to_h [afid]['h_time']
    
    for p , cos in dict_pairs_to_cosines_TP.items():
        (tid, fid) = p
        if afid == fid:
            h_all.append(h_classes)
            cos_all.append(cos)

# correlations    
corr_h_cos = stats.pearsonr(h_all, cos_all)

# printing the results
print( "###### Audio correlatons ######")
print(corr_h_rat)
print(corr_h_cos)

#%% TEXT COMPLEXITY CORRELATIONS



# mapping entropy to human ratings    
p_all = []
r_all = []
for tid_measure, vp in dict_tid_to_p.items():
    ws_counts = dict_tid_to_p [tid_measure]['ws_counts']
    cws_counts = dict_tid_to_p [tid_measure]['cws_counts']
    nouns_counts = dict_tid_to_p [tid_measure]['nouns_counts']
    adjs_counts = dict_tid_to_p [tid_measure]['adjs_counts']
    count_fr_ws = dict_tid_to_p [tid_measure]['count_fr_ws']
    count_fr_ns = dict_tid_to_p [tid_measure]['count_fr_ns']
    p_measure = ws_counts
    for p , r in dict_pairs_to_ratings_TP.items():
        (tid, fid) = p
        if tid_measure == tid:
            p_all.append(p_measure)
            r_all.append(r)

# correlations   
corr_p_rat = stats.pearsonr(p_all, r_all)

# mapping entropy to cosine distances  
p_all = []
cos_all = []
for tid_measure, vp in dict_tid_to_p.items():
    ws_counts = dict_tid_to_p [tid_measure]['ws_counts']
    cws_counts = dict_tid_to_p [tid_measure]['cws_counts']
    nouns_counts = dict_tid_to_p [tid_measure]['nouns_counts']
    adjs_counts = dict_tid_to_p [tid_measure]['adjs_counts']
    count_fr_ws = dict_tid_to_p [tid_measure]['count_fr_ws']
    count_fr_ns = dict_tid_to_p [tid_measure]['count_fr_ns']
    p_measure = ws_counts   
    for p , cos in dict_pairs_to_cosines_TP.items():
        (tid, fid) = p
        if tid_measure == tid:
            p_all.append(p_measure)
            cos_all.append(cos)

# correlations   
corr_p_cos = stats.pearsonr(p_all, cos_all)
print( "###### Text correlatons ######")
print(corr_p_rat)
print(corr_p_cos)

#%% set X, Y for linear regressor
  
h_all = []
p_all = []
r_all = []
for item_pair, item_rate in dict_pairs_to_ratings.items():
    (tid, fid) = item_pair
    print(fid in dict_fid_to_h)
    if fid in dict_fid_to_h:
        h_classes = dict_fid_to_h [fid]['h_class']
        ws_counts = dict_tid_to_p [tid]['ws_counts']
        h = h_classes
        p = ws_counts
        r = item_rate
        h_all.append(h)
        p_all.append(p)
        r_all.append(r)

x = np.array([h_all, p_all])
y = np.array([r_all])
X=x.T
Y=y.T

#%%
#predict the human rating (Y) based on a text and audio complexity index (X1,X2)
from sklearn import linear_model 

regr = linear_model.LinearRegression()
regr.fit(X,Y)
print(regr.coef_)
print(regr.intercept_)

# example point
X_ex = np.array([[0.9, 24]])
r_predicted = regr.predict(X)
print(r_predicted)