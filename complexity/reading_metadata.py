import os
import pandas as pd
root = "/worktmp2/hxkhkh/current/Dcase/retrieval-relevance-crowdsourcing/data"

p_audio_meta = os.path.join(root, "audio_metadata")
p_q_cosine = os.path.join(root, "qdata_with_cosine_sims")
p_q = os.path.join(root, "query_data")
p_r = os.path.join(root, "relevance_data")
p_text = os.path.join(root, "text_data")

# later read entropy for files in "dict_unique_audio_fids" using wav files from 
# "dic_audio_file_id_to_name", and save entropies.

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
path = p_r
files = os.listdir(path)

dict_unique_audio_fids = {}
dict_unique_text_tids = {}
dict_pairs_to_ratings = {}
for f in files:
    file = os.path.join(path,f)
    df = pd.read_csv(file, sep=',') 
    print (len (df))
    for row_ind in range(len(df)):
        row = df.iloc[row_ind]
        fid_row = row['fid']
        tid_row = row['tid']
        rating_row = row['rating']
        dict_pairs_to_ratings[(tid_row, fid_row)] = rating_row
        
        if fid_row in dict_unique_audio_fids:
            dict_unique_audio_fids[fid_row] += 1
        else:
            dict_unique_audio_fids[fid_row] = 1
        
        if tid_row in dict_unique_text_tids:
            dict_unique_text_tids[tid_row] += 1
        else:
            dict_unique_text_tids[tid_row] = 1
            
#%% reading q cosine
from ast import literal_eval
path = p_q_cosine
files = os.listdir(path)
dict_pairs_to_cosines = {}
for f in files:
    file = os.path.join(path,f)
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
        print(TP_fid, cosines[TP_fid])
        print(TN_fid, cosines[TN_fid])       
        for fid in C15_fids:     
            print(fid, cosines[fid])
  
#%% reading q
path = p_q
files = os.listdir(path)

for f in files:
    file = os.path.join(path,f)
    df = pd.read_csv(file, encoding="utf-8", converters={"fid_C15": literal_eval}) 
    
    for row_ind in range(len(df)):
        row = df.iloc[row_ind]


#%% reading text
path = p_text
files = os.listdir(path)

for f in files:
    file = os.path.join(path,f)
    df = pd.read_csv(file, sep=',') 
    print (len (df))
    for row_ind in range(len(df)):
        row = df.iloc[row_ind]
kh        
#%% testing correlations
rs = []
cs = []
for pair, r in dict_pairs_to_ratings.items():
    if pair in dict_pairs_to_cosines:
        print(r)
        c = dict_pairs_to_cosines [pair]
        
        rs.append(r)
        cs.append(c)

from scipy import stats    
res = stats.pearsonr(rs,cs)