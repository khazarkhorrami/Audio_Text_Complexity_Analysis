import numpy as np
import os
import pandas as pd
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import csv
import json

#%%
audio_path = '/worktmp2/hxkhkh/current/Dcase/data/clotho/audio/' 
root = "/worktmp2/hxkhkh/current/Dcase/retrieval-relevance-crowdsourcing/data"
p_audio_meta = os.path.join(root, "audio_metadata")
p_r = os.path.join(root, "relevance_data")
#%%
device = 'cuda' 
at = AudioTagging(checkpoint_path=None, device=device)
sed = SoundEventDetection(
    checkpoint_path=None, 
    device=device, 
    interpolate_mode='nearest')

#%%

def calculate_entropy (framewise_output):
    
    n_classes = np.size(framewise_output,1)
    n_timesteps = np.size(framewise_output,0)
    
    # Create class activations (0-1, random multiclass)
    # P = np.random.rand(n_timesteps, n_classes)
    P = framewise_output
    
    # Example 1: Entropy of classes
    # Get distribution of classes by averaging across time
    y = np.mean(P, axis=0)
    p = y / np.sum(y)  # normalize to distribution proper
    
    # Calculate entropy -sum(p*log*p))  (ignore zero classes)
    nonzero = np.where(p > 0)[0]
    H_classes = -np.sum(p[nonzero] * np.log2(p[nonzero]))
    
    # Normalize with the amount of classes (H becomes 0-1)
    H_classes = H_classes / np.log2(n_classes)
    
    # Example 2: Entropy in time
    # Get distribution of classes by averaging across classes
    y = np.mean(P, axis=1)
    p = y / np.sum(y)
    
    # Calculate entropy -sum(p*log*p))  (ignore zero classes)
    nonzero = np.where(p > 0)[0]
    H_time = -np.sum(p[nonzero] * np.log2(p[nonzero]))
    
    # Normalize with the amount of timesteps (H becomes 0-1)
    H_time = H_time / np.log2(n_timesteps)
    
    
    return H_classes, H_time

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
for split in splits:
    metafile = split + suf
    file = os.path.join(path,metafile)
    df = pd.read_csv(file, sep=',') 
    print (len (df))
    for row_ind in range(len(df)):
        row = df.iloc[row_ind]
        fid_row = row['fid']
        tid_row = row['tid']
        rating_row = row['rating']
        dict_pairs_to_ratings[(tid_row, fid_row)] = rating_row
        
        if fid_row in dict_unique_audio_fids_counts:
            dict_unique_audio_fids_counts[fid_row] += 1
        else:
            dict_unique_audio_fids_counts[fid_row] = 1
            dict_unique_audio_fids_split[fid_row] = split
        
        if tid_row in dict_unique_text_tids:
            dict_unique_text_tids[tid_row] += 1
        else:
            dict_unique_text_tids[tid_row] = 1
#%%
dict_fid_to_h = {}
for afid, split in dict_unique_audio_fids_split.items():
    aname = dic_audio_file_id_to_name [afid]
  
    af = os.path.join(audio_path, split, aname)
    (audio, _) = librosa.core.load(af, sr=32000, mono=True)
    audio = audio[None, :]  # (batch_size, segment_samples)
    try:
        framewise_output = sed.inference(audio)
        """(batch_size, time_steps, classes_num)"""    
        fo = framewise_output[0]
        h_classes, h_time = calculate_entropy (fo)
        dict_fid_to_h [afid] = {}
        dict_fid_to_h [afid]['h_class'] = h_classes
        dict_fid_to_h [afid]['h_time'] = h_time
    except: 
        pass

#%%

file_json = "/worktmp2/hxkhkh/current/Dcase/data/entropy/Aentropy.json"
with open(file_json, "w") as fp:
    json.dump(dict_fid_to_h, fp)

# with open(file_json, "r") as fp:
#     dtest = json.load(fp)
    