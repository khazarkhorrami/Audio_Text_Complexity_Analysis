import numpy as np
import os
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import csv
import json

#p: a measure of text complexity
root = "/worktmp2/hxkhkh/current/Dcase/retrieval-relevance-crowdsourcing/data"
p_r = os.path.join(root, "relevance_data")
p_text = os.path.join(root, "text_data")

#%%
def p_words(caption):
    # number of words
    ws = word_tokenize(caption)
    ws_counts = len(ws)
    
    # number of content words
    cws = get_content_words(caption)
    cws_counts = len(cws)
    
    # number of nouns
    nouns = detect_nouns (caption)
    nouns_counts = len(nouns)
    # number of adjs
    adjs = detect_adjectives (caption)
    adjs_counts = len(adjs)
    # number of frequent words
    count_fr_ws, count_fr_cws, count_fr_ns, count_fr_adjs = p2(caption)
    return ws_counts, cws_counts, nouns_counts, adjs_counts, count_fr_ws, count_fr_cws, count_fr_ns, count_fr_adjs

def get_content_words(caption):
    ws = word_tokenize(caption)
    ws = [w.lower() for w in ws]
    cws = detect_cws (ws)
    return cws

def detect_cws (input_words):
    tok = nltk.pos_tag(input_words, tagset='universal') 
    content_words = [n[0] for n in (tok) if n[1] =='NOUN' 
                     or n[1] =='VERB' or n[1] =='ADV' or n[1] =='ADJ'or n[1] =='NUM' or n[1] =='X']
    return content_words

def detect_nouns (caption):
    ws = word_tokenize(caption)
    ws = [w.lower() for w in ws]
    tok = nltk.pos_tag(ws, tagset='universal') 
    nouns = [n[0] for n in (tok) if n[1] =='NOUN']
    return nouns

def detect_adjectives (caption):
    ws = word_tokenize(caption)
    ws = [w.lower() for w in ws]
    tok = nltk.pos_tag(ws, tagset='universal') 
    adjs = [n[0] for n in (tok) if n[1] =='ADV' or n[1] =='ADJ' ]
    return adjs
#%%
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
#%%   by a simple comparison the correlation results for all clotho looks better         
path = p_text
files = os.listdir(path)

# Note: this includes all clotho caption data

dict_tid_to_caption = {}
for f in files:
    file = os.path.join(path,f)
    df = pd.read_csv(file, sep=',') 
    print (len (df))
    for row_ind in range(len(df)):
        row = df.iloc[row_ind]
        row_tid = row['tid']
        row_caption = row['text']
        dict_tid_to_caption [row_tid] = row_caption
        
# Note: this includes only 600 clotho captions used in crowdsourcing

# dict_tid_to_caption = {}
# for f in files:
#     file = os.path.join(path,f)
#     df = pd.read_csv(file, sep=',') 
#     print (len (df))
#     for row_ind in range(len(df)):
#         row = df.iloc[row_ind]
#         row_tid = row['tid']
#         row_caption = row['text']
#         if row_tid in dict_unique_text_tids:
#             dict_tid_to_caption [row_tid] = row_caption

#%%
dict_word_counts = {}
for tid, caption in dict_tid_to_caption.items():
    ws = word_tokenize(caption)
    for item in ws:
        if item in dict_word_counts:
            dict_word_counts[item] +=1
        else:
            dict_word_counts[item] = 1
            
word_counts_sorted = sorted(dict_word_counts.items(), key=lambda x:x[1], reverse=True )

dict_cword_counts = {}
for tid, caption in dict_tid_to_caption.items():
    cws = get_content_words (caption)
    for item in cws:
        if item in dict_cword_counts:
            dict_cword_counts[item] +=1
        else:
            dict_cword_counts[item] = 1
            
cword_counts_sorted = sorted(dict_cword_counts.items(), key=lambda x:x[1], reverse=True )


dict_noun_counts = {}
for tid, caption in dict_tid_to_caption.items():
    ns = detect_nouns (caption)
    for item in ns:
        if item in dict_noun_counts:
            dict_noun_counts[item] +=1
        else:
            dict_noun_counts[item] = 1
            
noun_counts_sorted = sorted(dict_noun_counts.items(), key=lambda x:x[1], reverse=True )


dict_adj_counts = {}
for tid, caption in dict_tid_to_caption.items():
    adjs = detect_adjectives(caption)
    for item in adjs:
        if item in dict_adj_counts:
            dict_adj_counts[item] +=1
        else:
            dict_adj_counts[item] = 1
            
adj_counts_sorted = sorted(dict_adj_counts.items(), key=lambda x:x[1], reverse=True )

# select_most_frequent_cws
m = 500
freq_words_and_counts =  word_counts_sorted [0:m]
freq_cwords_and_counts =  cword_counts_sorted [0:m]
freq_nouns_and_counts =  noun_counts_sorted [0:m]
freq_adjs_and_counts =  adj_counts_sorted [0:m]

dict_frequent_words = {}
for item in freq_words_and_counts:
    dict_frequent_words[item[0]] = item[1] 
    
dict_frequent_cwords = {}
for item in freq_cwords_and_counts:
    dict_frequent_cwords[item[0]] = item[1]  

dict_frequents_nouns = {}
for item in freq_nouns_and_counts:
    dict_frequents_nouns[item[0]] = item[1]

dict_frequents_adjs = {}
for item in freq_nouns_and_counts:
    dict_frequents_adjs[item[0]] = item[1]
  
def p2(caption):
    cws = get_content_words(caption)
    count_fr_ws = 0
    count_fr_cws = 0
    count_fr_ns = 0
    count_fr_adjs = 0
    for w in cws:
        if w in dict_frequent_words:
            count_fr_ws +=1
        if w in dict_frequent_cwords:
            count_fr_cws +=1
        if w in dict_frequents_nouns:
            count_fr_ns +=1
        if w in dict_frequents_adjs:
            count_fr_adjs +=1
    return count_fr_ws, count_fr_cws, count_fr_ns, count_fr_adjs
#%%
dict_tid_to_p = {}
for tid in dict_unique_text_tids:
    caption = dict_tid_to_caption [tid]
    ws_counts, cws_counts, nouns_counts, adjs_counts, count_fr_ws,count_fr_cws, count_fr_ns, count_fr_adjs = p_words(caption)
    dict_tid_to_p [tid] = {}
    dict_tid_to_p [tid]['ws_counts'] = ws_counts
    dict_tid_to_p [tid]['cws_counts'] = cws_counts
    dict_tid_to_p [tid]['nouns_counts'] = nouns_counts
    dict_tid_to_p [tid]['adjs_counts'] = adjs_counts
    dict_tid_to_p [tid]['count_fr_ws'] = count_fr_ws
    dict_tid_to_p [tid]['count_fr_cws'] = count_fr_cws
    dict_tid_to_p [tid]['count_fr_ns'] = count_fr_ns
    dict_tid_to_p [tid]['count_fr_adjs'] = count_fr_adjs
    
#%%

file_json = "/worktmp2/hxkhkh/current/Dcase/data/entropy/Tentropy.json"
with open(file_json, "w") as fp:
    json.dump(dict_tid_to_p, fp)
