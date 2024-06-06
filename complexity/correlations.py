from sklearn.linear_model import Ridge
import numpy as np
import os
import pandas as pd
import json
from scipy import stats
from sklearn import linear_model
from matplotlib import pyplot as plt
from ast import literal_eval
# %%
audio_path = '/worktmp2/hxkhkh/current/Dcase/data/clotho/audio/'
root = "/worktmp2/hxkhkh/current/Dcase/retrieval-relevance-crowdsourcing/data"
p_audio_meta = os.path.join(root, "audio_metadata")
p_r = os.path.join(root, "relevance_data")
p_q_cosine = os.path.join(root, "qdata_with_cosine_sims")

p_rall = '/worktmp2/hxkhkh/current/Dcase/data/crowdsourced_rated_pairs.csv'


file_json_Aentropy = "/worktmp2/hxkhkh/current/Dcase/data/entropy/Aentropy.json"
file_json_Adur = "/worktmp2/hxkhkh/current/Dcase/data/entropy/Adur.json"
file_json_Tentropy = "/worktmp2/hxkhkh/current/Dcase/data/entropy/Tentropy.json"
with open(file_json_Aentropy, "r") as fp:
    dict_fid_to_h = json.load(fp)
with open(file_json_Adur, "r") as fp:
    dict_fid_to_dur = json.load(fp)
with open(file_json_Tentropy, "r") as fp:
    dict_tid_to_p = json.load(fp)

# %% reading audio_meta
path = p_audio_meta
files = os.listdir(path)

data_audio = {}
data_audio['all'] = []

dic_audio_file_id_to_name = {}

for f in files:
    file = os.path.join(path, f)
    df = pd.read_csv(file, sep=',')
    for row_ind in range(len(df)):
        row = df.iloc[row_ind]
        dic_audio_file_id_to_name[row['fid']] = row['file_name']

    split = df.loc[:, 'split'][0]
    fID = df.loc[:, 'fid']
    fID_list = fID.tolist()
    fname = df.loc[:, 'file_name']
    fname_list = fname.tolist()

    data_audio['all'].extend(fID_list)
    data_audio[split] = fID_list

# %%
# reading all ratings:
# df = pd.read_csv(p_rall, encoding="utf-8", converters={"details": literal_eval})
# dict_pairs_to_rall = {}
# dict_pairs_to_rall_TP = {}
# dict_pairs_to_rall_TN = {}
# dict_pairs_to_rall_C15 = {}

# for row_ind in range(len(df)):
#     row = df.iloc[row_ind]
#     fid_row = row['fid']
#     tid_row = row['tid']
#     g_row = row['group']
#     rating_row = row['rating']
#     rall = np.std (row['details'])

#     dict_pairs_to_rall[(tid_row, fid_row)] = rall
#     if g_row == 'TP':
#         dict_pairs_to_rall_TP[(tid_row, fid_row)] = rall
#     elif g_row == 'TN':
#         dict_pairs_to_rall_TN[(tid_row, fid_row)] = rall
#     elif g_row == 'C15':
#         dict_pairs_to_rall_C15[(tid_row, fid_row)] = rall

# %% reading ratings
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
    file = os.path.join(path, metafile)
    df = pd.read_csv(file, sep=',')
    for row_ind in range(len(df)):
        row = df.iloc[row_ind]
        fid_row = row['fid']
        tid_row = row['tid']
        g_row = row['fid_group']
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

# %%
# reading played times (played) as well as all different ratings for each pair (details).
p_rtime = "/worktmp2/hxkhkh/current/Dcase/data/crowdsourced_relevance_ratings.csv"
df = pd.read_csv(p_rtime, encoding="utf-8",
                 converters={"details": literal_eval, "played": literal_eval})

dict_pairs_to_tall = {}
dict_pairs_to_tall_TP = {}  # mean played time
dict_pairs_to_tall_TN = {}
dict_pairs_to_tall_C15 = {}

dict_pairs_to_rall = {}
dict_pairs_to_rall_TP = {}
dict_pairs_to_rall_TN = {}
dict_pairs_to_rall_C15 = {}

t_tp = []
r_tp = []
for row_ind in range(len(df)):
    row = df.iloc[row_ind]
    fid_row = row['fid']
    tid_row = row['tid']
    g_row = row['group']
    rating_row = row['rating']
    # print(len(row['details']))
    rall = np.std(row['details'])  # DIFFERENCE BETWEEN RATINGS
    tall = np.array(row['played'])
    dur = dict_fid_to_dur[fid_row]/10
    # discarding outliers
    tmean = np.mean(tall)  # np.mean(tall* (tall < 30.0))
    tstd = np.std(tall)  # np.std(tall* (tall < 30.0))
    r_tp.extend(row['details'])
    t_tp.extend((dur - tall)/dur)

    dict_pairs_to_tall[(tid_row, fid_row)] = tmean
    if g_row == 'TP':
        dict_pairs_to_tall_TP[(tid_row, fid_row)] = tmean

    elif g_row == 'TN':
        dict_pairs_to_tall_TN[(tid_row, fid_row)] = tmean
        check = 1 * (np.array(tall) == 30)
        if sum(check) >= 1:
            print(sum(check))
            rat = rating_row
            r = row['details']
            t = tall

    elif g_row == 'C15':
        dict_pairs_to_tall_C15[(tid_row, fid_row)] = tmean

    dict_pairs_to_rall[(tid_row, fid_row)] = rall
    if g_row == 'TP':
        dict_pairs_to_rall_TP[(tid_row, fid_row)] = rall
    elif g_row == 'TN':
        dict_pairs_to_rall_TN[(tid_row, fid_row)] = rall
    elif g_row == 'C15':
        dict_pairs_to_rall_C15[(tid_row, fid_row)] = rall

# np.mean(m_TP)
# np.std(m_TP)
# np.mean(std_TP)
# np.std(std_TP)
corr_ = stats.pearsonr(t_tp, r_tp)
print(corr_)
# %% reading q cosine (machine rating)
splits = ['development', 'evaluation', 'validation']
suf = '_queries.csv'
path = p_q_cosine

dict_pairs_to_cosines = {}
dict_pairs_to_cosines_TP = {}
dict_pairs_to_cosines_TN = {}
dict_pairs_to_cosines_C15 = {}
for split in splits:
    metafile = split + suf
    file = os.path.join(path, metafile)
    df = pd.read_csv(file, encoding="utf-8",
                     converters={"fid_C15": literal_eval, "fid_scores": literal_eval})

    for row_ind in range(len(df)):
        row = df.iloc[row_ind]
        #scores = row.iloc[4]
        tid = row['tid']
        TP_fid = row["fid_TP"]
        TN_fid = row["fid_TN"]
        C15_fids = row["fid_C15"]
        cosines = row["fid_scores"]

        for k, v in cosines.items():
            dict_pairs_to_cosines[(tid, k)] = v
            if k == TP_fid:
                dict_pairs_to_cosines_TP[(tid, k)] = v
            elif k == TN_fid:
                dict_pairs_to_cosines_TN[(tid, k)] = v
            else:
                dict_pairs_to_cosines_C15[(tid, k)] = v

        # print(TP_fid, cosines[TP_fid])
        # print(TN_fid, cosines[TN_fid])
        # for fid in C15_fids:
        #     print(fid, cosines[fid])
# %%
# ........................................... mapping mpt to human ratings

mpt_all = [] # mean played time
dur_all = []
rat_all = []
rat_std = []
rat_machine = []
rat_HM_diff = []
for pair, rat in dict_pairs_to_rall_TP.items():
    (tid, fid) = pair
    rat_std.append(rat)
    mpt_all.append(dict_pairs_to_tall_TP[pair])
    dur_all.append(dict_fid_to_dur[fid])
    rat_all.append(dict_pairs_to_ratings_TP[pair])
    rat_machine.append(dict_pairs_to_cosines_TP[pair])
    rat_HM_diff.append(
        np.abs(dict_pairs_to_cosines_TP[pair] - 0.01 * dict_pairs_to_ratings_TP[pair]))

# correlations
corr_mpt_rat = stats.pearsonr(mpt_all, rat_all)
print("###### correlatons mpt with human ratings ######")
print('(', round(corr_mpt_rat[0], 3), ',', round(corr_mpt_rat[1], 3), ')')

corr_mpt_std_rat = stats.pearsonr(mpt_all, rat_std)
print("###### correlatons mpt with std human ratings ######")
print('(', round(corr_mpt_std_rat[0], 3),
      ',', round(corr_mpt_std_rat[1], 3), ')')

corr_mpt_Mrat = stats.pearsonr(mpt_all, rat_machine)
print("###### correlatons mpt with machine ratings ######")
print('(', round(corr_mpt_Mrat[0], 3), ',', round(corr_mpt_Mrat[1], 3), ')')

corr_mpt_HM = stats.pearsonr(mpt_all, rat_HM_diff)
print("###### correlatons mpt with machine-human ratings ######")
print('(', round(corr_mpt_HM[0], 3), ',', round(corr_mpt_HM[1], 3), ')')

# .............................................................................

corr_dur_rat = stats.pearsonr(dur_all, rat_all)
print("###### correlatons audio duration with human ratings ######")
print('(', round(corr_dur_rat[0], 3), ',', round(corr_dur_rat[1], 3), ')')

corr_dur_std_rat = stats.pearsonr(dur_all, rat_std)
print("###### correlatons audio duration with std human ratings ######")
print('(', round(corr_dur_std_rat[0], 3),
      ',', round(corr_dur_std_rat[1], 3), ')')

corr_dur_Mrat = stats.pearsonr(dur_all, rat_machine)
print("###### correlatons audio duration with machine ratings ######")
print('(', round(corr_dur_Mrat[0], 3), ',', round(corr_dur_Mrat[1], 3), ')')

corr_dur_HM = stats.pearsonr(dur_all, rat_HM_diff)
print("###### correlatons audio duration with machine-human ratings ######")
print('(', round(corr_dur_HM[0], 3), ',', round(corr_dur_HM[1], 3), ')')

# .............................................................................
corr_dur_mpt = stats.pearsonr(dur_all, mpt_all)
print("###### correlatons mpt with audio duration ######")
print('(', round(corr_dur_mpt[0], 3), ',', round(corr_dur_mpt[1], 3), ')')

# ............................................................................. plotting
# pname = "mpt_r"
# x = mpt_all
# y = rat_all
# corr_ = stats.pearsonr(x,y)
# plt.figure(figsize =(8, 8))
# plt.scatter(x, y, label =  str(round(corr_[0],3) ) + ', p < 0.01' )
# par = np.polyfit(x, y, 1, full=True)
# slope=par[0][0]
# intercept=par[0][1]
# plt.xlabel('Mean played time', fontsize = 20)
# plt.ylabel('Human rating', fontsize = 20)
# xl = [min(x), max(x)]
# yl = [slope*xx + intercept  for xx in xl]
# plt.plot(xl, yl, c ='r')
# plt.grid()
# plt.legend(fontsize = 18)
# plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/plots/' + pname + '.png' ,  format = 'png' , bbox_inches='tight')

# pname = "mpt_rall"
# x = mpt_all
# y = rat_std
# corr_ = stats.pearsonr(x,y)
# plt.figure(figsize =(8, 8))
# plt.scatter(x, y, label =  str(round(corr_[0],3) ) + ', p < 0.01' )
# par = np.polyfit(x, y, 1, full=True)
# slope=par[0][0]
# intercept=par[0][1]
# plt.xlabel('Mean played time', fontsize = 20)
# plt.ylabel('Difference between human ratings', fontsize = 20)
# xl = [min(x), max(x)]
# yl = [slope*xx + intercept  for xx in xl]
# plt.plot(xl, yl, c ='r')
# plt.grid()
# plt.legend(fontsize = 18)
# plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/plots/' + pname + '.png' ,  format = 'png' , bbox_inches='tight')
# %% AUDIO COMPLEXITY CORRELATIONS
kh
# ........................................... mapping Audio entropy to human ratings
h_all = []
r_all = []
for afid, vdict in dict_fid_to_h.items():
    h_classes = dict_fid_to_h[afid]['h_class']
    h_time = dict_fid_to_h[afid]['h_time']
    h_dur = dict_fid_to_dur[afid]
    h = h_classes
    for p, r in dict_pairs_to_ratings_TP.items():
        (tid, fid) = p
        if afid == fid:
            h_all.append(h)
            r_all.append(r)

# correlations
corr_h_rat = stats.pearsonr(h_all, r_all)
print("###### Audio correlatons with human ratings ######")
print('(', round(corr_h_rat[0], 5), ',', round(corr_h_rat[1], 3), ')')

# ........................................... plotting

# pname = "hc_r"
# x = h_all
# y = r_all
# corr_ = stats.pearsonr(x,y)
# plt.figure(figsize =(8, 8))
# plt.scatter(x, y, label =  str(round(corr_[0],3) ) + ', p < 0.01' )
# par = np.polyfit(x, y, 1, full=True)
# slope=par[0][0]
# intercept=par[0][1]
# plt.xlabel('Entropy over classes', fontsize = 20)
# plt.ylabel('Human rating', fontsize = 20)
# xl = [min(x), max(x)]
# yl = [slope*xx + intercept  for xx in xl]
# plt.plot(xl, yl, c ='r')
# plt.grid()
# plt.legend(fontsize = 18)
# plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/plots/' + pname + '.png' ,  format = 'png' , bbox_inches='tight')

# ........................................... mapping Audio entropy to std human ratings
h_all = []
r_rall = []
for afid, vdict in dict_fid_to_h.items():
    h_classes = dict_fid_to_h[afid]['h_class']
    h_time = dict_fid_to_h[afid]['h_time']
    h_dur = dict_fid_to_dur[afid]
    h = h_classes
    for p, r in dict_pairs_to_rall_TP.items():
        (tid, fid) = p
        if afid == fid:
            h_all.append(h)
            r_rall.append(r)

# correlations
corr_h_rall = stats.pearsonr(h_all, r_rall)
print("###### Audio correlatons with human ratings ######")
print('(', round(corr_h_rat[0], 5), ',', round(corr_h_rat[1], 3), ')')

# ........................................... plotting

# pname = "hc_rall"
# x = h_all
# y = r_rall
# corr_ = stats.pearsonr(x,y)
# plt.figure(figsize =(8, 8))
# plt.scatter(x, y, label =  str(round(corr_[0],3) ) + ', p < 0.01' )
# par = np.polyfit(x, y, 1, full=True)
# slope=par[0][0]
# intercept=par[0][1]
# plt.xlabel('Entropy over classes', fontsize = 20)
# plt.ylabel('Difference between human ratings', fontsize = 20)
# xl = [min(x), max(x)]
# yl = [slope*xx + intercept  for xx in xl]
# plt.plot(xl, yl, c ='r')
# plt.grid()
# plt.legend(fontsize = 18)
# plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/plots/' + pname + '.png' ,  format = 'png' , bbox_inches='tight')

# ........................................... mapping A entropy to machine ratings
h_all = []
cos_all = []
for afid, vdict in dict_fid_to_h.items():
    h_classes = dict_fid_to_h[afid]['h_class']
    h_time = dict_fid_to_h[afid]['h_time']
    h_dur = dict_fid_to_dur[afid]
    h = h_classes
    for p, cos in dict_pairs_to_cosines_TP.items():
        (tid, fid) = p
        if afid == fid:
            h_all.append(h)
            cos_all.append(cos)

# correlations
corr_h_cos = stats.pearsonr(h_all, cos_all)

print("###### Audio correlatons with machine ratings ######")
print('(', round(corr_h_cos[0], 5), ', ', round(corr_h_cos[1], 3), ')')

# ........................................... plotting

# pname = "hc_mr"
# x = h_all
# y = cos_all
# corr_ = stats.pearsonr(x,y)
# plt.figure(figsize =(8, 8))
# plt.scatter(x, y, label =  str(round(corr_[0],3) ) + ', p < 0.01' )
# par = np.polyfit(x, y, 1, full=True)
# slope=par[0][0]
# intercept=par[0][1]
# plt.xlabel('Entropy over classes', fontsize = 20)
# plt.ylabel('Machine rating', fontsize = 20)
# xl = [min(x), max(x)]
# yl = [slope*xx + intercept  for xx in xl]
# plt.plot(xl, yl, c ='r')
# plt.grid()
# plt.legend(fontsize = 18)
# plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/plots/' + pname + '.png' ,  format = 'png' , bbox_inches='tight')

# ........................................... mapping A entropy H-M
h_all = []
dif_HM_all = []
dur_all = []
for afid, vdict in dict_fid_to_h.items():
    h_classes = dict_fid_to_h[afid]['h_class']
    h_time = dict_fid_to_h[afid]['h_time']
    h_dur = dict_fid_to_dur[afid]
    h = h_classes
    for p, cos in dict_pairs_to_cosines_TP.items():
        (tid, fid) = p
        if afid == fid:
            h_all.append(h)
            hR = dict_pairs_to_ratings_TP[p]
            dif_HM_all.append(np.abs(cos - 0.01 * hR))
            dur_all.append(dict_fid_to_dur[fid])

# correlations
corr_h_dif = stats.pearsonr(h_all, dif_HM_all)

print("###### Audio correlatons with H-M ######")
print('(', round(corr_h_dif[0], 3), ', ', round(corr_h_dif[1], 3), ')')

corr_h_dur = stats.pearsonr(h_all, dur_all)

print("###### Audio correlatons with H-M ######")
print('(', round(corr_h_dur[0], 3), ', ', round(corr_h_dur[1], 3), ')')


# %% plot only correlations TP
# plt.figure(figsize =(8, 12))
# plt.suptitle('Correlations for TP pairs', fontsize = 20)
# plt.subplot(2,1,1)
# plt.ylabel('Entropy over classes', fontsize = 18)
# plt.xlabel('Human rating', fontsize = 18)
# plt.scatter(r_all, h_all, label =  str(round(corr_h_rat[0],3) ) + ', p < 0.05' )
# par = np.polyfit(r_all, h_all, 1, full=True)
# slope=par[0][0]
# intercept=par[0][1]
# xl = [min(r_all), max(r_all)]
# yl = [slope*xx + intercept  for xx in xl]
# plt.plot(xl, yl, c ='r')
# plt.ylim(0,1)
# plt.xlim(0,100)
# plt.grid()
# plt.legend(fontsize = 16)

# plt.subplot(2,1,2)
# plt.ylabel('Entropy over classes', fontsize = 18)
# plt.xlabel('Machine rating', fontsize = 18)
# plt.scatter(cos_all, h_all, label = str(round(corr_h_cos[0],3) ) + ', p < 0.05' )
# par = np.polyfit(cos_all, h_all, 1, full=True)
# slope=par[0][0]
# intercept=par[0][1]
# xl = [min(cos_all), max(cos_all)]
# yl = [slope*xx + intercept  for xx in xl]
# plt.plot(xl, yl, c ='r')
# plt.ylim(0,1)
# plt.xlim(0,1)
# plt.grid()
# plt.legend(fontsize = 16)
# pname = 'audio_h_TP.png'
# plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/' + pname ,  format = 'png' , bbox_inches='tight')

# %% TEXT COMPLEXITY CORRELATIONS
# ........................................... mapping entropy to human ratings
p_all = []
r_all = []

for tid_measure, vp in dict_tid_to_p.items():
    ws_counts = dict_tid_to_p[tid_measure]['ws_counts']
    cws_counts = dict_tid_to_p[tid_measure]['cws_counts']
    nouns_counts = dict_tid_to_p[tid_measure]['nouns_counts']
    adjs_counts = dict_tid_to_p[tid_measure]['adjs_counts']
    count_fr_ws = dict_tid_to_p[tid_measure]['count_fr_ws']
    count_fr_cws = dict_tid_to_p[tid_measure]['count_fr_cws']
    count_fr_ns = dict_tid_to_p[tid_measure]['count_fr_ns']
    count_fr_adjs = dict_tid_to_p[tid_measure]['count_fr_adjs']
    perp = dict_tid_to_p[tid_measure]['perp']
    # select which feature is p
    p_measure = ws_counts
    for p, r in dict_pairs_to_ratings_TP.items():
        (tid, fid) = p
        if tid_measure == tid:
            p_all.append(p_measure)
            r_all.append(r)


# correlations
corr_p_rat = stats.pearsonr(p_all, r_all)
print('(', round(corr_p_rat[0], 3), ',', round(corr_p_rat[1], 3), ')')

# ........................................... plotting
# pname = "p_nw_hr"
# x = p_all
# y = r_all
# corr_ = stats.pearsonr(x,y)
# plt.figure(figsize =(8, 8))
# plt.scatter(x, y, label =  str(round(corr_[0],3) ) + ', p < 0.01' )
# par = np.polyfit(x, y, 1, full=True)
# slope=par[0][0]
# intercept=par[0][1]
# plt.xlabel('Number of words', fontsize = 20)
# plt.ylabel('Human rating', fontsize = 20)
# xl = [min(x), max(x)]
# yl = [slope*xx + intercept  for xx in xl]
# plt.plot(xl, yl, c ='r')
# plt.grid()
# plt.legend(fontsize = 18)
# plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/plots/' + pname + '.png' ,  format = 'png' , bbox_inches='tight')


# ........................................... mapping entropy to cosine distances
p_all = []
cos_all = []
dif_all = []
dur_all = []
mpt_all = []
for tid_measure, vp in dict_tid_to_p.items():
    ws_counts = dict_tid_to_p[tid_measure]['ws_counts']
    cws_counts = dict_tid_to_p[tid_measure]['cws_counts']
    nouns_counts = dict_tid_to_p[tid_measure]['nouns_counts']
    adjs_counts = dict_tid_to_p[tid_measure]['adjs_counts']
    count_fr_ws = dict_tid_to_p[tid_measure]['count_fr_ws']
    count_fr_cws = dict_tid_to_p[tid_measure]['count_fr_cws']
    count_fr_ns = dict_tid_to_p[tid_measure]['count_fr_ns']
    count_fr_adjs = dict_tid_to_p[tid_measure]['count_fr_adjs']
    perp = dict_tid_to_p[tid_measure]['perp']
    # select which feature is p
    p_measure = ws_counts
    for p, cos in dict_pairs_to_cosines_TP.items():
        (tid, fid) = p
        if tid_measure == tid:
            p_all.append(p_measure)
            cos_all.append(cos)
            hR = dict_pairs_to_ratings_TP[p]
            dif_all.append(np.abs(cos - 0.01 * hR))
            dur_all.append(dict_fid_to_dur[fid])
            mpt_all.append(dict_pairs_to_tall_TP[p])

# correlations
corr_p_cos = stats.pearsonr(p_all, cos_all)
print('(', round(corr_p_cos[0], 3), ', ', round(corr_p_cos[1], 3), ')')

# ........................................... mapping P entropy to  H-M

corr_p_diff = stats.pearsonr(p_all, dif_all)
print('(', round(corr_p_diff[0], 3), ', ', round(corr_p_diff[1], 3), ')')

# ........................................... mapping P entropy to  dur A

corr_p_dur = stats.pearsonr(p_all, dur_all)
print('(', round(corr_p_dur[0], 3), ', ', round(corr_p_dur[1], 3), ')')

# ........................................... mapping P entropy to  mpt A

corr_p_mpt = stats.pearsonr(p_all, mpt_all)
print('(', round(corr_p_dur[0], 3), ', ', round(corr_p_dur[1], 3), ')')

# ........................................... plotting
# pname = "p_nw_hmdiff"
# x = p_all
# y = dif_all
# corr_ = stats.pearsonr(x,y)
# plt.figure(figsize =(8, 8))
# plt.scatter(x, y, label =  str(round(corr_[0],3) ) + ', p < 0.01' )
# par = np.polyfit(x, y, 1, full=True)
# slope=par[0][0]
# intercept=par[0][1]
# plt.xlabel('Number of words', fontsize = 20)
# plt.ylabel('Difference between human and machine', fontsize = 20)
# xl = [min(x), max(x)]
# yl = [slope*xx + intercept  for xx in xl]
# plt.plot(xl, yl, c ='r')
# plt.grid()
# plt.legend(fontsize = 18)
# plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/plots/' + pname + '.png' ,  format = 'png' , bbox_inches='tight')

# pname = "p_nw_mpt"
# x = p_all
# y = mpt_all
# corr_ = stats.pearsonr(x, y)
# plt.figure(figsize=(8, 8))
# plt.scatter(x, y, label=str(round(corr_[0], 3)) + ', p < 0.01')
# par = np.polyfit(x, y, 1, full=True)
# slope = par[0][0]
# intercept = par[0][1]
# plt.xlabel('Number of words', fontsize=20)
# plt.ylabel('Mean played time', fontsize=20)
# xl = [min(x), max(x)]
# yl = [slope*xx + intercept for xx in xl]
# plt.plot(xl, yl, c='r')
# plt.grid()
# plt.legend(fontsize=18)
# plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/plots/' +
#             pname + '.png',  format='png', bbox_inches='tight')

# %% plot only correlations TP
ylab = 'Perplexity'
plt.figure(figsize=(8, 12))
plt.suptitle('Correlations for TP pairs', fontsize=20)
plt.subplot(2, 1, 1)
plt.ylabel(ylab, fontsize=18)
plt.xlabel('Human rating', fontsize=18)
plt.scatter(r_all, p_all, label=str(round(corr_p_rat[0], 3)) + ', p < 0.05')
par = np.polyfit(r_all, p_all, 1, full=True)
slope = par[0][0]
intercept = par[0][1]
xl = [min(r_all), max(r_all)]
yl = [slope*xx + intercept for xx in xl]
plt.plot(xl, yl, c='r')
plt.xlim(0, 100)
plt.grid()
plt.legend(fontsize=16)

plt.subplot(2, 1, 2)
plt.ylabel(ylab, fontsize=18)
plt.xlabel('Machine rating', fontsize=18)
plt.scatter(cos_all, p_all, label='n.s.')
par = np.polyfit(cos_all, p_all, 1, full=True)
slope = par[0][0]
intercept = par[0][1]
xl = [min(cos_all), max(cos_all)]
yl = [slope*xx + intercept for xx in xl]
plt.plot(xl, yl, c='r')
plt.xlim(0, 1)
plt.grid()
plt.legend(fontsize=16)
pname = 'text_perplexity_TP.png'

plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/' +
            pname,  format='png', bbox_inches='tight')

# %% set X, Y for linear regressor
h_all = []
p_all = []
r_all = []
c_all = []
delta_r = []
for item_pair, item_rate in dict_pairs_to_ratings_TP.items():
    (tid, fid) = item_pair
    if fid in dict_fid_to_h:
        h_classes = dict_fid_to_h[fid]['h_class']
        ws_counts = dict_tid_to_p[tid]['ws_counts']
        perp = dict_tid_to_p[tid_measure]['perp']

        h = h_classes
        p = ws_counts
        r = item_rate
        h_all.append(h)
        p_all.append(p)
        r_all.append(r)
        c_all.append(dict_pairs_to_cosines_TP[(tid, fid)])
        delta_r.append(dict_pairs_to_rall_TP[(tid, fid)])

x = np.array([h_all, p_all])
y = np.array([delta_r], dtype='float') - 100 * np.array([c_all], dtype='float')
X = x.T
Y = y.T
# np.random.shuffle(Y) ... > 26.3
# set chance level by suffling and measure error (MSE)
# correlation between them(Y, Y_pred), spearman, Y is fused variable
# %%
###############################################################################
# New test
###############################################################################

h1_all = []
h2_all = []
h3_all = []
p1_all = []
p2_all = []
p3_all = []
p4_all = []
p5_all = []
p6_all = []
p7_all = []
p8_all = []
p9_all = []


r_all = []
rstd_all = []
rm_all = []
dif_hm = []
for p, r in dict_pairs_to_ratings_TP.items():
    (tid, fid) = p
    if tid in dict_tid_to_p and fid in dict_fid_to_h:
        # ..... audio complexity
        h1_all.append(dict_fid_to_h[fid]['h_class'])
        h2_all.append(dict_fid_to_h[fid]['h_time'])
        h3_all.append(dict_fid_to_dur[fid])
        # ..... text complexity
        p1_all.append(dict_tid_to_p[tid]['ws_counts'])
        p2_all.append(dict_tid_to_p[tid]['cws_counts'])
        p3_all.append(dict_tid_to_p[tid]['perp'])
        p4_all.append(dict_tid_to_p[tid]['nouns_counts'])
        p5_all.append(dict_tid_to_p[tid]['adjs_counts'])
        

        p6_all.append(dict_tid_to_p[tid]['count_fr_ws'])
        p7_all.append(dict_tid_to_p[tid]['count_fr_cws'])
        p8_all.append(dict_tid_to_p[tid]['count_fr_ns'])
        p9_all.append(dict_tid_to_p[tid]['count_fr_adjs'])
        # ..... rating
        r_all.append(r)
        rstd_all.append(dict_pairs_to_rall_TP[(tid, fid)] )
        cos = dict_pairs_to_cosines_TP[(tid, fid)]
        rm_all.append(cos)
        dif_hm.append(np.abs(cos - 0.01 * r))

#%%
# x = np.array([p1_all, p2_all, p3_all, p4_all, p5_all, p6_all, p7_all, p8_all, h1_all, h2_all])
# x = np.array([p1_all, h_all, t_all])
# x = np.array([p1_all, p2_all, p3_all, p4_all, p5_all, h_all, t_all])


x = np.array([ h2_all])
#x = np.array([h1_all, h2_all])

y = np.array([rstd_all], dtype='float')
X = x.T
Y = y.T

        
regr = linear_model.LinearRegression()
reg = regr.fit(X, Y)

Y_predicted = reg.predict(X)
res = stats.pearsonr(np.squeeze(Y), np.squeeze(Y_predicted))
res = stats.pearsonr(np.squeeze(Y), np.squeeze(Y_predicted))
print(res)
res = stats.spearmanr(Y, Y_predicted)
print(res)

regr.score(X, Y)
plt.scatter(X,Y, color = 'blue')
plt.plot(X,Y_predicted, color = 'red')
plt.xlabel("p2")
plt.ylabel("machine rating")
plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/plots/' + 'p2_train.png' ,  format = 'png' , bbox_inches='tight')



#%% LOO

lenD = len(X)
regr = linear_model.LinearRegression()

res_all_trails = []
inds = np.arange(len(X))
trials = 1
for count_trails in range(trials): 
    # shuffling  
    # np.random.shuffle(inds)
    # X = X[inds, :]
    # Y = Y[inds]
    #########-
    X_test = []
    Y_test = []
    Y_predicted_LOO = []
    
    for j_out in range(0, lenD):
        
        train_range = [i for i in range(0,lenD) if i != j_out ]
        Xf_test = X[j_out]
        X_test.extend(Xf_test)
        Y_test.extend(Y[j_out])
        
        Xtrain = X[train_range,:]
        Ytrain = Y[train_range,:]
        
        # reg= regr.fit(Xtrain, Ytrain)
        # print(regr.score(Xtrain, Ytrain))
        # Yf_predicted = reg.predict([Xf_test])
        M = Ytrain / Xtrain
        Yf_predicted = M * Xf_test
        Y_predicted_LOO.extend(Yf_predicted)

        

    res = stats.pearsonr(np.squeeze(Y_test), np.squeeze(Y_predicted_LOO) ) # np.squeeze(Y[0:-1])
    print(res)
    # res = stats.spearmanr(Y_test, Y_predicted_LOO)
    # print(res)
    res_all_trails.append(res[0])
    if res[1] > 0.05:
        print ('....................... not correlated .........................')
    
print ('#######    average is    #################')
print(np.mean(res_all_trails))

#%%

plt.scatter(Y_test,Y_predicted_LOO, label = str(round(res[0],3)))
plt.xlabel("Y_test")
plt.ylabel("Y_predicted_LOO")
plt.title('correlation between predicted and actual data points for \nY: difference between human ratings, X: entropy over time')
plt.grid()
plt.legend()
plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/plots/' + 'Y_Ypred_test.png' ,  format = 'png' , bbox_inches='tight')


plt.scatter(Y,Y_predicted, label= 'n.s')
plt.xlabel("Y_train")
plt.ylabel("Y_predicted")
plt.title('correlation between predicted and actual data points for \nY: difference between human ratings, X: entropy over time')
plt.grid()
plt.legend()
plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/plots/' + 'Y_Ypred_train.png' ,  format = 'png' , bbox_inches='tight')


plt.scatter(X,Y, label= 'n.s')
plt.scatter(X,Y_predicted, label= 'n.s')
plt.xlabel("X: entropy over time ")
plt.ylabel("Y: difference between human ratings")
plt.title('correlation between X and Y')
plt.grid()
plt.legend()
plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/plots/' + 'Y_X.png' ,  format = 'png' , bbox_inches='tight')


plt.scatter(X_test,Y_predicted_LOO, color = 'gray', alpha = 0.5, label = 'test')
plt.plot(X,Y_predicted, color = 'red', label = 'train')

plt.xlabel("h2: audio complexity over time")
plt.ylabel("predicted machine rating")
plt.legend()
plt.savefig('/worktmp2/hxkhkh/current/Dcase/docs/correlations/plots/' + 'h2_MR_test_corr.png' ,  format = 'png' , bbox_inches='tight')

# plt.scatter(X_test,Y_test)
# plt.plot(Y_test, Y_predicted)
#%% K-fold

nfolds = 581
lenD = len(X)
lenF = int(lenD / nfolds)
regr = linear_model.LinearRegression()

res_all_trails = []
inds = np.arange(len(X))
trials = 2
for count_trails in range(trials): 
    # shuffling  
    np.random.shuffle(inds)
    X = X[inds, :]
    Y = Y[inds]
    #########-
    #test = []
    Y_test = []
    Y_predicted = []
    
    for fo in range(0, lenD - lenF +1 , lenF):
        
        fold_range = [i for i in range(fo, fo+lenF)]
        train_range = [i for i in range(0,lenD) if i not in fold_range ]
        Xf = X[fold_range , :]
        Yf = Y[fold_range , :]
        Y_test.extend(Yf)
        
        #test.append(train_range)
        
        Xtrain = X[train_range,:]
        Ytrain = Y[train_range,:]
        
        regr.fit(Xtrain, Ytrain)
        Yf_predicted = regr.predict(Xf)
        Y_predicted.extend(Yf_predicted[0])
        
    # MSE = np.mean((np.sqrt((Y_test - Y_predicted)**2)))
    # print(MSE)  # 16.0 for random
    print(len(Y))
    print(len(Y_test))
    res = stats.pearsonr(np.squeeze(Y_test), np.squeeze(Y_predicted) ) # np.squeeze(Y[0:-1])
    print(res)
    # res = stats.spearmanr(Y_test, Y_predicted)
    # print(res)
    res_all_trails.append(res[0])
    if res[1] > 0.05:
        print ('....................... not correlated .........................')
    
print ('#######    average is    #################')
print(np.mean(res_all_trails))
# %%
data = {'h1_all':h1_all,'h2_all':h2_all, 'p1_all':p1_all,'p2_all':p2_all,'p3_all':p3_all, 'rhuman_all':r_all,'rmachine_all':rm_all  }

from scipy.io import savemat
savemat('/worktmp2/hxkhkh/current/Dcase/docs/correlations/plots/' + "data.mat", data)