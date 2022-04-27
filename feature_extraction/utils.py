
"""

"""

import pickle
import numpy 
import librosa
import nltk
#from gensim.models import word2vec
from sent2vec.vectorizer import Vectorizer
vectorizer = Vectorizer()
#################################### General ##################################

def serialize_features (input_file, filename):
    outfile = open(filename,'wb')
    pickle.dump(input_file ,outfile , protocol=pickle.HIGHEST_PROTOCOL)
    outfile.close()

################################## audio features #############################

def calculate_logmels (wavfile , number_of_mel_bands , window_len , window_hop , sr_target):
    
    win_len_sample = int (sr_target * window_len)
    win_hop_sample = int (sr_target * window_hop)
    n_fft = 1024    
    y, sr = librosa.load(wavfile)
    y = librosa.core.resample(y, sr, sr_target) 
    
    stft = librosa.stft(y, n_fft = n_fft, win_length = win_len_sample, hop_length = win_hop_sample, window='hann')
    esd = (abs(stft))**2

    m = librosa.filters.mel(sr=sr,n_fft = n_fft, n_mels = number_of_mel_bands)
    mel_feature = numpy.dot(m,esd)
    
    # replacing zeros with small/min number
    zeros_mel = mel_feature[mel_feature==0]          
    if numpy.size(zeros_mel)!= 0:
        
        mel_flat = mel_feature.flatten('F')
        mel_temp =[value for counter, value in enumerate(mel_flat) if value!=0]
    
        if numpy.size(mel_temp)!=0:
            min_mel = numpy.min(numpy.abs(mel_temp))
        else:
            min_mel = 1e-12 
           
        mel_feature[mel_feature==0] = min_mel           
    logmel_feature = numpy.transpose(10*numpy.log10(mel_feature))       
    return logmel_feature


################################## text features ##############################


# from gensim.models import KeyedVectors
# w2vfile = "../../model/word2vec/GoogleNews-vectors-negative300.bin"
# model = KeyedVectors.load_word2vec_format(w2vfile, binary=True)
# from sent2vec.vectorizer import Vectorizer
# vectorizer = Vectorizer(pretrained_weights= w2vfile)
# sentences= ['hello, how are you','I am fine']
# vectorizer.run(sentences, remove_stop_words=['not'], add_stop_words=[])
# vectors_w2v = vectorizer.vectors


def bert_vectors (list_of_sentences):
    
    vectorizer.vectors = []
    vectors = vectorizer.run(list_of_sentences)
    vectors = vectorizer.vectors
    vectors_bert = numpy.array(vectors)
    # vectorizer.bert(list_of_sentences)
    #vectors_bert = vectorizer.vectors   
    return vectors_bert 

def string_to_nouns (string):
    #string = string.lower()
    words = nltk.word_tokenize(string)
    tok = nltk.pos_tag(words)

    nouns = [tok[i][0].lower() for i in numpy.arange(len(tok)) if tok[i][1] =='NN' or tok[i][1] =='NNS'
             or tok[i][1] =='VB' or tok[i][1] =='VBD' or tok[i][1] =='VBG' or tok[i][1] =='VBN' or tok[i][1] =='VBP'
             or tok[i][1] =='JJ'or tok[i][1] =='JJR'or tok[i][1] =='JJS']
    return nouns

def w2v_similarity_with (model, noun_list_ref,noun_list_can):
    max_similarities = []
    for n_r in noun_list_ref:        
        noun_vec = []
        for n_c in noun_list_can:                        
            sim = model.similarity(n_r,n_c)
            noun_vec.append(sim)            
        max_similarities.append(numpy.max(noun_vec))
    return round(numpy.mean(max_similarities),3)


def w2v_similarity_without (model, noun_list_ref,noun_list_can):
    max_similarities = []

    for n_r in noun_list_ref:        
        noun_vec = []
        for n_c in noun_list_can:
            if n_c != n_r :
                try:
                    sim = model.similarity(n_r,n_c)
                    noun_vec.append(sim)
                except:
                    print("An exception occurred in nouns" + n_c + " and " + n_r) 
        max_similarities.append(numpy.max(noun_vec))

    return round(numpy.mean(max_similarities),3)