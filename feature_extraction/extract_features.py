
"""

"""
import config as cfg
from utils import calculate_logmels, serialize_features, bert_vectors
from data_preprocessing import get_captions_dictionary
import os


class Features:
    
    def __init__(self):
    
        # paths
        self.dataset_name = cfg.paths['dataset_name']
        self.path_clotho_audio = cfg.paths['path_clotho_audio']
        self.path_clotho_captions = cfg.paths['path_clotho_captions']
        self.path_clotho_meta = cfg.paths['path_clotho_meta']
        self.feature_path_audio = cfg.paths['feature_path_audio']
        self.feature_path_captions = cfg.paths['feature_path_captions']

        # Audio features parameters
        self.number_of_mel_bands = cfg.audio_feature_params['number_of_mel_bands']
        self.window_len_in_seconds = cfg.audio_feature_params['window_len_in_seconds']
        self.window_hop_in_seconds = cfg.audio_feature_params['window_hop_in_seconds']
        self.sr_target = cfg.audio_feature_params['sr_target']

        # Text features parameters
        self.pretrained_text_model = cfg.text_feature_params['pretrained_model']
      
        # action parameters
        self.extracting_audio_features = cfg.action_params['extracting_audio_features']
        self.extracting_textual_features = cfg.action_params['extracting_textual_features']
        self.split = cfg.action_params['split']


    def save_features (self, input_file , feature_fullpath , save_name):
        filename = os.path.join(feature_fullpath, save_name)
        serialize_features (input_file, filename)
        
    def read_data_paths (self):
       
       if self.dataset_name == 'clotho':
           
           if self.split == 'development':
               self.audio_path = os.path.join( self.path_clotho_audio , "development" )
               self.caption_path = os.path.join( self.path_clotho_captions , "clotho_captions_development.csv") 
               self.meta_path =  os.path.join(self.path_clotho_meta , "clotho_metadata_development.csv")
               
           elif self.split == 'evaluation':
               self.audio_path = os.path.join( self.path_clotho_audio , "evaluation" )
               self.caption_path = os.path.join( self.path_clotho_captions , "clotho_captions_evaluation.csv") 
               self.meta_path =  os.path.join(self.path_clotho_meta , "clotho_metadata_evaluation.csv")
               
           elif self.split == 'validation':
               self.audio_path = os.path.join( self.path_clotho_audio , "validation" )
               self.caption_path = os.path.join( self.path_clotho_captions , "clotho_captions_validation.csv") 
               self.meta_path =  os.path.join(self.path_clotho_meta , "clotho_metadata_validation.csv")
      
    def extract_textual_features(self):
        
        self.read_data_paths()
        if self.dataset_name == "clotho":
            
            feature_fullpath = os.path.join(self.feature_path_captions, self.split)  
            os.makedirs(feature_fullpath, exist_ok= True)
            
            all_captions_dictionaries = get_captions_dictionary (self.caption_path)
            vectors_bert_all = []
            if self.pretrained_text_model == "bert":
                for wav_name in all_captions_dictionaries:
                    list_of_sentences = all_captions_dictionaries[wav_name]
                    vectors_bert = bert_vectors (list_of_sentences)
                    vectors_bert_all.append(vectors_bert)
                    # save_name = wav_name[0:-4]               
                    # self.save_features(vectors_bert, feature_fullpath , save_name)
        
        return all_captions_dictionaries, vectors_bert_all
    
    def find_logmel_features(self, wavfile):
        logmel_feature = calculate_logmels (wavfile , self.number_of_mel_bands , self.window_len_in_seconds , self.window_hop_in_seconds , self.sr_target)
        return logmel_feature
    
     
    def extract_audio_features (self):
        self.read_data_paths()
        
        if self.dataset_name == "clotho":
            
            feature_fullpath = os.path.join(self.feature_path_audio, self.split)  
            os.makedirs(feature_fullpath, exist_ok= True)
            
            wav_names = os.listdir(self.audio_path)
            for counter, wav_name in enumerate(wav_names):
                print(counter)
                
                wavfile = os.path.join(self.audio_path, wav_name)
                logmel = self.find_logmel_features(wavfile)
                save_name = wav_name[0:-4] 
                self.save_features(logmel, feature_fullpath , save_name)
    
    def __call__(self):
        
        if self.extracting_audio_features:
            self.extract_audio_features()
        elif self.extracting_textual_features:
            self.extract_visual_features()
            
            