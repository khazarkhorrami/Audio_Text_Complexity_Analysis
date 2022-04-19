
"""

"""
import config as cfg
from utils import calculate_logmels1, calculate_vgg_b5, serialize_features
from data_preprocessing import get_SPOKENCOCO_imagenames
import os
import pathlib

from keras.models import Model
from keras.applications.vgg16 import VGG16

class Features:
    
    def __init__(self):
    
        # paths
        self.dataset_name = cfg.paths['clotho']
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
        self.text_feature_name = cfg.text_feature_params['GN_negative300']
      
        # action parameters
        self.extracting_audio_features = cfg.action_parameters['extracting_audio_features']
        self.extracting_textual_features = cfg.action_parameters['extracting_textual_features']
        self.split = cfg.action_parameters['split']


    def save_features (self, input_file , feature_fullpath , save_name):
        filename = os.path.join(feature_fullpath, save_name)
        serialize_features (input_file, filename)
        
    def read_file_paths (self, dataset_name):
       
       if dataset_name == "clotho":
           
           if self.split == "development":
               self.audio_path = os.path.join( self.path_clotho_audio , "development" )
               self.caption_path = os.path.join( self.path_clotho_captions , "clotho_captions_development.csv") 
               self.meta_path =  os.path.join(self.path_clotho_meta , "clotho_metadata_development.csv")
               
           elif self.split == "evaluation":
               self.audio_path = os.path.join( self.path_clotho_audio , "evaluation" )
               self.caption_path = os.path.join( self.path_clotho_captions , "clotho_captions_evaluation.csv") 
               self.meta_path =  os.path.join(self.path_clotho_meta , "clotho_metadata_evaluation.csv")
               
           elif self.split == "validation":
               self.audio_path = os.path.join( self.path_clotho_audio , "validation" )
               self.caption_path = os.path.join( self.path_clotho_captions , "clotho_captions_validation.csv") 
               self.meta_path =  os.path.join(self.path_clotho_meta , "clotho_metadata_validation.csv")
      
    def extract_visual_features(self, dataset_name):

        self.read_file_paths (dataset_name)
        os.makedirs(self.feature_path_visual , exist_ok= True)
        
        if self.visual_feature_name == 'vgg' and self.visual_feature_subname == 'block5_conv3':
            model = VGG16()
            layer_name = 'block5_conv3'  
            model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
            
        if dataset_name == "SPOKEN-COCO":
            image_fullnames = get_SPOKENCOCO_imagenames (self.json_file)
            for image_fullname in image_fullnames:
                
                image_fullpath = os.path.join(self.path_MSCOCO, image_fullname)
                vf_output = calculate_vgg_b5 (model, image_fullpath)
                
                image_name = image_fullname.split('/')[1]
                # remove ".jpg" from name
                save_name = image_name[:-4] 
                self.save_features(vf_output, self.feature_path_visual , save_name)
    
    def find_logmel_features(self, wavfile):
        logmel_feature = calculate_logmels1 (wavfile , self.number_of_mel_bands , self.window_len_in_seconds , self.window_hop_in_seconds , self.sr_target)
        return logmel_feature
    
     
    def extract_audio_features (self, dataset_name):
        self.read_file_paths (dataset_name)
        
        if dataset_name == "SPOKEN-COCO":
            folders = os.listdir(self.audio_path)
            for folder_name in folders:
                print(folder_name)
                feature_fullpath = os.path.join(self.feature_path, folder_name)  
                os.makedirs(feature_fullpath, exist_ok= True)
                
                files = os.listdir(os.path.join(self.audio_path,folder_name))
                for file_name in files:
                    wavfile = os.path.join(self.audio_path, folder_name , file_name)
                    logmel = self.find_logmel_features(wavfile)
                    save_name = file_name[0:-4] 
                    self.save_features(logmel, feature_fullpath , save_name)
    
    def __call__(self):
        
        if self.extracting_audio_features:
            self.extract_audio_features(self.dataset_name)
        elif self.extracting_visual_features:
            self.extract_visual_features(self.dataset_name)
            
            