
###################### initial configuration  #################################


paths = {
  "dataset_name" : "clotho",
    
  "path_clotho_audio": "../../data/clotho/audio/",
  "path_clotho_captions" : "../../data/clotho/captions/",
  "path_clotho_meta" : "../../data/clotho/metadata/",
  
  "feature_path_audio": "../../features/clotho/audio",
  "feature_path_captions": "../../features/clotho/captions", 
}


audio_feature_params = {
    "number_of_mel_bands" : 64,
    "window_len_in_seconds" : 0.040,
    "window_hop_in_seconds" : 0.020,
    "sr_target" : 22050
    }

text_feature_params = {
    "pretrained_model" : "/../../model/word2vec/GoogleNews-vectors-negative300.bin",
    }

action_params = {
    "extracting_audio_features" : True,
    "extracting_textual_features" : False,
    "split" : "validation",
    }