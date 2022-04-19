
###################### initial configuration  #################################


paths = {
  "dataset_name" : "clotho",
    
  "path_clotho_audio": "../../data/clotho/audio/",
  "path_clotho_captions" : "../../data/clotho/captions/",
  "path_clotho_meta" : "../../data/clotho/metadata/",
  
  "feature_path_audio": "../../features/clotho/audio",
  "feature_path_captions": "../../features/clotho/captions", 
}


audio_feature_parameters = {
    "number_of_mel_bands" : 40,
    "window_len_in_seconds" : 0.025,
    "window_hop_in_seconds" : 0.01,
    "sr_target" : 16000
    }

text_feature_parameters = {
    "text_feature_name" : "GN_negative300",
    }

action_parameters = {
    "extracting_audio_features" : False,
    "extracting_textual_features" : False,
    "split" : "dev",
    }