
###################### initial configuration  #################################


paths = {
    
  "dataset_name" : "clotho",
  
  "path_clotho_audio": "../../data/clotho/audio/",
  "path_clotho_captions" : "../../data/clotho/captions/",
  "path_clotho_meta" : "../../data/clotho/metadata/",

  "feature_path_audio": "../../features/clotho/audio",
  "feature_path_captions": "../../features/clotho/captions", 
  
  "modeldir": "../../model/model03/",
}


action_parameters = {
  "use_pretrained": False,
  "training_mode": True,
  "evaluating_mode": True,
  "save_model":True,
  "save_best_recall" : False,
  "save_best_loss" : True,
  "find_recall" : True,
  "number_of_epochs" : 50,
  "chunk_length":10000
}

feature_settings = {
    "model_name": "CNN0",   
    "length_sequence" : 1024,
    "Xshape" : (1024,64),
    "Yshape" : (768)
    }
