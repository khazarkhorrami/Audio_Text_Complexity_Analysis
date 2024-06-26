import os
import numpy
import scipy.io
from matplotlib import pyplot as plt

from tensorflow import keras
from data_preprocessing import prepare_XY , get_captions_dictionary , load_data
from utils import  prepare_triplet_data ,  triplet_loss , prepare_MMS_data , mms_loss, calculate_recallat
from model import VGS
import config as cfg
    
class train_validate (VGS):
    
    def __init__(self):
        VGS.__init__(self)

        # paths
        self.dataset_name = cfg.paths['dataset_name']
        self.model_dir = cfg.paths['modeldir']
        
        self.path_clotho_audio = cfg.paths['path_clotho_audio']
        self.path_clotho_captions = cfg.paths['path_clotho_captions']
        self.path_clotho_meta = cfg.paths["path_clotho_meta"]

        self.feature_path_audio = cfg.paths["feature_path_audio"]
        self.feature_path_captions = cfg.paths["feature_path_captions"]
        
        # action parameters
        self.use_pretrained = cfg.action_parameters['use_pretrained']
        self.training_mode = cfg.action_parameters['training_mode']
        self.evaluating_mode = cfg.action_parameters['evaluating_mode']
        self.saving_mode = cfg.action_parameters['save_model']
        self.save_best_recall = cfg.action_parameters['save_best_recall']
        self.save_best_loss = cfg.action_parameters['save_best_loss']
        self.find_recall = cfg.action_parameters['find_recall']
        self.number_of_epochs = cfg.action_parameters['number_of_epochs']
        self.chunk_length = cfg.action_parameters['chunk_length']
        
        # model setting
        self.model_name = cfg.feature_settings ['model_name']
        self.length_sequence = cfg.feature_settings['length_sequence']
        self.Xshape = cfg.feature_settings['Xshape']
        self.Yshape = cfg.feature_settings['Yshape']
        self.input_dim = [self.Xshape,self.Yshape] 
        self.loss = "Triplet"
        
        self.length_sequence = self.Xshape[0]
        self.split = 'validation'
        self.captionID = 0
        self.feature_names = []
        
        super().__init__() 
        
        
    def initialize_model_parameters(self):
        
        if self.use_pretrained:
            data = scipy.io.loadmat(self.model_dir + 'valtrainloss.mat', variable_names=['allepochs_valloss','allepochs_trainloss','all_avRecalls', 'all_vaRecalls'])
            allepochs_valloss = data['allepochs_valloss'][0]
            allepochs_trainloss = data['allepochs_trainloss'][0]
            all_avRecalls = data['all_avRecalls'][0]
            all_vaRecalls = data['all_vaRecalls'][0]
            
            allepochs_valloss = numpy.ndarray.tolist(allepochs_valloss)
            allepochs_trainloss = numpy.ndarray.tolist(allepochs_trainloss)
            all_avRecalls = numpy.ndarray.tolist(all_avRecalls)
            all_vaRecalls = numpy.ndarray.tolist(all_vaRecalls)
            recall_indicator = numpy.max(allepochs_valloss)
            val_indicator = numpy.min(allepochs_valloss)
        else:
            allepochs_valloss = []
            allepochs_trainloss = []
            all_avRecalls = []
            all_vaRecalls = []
            recall_indicator = 0
            val_indicator = 1000
            
        saving_params = [allepochs_valloss, allepochs_trainloss, all_avRecalls, all_vaRecalls, val_indicator , recall_indicator ]       
        return saving_params 

        
    # def prepare_chunked_names (self , captionID):
    
    #     n = len(self.feature_names)
    #     Ynames_all, Xnames_all , Znames_all = [],[],[]       
    #     for start_chunk in range(0, n , self.chunk_length):
    #         end_chunk = start_chunk + self.chunk_length
    #         chunk = self.feature_names [start_chunk:end_chunk ]            
    #         Ynames, Xnames , Znames = expand_feature_filenames2(chunk , captionID)
    #         Ynames_all.append(Ynames)
    #         Xnames_all.append(Xnames)
    #         Znames_all.append(Znames)
    #     return Ynames_all, Xnames_all , Znames_all
    
    def set_data_paths (self):
       
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

    def read_features(self):     
        self.set_data_paths()
        if self.dataset_name == "clotho":
            
            textual_feature_fullpath = os.path.join(self.feature_path_captions, self.split) 
            audio_feature_fullpath = os.path.join(self.feature_path_audio, self.split)              
            all_captions_dictionaries = get_captions_dictionary (self.caption_path)
            
            Ydata_initial = []
            Xdata_initial = []
            Zdata = []
            for wav_name in all_captions_dictionaries:
                file_name = wav_name[0:-4]
                textual_feature_filename = os.path.join(textual_feature_fullpath, file_name)
                audio_feature_filename = os.path.join(audio_feature_fullpath, file_name)
                list_of_sentences = all_captions_dictionaries[wav_name]
                
                tf_allcaptions = load_data(textual_feature_filename)
                tf = tf_allcaptions[self.captionID]        
                af = load_data(audio_feature_filename)
                
                Ydata_initial.append(tf)
                Xdata_initial.append(af)
                Zdata.append(list_of_sentences [self.captionID])   
                
        Ydata, Xdata = prepare_XY (Ydata_initial, Xdata_initial, self.length_sequence)        
        return  Ydata, Xdata , Zdata                   
                    
        
    def train_model(self): 
        self.split = 'development'
        self.set_data_paths()
        
        for capID in range(5):
            print('......... capID ...........' , str(capID))
            self.captionID = capID       
            Ydata, Xdata , Zdata = self.read_features()
            if self.loss == "MMS":
                [Yin, Xin] , bin_target = prepare_MMS_data (Ydata, Xdata , shuffle_data = False)
            if self.loss == "Triplet":                   
                Yin, Xin, bin_target = prepare_triplet_data (Ydata, Xdata)
                    
            history = self.vgs_model.fit([Yin, Xin ], bin_target, shuffle=False, epochs=1,batch_size=96)                      
            del Yin, Xin, Ydata, Xdata
            training_output = history.history['loss'][0]
        return training_output

    def evaluate_model (self) :
        self.split = 'evaluation'
        self.set_data_paths()

        epoch_cum_val = 0
        epoch_cum_recall_av = 0
        epoch_cum_recall_va = 0
        
        for capID in range(5):
            print('......... capID ...........' , str(capID))
            self.captionID = capID       
            Ydata, Xdata , Zdata = self.read_features()
            if self.loss == "MMS":
                [Yin, Xin] , bin_target = prepare_MMS_data (Ydata, Xdata , shuffle_data = False)
            if self.loss == "Triplet":                   
                Yin, Xin, bin_target = prepare_triplet_data (Ydata, Xdata)
                
            loss = self.vgs_model.evaluate([Yin, Xin ], bin_target, batch_size=96) 
            epoch_cum_val += loss
            #............................................................. Recall
            if self.find_recall:
                print('......... finding recall ...........' )
                number_of_samples = len(Ydata)
                visual_embeddings = self.visual_embedding_model.predict(Ydata)
                audio_embeddings = self.audio_embedding_model.predict(Xdata)
                if self.model_name == "CNN0":
                    visual_embeddings_mean = visual_embeddings
                    audio_embeddings_mean = audio_embeddings                 
  
                elif self.model_name == "CNNatt":                 
                    visual_embeddings_mean = numpy.mean(visual_embeddings, axis = 1)
                    audio_embeddings_mean = numpy.mean(audio_embeddings, axis = 1)
                ########### calculating Recall@10                    
                poolsize =  number_of_samples
                number_of_trials = 2
                recall_av_vec = calculate_recallat( audio_embeddings_mean, visual_embeddings_mean, number_of_trials,  number_of_samples , poolsize )          
                recall_va_vec = calculate_recallat( visual_embeddings_mean , audio_embeddings_mean, number_of_trials,  number_of_samples , poolsize ) 
                recall_av = numpy.mean(recall_av_vec)/(poolsize)
                print(recall_av)
                recall_va = numpy.mean(recall_va_vec)/(poolsize)
                print(recall_va)
                epoch_cum_recall_av += recall_av
                epoch_cum_recall_va += recall_va               
                del Xdata, audio_embeddings
                del Ydata, visual_embeddings            
            del Xin, Yin
            
        final_recall_av = epoch_cum_recall_av /  (self.captionID + 1) 
        final_recall_va = epoch_cum_recall_va / (self.captionID + 1) 
        final_valloss = epoch_cum_val/ (self.captionID + 1) 
        
        validation_output = [final_recall_av, final_recall_va , final_valloss]
        print(validation_output)
        return validation_output
    
    def save_model(self, initialized_output , training_output, validation_output):
        
        os.makedirs(self.model_dir, exist_ok=1)
        [allepochs_valloss, allepochs_trainloss, all_avRecalls, all_vaRecalls, val_indicator , recall_indicator ] = initialized_output
        [final_recall_av, final_recall_va , final_valloss] = validation_output 
        [epoch_recall_av, epoch_recall_va , epoch_valloss] = validation_output
               
            
        if self.save_best_recall:
            epoch_recall = ( epoch_recall_av + epoch_recall_va ) / 2
            if epoch_recall >= recall_indicator: 
                recall_indicator = epoch_recall
                # weights = vgs_model.get_weights()
                # vgs_model.set_weights(weights)
                self.vgs_model.save_weights('%smodel_weights.h5' % self.model_dir)
        else :
            if epoch_valloss <= val_indicator: 
                val_indicator = epoch_valloss
                # weights = vgs_model.get_weights()
                # vgs_model.set_weights(weights)
                self.vgs_model.save_weights('%smodel_weights.h5' % self.model_dir)
                      
        allepochs_trainloss.append(training_output)  
        allepochs_valloss.append(epoch_valloss)
        if self.find_recall: 
            all_avRecalls.append(epoch_recall_av)
            all_vaRecalls.append(epoch_recall_va)
        save_file = self.model_dir + 'valtrainloss.mat'
        scipy.io.savemat(save_file, 
                          {'allepochs_valloss':allepochs_valloss,'allepochs_trainloss':allepochs_trainloss,'all_avRecalls':all_avRecalls,'all_vaRecalls':all_vaRecalls })  
        
        self.make_plot( [allepochs_trainloss, allepochs_valloss , all_avRecalls, all_vaRecalls ])

        
    def make_plot (self, plot_lists):
        
        plt.figure()
        plot_names = ['training loss','validation loss','audio_to_text','text_to_audio']
        for plot_counter, plot_value in enumerate(plot_lists):
            plt.subplot(2,2,plot_counter+1)
            plt.plot(plot_value)
            plt.ylabel(plot_names[plot_counter])
            plt.grid()

        plt.savefig(self.model_dir + 'evaluation_plot.pdf', format = 'pdf')
 
    def define_and_compile_models (self):
        self.vgs_model, self.visual_embedding_model, self.audio_embedding_model = self.build_model(self.model_name, self.input_dim)
        if self.loss == "MMS":
            self.vgs_model.compile(loss=mms_loss, optimizer= keras.optimizers.Adam(lr=1e-03))
        elif self.loss == "Triplet":
            self.vgs_model.compile(loss=triplet_loss, optimizer= keras.optimizers.Adam(lr=1e-03))
        print(self.vgs_model.summary())
        
    def __call__(self):
    
        self.define_and_compile_models()
  
        initialized_output = self.initialize_model_parameters()
        
        if self.use_pretrained:
            self.vgs_model.load_weights(self.model_dir + 'model_weights.h5')

        for epoch_counter in numpy.arange(self.number_of_epochs):
            
            print('......... epoch ...........' , str(epoch_counter))
            
            if self.training_mode:

                training_output = self.train_model()
                
            else:
                training_output = 0
                
            if self.evaluating_mode:
                validation_output = self.evaluate_model()
            else: 
                validation_output = [0, 0 , 0 ]
                
            if self.saving_mode:
                self.save_model(initialized_output, training_output, validation_output)
                

        