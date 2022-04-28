# model00 : similaro to simple VGS network
# model01 : upgraded CNN network
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import  MaxPooling1D, MaxPooling2D, Conv1D,Conv2D, ReLU, Add, GRU, Bidirectional
from tensorflow.keras.layers import Softmax, Permute, AveragePooling1D, Concatenate, dot, LeakyReLU

class VGS:
    
    def __init__(self):
        pass

    def build_residual_audio_model (self, input_dim): 
        [Xshape, Yshape] = self.input_dim
        speech_sequence = Input(shape=Xshape) #Xshape = (1024,64)
  
        # speech channel
        
        #speech_sequence_masked = Masking (mask_value=0., input_shape=X2shape)(speech_sequence)
        strd = 2
        
        x0 = Conv1D(128,1,strides = 1, padding="same")(speech_sequence)
        x0 = BatchNormalization(axis=-1)(x0)
        x0 = ReLU()(x0) 
          
        # layer 1  
        in_residual = x0  
        x1 = Conv1D(128,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)       
        x2 = Conv1D(128,9,strides = strd, padding="same")(x1)  
        downsample = Conv1D(128,9,strides = strd, padding="same")(in_residual)
        out = Add()([downsample,x2])
        out_1 = ReLU()(out) # (500, 128) 
        
        # layer 2
        in_residual = out_1  
        x1 = Conv1D(256,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)    
        x2 = Conv1D(256,9,strides = strd, padding="same")(x1)  
        downsample = Conv1D(256,9,strides = strd, padding="same")(in_residual)
        out = Add()([downsample,x2])
        out_2 = ReLU()(out) # (256, 256)
        
        # layer 3
        in_residual = out_2  
        x1 = Conv1D(512,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)    
        x2 = Conv1D(512,9,strides = strd, padding="same")(x1)  
        downsample = Conv1D(512,9,strides = strd, padding="same")(in_residual)
        out = Add()([downsample,x2])
        out_3 = ReLU()(out) # (128, 512)
    
        # layer 4
        # in_residual = out_3  
        # x1 = Conv1D(1024,9,strides = 1, padding="same")(in_residual)
        # x1 = BatchNormalization(axis=-1)(x1)
        # x1 = ReLU()(x1)       
        # x2 = Conv1D(1024,9,strides = strd, padding="same")(x1)  
        # downsample = Conv1D(1024,9,strides = strd, padding="same")(in_residual)
        # out = Add()([downsample,x2])
        # out_4 = ReLU()(out)   # (64, 1024)  
        
        # pooling
        
        # out_speech_channel  = AveragePooling1D(512,padding='same')(out_3) #(N,1, 1024)
        # out_speech_channel = Reshape([out_speech_channel.shape[2]])(out_speech_channel)  #(N, 1024)      
        # out_speech_channel = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_speech')(out_speech_channel)
        out_speech_channel = out_3               
        audio_model = Model(inputs= speech_sequence, outputs = out_speech_channel )
        #audio_model.summary()
        return speech_sequence , out_speech_channel , audio_model 
    
          
    def build_CNN01_audio_model (self , input_dim):
        #Xshape = (1024,64)
        [Xshape, Yshape] = self.input_dim
        
        dropout_size = 0.1
        activation_C='relu'
    
        audio_sequence = Input(shape=Xshape)
        audio_sequence_extended =  Reshape([audio_sequence.shape[1],audio_sequence.shape[2], 1 ])(audio_sequence) 
        forward1 = Conv2D(32,3,padding="same",activation=activation_C,name = 'conv1')(audio_sequence_extended)
        dr1 = Dropout(dropout_size)(forward1)
        bn1 = BatchNormalization(axis=-1)(dr1)
        
        pool1 = MaxPooling2D(4,strides = 2, padding='same')(bn1)
        
        forward2 = Conv2D(32,3,padding="same",activation=activation_C,name = 'conv2')(pool1)
        dr2 = Dropout(dropout_size)(forward2)
        bn2 = BatchNormalization(axis=-1)(dr2)
         
        pool2 = MaxPooling2D(4,strides = 2, padding='same')(bn2)
          
        forward3 = Conv2D(64,5,padding="same",activation=activation_C,name = 'conv3')(pool2)
        dr3 = Dropout(dropout_size)(forward3)
        bn3 = BatchNormalization(axis=-1)(dr3) 
        
        pool3 = MaxPooling2D(4,strides = 2,padding='same')(bn3)
          
        forward4 = Conv2D(128,7,padding="same",activation=activation_C,name = 'conv4')(pool3)
        dr4 = Dropout(dropout_size)(forward4)
        bn4 = BatchNormalization(axis=-1)(dr4) 
        pool4 = MaxPooling2D(4,strides = 2,padding='same')(bn4)
           
        forward5 = Conv2D(128,7,padding="same",activation=activation_C,name = 'conv5')(pool4)
        dr5 = Dropout(dropout_size)(forward5)
        bn5 = BatchNormalization(axis=-1,name='audio_branch')(dr5) # (N,64,512)
        pool5 = MaxPooling2D(4,strides = 4,padding='same')(bn5)
        
        out_audio_channel = Reshape([pool5.shape[1], pool5.shape[3]]) (pool5)
        audio_model = Model(inputs= audio_sequence, outputs = out_audio_channel )
        
        return audio_sequence , out_audio_channel , audio_model

    def build_baseline_audio_model (self , input_dim):
        #Xshape = (1024,64)
        [Xshape, Yshape] = self.input_dim
        
    
        audio_sequence = Input(shape=Xshape)
        audio_sequence_reshaped =  Reshape([audio_sequence.shape[1],audio_sequence.shape[2], 1 ])(audio_sequence) 
        
        # Conv2D block
        bn = BatchNormalization(axis=-1)(audio_sequence_reshaped)
        forward = Conv2D(32,3,padding="same")(bn)
        act = LeakyReLU(alpha=0.1)(forward)     
        
        # pooling
        pool = MaxPooling2D(4,strides = (2, 4), padding='same')(act)
        
        # Conv2D block
        bn = BatchNormalization(axis=-1)(pool)
        forward = Conv2D(128,3,padding="same")(bn)
        act = LeakyReLU(alpha=0.1)(forward)
        
        # Conv2D block
        bn = BatchNormalization(axis=-1)(act)
        forward = Conv2D(128,3,padding="same")(bn)
        act = LeakyReLU(alpha=0.1)(forward)
        
        # pooling
        pool = MaxPooling2D(4,strides = (2, 4), padding='same')(act)
        
        # Conv2D block
        bn = BatchNormalization(axis=-1)(pool)
        forward = Conv2D(128,3,padding="same")(bn)
        act = LeakyReLU(alpha=0.1)(forward)
        
        # Conv2D block
        bn = BatchNormalization(axis=-1)(act)
        forward = Conv2D(128,3,padding="same")(bn)
        act = LeakyReLU(alpha=0.1)(forward)
        
        # pooling
        pool = MaxPooling2D(4,strides = (1, 4), padding='same')(act)
        
        dr = Dropout(0.3)(pool)
        
        input_gru = Reshape([dr.shape[1], dr.shape[3]])(dr)
        gru = Bidirectional (GRU (128, activation="tanh", return_sequences=True) ) (input_gru)
        
        out_audio_channel = gru
        audio_model = Model(inputs= audio_sequence, outputs = out_audio_channel )
        
        return audio_sequence , out_audio_channel , audio_model    
        
    def build_baseline_textual_model (self, input_dim):
        # Yshape = (768)
        [Xshape, Yshape] = self.input_dim
        
        textual_sequence = Input(shape=Yshape)
        
        #textual_sequence_norm = BatchNormalization(axis=0, name = 'bn0_textual')(textual_sequence)
        
        # dropout_size = 0.3
        # forward_textual = Conv1D(512,3,strides=1,padding = "same", activation='linear', name = 'conv_textual')(textual_sequence_norm)
        # dr_textual = Dropout(dropout_size,name = 'dr_textual')(forward_textual)
        # bn_textual = BatchNormalization(axis=-1,name = 'bn1_textual')(dr_textual)
        # pool_textual = MaxPooling1D(3,strides = 2,padding='same')(bn_textual)
        
        
        out_textual_channel = textual_sequence
        textual_model = Model(inputs= textual_sequence, outputs = out_textual_channel )
        return textual_sequence , out_textual_channel , textual_model
   

    def CNNatt (self , input_dim):
        #input_dim = [(512, 40), (14, 14, 512)]
        speech_sequence , out_speech_channel , audio_model = self.build_audio_model (input_dim)
        textual_sequence , out_textual_channel , textual_model = self. build_textual_model ( input_dim)          
        
        A = out_speech_channel
        I = out_textual_channel
        
        textual_embedding_model = Model(inputs=textual_sequence, outputs = I, name='textual_embedding_model')
        audio_embedding_model = Model(inputs= speech_sequence, outputs = A, name='audio_embedding_model')  
        
        #### Attention I for query Audio
        # checks which part of image gets more attention based on audio query.   
        keyImage = out_textual_channel
        valueImage = out_textual_channel
        queryAudio = out_speech_channel
        
        scoreI = keras.layers.dot([queryAudio,keyImage], normalize=False, axes=-1,name='scoreI')
        weightID = Dense(196,activation='sigmoid')(scoreI)
        weightI = Softmax(name='weigthI')(scoreI)
        
        valueImage = Permute((2,1))(valueImage)
        attentionID = keras.layers.dot([weightID, valueImage], normalize=False, axes=-1,name='attentionID')
        attentionI = keras.layers.dot([weightI, valueImage], normalize=False, axes=-1,name='attentionI')
        
        poolAttID = AveragePooling1D(512, padding='same')(attentionID)
        poolAttI = AveragePooling1D(512, padding='same')(attentionI)
        
        poolqueryAudio = AveragePooling1D(512, padding='same')(queryAudio)
        
        outAttAudio = Concatenate(axis=-1)([poolAttI,poolAttID, poolqueryAudio])
        outAttAudio = Reshape([1536],name='reshape_out_attAudio')(outAttAudio)
    
        ###  Attention A  for query Image
        # checks which part of audio gets more attention based on image query.
        keyAudio = out_speech_channel
        valueAudio = out_speech_channel
        queryImage = out_textual_channel
        
        scoreA = keras.layers.dot([queryImage,keyAudio], normalize=False, axes=-1,name='scoreA')
        weightAD = Dense(64,activation='sigmoid')(scoreA)
        weightA = Softmax(name='weigthA')(scoreA)
        
        valueAudio = Permute((2,1))(valueAudio)
        attentionAD = keras.layers.dot([weightAD, valueAudio], normalize=False, axes=-1,name='attentionAD')
        attentionA = keras.layers.dot([weightA, valueAudio], normalize=False, axes=-1,name='attentionA')
        
        poolAttAD = AveragePooling1D(512, padding='same')(attentionAD)
        poolAttA = AveragePooling1D(512, padding='same')(attentionA)
        
        poolqueryImage = AveragePooling1D(512, padding='same')(queryImage)
        
        outAttImage = Concatenate(axis=-1)([poolAttA,poolAttAD, poolqueryImage])
        outAttImage = Reshape([1536],name='reshape_out_attImage') (outAttImage)
        
        # combining audio and textual channels 
        
        A_e = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(outAttAudio)
        I_e = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_textual')(outAttImage)
        
        
        if self.loss == "Triplet":
            mapIA = keras.layers.dot([I_e,A_e],axes=-1,normalize = True,name='dot_final')
            final_model = Model(inputs=[textual_sequence, speech_sequence], outputs = mapIA , name='vgs_model')
            
        elif self.loss == "MMS":
            s_output = Concatenate(axis=1)([Reshape([1 , I_e.shape[1]])(I_e) ,  Reshape([1 ,A_e.shape[1]])(A_e)])
            final_model = Model(inputs=[textual_sequence,  speech_sequence], outputs = s_output )
    
        return final_model, textual_embedding_model, audio_embedding_model 
         
    def CNN0 (self, input_dim):
    
        #input_dim = [(1024, 64), (1,768)]
        audio_sequence , out_audio_channel , audio_model = self.build_baseline_audio_model  ( input_dim)
        textual_sequence , out_textual_channel , textual_model = self. build_baseline_textual_model ( input_dim)  
        
        
        T_e =  out_textual_channel #AveragePooling1D(64,padding='same') (out_textual_channel)
        #T = Reshape([out_textual_channel.shape[2]])(T) # (N, 512)
        
        A_e = out_audio_channel
        A_e = AveragePooling1D(256,padding='same') (A_e)
        A_e = Reshape([A_e.shape[2]])(A_e) # (N, 512)   
        #A_e = Dense(512,activation='linear',name='dense_audio')(A_e)       
        A_e = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(A_e)
        
        T_e = Dense(256,activation='linear',name='dense_textual')(T_e) 
        T_e = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_textual')(T_e)
        
        textual_embedding_model = Model(inputs=textual_sequence, outputs = T_e, name='textual_embedding_model')
        audio_embedding_model = Model(inputs= audio_sequence, outputs = A_e, name='audio_embedding_model')  
        
        if self.loss == "Triplet":
            # mapIA = keras.layers.dot([I,A],axes=-1,normalize = True,name='dot_matchmap') 
            # def final_layer(tensor):
            #     x= tensor 
            #     score = K.mean( (K.mean(x, axis=1)), axis=-1)
            #     output_score = Reshape([1],name='reshape_final')(score)          
            #     return output_score
            # lambda_layer = Lambda(final_layer, name="final_layer")(mapIA)
            # final_model = Model(inputs=[textual_sequence, speech_sequence], outputs = lambda_layer )
            
            mapTA = dot([T_e,A_e],axes=-1,normalize = True,name='dot_matchmap')       
            final_model = Model(inputs=[textual_sequence, audio_sequence], outputs = mapTA )
            
        elif self.loss == "MMS":
            s_output = Concatenate(axis=1)([Reshape([1 , T_e.shape[1]])(T_e) ,  Reshape([1 ,A_e.shape[1]])(A_e)])
            final_model = Model(inputs=[textual_sequence,  audio_sequence], outputs = s_output )
        return final_model, textual_embedding_model, audio_embedding_model
  
        return  final_model, textual_embedding_model, audio_embedding_model   
          
    def build_model (self, model_name, input_dim): 
            
        if model_name == 'CNN0':
            final_model, textual_embedding_model, audio_embedding_model = self.CNN0(input_dim)
        elif model_name == 'CNNatt':
            final_model, textual_embedding_model, audio_embedding_model = self.CNNatt( input_dim)
    
        return final_model, textual_embedding_model, audio_embedding_model         
    

    
     


