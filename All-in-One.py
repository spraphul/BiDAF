import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

class SimilarityMatrix(keras.layers.Layer):
    
    def __init__(self,dims, **kwargs):
        self.dims = dims
        super(SimilarityMatrix, self).__init__(**kwargs)
    
    def similarity(self, context, query):
        e = context*query
        c = K.concatenate([context, query, e], axis=-1)
        dot = K.squeeze(K.dot(c, self.W), axis=-1)
        return keras.activations.linear(dot + self.b)
    
    def build(self, input_shape):
        dimension = 3*self.dims
        self.W = self.add_weight(name='Weights',
                                shape=(dimension,1),
                                initializer='uniform',
                                trainable=True)
        
        self.b = self.add_weight(name='Biases',
                                shape=(),
                                initializer='ones',
                                trainable =True)
        
        super(SimilarityMatrix, self).build(input_shape)
        
    def call(self, inputs):
        C, Q = inputs
        C_len = K.shape(C)[1]
        Q_len = K.shape(Q)[1]
        C_rep = K.concatenate([[1,1],[Q_len],[1]], 0)
        Q_rep = K.concatenate([[1],[C_len],[1,1]],0)
        C_repv = K.tile(K.expand_dims(C, axis=2),C_rep)
        Q_repv = K.tile(K.expand_dims(Q, axis=1), Q_rep)
        
        return self.similarity(C_repv, Q_repv)
    
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        C_len = input_shape[0][1]
        Q_len = input_shape[1][1]
        return (batch_size, C_len, Q_len)
    
    def get_config(self):
        cofig = super().get_config()
        return config


class Context2QueryAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Context2QueryAttention, self).__init__(**kwargs)
        
    
    def build(self, input_shape):
        super(Context2QueryAttention, self).build(input_shape)
    
    def call(self, inputs):
        mat,query = inputs
        attention = keras.layers.Softmax()(mat)
        return K.sum(K.dot(attention, query), -2)
    
    def compute_output_shape(self,input_shape):
        mat_shape, query_shape = input_shape
        return K.concatenate([mat_shape[:-1],query_shape[-1:]])
    
    def get_config(self):
        config = super().get_config()
        return config
        




class Query2ContextAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Query2ContextAttention, self).__init__(**kwargs)
        
    
    def build(self, input_shape):
        super(Query2ContextAttention, self).build(input_shape)
    
    def call(self, inputs):
        mat,context = inputs
        attention = keras.layers.Softmax()(K.max(mat, axis=-1))
        prot = K.expand_dims(K.sum(K.dot(attention,context),-2),1)
        final = K.tile(prot, [1,K.shape(mat)[1],1])
        return final
    
    def compute_output_shape(self,input_shape):
        mat_shape, cont_shape = input_shape
        return K.concatenate([mat_shape[:-1],cont_shape[-1:]])
    
    def get_config(self):
        config = super().get_config()
        return config
        
        
 
 
 class MegaMerge(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MegaMerge, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(MegaMerge, self).build(input_shape)
    
    def call(self, inputs):
        context, C2Q, Q2C = inputs
        CC2Q = context*C2Q
        CQ2C = context*Q2C
        final = K.concatenate([context, C2Q, CC2Q, CQ2C], axis=-1)
        return final
    
    def compute_output_shape(self, input_shape):
        C_shape,_,_ = input_shape
        return K.concatenate([C_shape[:-1], 4*C_shape[-1:]])
    
    def get_config(self):
        config = super().get_config()
        return config
        
 
 
 
 class HighwayLSTMs(keras.layers.Layer):
    def __init__(self, dims, **kwargs):
        self.dims = dims
        super(HighwayLSTMs, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.LSTM = keras.layers.Bidirectional(keras.layers.LSTM(self.dims, return_sequences=True))
        super(HighwayLSTMs, self).build(input_shape)
    
    def call(self, inputs):
        h = self.LSTM(inputs)
        flat_inp = keras.layers.Flatten()(inputs)
        trans_prob = keras.layers.Dense(1, activation='softmax')(flat_inp)
        trans_prob = K.tile(trans_prob, [1,2*self.dims])
        trans_prob = keras.layers.RepeatVector(K.shape(inputs)[-2])(trans_prob)
        out = h + trans_prob*inputs
        return out
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        return config
