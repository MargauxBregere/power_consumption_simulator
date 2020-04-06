from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import os
import time
import numpy as np
import glob
from numpy import zeros
from numpy import ones
import random

tf.keras.backend.set_floatx('float64')
tf.config.experimental_run_functions_eagerly(True)
                                                                                                                                                                                                                                                 

class CVAE(tf.keras.Model):
    def __init__(self, 
                 input_dim=48,                              # dimension of the encoder inputs data (without context) and decoder outputs 
                 latent_dim=4,                              # decoder inputs sampled from N(mu,Sigma), with mu a latent_dim-vector and Sigma a latent_dim*laten_dim matrix
                 cond_dim=0,                                # dimension of the context
                 full_covariance_matrix=False,              # full_covariance_matrix=True if Sigma is a full matrix, and False if it is a diagonal matrix
                 nb_layers_encoder=1,                       # number of hidden layers for the encoder neural network
                 units_layers_encoder=None,                 # hidden layers size for encoder 
                 nb_layers_decoder=1,                       # number of hidden deocoder for the encoder neural network
                 units_layers_decoder=None,                 # hidden layers size for decoder
                 initializer_encoder='zeros',               # weights initialization of encoder 
                 weights_encoder=None,
                 initializer_decoder='zeros',               # weights initialization of decoder
                 weights_decoder=None,
                 activation='relu',                         # neurals activation function
                 optimizer=tf.keras.optimizers.Adam(1e-3), 
                 loss='L1'):                               # distance considered for the reconstruction error 
        
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.full_covariance_matrix = full_covariance_matrix
        self.nb_layers_encoder = nb_layers_encoder
        self.nb_layers_decoder = nb_layers_decoder        
        self.units_layers_encoder = units_layers_encoder
        self.units_layers_decoder = units_layers_decoder
        self.initializer_encoder = initializer_encoder
        self.weights_encoder = weights_encoder
        self.initializer_decoder = initializer_decoder
        self.weights_decoder = weights_decoder
        self.activation = activation 
        self.optimizer = optimizer
        self.loss = loss
        
        if (units_layers_encoder is None) & (full_covariance_matrix == False):
            units_layers_encoder = [2*latent_dim]
        if (units_layers_encoder is None) & (full_covariance_matrix == True):
            units_layers_encoder = [latent_dim + (latent_dim*(latent_dim+1)/2)]
        if units_layers_decoder is None:
            units_layers_decoder = [input_dim]

        # encoder
        self.encoder_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_dim+cond_dim,)),
            ]
        )
        
        # check if the number of layers and their number of units have been correctly given
        if len(units_layers_encoder) != nb_layers_encoder:
            raise Exception('The number of layers of the encoder is not equal to the '
                            'length of the lists of the numbers of units per layer.')
 
        if (units_layers_encoder[nb_layers_encoder-1] != 2*latent_dim) & (full_covariance_matrix is False):
            raise Exception('The last layer must have a number of units equal to twice the latent dimension - to '
                            'encode the mean and the diagonal of the matrix covariance of the latent distribution.')
            
        if (units_layers_encoder[nb_layers_encoder-1] != latent_dim + (latent_dim*(latent_dim+1)/2)) \
                & (full_covariance_matrix is True):
            raise Exception('The last layer of the encoder must have a number of units equal to the latent dimension - '
                            'to encode the mean off the latent distribution - plus (latent_dim*(latent_dim+1)/2) - to'
                            ' encode the covariance matrix of the latent distribution.')

        # Specify the weights of the initial model
        if initializer_encoder == 'specified':
                    
            # Check the dimensions of the specified weights
            for layer in range(nb_layers_encoder):
                if layer not in weights_encoder.keys():
                    raise Exception('Encoder weights must be a dictionary which keys correspond to integers 0 '
                                    '(for first layer),...,nb_layers_encoder -1 (for last layer).')
                    
            if weights_encoder[0].shape != ((input_dim + cond_dim),units_layers_encoder[0]):
                raise Exception('The shape weights specified for the first layer of the encoder must be '
                                '(input_dim + cond_dim),units_layers_encoder[0])')
                
            for layer in range(1, nb_layers_encoder):
                if weights_encoder[layer].shape != (units_layers_encoder[layer-1], units_layers_encoder[layer]):
                    raise Exception('The shape weights specified for the layers of the encoder must be '
                                    '(units_layers_encoder[layer-1],units_layers_encoder[layer])')
                            
            def init_e(shape, dtype=None):
                ker = tf.reshape(weights_encoder[layer], shape=shape)
                return tf.dtypes.cast(ker, tf.float64)
                
        else:
            init_e = initializer_encoder

        for layer in range(nb_layers_encoder):
            # Activation is linear for the last layer and specify in the model definition of the other layers
            if layer == (nb_layers_encoder-1):
                act = None
            else:
                act = activation
            self.encoder_net.add(tf.keras.layers.Dense(units_layers_encoder[layer],
                                                       activation=act, kernel_initializer=init_e)),

        # decoder
        self.decoder_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim+cond_dim,)),
            ]
        )
        # check if the number of layers and their number of units have been correctly given
        if len(units_layers_decoder) != nb_layers_decoder:
            raise Exception('The number of layers of the decoder is not equal to the length of the lists '
                            'of the numbers of units per layer.')
 
        if units_layers_decoder[nb_layers_decoder-1] != input_dim:
            raise Exception('The last layer of decoder must have a number of units equal to the input dimension.')

        # Specify the weights of the initial model
        if initializer_decoder == 'specified':
                    
            # Check the dimensions of the specified weights
            for layer in range(nb_layers_decoder):
                if layer not in weights_decoder.keys():
                    raise Exception('Decoder weights must be a dictionary which keys correspond to integers 0 '
                                    '(for first layer),...,nb_layers_decoder -1 (for last layer).')
            
            if weights_decoder[0].shape != ((latent_dim+cond_dim), units_layers_decoder[0]):
                raise Exception('The shape weights specified for the first layer of the decoder must be '
                                '(latent_dim + cond_dim),units_layers_decoder[0])')
                
            for layer in range(1, nb_layers_decoder):
                if weights_decoder[layer].shape != (units_layers_decoder[layer-1], units_layers_decoder[layer]):
                    raise Exception('The shape weights specified for the layers of the decoder must be '
                                    '(units_layers_decoder[layer-1],units_layers_decoder[layer])')
                            
            def init_d(shape, dtype=None):
                ker = tf.reshape(weights_decoder[layer], shape=shape)
                return tf.dtypes.cast(ker, tf.float64)
                
        else:
            init_d = initializer_decoder

        for layer in range(nb_layers_decoder):
            # Activation is linear for the last layer and specify in the model definition of the other layers
            if layer == (nb_layers_decoder-1):
                act = None
            else:
                act = activation
            self.decoder_net.add(tf.keras.layers.Dense(units_layers_decoder[layer],
                                                       activation=act, kernel_initializer=init_d)),

    # generates outputs for context c (samples eps from N(0,I_d) and decodes it knowing c)
    @tf.function
    def sample(self, eps=None, c=None):
        if eps is None:
            eps = tf.random.normal(shape=(1, self.latent_dim), dtype=tf.dtypes.float64)
        if c is None:
            c = tf.random.uniform(shape=(1, self.cond_dim), dtype=tf.dtypes.float64)
        return self.decode(eps, c)

    # outputs mean and covariance matrix associated with input x and context c
    def encode(self, x, c):
        if self.cond_dim != 0:
            x = tf.concat([x, c], 1)
        if self.full_covariance_matrix is True:
            mean, logscale_v = tf.split(self.encoder_net(x),
                                        num_or_size_splits=[self.latent_dim,
                                                            round(self.latent_dim*(self.latent_dim+1)*.5)], axis=1)
            d = 0
            e = self.latent_dim
            _, logscale, _ = tf.split(logscale_v,
                                      num_or_size_splits=[d, e,
                                                          (round(self.latent_dim*(self.latent_dim+1)*.5)-d-e)], axis=1)
            for i in range(1, self.latent_dim):
                d = d + e
                e = self.latent_dim - i
                _, line_i, _ = tf.split(logscale_v,
                                        num_or_size_splits=[d, e,
                                                            (round(self.latent_dim*(self.latent_dim+1)*.5)-d-e)],
                                        axis=1)
                line_i = tf.concat([tf.zeros(shape=[1, i], dtype=tf.dtypes.float64), line_i], 1)
                logscale = tf.concat([logscale, line_i], 0)
            return mean, logscale
        else:
            mean, logscale = tf.split(self.encoder_net(x), num_or_size_splits=2, axis=1)
            return mean, logscale

        
    # samples data from N(mean, exp(logscale))
    def reparameterize(self, mean, logscale):
        eps = tf.random.normal(shape=mean.shape, dtype=tf.dtypes.float64)
        if self.full_covariance_matrix is True:
            return eps @ tf.exp(logscale) + mean
        else:
            return eps * tf.exp(logscale) + mean

    # decodes 
    def decode(self, z, c):
        if self.cond_dim != 0:
            z = tf.concat([z, c], 1)
        logits = self.decoder_net(z)
        return logits

    # computes gradient and optimizes model
    @tf.function
    def compute_apply_gradients(self, x, c=None, kl_coeff=1):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, c, kl_coeff)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        
    # completes a train step: for all d in df_train_tf, updates model
    @tf.function
    def train_step(self, df_train_tf, cond_train_tf=None, kl_coeff=10):
        for d in range(df_train_tf.shape[0]):
            x = tf.reshape(df_train_tf[d, ].numpy(), shape=[1, self.input_dim])
            if self.cond_dim != 0:
                cond = tf.reshape(cond_train_tf[d, ].numpy(), shape=[1, self.cond_dim])
            else:
                cond = tf.Variable(np.empty((1, self.cond_dim), dtype=np.float64))
            self.compute_apply_gradients(x, cond, kl_coeff)
    
    # trains cvae 
    @tf.function
    def train(self, epochs, df_train_tf, cond_train_tf=None, kl_coeff=10):
        for epoch in range(epochs):
            self.train_step(df_train_tf, cond_train_tf, kl_coeff)
    
    # computed KL divergence between N(mu,Sigma), where mu and Sigma are the encoder outputs for input x and N(0,I_d)
    @tf.function
    def compute_kl_err(self, x, c):
        mean, logscale = self.encode(x, c)
        if self.full_covariance_matrix is True:
            regularization = tfp.distributions.kl_divergence(
                tfp.distributions.MultivariateNormalTriL(loc=mean, scale_tril=tf.exp(logscale)),
                tfp.distributions.MultivariateNormalDiag(loc=zeros(self.latent_dim), scale_diag=ones(self.latent_dim)))
        else:
            regularization = tfp.distributions.kl_divergence(
                tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(logscale)),                
                tfp.distributions.MultivariateNormalDiag(loc=zeros(self.latent_dim), scale_diag=ones(self.latent_dim)))
                
        return regularization
    
    
    # computes loss = sum of KL divergence and reconstruction error
    @tf.function
    def compute_loss(self, x, c, kl_coeff=1):
        mean, logscale = self.encode(x, c)
        z = self.reparameterize(mean, logscale)
        x_hat = self.decode(z, c)
        if self.loss == 'L2':
            mse = tf.multiply((x-x_hat), (x-x_hat))
            reconstructed_err = tf.reduce_sum(mse, axis=[1])
            
        elif self.loss == 'L1':
            mae = tf.math.abs((x-x_hat))
            reconstructed_err = tf.reduce_sum(mae, axis=[1])**2
            
        else:
            raise Exception('Specify reconstruction error loss.')  
        return kl_coeff*self.compute_kl_err(x, c) + reconstructed_err
    
    # computes mean squared error between x and x_hat, where x_hat are decoder outputs for inputs sampled from N(0,I_d)
    @tf.function
    def compute_mse_test(self, x, c):
        z = tf.random.normal(shape=[1,self.latent_dim], dtype=tf.dtypes.float64)
        x_hat = self.decode(z, c)
        reconstruction_err = tf.reduce_mean(tf.multiply((x-x_hat), (x-x_hat)))
        return tf.reduce_mean(100*tf.math.sqrt(reconstruction_err))
    
    # computes mean squared error between x and x_hat, where x_hat are decoder outputs for inputs sampled from N(mu,Sigma)
    # with mu and Sigma the outputs of encoder for input x
    @tf.function
    def compute_mse(self, x, c):
        mean, logscale = self.encode(x, c)
        z = self.reparameterize(mean, logscale)
        x_hat = self.decode(z, c)
        reconstruction_err = tf.reduce_mean(tf.multiply((x-x_hat), (x-x_hat)))
        return tf.reduce_mean(100*tf.math.sqrt(reconstruction_err))

    
    
# generates renormalized (bet ween conso_min and conso _max) sample for a model = decoder neural network (previously trained on date between 0 and 1)
# feeded with noise eps and condition variables c
# GMT is the name of the sample columns

def simulate_cvae(model, GMT, eps, c, conso_min, conso_max):
    z = tf.concat([eps, c], 1) 
    conso = pd.DataFrame(tf.reshape(tf.multiply(model(z),(conso_max-conso_min))+conso_min, shape=[len(GMT), ]).numpy()).transpose()
    conso.columns = GMT
    conso.columns = conso.columns.astype(str)
    return(conso)

