import numpy as np
import h5py as h5
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats


from keras.models import Model, load_model
import keras.layers as layers
import keras
from keras import ops
from keras.layers import Dense, LeakyReLU, Lambda, Input, Conv1D, MaxPooling1D, AveragePooling1D, UpSampling1D, Reshape, Flatten,Cropping1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from tensorflow import math as K
import tensorflow as tf
from sklearn.model_selection import train_test_split

original_dim = 119
latent_dim = 6
batch_size = 32
epochs = 10

infile = h5.File("C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\pca\\clumpy_models_201410_tvavg.hdf5", 'r')
data = infile['flux_tor'][:]
data = np.reshape(data, (data.shape[0], 119, 1))
print(data.shape)
x_temp, x_test, _, _ = train_test_split(data, data, test_size=0.05)
x_train, x_valid, _, _ = train_test_split(x_temp,
                                          x_temp,
                                          test_size=0.1)
def vae_loss(input_img, output):
    mean,log_var, z = encoder(input_img)
    reconstruction = decoder(z)
    reconstruction_loss = tf.reduce_sum(tf.square(reconstruction-input_img))
    kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.square(tf.exp(log_var)), axis=-1)
    total_loss = tf.reduce_mean(reconstruction_loss + 10*kl_loss)
    return total_loss

class Sampling(layers.Layer):
    """Uses (mean, log_var) to sample z, the vector encoding a digit."""
 
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(shape=(latent_dim,))
        return mean + tf.exp(0.5 * log_var) * epsilon

checkpointer = ModelCheckpoint('model_checkpoint.keras', verbose=1, save_best_only=True)

# encoder hidden layers

# encoder hidden layers
encoder_inputs = Input(shape=(original_dim,1,))
x = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv1D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
mean = layers.Dense(latent_dim, name="mean")(x)
log_var = layers.Dense(latent_dim, name="log_var")(x)
z = Sampling()([mean, log_var])
encoder = keras.Model(encoder_inputs, [mean, log_var, z], name="encoder")

x = layers.Dense(30*128, activation="relu")(z)
x = layers.Reshape((30, 128))(x)
x = layers.Conv1DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv1DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder_outputs = layers.Cropping1D((0,1))(decoder_outputs)


vae = Model(inputs=encoder_inputs, outputs=decoder_outputs)
vae.summary()
# create a placeholder for an encoded input
encoded_input = Input(shape=(latent_dim,))
# retrieve the last layers of the autoencoder model
decoded_output = vae.layers[-6](encoded_input)
decoded_output = vae.layers[-5](decoded_output)
decoded_output = vae.layers[-4](decoded_output)
decoded_output = vae.layers[-3](decoded_output)
decoded_output = vae.layers[-2](decoded_output)
decoded_output = vae.layers[-1](decoded_output)
# create the decoder model
decoder = Model(inputs=encoded_input, outputs=decoded_output, name='decoder')
decoder.summary()
encoder.summary()

vae.compile(optimizer=keras.optimizers.Adam(), loss=vae_loss, metrics=['mse'])
results = vae.fit(x_train, x_train,
                      shuffle=True,
                      batch_size = batch_size, 
                      epochs = epochs,
                      validation_data = (x_valid,x_valid),
                      callbacks=checkpointer)

vae.save("C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\autoencoder\\model.keras")
encoder.save("C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\autoencoder\\model_encoder.keras")
decoder.save("C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\autoencoder\\model_decoder.keras")