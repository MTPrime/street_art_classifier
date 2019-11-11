from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import pickle
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import callbacks
from tensorflow import keras
from tensorflow.keras.models import load_model

def encoder(input_img, latent_dim=50):
    """
    Encoder model. Encodes a 100x100x3 image down to a vector of 50 latent topics.
    """
    #encoder
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) #100 100 16
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 50 50 16
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1) #50 50 32
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 25 25 32

    flat = Flatten()(pool2)
    latent = Dense(latent_dim, name='latent_vector')(flat)
    


    return latent

def decoder(latent_inputs, latent_dim=50):
    """
    Takes the latent vector from the encoder model and decodes it back to original images. 100x100x3
    """

    dense_layer = Dense((25*25*32))(latent_inputs)
    shaped = Reshape((25,25,32))(dense_layer)

    de_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(shaped) #25 25 32
    up1 = UpSampling2D((2,2))(de_conv1) #50 50 32
    de_conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(up1) #50 50 16
    up2 = UpSampling2D((2,2))(de_conv2) #100 100 
    decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(up2) # 100 100 3
    outputs = Activation('sigmoid', name='decoder_output')(decoded)
    

    return outputs


if __name__ == '__main__':

    #Inputs
    input_shape = (100, 100, 3)
    batch_size = 128
    epochs = 1000
    latent_dim =50
    inputs = Input(shape=input_shape, name='encoder_input')
    latent_inputs  = Input(shape=(latent_dim,), name='decoder_input')

    #Building Encoder
    encoder = Model(inputs, encoder(inputs), name='encoder')
    encoder.summary()

    #Building Decoder
    decoder = Model(latent_inputs, decoder(latent_inputs), name='decoder')
    decoder.summary()

    #Building autoencoder
    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.compile(loss='mse', optimizer='adam')

    print(autoencoder.summary())

    # Import data
    with open('data/training_img.pkl', 'rb') as f:
        image_array = pickle.load(f)
    train_data = np.array(image_array)

    #Train, Test, Split
    x_train, train_ground, x_test, test_ground = train_test_split(train_data,
                                                            train_data, 
                                                            test_size=0.2, 
                                                            random_state=42)
 
    #Tensorboard for model checkpoints
    tensorboard = callbacks.TensorBoard(
        log_dir='logdir',
        histogram_freq=0, 
        write_graph=True,
        update_freq='epoch')
    
    savename = 'best_encoder_decoder.h5'

    mc = callbacks.ModelCheckpoint(
        savename,
        monitor='val_loss', 
        verbose=0, 
        save_best_only=True,
        mode='auto', 
        save_freq='epoch')

    #Train Autoencoder 
    autoencoder.fit(x_train, 
                    x_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    verbose=1, 
                    validation_data=(x_test, x_test),
                    callbacks=[mc, tensorboard])