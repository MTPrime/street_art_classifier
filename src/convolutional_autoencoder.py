from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pickle
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import callbacks
from tensorflow import keras
from tensorflow.keras.models import load_model


# class Autoencoder(object):
    
#     def __init__(self):    
        
#         # Encoding
#         input_layer = Input(shape=(100, 100, 3)) 
#         encoding_conv_layer_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
#         encoding_pooling_layer_1 = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_1)
#         encoding_conv_layer_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoding_pooling_layer_1)
#         encoding_pooling_layer_2 = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_2)
#         encoding_conv_layer_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoding_pooling_layer_2)
#         code_layer = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_3)
        
#         # Decoding
#         decodging_conv_layer_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(code_layer)
#         decodging_upsampling_layer_1 = UpSampling2D((2, 2))(decodging_conv_layer_1)
#         decodging_conv_layer_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(decodging_upsampling_layer_1)
#         decodging_upsampling_layer_2 = UpSampling2D((2, 2))(decodging_conv_layer_2)
#         decodging_conv_layer_3 = Conv2D(16, (3, 3), activation='relu')(decodging_upsampling_layer_2)
#         decodging_upsampling_layer_3 = UpSampling2D((2, 2))(decodging_conv_layer_3)
#         output_layer = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decodging_upsampling_layer_3)
        
#         self._model = Model(input_layer, output_layer)
#         self._model.compile(optimizer='adam', loss='mean_squared_error')
        
#     def train(self, input_train, input_test, batch_size, epochs):#, callbacks):    
#         self._model.fit(input_train, 
#                         input_train,
#                         epochs = epochs,
#                         batch_size=batch_size,
#                         shuffle=True,
#                         validation_data=(
#                                 input_test, 
#                                 input_test))#,
#                         #callbacks=callbacks)
        
#     def getEncodedImage(self, image):
#         encoded_image = self._encoder_model.predict(image)
#         return encoded_image

#     def getDecodedImage(self, encoded_imgs):
#         decoded_image = self._model.predict(encoded_imgs)
#         return decoded_image

def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    code_layer = MaxPooling2D((2, 2), padding='same')(conv3)

    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(code_layer) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

if __name__ == '__main__':

    # Import data
    with open('training_img.pkl', 'rb') as f:
        image_array = pickle.load(f)
    train_data = np.array(image_array)

    x_train, train_ground, x_test, test_ground = train_test_split(train_data,
                                                             train_data, 
                                                             test_size=0.2, 
                                                             random_state=42)

    batch_size = 128
    epochs = 2
    inChannel = 3
    x, y = 100, 100
    input_img = Input(shape = (x, y, inChannel))
    autoencoder = Model(input_img, autoencoder(input_img))
    autoencoder.compile(loss='mean_squared_error', optimizer = 'adam')

    print(autoencoder.summary())

    tensorboard = callbacks.TensorBoard(
        log_dir='logdir',
        histogram_freq=0, 
        write_graph=True,
        update_freq='epoch')
    
    savename = 'best_autoencoder_model.h5'

    mc = callbacks.ModelCheckpoint(
        savename,
        monitor='val_accuracy', 
        verbose=0, 
        save_best_only=True, 
        mode='auto', 
        save_freq='epoch')

    autoencoder_train = autoencoder.fit(x_train, x_train, 
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=1,
                                        validation_data=(x_test, x_test),
                                        callbacks=[mc, tensorboard])



    # Convolutional implementation
    # autoencoder = Autoencoder()
    


    # autoencoder.train(x_train, x_test, 50, 2)#, callbacks=[mc, tensorboard])
    # decoded_imgs = autoencoder.getDecodedImage(x_test)

    # # Convolutional implementation results
    # # plt.figure(figsize=(20, 20))
    # # for i in range(10):
    # #     # Original
    # #     subplot = plt.subplot(2, 10, i + 1)
    # #     plt.imshow(x_test[i].reshape(100, 100,3))
    # #     plt.gray()
    # #     subplot.get_xaxis().set_visible(False)
    # #     subplot.get_yaxis().set_visible(False)

    # #     # Reconstruction
    # #     subplot = plt.subplot(2, 10, i + 11)
    # #     plt.imshow(decoded_imgs[i].reshape(100,100,3))
    # #     plt.gray()
    # #     subplot.get_xaxis().set_visible(False)
    # #     subplot.get_yaxis().set_visible(False)
    # # plt.show()
    autoencoder.save_weights('autoencoder_weights.h5')
    autoencoder.save('autoencoder_model.h5')