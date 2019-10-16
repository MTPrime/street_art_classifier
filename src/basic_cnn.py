import numpy as np
np.random.seed(1337)  # for reproducibility

import os 
os.environ['TF_KERAS'] = '1'

# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.utils import to_categorical
# # from keras_radam import RAdam
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import load_model


import tensorflow as tf
from tensorflow.python.keras.layers import Dropout, Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# def load_and_featurize_data():
#     # the data, shuffled and split between train and test sets
#     (X_train, y_train), (X_test, y_test) = mnist.load_data()

#     # reshape input into format Conv2D layer likes
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

#     # don't change conversion or normalization
#     X_train = X_train.astype('float32') # data was uint8 [0-255]
#     X_test = X_test.astype('float32')  # data was uint8 [0-255]
#     X_train /= 255 # normalizing (scaling from 0 to 1)
#     X_test /= 255  # normalizing (scaling from 0 to 1)

#     print('X_train shape:', X_train.shape)
#     print(X_train.shape[0], 'train samples')
#     print(X_test.shape[0], 'test samples')

#     # convert class vectors to binary class matrices (don't change)
#     Y_train = to_categorical(y_train, nb_classes) # cool
#     Y_test = to_categorical(y_test, nb_classes)   
#     # in Ipython you should compare Y_test to y_test
#     return X_train, X_test, Y_train, Y_test

# def define_model(nb_filters, kernel_size, input_shape, pool_size):
#     model = Sequential() # model is a linear stack of layers (don't change)

#     # note: the convolutional layers and dense layers require an activation function
#     # see https://keras.io/activations/
#     # and https://en.wikipedia.org/wiki/Activation_function
#     # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'

#     model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
#                         padding='same', 
#                         input_shape=input_shape)) #first conv. layer  KEEP
#     model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers

#     model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='same')) #2nd conv. layer KEEP
#     model.add(Activation('relu'))

#     model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
#     model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

#     model.add(Flatten()) # necessary to flatten before going into conventional dense layer  KEEP
#     print('Model flattened out to ', model.output_shape)

#     # now start a typical neural network
#     model.add(Dense(32)) # (only) 32 neurons in this layer, really?   KEEP
#     model.add(Activation('relu'))

#     model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

#     model.add(Dense(nb_classes)) # 10 final nodes (one for each class)  KEEP
#     model.add(Activation('softmax')) # softmax at end to pick between classes 0-9 KEEP

#     # model.compile(RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])
        
#     model.compile(loss='categorical_crossentropy',
#                 optimizer='adam',
#                 metrics=['accuracy'])
#     return model

if __name__ == '__main__':
    # # # important inputs to the model: don't changes the ones marked KEEP 
    batch_size = 30  # number of training samples used at a time to update the weights
    nb_classes = 2   # number of output possibilites: [0 - 9] KEEP
    nb_epoch = 10    # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 64, 64  # the size of the MNIST images KEEP
    input_shape = (img_rows, img_cols, 3)  # 1 channel image input (grayscale) KEEP
    nb_filters = 12  # number of convolutional filters to use
    pool_size = (2, 2) #2,2 pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (3, 3) # convolutional kernel size, slides over image to learn features

    # model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='same')) #2nd conv. layer KEEP

    # X_train, X_test, Y_train, Y_test = load_and_featurize_data()
    
    
    # model = define_model(nb_filters, kernel_size, input_shape, pool_size)
    
    # # during fit process watch train and test error simultaneously
    # model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
    #         verbose=1, validation_data=(X_test, Y_test))

    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1]) # this is the one we care about

    


    # model = define_model(nb_filters, kernel_size, input_shape, pool_size)    

    # Image Processesing
    
    model = Sequential()
    model.add(Conv2D(12, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(12, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(12, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    batch_size = 10
    
    datagen = ImageDataGenerator()

    train_generator = datagen.flow_from_directory('data/train_test_split/train',  
                                                target_size=(64, 64),
                                                batch_size=batch_size,
                                                class_mode='categorical')
    

    # train_datagen = ImageDataGenerator(
    #         rescale=1./255,
    #         shear_range=0.2,
    #         zoom_range=0.2,
    #         horizontal_flip=True)

    test_generator = datagen.flow_from_directory('data/train_test_split/test',
                                                target_size=(64,64),
                                                batch_size=batch_size,
                                                class_mode='categorical')
                                                

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    # train_generator = train_datagen.flow_from_directory(
    #         'data/train_test_split/train',  # this is the target directory
    #         target_size=(150, 150),  # all images will be resized to 150x150
    #         batch_size=batch_size,
    #         class_mode='categorical')  

    # this is a similar generator, for validation data
    validation_generator = datagen.flow_from_directory(
            'data/train_test_split/val',
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode='categorical')

    
    model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=3,
            validation_data=validation_generator,
            validation_steps=200)
    
    # model.save_weights('3_epoch_model_weights.h5')
    # model.save('3_epoch_model.h5')
    # model = load_model('models/3_epic_model.hf')