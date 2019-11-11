from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D


def create_model(input_size, n_categories):
    """
    Create a simple baseline CNN
    """

    nb_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)

    model = Sequential()
    # 2 convolutional layers followed by a pooling layer followed by dropout
    model.add(Convolution2D(nb_filters, kernel_size,
                            padding='valid',
                            input_shape=input_size))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    # transition to an mlp
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_categories))
    model.add(Activation('softmax'))
    return model

def build_model(input_size, n_categories):
    nb_filters = 12
    pool_size = (2, 2)
    kernel_size = (3, 3)
   
    model = Sequential() 

    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                    input_shape=input_size,
                    name="conv-1")) 

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, name='pool-1'))

    model.add(Convolution2D(nb_filters, 
                    (kernel_size[0], kernel_size[1]), 
                    name='conv-2')) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, name='pool-2'))

    model.add(Convolution2D(nb_filters, 
                    (kernel_size[0], kernel_size[1]), 
                    name='conv-3')) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, name='pool-3'))

    #
    model.add(Flatten())  
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_categories))
    model.add(Activation('softmax'))

    return model
    