import tensorflow as tf
from tensorflow.python.keras.layers import Dropout, Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    img_rows, img_cols = 150, 150  # the size of the MNIST images KEEP
    input_shape = (img_rows, img_cols, 3)  # 1 channel image input (grayscale) KEEP
    ts = (64,64)

    #Image Processing
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
    batch_size = 16
    
    datagen = ImageDataGenerator(rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    train_generator = datagen.flow_from_directory('data/train_test_split/train',  
                                                target_size=ts,
                                                batch_size=batch_size,
                                                class_mode='categorical')
    

    test_generator = datagen.flow_from_directory('data/train_test_split/test',
                                                target_size=ts,
                                                batch_size=batch_size,
                                                class_mode='categorical')
                                                

    # this is a similar generator, for validation data
    validation_generator = datagen.flow_from_directory(
            'data/train_test_split/val',
            target_size=ts,
            batch_size=batch_size,
            class_mode='categorical')

    
    model.fit_generator(
            train_generator,
            steps_per_epoch=None,
            epochs=1,
            validation_data=validation_generator)
