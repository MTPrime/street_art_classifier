import tensorflow as tf
from tensorflow.python.keras.layers import Dropout, Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


if __name__ == '__main__':
        img_rows, img_cols = 64, 64  # the size of the MNIST images KEEP
        input_shape = (img_rows, img_cols, 3)  # 1 channel image input (grayscale) KEEP
        ts = (64,64)
        batch_size = 16


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
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation('sigmoid'))

        model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])


        datagen = ImageDataGenerator(rescale=1./255
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True
        )
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = datagen.flow_from_directory('data/train_test_split/train',  
                                                target_size=ts,
                                                batch_size=batch_size,
                                                class_mode='categorical', 
                                                classes=['realistic', 'wildstyle'])


        test_generator = test_datagen.flow_from_directory('data/train_test_split/test',
                                                target_size=ts,
                                                batch_size=batch_size,
                                                classes=['realistic', 'wildstyle'],
                                                class_mode='categorical')
                                                

        # this is a similar generator, for validation data
        validation_generator = test_datagen.flow_from_directory(
                'data/train_test_split/val',
                target_size=ts,
                batch_size=batch_size,
                classes=['realistic', 'wildstyle'],
                class_mode='categorical')
        #Checkpoint
        filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', mode='max', save_weights_only=True, save_best_only=True, save_freq=1)
        callbacks_list = [checkpoint]

        model.fit_generator(
                train_generator,
                steps_per_epoch=2000,
                epochs=1,
                validation_steps=800,
                validation_data=validation_generator)
        
        model.evaluate_generator(test_generator)
        model.save_weights('bi_class_weights_1_epochs.h5')
        model.save('bi_class_model_1_epochs.h5')