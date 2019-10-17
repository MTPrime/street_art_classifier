from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
# from keras_radam import RAdam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

def create_data_generators(directory_path='data/train_test_split/', input_shape=(64, 64), batch_size=16):
    
    train_path = directory_path +'train'
    test_path = directory_path +'test'
    val_path = directory_path +'val'

    # train_datagen = ImageDataGenerator(
    #     rescale=1./255,
    #     rotation_range=6,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     brightness_range=[0.2, 0.8],
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True
    #     )

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
            )
    
    test_datagen = ImageDataGenerator(
        rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical')
        
    val_generator = test_datagen.flow_from_directory(
        val_path,
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, test_generator, val_generator

def build_model(opt='adam', input_shape=(64, 64, 3), nb_classes = 5, neurons = 64, nb_filters = 12, pool_size = (2, 2), kernel_size = (3, 3)):
   
    model = Sequential() 

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                    input_shape=input_shape,
                    name="conv-1")) 

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, name='pool-1'))

    model.add(Conv2D(nb_filters, 
                    (kernel_size[0], kernel_size[1]), 
                    name='conv-2')) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, name='pool-2'))

    

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(neurons))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    if nb_classes == 2:
        model.add(Activation('sigmoid'))
    else:
        model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    batch_size = 10  
    nb_classes = 2   
    nb_epoch = 1    
    img_rows, img_cols = 150, 150  
    input_shape = (img_rows, img_cols, 3)  
    nb_filters = 12  
    pool_size = (2, 2)
    kernel_size = (3, 3) 
    neurons=64

    train_generator, test_generator, val_generator = create_data_generators(directory_path='data/train_test_split/', 
                                                                            input_shape=(img_rows,img_cols), 
                                                                            batch_size=batch_size)

    art_model = build_model(opt='adam', 
                            input_shape=input_shape, 
                            nb_classes = nb_classes, 
                            neurons = neurons, 
                            nb_filters = nb_filters, 
                            pool_size = pool_size, 
                            kernel_size = kernel_size)
    
    art_model.fit_generator(
            train_generator,
            steps_per_epoch=1000,
            epochs=nb_epoch,
            validation_data=val_generator,
            validation_steps=800//batch_size,
            use_multiprocessing=True)
    
    art_model.save_weights('street_art_cnn.h5')
    art_model.save('street_art_cnn.h5')