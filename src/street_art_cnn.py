from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
# from keras_radam import RAdam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

def create_data_generators(directory_path='data/train_test_split/', input_shape=(64, 64), batch_size=16):
    
    train_path = directory_path +'train'
    test_path = directory_path +'test'
    val_path = directory_path +'val'


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
        class_mode='categorical',
        shuffle=False)
        
    val_generator = test_datagen.flow_from_directory(
        val_path,
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

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

    model.add(Conv2D(nb_filters, 
                    (kernel_size[0], kernel_size[1]), 
                    name='conv-3')) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, name='pool-3'))

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
    batch_size = 20  
    nb_classes = 6   
    nb_epoch = 5              
    img_rows, img_cols = 64, 64  
    input_shape = (img_rows, img_cols, 3)  
    nb_filters = 12  
    pool_size = (2, 2)
    kernel_size = (3, 3) 
    neurons=128

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

    art_model.load_weights('./models/street_art_cnn_weights_86.h5', by_name=True)
    # art_model = load_model('./models/street_art_cnn.h5')
    
    #Checkpoint
    filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', mode='max', save_weights_only=True, save_best_only=True, period=1)
    callbacks_list = [checkpoint]

    
    art_model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=nb_epoch,
            validation_data=val_generator,
            validation_steps=200,
            use_multiprocessing=True)
    
    art_model.save_weights('6_class_weights.h5')
    art_model.save('6_class_model.h5')
