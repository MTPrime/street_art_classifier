from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
import matplotlib.pyplot as plt
from skimage import color, transform, restoration, io, feature, filters
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

#Needs to be run in tensorflow docker from notebook. The code below changes the file path to allow for this.

import os,sys
sys.path.append(os.path.abspath('..'))
from src.street_art_cnn import create_data_generators

def plot_confusion_matrix(generator, y_pred, class_labels):
    """
        Plots each class in a confusion matrix using Seaborn.
    """
    cm = confusion_matrix(generator.classes, y_pred)
    sns.set(font_scale=2.5)
    fig, ax = plt.subplots(figsize=(15,15))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(class_labels); 
    ax.yaxis.set_ticklabels(class_labels);

    plt.show()

def calculate_y_correct(model, generator):
    """
    Takes a model and a generator and calculates the various variables needed to pull out incorrect images,
    calculate accuracies, and plot confusion matrixes
    OUTPUTS:
        y - class labels of data.
        yhat - predicted class probabilities
        y_correct - Indexes of predictions that were correct. Used for masking
        y_incorrect - Indexes of predictions that were incorrect. Used for masking
        y_pred - class labels of predictions
    """

    #Getting y and yhat. Reseting generator to ensure consistent indexes
    generator.reset()
    y = generator.labels
    generator.reset()
    yhat = model.predict_generator(generator)

    #Changes yhat to class labels
    yhat_clean = np.zeros_like(yhat)
    yhat_clean[np.arange(len(yhat)), yhat.argmax(1)] = 1

    y_correct = []
    y_incorrect = []
    for i, v in enumerate(yhat_clean):
        
        if np.argmax(v) == y[i]:
            y_correct.append(True)
            y_incorrect.append(False)
            
        else:
            y_correct.append(False)
            y_incorrect.append(True)
    
    y_pred = np.argmax(yhat, axis=1)

    return y, yhat, y_correct, y_incorrect, y_pred

def display_image_in_actual_size(im_path):

    dpi = 80
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

def plot_incorrect(generator, yhat,y,  y_incorrect, indx, num_classes, image_size=(150,150)):
    generator.reset()
    class_names = list(generator.class_indices.keys())
    class_dict = {v: k for k, v in generator.class_indices.items()}

    for i in range(num_classes):
        print(class_names[i].capitalize() + ': ' + str(yhat[y_incorrect][indx][i]))
    actual = y[y_incorrect][indx]
    print("Actual - " + class_dict[actual].capitalize())
    
    files = np.asarray(generator.filepaths)
    img = io.imread(files[y_incorrect][indx])
    img = transform.resize(img, image_size)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show
    plt.imshow(img)

    #Displays the full image
    display_image_in_actual_size(files[y_incorrect][indx])

if __name__ =='__main__':
    batch_size = 16
    img_rows, img_cols = 150, 150

    train_generator, test_generator, val_generator = create_data_generators(directory_path='../data/train_test_split/', 
                                                                        input_shape=(img_rows,img_cols), 
                                                                        batch_size=batch_size)

    model = load_model('../models/3_epoch_model_150_87.h5')

    y, yhat, y_correct, y_incorrect, y_pred = calculate_y_correct(model, val_generator)

    plot_incorrect(val_generator, yhat,y, y_incorrect, indx=75, num_classes=2, image_size=(img_rows,img_cols))
    
    class_labels = ['3D', 'Brush', 'Cartoon', 'Realistic', 'Wildstyle']
    plot_confusion_matrix(val_generator, y_pred, class_labels)