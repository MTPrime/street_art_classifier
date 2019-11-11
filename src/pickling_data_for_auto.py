import scipy
import glob
from skimage import color, transform, restoration, io, feature, filters
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np 
import pandas as pd

#Surpresses Scientific Notation in Numpy
np.set_printoptions(suppress=True)

def format_image(file):
    """
    Reads in an image from a file path and resizes it to be 100 x 100.
    Returns formatted image
    """
    img = io.imread(file)
    out = transform.resize(img, (100, 100))
    return out

def create_pickled_np_array(folders, dir_path='data/img_for_auto/', name='training_img.pkl'):
    """
    Reads in all the image files in a series of folders and converts them to a single numpy array, which is then pickled.
    Numpy array is also returned to work with immediately.

    Inputs:
        folders: (list) folders where images are being pulled from. ie 'Wildstyle'
        dir_path: (str) directory path where the folders are located
        name: (str) savename for the final pickle file

    Output:
        array: (np array) Array of formatted images
        file_list: (list) List of file paths for each image. Used to look up original images later from index. 

    """
    array = []
    file_list = []
    for folder in folders:
        for file in glob.glob(dir_path + folder + "/*"):
            formatted = format_image(file)
            if formatted.shape == (100,100,3):
                file_list.append(file)
                array.append(formatted)

    with open(name, 'wb') as f:
        pickle.dump(array, f)
        print(name)
    return np.array(array), file_list

if __name__ == '__main__':
    folders = ['3d', 'abstract', 'brush', 'bubble', 'cartoon', 'realistic', 'stencil', 'wildstyle']

    image_array, file_list = create_pickled_np_array(folders)

    #Saving the file paths with the same index as the pickle file. Glob reads in randomly, so this is necessary to be consistent.
    df = pd.DataFrame(file_list, columns=['file_path'])
    df.to_csv('file_paths_full.csv', header=True)

    