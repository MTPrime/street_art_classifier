import scipy
import glob
from skimage import color, transform, restoration, io, feature, filters
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np 

np.set_printoptions(suppress=True)

def format_image(file):
    img = io.imread(file)
    out = transform.resize(img, (100, 100))
    return out

def create_pickled_np_array(folders, dir_path='data/img_for_auto/', name='training_img.pkl'):
    array = []
    for folder in folders:
        for file in glob.glob(dir_path + folder + "/*"):
            formatted = format_image(file)
            if formatted.shape == (100,100,3):
                array.append(formatted)

    with open(name, 'wb') as f:
        pickle.dump(array, f)
        print(name)
    return np.array(array)
if __name__ == '__main__':
    folders = ['3d', 'abstract', 'brush', 'bubble', 'cartoon', 'realistic', 'stencil', 'wildstyle']

    # image_array= create_pickled_np_array(folders)

    with open('training_img.pkl', 'rb') as f:
        image_array = pickle.load(f)
    image_array = np.array(image_array)