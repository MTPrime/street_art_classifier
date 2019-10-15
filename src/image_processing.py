from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage import color, transform, restoration, io, feature
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import pandas as pd 

def images_resized(images):
    images_resized = []
    for i in images:
        images_resized.append(transform.resize(i, (300, 300)))

    return images_resized

def featurize_img(img):
    img_resized = (transform.resize(img, (300, 300)))
    img_gray= color.rgb2gray(img_resized)
    img_canny = feature.canny(img_gray, sigma=1)
    img_tv = restoration.denoise_tv_chambolle(img_canny, weight = .8)
    return img_tv

def balance_classes(datagen, class_names):
    """ 
    Will loop through the train test val folders and will create augmented images for the minority class 
    until all classes are balanced. 

    Works, but smells of muffins in my opinion.
    """
    folders = ['test', 'train', 'val']
    image_count_dict = dict()
    for folder in folders:
        for c in class_names:
            directory_name = './data/train_test_split/' + folder + '/' + c
            image_count_dict[c] = len([name for name in os.listdir(directory_name) if os.path.isfile(os.path.join(directory_name, name))])
        
        print(folder)
        print("\n\n\n\n")

        while min(image_count_dict.values()) != max(image_count_dict.values()):
            for c in class_names:
                directory_name = './data/train_test_split/' + folder + '/' + c
                image_count_dict[c] = len([name for name in os.listdir(directory_name) if os.path.isfile(os.path.join(directory_name, name))])
            count = max(image_count_dict.values()) - min(image_count_dict.values())
            minority_class = min(image_count_dict)
            
            max_count = max(image_count_dict.values())
            min_count = min(image_count_dict.values())
            print(max_count)
            print(min_count)

            minority_file_path = './data/train_test_split/' + folder + '/' + minority_class +"/"
            minority_list = [name for name in os.listdir(directory_name) if os.path.isfile(os.path.join(directory_name, name))]

            print(minority_list[44])
            for i in range(count):
                try:
                    image_path = minority_file_path + minority_list[i]
                    print(image_path)
                    print(i)
                    img = io.imread(image_path)  # this is a PIL image
                    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
                    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

                    # file_path_save = 'data/train_test_split/' + folder + "/" + minority_class +"/"
            
                    for batch in datagen.flow(x, batch_size=1,save_to_dir=minority_file_path, save_prefix='altered', save_format='jpg'):
                        break
                except:
                  continue

if __name__ == '__main__':
    

    datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    
    # df_all_images = pd.read_csv("data/meta_data_cleaned.csv", index_col=0)
    # df_images = df_all_images[df_all_images['Style'] == 'Realistic']['File_Path'].reset_index()
    # count = 400
    
    class_names = ['wildstyle', 'realistic']

    balance_classes(datagen,class_names)

    # # simple version for working with CWD
    # directory_name = './data/train_test_split/train/realistic' 
    # test_list= [name for name in os.listdir(directory_name) if os.path.isfile(os.path.join(directory_name, name))]

    # print(len([name for name in os.listdir('./data/train_test_split/train/realistic') if os.path.isfile(os.path.join('./data/train_test_split/train/realistic', name))]))

    # counter = 0
    # for i in os.listdir("data/train_test_split/train/realistic"):
    #     counter +=1
    #     # print(os.path.isfile(i))