from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import pickle
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import callbacks
from tensorflow import keras
from tensorflow.keras.models import load_model
import pandas as pd
import os,sys
# sys.path.append(os.path.abspath('..'))
from src.pickling_data_for_auto import format_image
from sklearn.metrics.pairwise import cosine_similarity
from src.finding_dominate_colors import resize
import matplotlib.pyplot as plt

def generate_encoded_dataframe(
                            pickled_file='data/training_img.pkl', 
                            saved_model='best_encoder_decoder.h5', 
                            file_paths='data/file_paths_full.csv', 
                            savename='encoded_dataframe.csv'):

    autoencoder = load_model(saved_model)

    # Import data
    with open(pickled_file, 'rb') as f:
        image_array = pickle.load(f)
    train_data = np.array(image_array)
    df = pd.read_csv(file_paths)

    #Accesses the encoder model inside the autoencoder model. Then runs a prediction
    predictions = autoencoder.get_layer('encoder').predict(train_data)
    
    final_df = pd.DataFrame(predictions) 
    final_df['file_path'] = df['file_path']

    final_df.to_csv(savename, header=True, index=None)
    return final_df

def process_new_image(img, saved_model='models/best_encoder_decoder.h5'):
    formatted = format_image(img)
    autoencoder = load_model(saved_model) 
    encoding = autoencoder.get_layer('encoder').predict(formatted.reshape(1, 100,100,3))
    return encoding

def make_recommendations(img_file_path, df_filepath='encoded_dataframe.csv'):
    df = pd.read_csv(df_filepath)
    files = df.pop('file_path')
    encoded_np = df.to_numpy()
    encoded_img = process_new_image(img_file_path)
    recommendations = cosine_similarity(encoded_img.reshape(1,-1), encoded_np)
    sorted_recommendations = np.argsort(recommendations)
    #Slice reverses the array to put the most similar at the start and then takes the top 5. Use [-2:-7:-1] if the image is in the training data
    top_5 = sorted_recommendations[0][:-6:-1] 
    #Retrieve the file path for 10 closest images
    file_paths = files.iloc[top_5]
    recommended_images = [resize(i) for i in file_paths]

    
    fig, axs = plt.subplots(1,5, figsize=(20,20))
    fig.suptitle("Recommended Images", fontsize=36, y=.63)
    plt.tight_layout()
    for i, ax in enumerate(axs.flat):
        ax.grid(False)
        ax.axis('off')
        ax.imshow(recommended_images[i])
    plt.show()

if __name__ == '__main__':
    
    generate_encoded_dataframe(
                            pickled_file='data/training_img.pkl', 
                            saved_model='best_encoder_decoder.h5', 
                            file_paths='data/file_paths_full.csv', 
                            savename='encoded_dataframe.csv')

    img_file_path = 'images/my_images/cartoon_meowl.JPG'
    make_recommendations(img_file_path, df_filepath='encoded_dataframe.csv')
