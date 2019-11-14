import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io, transform
from sklearn.metrics.pairwise import cosine_similarity

from tensorflow import keras
from tensorflow.keras.models import load_model


def format_image(file_path):
    """
    Reads in an image from a file path and resizes it to be 100 x 100.
    Returns formatted image of the correct shape.
    """
    img = io.imread(file_path)
    out = transform.resize(img, (100, 100), anti_aliasing=True)
    return out

def process_new_image(img, autoencoder):
    formatted = format_image(img)
    encoding = autoencoder.get_layer('encoder').predict(formatted.reshape(1, 100,100,3))
    return encoding

def make_recommendations(img_file_path, autoencoder, df_filepath='data/encoded_dataframe.csv'):
    df = pd.read_csv(df_filepath)
    files = df.pop('file_path')
    encoded_np = df.to_numpy()
    encoded_img = process_new_image(img_file_path, autoencoder)
    recommendations = cosine_similarity(encoded_img.reshape(1,-1), encoded_np)
    sorted_recommendations = np.argsort(recommendations)

    #Slice reverses the array to put the most similar at the start and then takes the top 3.
    top_3 = sorted_recommendations[0][:-4:-1] 
    #Retrieve the file path for 10 closest images
    file_paths = files.iloc[top_3]
    recommended_images = [format_image(i) for i in file_paths]
    return recommended_images
    # fig, axs = plt.subplots(1,5, figsize=(20,20))
    # fig.suptitle("Recommended Images", fontsize=36, y=.63)
    # plt.tight_layout()
    # for i, ax in enumerate(axs.flat):
    #     ax.grid(False)
    #     ax.axis('off')
    #     ax.imshow(recommended_images[i])
    # plt.show()

def classify_new_image(img, model):
    """
    Used to classify new images.
    """

    formatted_img = format_image(img)
    yhat = model.predict(formatted_img.reshape(-1,100,100,3))
    out = np.around(yhat[0],2)
    return out
    # for i in range(len(class_names)):
    #     print("{} : {:.2f}".format(class_names[i].capitalize(), yhat[0][i]))
    # raw_img = io.imread(img)
    # fig, ax = plt.subplots()
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.imshow(raw_img)

if __name__ == '__main__':
    pass
    # encoder_model='data/best_encoder_decoder.h5'
    # classifier_model='data/5_class_model_best.h5'
    # classifier = load_model(classifier_model)
    # autoencoder = load_model(encoder_model) 
