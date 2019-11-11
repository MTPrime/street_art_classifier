# Here is a brief overview of the files in this folder.

### Data Collecting
1. art_collecting.py - Scrapes the fatcap.com website using beautiful soup and saves the images as well as their meta data. There are three main functions in this file. One to get all the url links for the images, one to scrape the individual image pages and one to save the images.

### Data Processing
1. folder_splitting.py - Takes a folder where images are stored in subfolders by type and splits the files into a train, test, split folder. Run before image_processing.py

2. image_processing.py - This file will balance the classes found in the train/test/split file tree using Keras image augmentation. 

3. pickling_data_for_auto.py - This file collects all the images using glob, converts them to 100x100x3 numpy arrays and then saves the complete numpy array as a pickled file. This speeds up the image import stage of modeling and insures a consistent order since glob is only run one time. (Glob can be inconsistent in how it reads in files) This is important for later, when using the indexes to pull out image titles and metadata.


### Modeling
1. street_art_cnn.py - The cnn model for classifying street art. Running this script creates a new classification model

2. encoder_and_decoder.py - An encoder and decoder model that are combined into a single autoencoder model. Running this script will generate a new model.

3. find_recommendations.py - Uses the autoencoder model and cosine similarity to find similar images to the target image.

### Visualizing
1. plotting_and_visualizing.py - A class for plotting incorrect classifications and confusion matrixes for the CNN. Also functions for plotting average pictures.

2. finding_dominate_colors.py - There are several functions in this script. One finds the average pixel intensity by class and generates an image for each class. Another scipt uses K-Means to find the dominant colors from each image. It then adds those dominant colors together as a single image for each class and takes the K-Means on that single image. This finds the dominant colors by class. 
