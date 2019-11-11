from skimage import color, transform, restoration, io, feature, filters
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np 
import glob 
import argparse

def resize(file):
    img = io.imread(file)
    out = transform.resize(img, (100, 100), anti_aliasing=True)
    return out
    
def dominant_colors_by_classes(folder, num_colors=3):
    """
    Gets the X number of dominant colors from every image in a folder through KMeans Clustering.
    Appends all the colors into a single np array in order to create an average image of dominant colors
    for a class of images.
    """

    picture_list = [resize(file) for file in glob.glob(folder +'/*')]

    for i, img in enumerate(picture_list):
        try:
            nrow, ncol, _ = img.shape 
            lst_of_pixels = [img[irow][icol] for irow in range(nrow) for icol in range(ncol)]

            kmeans_img = KMeans(n_clusters=num_colors).fit(lst_of_pixels)  # looking for the 3 dominant colors
            cluster_centers = kmeans_img.cluster_centers_ 
            if i ==0:
                average_img = cluster_centers
            else:
                average_img = np.append(average_img,cluster_centers, axis=0)
        except:
             continue
    return average_img

def save_avg_dominate_color(arr, img_col, img_row, name):
    """
    Graphs and saves the average dominant color image.
    """
    fig, ax = plt.subplots(figsize=(20,20), frameon=False)
    ax.grid(False)
    ax.axis('off')
    plt.imshow(arr.reshape(img_col, img_row, 3))
    plt.tight_layout()
    plt.savefig(name,bbox_inches = 'tight',pad_inches=0.0)

def resize_and_recolor(file):
    img = io.imread(file)
    out = transform.resize(img, (100, 100), anti_aliasing=True)
    out = color.rgb2gray(out)
    return out

def pixel_intensity_by_class(folders, savename='pixel_intensity_by_class_test.jpg', img_dir = '../data/img/'):
    """
    Creates an average pixel intensity image for each class and graphs them together.
    """

    avg_image = []
    for folder in folders:
        picture_list = [resize_and_recolor(file) for file in glob.glob(img_dir + folder +'/*')]
        array = np.array(picture_list)
        avg_image.append(np.mean(array, axis=0).reshape(100,100))
        
    fig, axs = plt.subplots(1,5, figsize=(20,20))
    fig.suptitle("Pixel Intensity", fontsize=36, y=.63)
    plt.tight_layout()
    for i, ax in enumerate(axs.flat):
        ax.grid(False)
        ax.axis('off')
        ax.imshow(avg_image[i], cmap='ocean')
        ax.set_title(folders[i].capitalize(), fontsize=24)
    plt.savefig(savename)

def color_palette_by_class(classes, savename):
    """
    Takes each class's composite photo of dominate colors and graphs the top 10 most dominate.
    """

    colors = []
    for file in classes:
        path = '../images/class_colors/10_dominate_colors/' + file + '.jpg' 
        
        img = io.imread(path)
        img_resized = transform.resize(img, (300,300))
        
        #Creates a list of the image's pixels. Each item in the list is an RGB value for a pixel
        pixel_list = [img_resized[irow][icol] for irow in range(nrow) for icol in range(ncol)]

        kmeans_img = KMeans(n_clusters=10).fit(pixel_list)  # looking for the 3 dominant colors
        img_cluster_centers = kmeans_img.cluster_centers_ 
        colors.append(img_cluster_centers.reshape(1,10,3))

    fig, axs = plt.subplots(1,5, figsize=(20,20))
    fig.suptitle("Colors by Classes", fontsize=36, y=.6)
    plt.tight_layout()
    for i, ax in enumerate(axs.flat):
        ax.grid(False)
        ax.axis('off')
        ax.imshow(colors[i])
        ax.set_title(classes[i].capitalize(), fontsize=24)
    plt.savefig(savename)

if __name__ =='__main__':
    #These scripts are intended to be run in a jupyter notebook.
    pass
