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
    
def dominant_colors_by_classes(folder):
    picture_list = [resize(file) for file in glob.glob(folder +'/*')]

    for i, img in enumerate(picture_list):
        try:
            nrow, ncol, _ = img.shape 
            lst_of_pixels = [img[irow][icol] for irow in range(nrow) for icol in range(ncol)]

            kmeans_img = KMeans(n_clusters=3).fit(lst_of_pixels)  # looking for the 3 dominant colors
            cluster_centers = kmeans_img.cluster_centers_ 
            if i ==0:
                average_img = cluster_centers
            else:
                average_img = np.append(average_img,cluster_centers, axis=0)
        except:
             continue
    return average_img

def save_avg_dominate_color(arr, img_col, img_row, name):
    fig, ax = plt.subplots(figsize=(20,20), frameon=False)
    ax.grid(False)
    ax.axis('off')
    plt.imshow(arr.reshape(img_col, img_row, 3))
    plt.tight_layout()
    plt.savefig(name,bbox_inches = 'tight',pad_inches=0.0)

if __name__ =='__main__':
    pass
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-folder", "--folder", required=True, help="the picture folder")
    # ap.add_argument("-name", "--name", required=True, help="save name")
    # args = vars(ap.parse_args())

    # unsorted_list = np.loadtxt(args['in']) #, dtype=int)
    # x = bubble_sort(unsorted_list)
    # np.savetxt(args['out'], x, fmt='%i')