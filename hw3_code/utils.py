from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage import correlate
import sklearn.cluster
from scipy.spatial.distance import cdist

def computeHistogram(img_file, F, textons):
    ### YOUR CODE HERE
    img = img_as_float(io.imread(img_file))
    img = rgb2gray(img)

    # Apply filter bank to the image
    filtered_images = []
    for filter in F:
        filtered_images.append(correlate(img, filter))

    # Initialize histogram
    hist = np.zeros(textons.shape[0])

    # Flatten and reshape filtered images and textons arrays
    for i, image in enumerate(filtered_images):
        filtered_images[i] = image.flatten()
    
    textons_flat = textons.reshape(textons.shape[0], -1)

    # Compute distances between each filtered image and textons
    for image in filtered_images:
        distances = cdist(image.reshape(1, -1), textons_flat, 'euclidean')
        closest_index = np.argmin(distances)
        hist[closest_index] += 1

    return hist
    ### END YOUR CODE
    
def createTextons(F, file_list, K):

    ### YOUR CODE HERE
    features = []
    
    # Apply filter bank to the images
    for file in file_list:
        img = img_as_float(io.imread(file))
        img = rgb2gray(img)
        
        for filter in F:
            img = correlate(img, filter)
            features.append(img.flatten())
            
    features = np.array(features)
    
    # Cluster the features to get the textons
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(features)
    
    # Get the textons from the cluster centers
    textons = kmeans.cluster_centers_
    
    return textons
    ### END YOUR CODE
