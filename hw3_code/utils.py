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
    filtered_responses = [correlate(img, filt) for filt in F]

    hist = np.zeros(textons.shape[0])

    # Flatten and reshape filtered responses and textons arrays
    for i, response in enumerate(filtered_responses):
        filtered_responses[i] = response.flatten()
    
    textons_flat = textons.reshape(textons.shape[0], -1)

    # Compute distances between each filtered response and textons
    for response in filtered_responses:
        distances = cdist(response.reshape(1, -1), textons_flat, 'sqeuclidean')
        closest_idx = np.argmin(distances)
        hist[closest_idx] += 1

    return hist
    ### END YOUR CODE
    
def createTextons(F, file_list, K):

    ### YOUR CODE HERE
    features = []
    
    for file in file_list:
        img = img_as_float(io.imread(file))
        img = rgb2gray(img)
        
        for filter in F:
            img = correlate(img, filter)
            features.append(img.flatten())
            
    features = np.array(features)
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(features)
    textons = kmeans.cluster_centers_
    
    return textons
    ### END YOUR CODE
