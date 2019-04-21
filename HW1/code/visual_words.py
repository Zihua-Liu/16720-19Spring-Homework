import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import util
import random

image_dicts_path = "./image_dicts/"
K = 200

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    if len(image.shape) == 2:
        image = np.tile(image[:, newaxis], (1, 1, 3))

    if image.shape[2] == 4:
        image = image[:,:,0:3]

    image = skimage.color.rgb2lab(image)

    scales = [1,2,4,8,8*np.sqrt(2)]
    for i in range(len(scales)):
        for c in range(3):
            #img = skimage.transform.resize(image, (int(ss[0]/scales[i]),int(ss[1]/scales[i])),anti_aliasing=True)
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i])
            if i == 0 and c == 0:
                imgs = img[:,:,np.newaxis]
            else:
                imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scales[i])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[0,1])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[1,0])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)

    return imgs

def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    filter_response = extract_filter_responses(image)
    H, W = filter_response.shape[0], filter_response.shape[1]
    filter_response = filter_response.reshape(H * W, -1)
    dists = scipy.spatial.distance.cdist(filter_response, dictionary)
    wordmap = np.argmin(dists, axis = 1).reshape(H, W)
    return wordmap



def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''
    i, alpha, image_path = args

    out_path = os.path.join(image_dicts_path, "{}.npy".format(i))

    if os.path.exists(out_path):
        print("{}.npy Already Exists!".format(i))
        return

    image = skimage.io.imread(image_path)
    image = image.astype('float') / 255
    filter_response = extract_filter_responses(image)
    random_sample = np.random.choice(filter_response.shape[0] * filter_response.shape[1], alpha)
    sampled_response = filter_response.reshape(filter_response.shape[0] * filter_response.shape[1], -1)[random_sample, :]
    np.save(out_path, sampled_response)
    print("Worker {} Finished!".format(i))
    return
    


def compute_dictionary(num_workers = 2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''
    if os.path.exists("dictionary.npy"):
        print("Dictionary Already Exists!")
        return

    train_data = np.load("../data/train_data.npz")
    files = train_data['files']
    labels = train_data['labels']

    alpha = 300

    pool = multiprocessing.Pool()

    args_list = []
    for i, (file, label) in enumerate(zip(files, labels)):
        args_list.append((i, alpha, os.path.join("../data", file)))

    pool.map(compute_dictionary_one_image, args_list)
    print("-" * 50)
    

    filter_responses = []
    for i, (file, label) in enumerate(zip(files, labels)):
        filter_responses.append(np.load(os.path.join(image_dicts_path, "{}.npy".format(i))))
    filter_responses = np.array(filter_responses).reshape(-1, 60)
    print("Number of Points: {} | Dimensions: {}".format(filter_responses.shape[0], filter_responses.shape[1]))
    print("Running K Means...")
    kmeans = sklearn.cluster.KMeans(n_clusters = K, n_jobs = 4).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    print("Finished!")
    np.save("dictionary.npy", dictionary)
    print("Dictionary saved as dictionary.npy")
    return


