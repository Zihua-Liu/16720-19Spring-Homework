import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import skimage.io
import multiprocessing

def build_recognition_system(num_workers = 2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''
    print("Building System...")
    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    files = train_data['files']
    labels = train_data['labels']

    args_list = []
    for i, (file, label) in enumerate(zip(files, labels)):
        path_img = os.path.join("../data/", file)
        args_list.append((i, path_img, dictionary, 3, visual_words.K))

    pool = multiprocessing.Pool()
    features = np.array(pool.map(call_get_image_feature, args_list)).reshape(-1, visual_words.K * (4 ** 3 - 1) // 3)
    print("-" * 50)
    np.savez("trained_system", dictionary = dictionary, features = features, labels = labels, SPM_layer_num = 3)
    print("Recognition System Build Complete!")


def evaluate_recognition_system(num_workers = 2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''
    print("Evaluating System...")

    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    
    dictionary = trained_system['dictionary']
    trained_features = trained_system['features']
    trained_labels = trained_system['labels']
    SPM_layer_num = trained_system['SPM_layer_num']

    test_images = test_data['files']
    test_labels = test_data['labels']

    args_list = []
    for i, (image_path, label) in enumerate(zip(test_images, test_labels)):
        args_list.append((i, os.path.join("../data", image_path), label, dictionary, trained_features, trained_labels, SPM_layer_num, visual_words.K))
    pool = multiprocessing.Pool()
    result = pool.map(evaluate_image, args_list)
    
    conf = np.zeros((8, 8))

    for item in result:
        pred, label = item['pred'], item['label']
        conf[label][pred] += 1.0
    np.save("conf.npy", conf)
    print(conf)
    acc = np.trace(conf) / np.sum(conf)
    print("Accuracy: {}".format(acc))
    return conf, acc

def evaluate_image(args):
    i, image_path, label, dictionary, trained_features, trained_labels, num_layers, K = args
    image = skimage.io.imread(image_path)
    image = image.astype('float') / 255
    wordmap = visual_words.get_visual_words(image, dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap, num_layers, K)
    similarity = distance_to_set(trained_features, feature)
    pred = trained_labels[np.argmax(similarity)]
    print("Image {} Finished!".format(i))
    return {"pred":pred, "label":label}



def call_get_image_feature(args):
    i, file_path, dictionary, layer_num, K = args
    feature = get_image_feature(file_path, dictionary, layer_num, K)
    print("Image {} Finished!".format(i))
    return feature


def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    

    image = skimage.io.imread(file_path)
    image = image.astype('float') / 255
    wordmap = visual_words.get_visual_words(image, dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
    return feature
    



def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    return np.sum(np.minimum(word_hist, histograms), axis=1)



def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    # ----- TODO -----
    hist, bin_edges = np.histogram(wordmap.reshape(-1), bins = range(dict_size + 1), density = True)
    return hist



def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    
    hist = np.array([])
    H, W = wordmap.shape[0], wordmap.shape[1]
    # for l in range(layer_num):
        # n_cells = 2 ** l
        # h, w = H // n_cells, W // n_cells
        # weights = 2.0 ** (-layer_num + 1) if l in [0, 1] else 2.0 ** (l - layer_num)
        # for i in range(n_cells):
            # for j in range(n_cells):
                # sub_fig = wordmap[i * h:(i + 1) * h, j * w:(j + 1) * w]
                # sub_hist, _ = np.histogram(sub_fig.reshape(-1), bins = range(dict_size + 1))
                # hist = np.concatenate((hist, sub_hist * weights))

    n_cells = 2 ** (layer_num - 1)
    h, w = H // n_cells, W // n_cells
    sub_hists = np.zeros((n_cells, n_cells, dict_size))
    sub_hists_list = [None for l in range(layer_num)]
    for i in range(n_cells):
        for j in range(n_cells):
            sub_fig = wordmap[i * h:(i + 1) * h, j * w:(j + 1) * w]
            sub_hist, _ = np.histogram(sub_fig.reshape(-1), bins = range(dict_size + 1))
            sub_hists[i][j] = sub_hist

    sub_hists_list[-1] = sub_hists
    for l in range(layer_num - 2, -1, -1):
        n_cells = 2 ** l
        h, w = H // n_cells, W // n_cells
        sub_hists = np.zeros((n_cells, n_cells, dict_size))
        for i in range(n_cells):
            for j in range(n_cells):
                sub_hists[i][j] = sub_hists_list[l + 1][i * 2: (i + 1) * 2, j * 2: (j + 1) * 2].reshape(4, -1).sum(axis = 0)
        sub_hists_list[l] = sub_hists

    for l in range(layer_num):
        weights = 2.0 ** (-layer_num + 1) if l in [0, 1] else 2.0 ** (l - layer_num)
        sub_hists = sub_hists_list[l]
        for i in range(len(sub_hists)):
            for j in range(len(sub_hists[i])):
                hist = np.concatenate((hist, sub_hists[i][j] * weights))

    hist = hist / hist.sum()
    return hist







    

