import numpy as np
import torchvision
import util
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import visual_words
import visual_recog
import skimage.io

if __name__ == '__main__':

    num_cores = util.get_num_CPU()

    # path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    # path_img = "../data/waterfall/sun_abcxnrzizjgcwkdn.jpg"
    # path_img = "../data/park/labelme_gmgfmbtnkpwaugo.jpg"
    # image = skimage.io.imread(path_img)
    # image = image.astype('float') / 255
    # # filter_responses = visual_words.extract_filter_responses(image)
    # util.display_filter_responses(filter_responses)

    # visual_words.compute_dictionary(num_workers=num_cores)

    # dictionary = np.load('dictionary.npy')
    # wordmap = visual_words.get_visual_words(image, dictionary)
    # filename = "wordmap2.jpg"
    # util.save_wordmap(wordmap, filename)
    visual_recog.build_recognition_system(num_workers = num_cores // 2)
    visual_recog.evaluate_recognition_system(num_workers = num_cores // 2)

    # conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    # print(conf)
    # print(np.diag(conf).sum()/conf.sum())



    #visual_recog.get_feature_from_wordmap(wordmap, visual_words.K)
    #visual_recog.get_feature_from_wordmap_SPM(wordmap, 3, visual_words.K)

