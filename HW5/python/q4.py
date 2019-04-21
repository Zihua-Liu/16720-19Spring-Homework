import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    denoise_image = skimage.restoration.denoise_bilateral(image, multichannel = True)
    greyscale_image = skimage.color.rgb2gray(image)
    threshold = skimage.filters.threshold_otsu(greyscale_image)
    bw = greyscale_image < threshold
    bw = skimage.morphology.closing(bw, skimage.morphology.square(5))
    label_image = skimage.morphology.label(bw, connectivity = 2)
    props = skimage.measure.regionprops(label_image)
    mean_size = sum([prop.area for prop in props]) / len(props)
    bboxes = [prop.bbox for prop in props if prop.area > mean_size / 3]
    bw = (~bw).astype(np.float)
    return bboxes, bw