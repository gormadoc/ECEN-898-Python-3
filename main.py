#!/usr/bin/env python

import sys, getopt, os
import math
import numpy as np
import cv2
import copy
import random
from timeit import default_timer as timer
from utils import *

def main(argv):
    usage = "main.py -i <image)> -n <noise> -v <verbosity> "
    try:
        opts, args = getopt.getopt(argv, "hi:s:n:m:v:k:c:")
    except getopt.GetoptError:
        print(usage)
        print("Use\n\tmain.py -h\nto learn how to use this and run default settings")
        sys.exit(2)
    
    # assume a square kernel
    kernel_size = 5
    sigma = 2
    image = 'test/test_img002.png'
    verbose = False
    noise = 0
    
    for opt, arg in opts:
        log("{0} {1}".format(opt, arg))
        if opt == '-h':
            print(usage)
            print("Example usage: main.py -i 1 -n x -v False")
            print("\tImage (str): 1, 2, 3")
            print("\tNoise (float): percent of maximum pixel for range of noise added to the accumulator")
            print("\tVerbosity (str): True, False")
            print("No required arguments")
        elif opt == '-i':
            if "1" in arg.lower():
                image = 'test/test_img001.png'
            elif "2" in arg.lower():
                image = 'test/test_img002.png'
            elif "3" in arg.lower():
                image = 'test/test_img003.png'
            else:
                print("Use\n\tmain.py -h\nto learn how to use the image argument; defaulting to 001")
        elif opt == '-v':
            if "false" in arg.lower() or "f" == arg.lower():
                verbose = False
            elif "true" in arg.lower() or "t" == arg.lower():
                verbose = True
            else:
                print("Use\n\tmain.py -h\nto learn how to use the verbosity argument; defaulting to False")
                verbose = False
        elif opt == '-n':
            try:
                noise = float(arg)
            except ValueError:
                print("Noise must be a number 0-100")
                sys.exit(2)
            if noise < 0 or noise > 100:
                print("Noise must be a number 0-100")
                sys.exit(2)
                
    # prepare file names
    file_suffix = image.rsplit("/", 1)[1].rsplit(".", 1)[0] + '_n_' + str(noise)
    logfile = 'out/' + file_suffix + '_info.txt'
    open(logfile, 'w').close()
    
    # Force kernels to be odd
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    
    # load and blur reference images
    start = timer()
    refs = []
    for filename in os.listdir("ref/"):
        img = cv2.imread("ref/" + filename, cv2.IMREAD_GRAYSCALE)
        refs.append(blur(img, kernel_size, sigma).round(decimals=2))
        if verbose:
            cv2.imwrite('out/' + filename+"blurred.png", refs[-1])
    
    # load test image
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    x = blur(img, kernel_size, sigma).round(decimals=2)
    if verbose:
        cv2.imwrite("out/test_blurred.png", x)
    end = timer()
    log("Time taken for loading and pre-processing images: {}".format(end-start))
    
    # build r-table
    start = timer()
    table = buildRtable(refs, (refs[0].shape[0]/2,refs[0].shape[1]/2), (50,35), verbose)
    end = timer()
    log("Time taken to build r-table: {}".format(end-start))
    
    # build accumulator
    start = timer()
    rotations = range(-10, 11, 1)
    scales = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, .9, .95, 1, 1.1, 1.2]
    accum = genAccumulator(x, table, (55,40), rotations, scales, verbose)
    end = timer()
    log("Time taken to build accumulator: {}".format(end-start))
    
    print(accum.shape)
    
    if verbose:
        for s in scales:
            path = "out/" + str(s)
            if not os.path.exists(path):
                os.mkdir(path)
            for t in rotations:
                cv2.imwrite("out/" + str(s) + "/test_votes_{0}_{1}.png".format(t,s), (accum[:,:,rotations.index(t),scales.index(s)]*255/np.max(accum[:,:,rotations.index(t),scales.index(s)])).astype(np.uint8))
            
    # find peaks
    start = timer()
    peaks = getPeaks(accum, np.max(accum)/2)
    end = timer()
    log("Time taken to find peaks: {}".format(end-start))
    
    if verbose:
        for s in scales:
            path = "out/" + str(s)
            if not os.path.exists(path):
                os.mkdir(path)
            for t in rotations:
                cv2.imwrite("out/" + str(s) + "/test_votes_{0}_{1}_peaks.png".format(t,s), (peaks[:,:,rotations.index(t),scales.index(s)]).astype(np.uint8))
    
    # choose the most likely peak
    max = np.amax(peaks, axis=None)
    maxi = np.argmax(peaks, axis=None)
    index_max = np.unravel_index(maxi, peaks.shape)
    print(index_max)
    print("Scale for max: {0}, rotation for max: {1}".format(scales[index_max[3]], rotations[index_max[2]]))
    print("Value of max: {0}".format(peaks[index_max]))
    
    if verbose:
        cv2.imwrite("out/test_peaks.png", (peaks[:,:,index_max[2],index_max[3]]).astype(np.uint8))
    
    # save the final image
    img_BGR = cv2.imread(image, cv2.IMREAD_COLOR)
    box_size = (360,280)
    result = displayResult(img_BGR, (index_max[0], index_max[1]), box_size, rotations[index_max[2]], scales[index_max[3]])
    green_dot = copy.deepcopy(img_BGR)
    print(index_max[0], index_max[1])
    cv2.imwrite("out/test_marked1.png", green_dot)
    green_dot[index_max[0]-1:index_max[0]+2, index_max[1]-1:index_max[1]+2] = (0,255,0)
    cv2.imwrite("out/test_marked2.png", green_dot)
            
if __name__ == "__main__":
    main(sys.argv[1:])