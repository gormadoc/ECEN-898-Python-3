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
    kernel_size = 20
    sigma = 5
    image = 'test/test_img001.png'
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
    
    # load reference images
    refs = []
    for filename in os.listdir("ref/"):
        refs.append(cv2.cvtColor(cv2.imread("ref/" + filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY))
    
    # build r-table
    start = timer()
    table = buildRtable(refs, (140,180), verbose)
    end = timer()
    log("Time taken to build r-table: {}".format(end-start))
    
    
    # build accumulator
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    start = timer()
    accum = genAccumulator(img, table, verbose)
    end = timer()
    log("Time taken to build accumulator: {}".format(end-start))
    
    
    
            
if __name__ == "__main__":
    main(sys.argv[1:])