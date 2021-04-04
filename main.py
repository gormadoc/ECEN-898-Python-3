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
    usage = "main.py -i <image)> -s <scales> -r <rotation> -n <noise> -v <verbosity> "
    try:
        opts, args = getopt.getopt(argv, "hi:n:s:r:v:")
    except getopt.GetoptError:
        print(usage)
        print("Use\n\tmain.py -h\nto learn how to use this and run default settings")
        sys.exit(2)
    
    # assume a square kernel
    kernel_size = (9,9,5)
    sigma = (2,2,2)
    kernel_size_test = (9,9,11)
    sigma_test = (3,3,3)
    image = 'test/test_img003.png'
    image_n = 3
    verbose = False
    noise = 0
    scales = list(np.linspace(0.5, 0.8, 7))
    rotations = list(np.linspace(-2.5, 2.5, num=11))
    
    for opt, arg in opts:
        log("{0} {1}".format(opt, arg))
        if opt == '-h':
            print(usage)
            print("Example usage (defaults): main.py -i 1 -s 0.5,0.8 -r 2.5 -n 0 -v True")
            print("\tImage (str): 1, 2, 3")
            print("\tScale (float (,float)): seven scales from val1 to val2; if no val2 only scale at val1")
            print("\tRotation maximum (int): eleven rotations from -value to +value, unless value=0")
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
                print("Use\n\tmain.py -h\nto learn how to use the image argument; defaulting to 3")
                continue
            image_n = int(arg)
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
        elif opt == '-s':
            try:
                start = float(arg.rsplit(",")[0])
                end = float(arg.rsplit(",")[1])
                scales = list(np.linspace(start,end, 7))
            except ValueError:
                print("Scales must be floats.")
                sys.exit(2)
            except IndexError:
                print("Single-value scale: {}".format(start))
                end = start
                scales = list(np.linspace(start, start, 1))
                sys.exit(2)
        elif opt == '-r':
            try:
                val = float(arg)
            except ValueError:
                print("Rotation must be a float.")
                sys.exit(2)
            if val == 0.0:
                rotations = list(np.linspace(0, 0, num=1))
            else:
                rotations = list(np.linspace(-val, val, num=11))
                
    # prepare file names
    file_suffix = image.rsplit("/", 1)[1].rsplit(".", 1)[0] + '_s_{0:.3f},{1:.3f}_r_{2:.3f}_n_{3}'.format(scales[0], scales[-1], rotations[-1],noise)
    logfile = 'out/' + file_suffix + '_info.txt'
    open(logfile, 'w').close()
    log("Scales: {0}\nRotations: {1}".format(scales, rotations), logfile)
    
    # Force kernels to be odd
    for size in kernel_size:
        if size % 2 == 0:
            size = size + 1
    
    # load and blur reference images
    start = timer()
    refs = []
    i = 1
    for filename in os.listdir("ref/"):
        img = cv2.imread("ref/" + filename, cv2.IMREAD_GRAYSCALE)
        refs.append(blur(img, kernel_size[i], sigma[i]).round(decimals=2))
        if verbose:
            cv2.imwrite('out/' + filename + "blurred.png", refs[-1])
    
    # load test image
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    x = blur(img, kernel_size_test[image_n-1], sigma_test[image_n-1]).round(decimals=2)
    if verbose:
        cv2.imwrite("out/{0}_blurred.png".format(file_suffix), x)
    end = timer()
    log("Time taken for loading and pre-processing images: {}".format(end-start), file=logfile)
    
    # build r-table
    start = timer()
    table = buildRtable(refs, (refs[0].shape[0]/2,refs[0].shape[1]/2), (50,35), verbose)
    end = timer()
    log("Time taken to build r-table: {}".format(end-start), logfile)
    
    # build accumulator
    start = timer()
    
    accum = genAccumulator(x, table, (55,40), rotations, scales, verbose)
    if noise != 0.0:
        for a in accum:
            sc =  noise/100*np.amax(a)
            a += np.random.randint(-sc, high=sc, size=a.shape).astype(int)
            np.clip(a, 0, 255)
    end = timer()
    log("Time taken to build accumulator: {}".format(end-start), logfile)
    
    if verbose:
        for s in scales:
            path = "out/{0:.3f}".format(s)
            if not os.path.exists(path):
                os.mkdir(path)
            for t in rotations:
                cv2.imwrite("out/" + str(s) + "/{0}_votes_{1:.3f}_{2:.3f}.png".format(file_suffix,t,s), (accum[:,:,rotations.index(t),scales.index(s)]*255/np.max(accum[:,:,rotations.index(t),scales.index(s)])).astype(np.uint8))
            
    # find peaks
    start = timer()
    peaks = getPeaks(accum, np.max(accum)/2)
    end = timer()
    log("Time taken to find peaks: {}".format(end-start), logfile)
    
    if verbose:
        for s in scales:
            path = "out/{0:.3f}".format(s)
            if not os.path.exists(path):
                os.mkdir(path)
            for t in rotations:
                cv2.imwrite("out/" + str(s) + "/{0}_peaks_{1:.3f}_{2:.3f}.png".format(file_suffix,t,s), (peaks[:,:,rotations.index(t),scales.index(s)]).astype(np.uint8))
    
    # choose the most likely peak
    start = timer()
    max = np.amax(peaks, axis=None)
    maxi = np.argmax(peaks, axis=None)
    index_max = np.unravel_index(maxi, peaks.shape)
    log("Scale for max: {0}, rotation for max: {1}".format(scales[index_max[3]], rotations[index_max[2]]), logfile)
    log("Value of max: {0}, position of maximum: {1}".format(peaks[index_max], (index_max[0], index_max[1])), logfile)
    end = timer()
    log("Time taken to choose most likely peak: {}".format(end-start), logfile)
    
    if verbose:
        cv2.imwrite("out/peaks_{0}.png".format(file_suffix), (peaks[:,:,index_max[2],index_max[3]]).astype(np.uint8))
    
    # save the final image
    img_BGR = cv2.imread(image, cv2.IMREAD_COLOR)
    box_size = (360,280)
    result = displayResult(img_BGR, (index_max[0], index_max[1]), box_size, rotations[index_max[2]], scales[index_max[3]])
    green_dot = copy.deepcopy(img_BGR)
    cv2.imwrite("out/{0}_result.png".format(file_suffix), green_dot)
    
    # save the "face" model
    face = np.zeros(refs[0].shape)
    ref_point = np.array((refs[0].shape[0]/2,refs[0].shape[1]/2))
    for theta in table:
        for rho in table[theta]:
            edge = ref_point + np.array(rho)
            face[edge.astype(int)[0],edge.astype(int)[1]] += table[theta][rho]
    face = (face*255/np.amax(face)).astype(int)
    cv2.imwrite("out/face_model.png", face)
        
            
if __name__ == "__main__":
    main(sys.argv[1:])