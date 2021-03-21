#!/usr/bin/env python

# Canny edge detection
'''
1. Gaussian filter smoothing (done)
2. Find intensity gradient
3. Pre-massage with gradient magnitude thresholding or lower bound cut-off suppression
4. Apply double threshold to find potential edges
5. Track edge by hysteresis: finalize detection by suppressing weak edges not connected to strong edges
'''

import sys, getopt, os
import math
import numpy as np
import copy
import cv2


def log(message, file=None):
    if not file:
        print(message)
    else:
        with open(file, 'a') as f:
            f.write(message + '\n')

def pad_array(img, amount, method='replication'):
    method = method
    amount = amount
    t_img = np.array(img)
    re_img = np.zeros([img.shape[0]+2*amount, img.shape[1]+2*amount])
    re_img[amount:img.shape[0]+amount, amount:img.shape[1]+amount] = t_img
    if method == 'zero':
        pass # already that way
    elif method == 'replication':
        re_img[0:amount,amount:img.shape[1]+amount] = np.flip(img[0:amount, :], axis=0) # left
        re_img[-1*amount:-1, amount:img.shape[1]+amount] = np.flip(img[-1*amount:-1, :], axis=0) # right
        re_img[:, 0:amount] = np.flip(re_img[:, amount:2*amount], axis=1) # top
        re_img[:, -1*amount:] = np.flip(re_img[:, -2*amount:-amount], axis=1) # bottom
        
    return re_img
        
def image_filter2d(img, kernel):
    # establish useful values
    imx = img.shape[0]
    imy = img.shape[1]
    kx = kernel.shape[0]
    ky = kernel.shape[1]
    if kx % 2 == 1:
        center = [math.ceil(kx/2), math.ceil(ky/2)]
    else:
        center = [int(kx/2) + 1, int(ky/2) + 1]
        
    # pad arrays and put image in center
    re_img = np.zeros([imx+2*kx, imy+2*ky])
    pad_img = np.zeros([imx+2*kx, imy+2*ky])+np.max(np.max(img))/2
    pad_img[kx:imx+kx, ky:imy+ky] = img
    
    # Perform sum of products
    for row in range(kx, imx+kx):
        for col in range(ky, imy+ky):
            for a in range(0, kx):
                for b in range(0, ky):
                    re_img[row, col] = re_img[row,col] + pad_img[row+a-center[0]+1, col+b-center[1]+1]*kernel[a,b]
    return re_img[kx:imx+kx, ky:imy+ky]
    
def gradient_calc(image):
    # get some arrays ready
    sobx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    soby = sobx.transpose()
    phi = np.zeros(image.shape)
    M = np.zeros(image.shape)
    
    # need to slightly pad but we don't need to calculate the new borders
    img = pad_array(image, 1)
    x, y = img.shape
    for i in range(1, x-2):
        for j in range(1, y-2):
            # calculate both at once rather than separately
            dx = -1*img[i-1,j-1] -2*img[i-1,j] -1*img[i-1,j+1] +img[i+1,j-1] +2*img[i+1,j] +img[i+1,j+1] 
            dy = -1*img[i-1,j-1] -2*img[i,j-1] -1*img[i+1,j-1] +img[i-1,j+1] +2*img[i,j+1] +img[i+1,j+1]
            
            # division by zero is undefined
            if dx == 0 and dy > 0:
                phi[i,j] = 90
            elif dx == 0 and dy < 0:
                phi[i,j] = -90
            else:
                phi[i,j] = np.arctan2(dy,dx)/np.pi*180
                
            # magnitude
            M[i-1, j-1] = (dx**2+dy**2)**0.5
    return phi, M
        

def Gaussian2D(size, sigma):
    # simplest case is where there is no Gaussian
    if size==1:
        return np.array([[0,0,0],[0,1,0],[0,0,0]])

    # parameters
    peak = 1/2/np.pi/sigma**2
    width = -2*sigma**2
    
    # Gaussian filter
    H = np.zeros([size, size])

    # populate the Gaussian
    if size % 2 == 1:
        k = (size - 1)/2
        for i in range(1, size+1):
            i_part = (i-(k+1))**2
            for j in range(1, size+1):
                H[i-1, j-1] = peak*math.exp((i_part + (j-(k+1))**2)/width)
    else:
        k = size / 2
        for i in range(1, size+1):
            i_part = (i-(k+0.5))**2
            for j in range(1, size+1):
                H[i-1, j-1] = peak*math.exp((i_part + (j-(k+0.5))**2)/width)

    # normalize the matrix
    H = H / np.sum(np.concatenate(H))
    return H


def neighbors(image, p, connectedness=8):
    X,Y = image.shape
    x = p[0]
    y = p[1]
    n = []
    if connectedness == 8:
        for i in [-1, 0, 1]:
            # check within x bounds
            if x+i > -1 and x+i < X:
                #print(x+i)
                for j in [-1, 0, 1]:
                    # check within y bounds
                    if y+j > -1 and y+j < Y:
                        #print(y+j)mi
                        # p is not a neighbor of p
                        if i != 0 or j != 0:
                            n.append((x+i,y+j))
    elif connectedness == 4:
        if x > 0:
            n.append((x-1, y))
        if x < X-1:
            n.append((x+1, y))
        if y > 0:
            n.append((x, y-1))
        if y < Y-1:
            n.append((x, y+1))
    return n
    
    
def buildRtable(images, point, verbose=False):
    r_table = {}
    for img in images:
        # gradient calculations
        phi,M = gradient_calc(img)
        threshold = (40, 20)
        
        # we can ready some queues for threshold information
        strong_queue = []
        weak_list = []
    
        # non-maxima suppression and edge candidate detection
        N = copy.deepcopy(M)
        for row in range(0, M.shape[0]-1):
            for col in range(0, M.shape[1]-1):
                p = phi[row,col]
                # eight cases decomposed into four by arctan range (-90deg<->90deg)
                if p < 22.5 and p >= -22.5: # 4,6
                    coords = [[row-1, col], [row+1, col]]
                elif p < 67.5 and p >= 22.5: # 1,9
                    coords = [[row-1, col-1], [row+1, col+1]]
                elif p <=90 and p >= 67.5 or p <= -67.5 and p >= -90: # 2, 8
                    coords = [[row, col+1], [row, col-1]]
                else: # 3, 7
                    coords = [[row-1, col+1], [row+1, col-1]]
                if M[row, col] <= M[coords[0][0], coords[0][1]] or M[row, col] <= M[coords[1][0], coords[1][1]]:
                    N[row,col] = 0
                
                # threshold control; values just for informative picture
                if N[row,col] > threshold[1]:
                    N[row,col] = 128
                    strong_queue.append((row,col))
                elif N[row,col] > threshold[0]:
                    N[row,col] = 64
                    weak_list.append((row, col))
                else:
                    N[row,col] = 0
        
        # edge strengthening
        while len(strong_queue) > 0:
            # get pixel at head
            px = strong_queue[-1]
        
            # remove the pixel from the queue
            strong_queue.pop(-1)
                
            # begin processing
            N[px[0], px[1]] = 255
            for i in range(-1,2):
                for j in range(-1,2):
                    # use our weak flag value to speed things up
                    if N[px[0]+i, px[1]+j] == 64:
                        # set it to the strong-but-not-processed value (which is unused due to the queue)
                        N[px[0]+i, px[1]+j] = 128
                        
                        # pop the pixel from the weak_list and add it to the queue
                        weak_list.pop(weak_list.index((px[0]+i, px[1]+j)))
                        strong_queue.append((px[0]+i, px[1]+j))
        
        # cull unverified weak edges
        for px in weak_list:
            N[px[0], px[1]] = 0
            
        if verbose:
            cv2.imwrite("out/ref_edges.png", N)
            
        # build r-table
        for i in range(0, N.shape[0]):
            for j in range(0, N.shape[1]):
                if N[i,j] == 255:
                    theta = phi[i,j]
                    rho = (i-point[0], j-point[1]) # just a displacement vector
                    if theta in r_table.keys():
                        if rho not in r_table[theta]:
                            r_table[theta][rho] = M[i,j]
                        else:
                            r_table[theta][rho] += M[i,j]
                    else:
                        r_table[theta] = {rho: M[i,j]}
    return r_table
    
    
def genAccumulator(image, r_table, verbose=False):
    ''' Find boundaries in image '''
    # gradient calculations
    phi,M = gradient_calc(image)
    threshold = (40, 20)
    
    # we can ready some queues for threshold information
    strong_queue = []
    weak_list = []

    # non-maxima suppression and edge candidate detection
    N = copy.deepcopy(M)
    for row in range(1, M.shape[0]-1):
        for col in range(1, M.shape[1]-1):
            p = phi[row,col]
            # eight cases decomposed into four by arctan range (-90deg<->90deg)
            if p < 22.5 and p >= -22.5: # 4,6
                coords = [[row-1, col], [row+1, col]]
            elif p < 67.5 and p >= 22.5: # 1,9
                coords = [[row-1, col-1], [row+1, col+1]]
            elif p <=90 and p >= 67.5 or p <= -67.5 and p >= -90: # 2, 8
                coords = [[row, col+1], [row, col-1]]
            else: # 3, 7
                coords = [[row-1, col+1], [row+1, col-1]]
            if M[row, col] <= M[coords[0][0], coords[0][1]] or M[row, col] <= M[coords[1][0], coords[1][1]]:
                N[row,col] = 0
            
            # threshold control; values just for informative picture
            if N[row,col] > threshold[1]:
                N[row,col] = 128
                strong_queue.append((row,col))
            elif N[row,col] > threshold[0]:
                N[row,col] = 64
                weak_list.append((row, col))
            else:
                N[row,col] = 0
    
    # edge strengthening
    while len(strong_queue) > 0:
        # get pixel at head
        px = strong_queue[-1]
    
        # remove the pixel from the queue
        strong_queue.pop(-1)
            
        # begin processing
        N[px[0], px[1]] = 255
        for i in range(-1,2):
            for j in range(-1,2):
                # use our weak flag value to speed things up
                if N[px[0]+i, px[1]+j] == 64:
                    # set it to the strong-but-not-processed value (which is unused due to the queue)
                    N[px[0]+i, px[1]+j] = 128
                    
                    # pop the pixel from the weak_list and add it to the queue
                    weak_list.pop(weak_list.index((px[0]+i, px[1]+j)))
                    strong_queue.append((px[0]+i, px[1]+j))
    
    # cull unverified weak edges
    for px in weak_list:
        N[px[0], px[1]] = 0
        
    if verbose:
        cv2.imwrite("out/test_edges.png", N)
    
    # build vote-space
    P = np.zeros(image.shape)
    for i in range(0, N.shape[0]):
        for j in range(0, N.shape[1]):
            if N[i,j] == 255:
                theta = phi[i,j]
                if theta  in r_table.keys():
                    for rho in r_table[theta]:
                        p = (i-rho[0], j-rho[1])
                        if -1 < p[0] < N.shape[0] and -1 < p[1] < N.shape[1]:
                            P[p] += r_table[theta][rho]
                            
    if verbose:
            cv2.imwrite("out/test_votes.png", P)
            
    return P

def getPeaks(accumulator):
    return []
    
    
def displayResult(image, center):
    box = (10, 10, 10, 10)
    return cv2.rectangle(image, (center-box[0]/2, center-box[1]/2), (center+box[0]/2, center+box[1]/2), (0,0,255), 2)