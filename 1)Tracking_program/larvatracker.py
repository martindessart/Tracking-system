#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:15:58 2020

@author: miguel
"""

#%%
from __future__ import print_function
import numpy as np
import pandas as pd
import tracktor_py38 as tr
import cv2
import sys
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import statistics as st
import string
import random
from numpy.random import default_rng
from scipy import signal

#%%
n_inds = 2
if n_inds<=26:
    t_id=list(string.ascii_uppercase)[0:n_inds]
else:
    t_id=list(string.ascii_uppercase)[0:26]+list(string.ascii_lowercase)[0:n_inds-26]
colours= np.zeros((n_inds, 3))
for i in range(0,n_inds):
    colours[i]=tr.random_color()
scaling =.75

mot = True

# name of source video and paths
video = 'IR'
input_vidpath = '/home/pineirua/Dropbox/Tours/Recherche/These_Martin/videos/' + video + '.mp4'
output_vidpath = '/home/pineirua/Dropbox/Tours/Recherche/These_Martin/videos/' + video + '_tracked.mp4'
output_filepath = '/home/pineirua/Dropbox/Tours/Recherche/These_Martin/videos/' + video + '_tracked.csv'
codec = 'DIVX'  # try other codecs if the default doesn't work ('DIVX', 'avc1', 'XVID') note: this list is non-exhaustive




#%%
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
# grab references to the global variables
    global refPt, cropping
 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
 
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
    # record the ending (x, y) coordinates and indicate that
    # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
 
    # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

cap = cv2.VideoCapture(input_vidpath)
ret, image = cap.read()
image = cv2.resize(image, None, fx = scaling, fy = scaling, interpolation = cv2.INTER_CUBIC)

# load the image, clone it, and setup the mouse callback function

clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
 
# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
 
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image=clone.copy()
        
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break
#if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    cv2.imshow("ROI", roi)
    model = np.zeros((roi.shape[0], roi.shape[1]))
    cv2.waitKey(0)
 # close all open windows
cv2.destroyAllWindows()




#%%

# Background calculation
cap = cv2.VideoCapture(input_vidpath)

# Randomly select 25 frames
#frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=35)

medianFrame_par=[]
count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(10) :
    rng = default_rng()
    frameIds=rng.choice(count, size=10, replace=False)
    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx = scaling, fy = scaling, interpolation = cv2.INTER_CUBIC)
        frame=frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    # Calculate the median along the time axis
    medianFrame_par.append(np.median(frames, axis=0))   
medianFrame=np.max(frames, axis=0)
# Display median frame
normaFrame=(medianFrame-medianFrame.mean(keepdims=True))/medianFrame.std(keepdims=True)
cv2.imshow('frame', medianFrame)
cv2.waitKey(0)
cv2.destroyAllWindows()




#%%
#Image Clean Truncate Threshold

cap = cv2.VideoCapture(input_vidpath)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));

trackbar_value = 'Value'
trackbar_frcount = 'Frame Count'
window_name = 'Threshold'


def Threshold_Demo(val):
    
    global thresh_value
    
    frcount=cv2.getTrackbarPos(trackbar_frcount, window_name)
    cap.set(cv2.CAP_PROP_POS_FRAMES,frcount);

    ret, frame = cap.read()
    if ret != True:
        sys.exit('Caca frame')
    frame = cv2.resize(frame, None, fx = scaling, fy = scaling, interpolation = cv2.INTER_CUBIC)
    
    frame=frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_clean=cv2.subtract(medianFrame,src_gray)
    
        
    thresh_value=cv2.getTrackbarPos(trackbar_value, window_name)
    _,prethresh = cv2.threshold(img_clean,thresh_value,255,3)
    
    cv2.imshow(window_name, prethresh)
    
cv2.namedWindow(window_name)
cv2.createTrackbar(trackbar_frcount, window_name , 231, count, Threshold_Demo)
cv2.createTrackbar(trackbar_value, window_name , 1, 254, Threshold_Demo)


# Call the function to initialize
Threshold_Demo(0)
# Wait until user finishes program
cv2.waitKey()
# close all open windows
cap.release()
cv2.destroyAllWindows()


#%%
#Adaptive threshold

cap = cv2.VideoCapture(input_vidpath)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
meas_last = list(np.zeros((n_inds,2)))
meas_now = list(np.zeros((n_inds,2)))
cova_last=list(np.zeros((n_inds,2,2)))
cova_now=list(np.zeros((n_inds,2,2)))

max_offset = 50
max_bs = 101
max_value = 255
trackbar_blocksize = 'Blocksize'
trackbar_offset = 'Offset'
max_type = 1
trackbar_frcount = 'Frame Count'
trackbar_maxarea = 'Max Area'
trackbar_minarea = 'Min Area'
window_name = 'Threshold'

def Threshold_Demo(val):
    
    global block_size,offset,thresh_value,max_area,min_area,area,max_area_real,min_area_real
    
    frcount=cv2.getTrackbarPos(trackbar_frcount, window_name)
    cap.set(cv2.CAP_PROP_POS_FRAMES,frcount);

    ret, frame = cap.read()
    if ret != True:
        sys.exit('Caca frame')
    frame = cv2.resize(frame, None, fx = scaling, fy = scaling, interpolation = cv2.INTER_CUBIC)
    frame=frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_clean=cv2.absdiff(src_gray,medianFrame)
    
    threshold_blocksize = cv2.getTrackbarPos(trackbar_blocksize, window_name)
    threshold_offset = cv2.getTrackbarPos(trackbar_offset, window_name)
    max_area=cv2.getTrackbarPos(trackbar_maxarea, window_name)
    min_area=cv2.getTrackbarPos(trackbar_minarea, window_name)
    
    _,prethresh = cv2.threshold(img_clean,thresh_value,255,3)
    prethresh=(255-prethresh)
    thresh = cv2.adaptiveThreshold(prethresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, threshold_blocksize, threshold_offset)
    
    final, contours, _, _ ,_,_,_= tr.detect_and_draw_contours(frame, thresh, meas_last, meas_now,cova_last, cova_now, min_area, max_area,1,max_area)
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    cv2.putText(final, str(int(len(contours))), (5,30), font, 1, (255,255,255), 2)
    cv2.imshow(window_name, final)
    block_size=threshold_blocksize
    offset=threshold_offset
    l=len(contours)
    area=np.zeros(l)
    for i in range(l):
        area[i]=cv2.contourArea(contours[i])
    min_area_real=st.mean(area)
    max_area_real=1.05*max(area)
    
    
    #thresh_value=threshold_value
parser = argparse.ArgumentParser(description='Code for Basic Thresholding Operations tutorial.')
parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
args = parser.parse_args()


# Convert the image to Gray

cv2.namedWindow(window_name)
cv2.createTrackbar(trackbar_frcount, window_name ,0,count, Threshold_Demo)
cv2.createTrackbar(trackbar_maxarea, window_name , 10, 1000, Threshold_Demo)
cv2.createTrackbar(trackbar_minarea, window_name , 1, 200, Threshold_Demo)
#cv2.createTrackbar(trackbar_type, window_name , 0, max_type, Threshold_Demo)
cv2.createTrackbar(trackbar_blocksize, window_name , 51, max_bs, Threshold_Demo)
# Create Trackbar to choose Threshold value
cv2.createTrackbar(trackbar_offset, window_name , 5, max_offset, Threshold_Demo)
#cv2.createTrackbar(trackbar_value, window_name , 1, max_value, Threshold_Demo)

# Call the function to initialize
Threshold_Demo(0)
# Wait until user finishes program
cv2.waitKey()
# close all open windows
cap.release()
cv2.destroyAllWindows()


#%%
## TRACKER


cap = cv2.VideoCapture(input_vidpath)
if cap.isOpened() == False:
    sys.exit('Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file')

## Video writer class to output video with contour and centroid of tracked object(s)
# make sure the frame size matches size of array 'final'
fourcc = cv2.VideoWriter_fourcc(*codec)
output_framesize = (roi.shape[1],roi.shape[0])
out = cv2.VideoWriter(filename = output_vidpath, fourcc = fourcc, fps = 25.0, frameSize = output_framesize, isColor = True)


## Individual location(s) measured in the last and current step
meas_last = list(np.zeros((n_inds,2)))
meas_now = list(np.zeros((n_inds,2)))
cova_last=list(np.random.rand(n_inds,2,100))
cova_now=list(np.random.rand(n_inds,2,100))
for i in range(len(cova_last)):
    cova_last[i]=np.cov(cova_last[i])
    cova_now[i]=np.cov(cova_now[i])

metric='euclidian'

df = []
last = 0
seuil=0
#cap.set(cv2.CAP_PROP_POS_FRAMES,1);
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    this = cap.get(1)
    
    if ret == True:
        # Preprocess the image for background subtraction
        frame = cv2.resize(frame, None, fx = scaling, fy = scaling, interpolation = cv2.INTER_CUBIC)
        frame=frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

    
        # Apply mask to ignore area outside the petri dish
        
        
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_clean=cv2.absdiff(gray,medianFrame)
        _,prethresh = cv2.threshold(img_clean,thresh_value,255,3)
        prethresh=(255-prethresh)
        
        thresh = cv2.adaptiveThreshold(prethresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, offset)
        
        contours_test,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        i=0
        j=0
        while i < len(contours_test):
            area = cv2.contourArea(contours_test[i])
            if area > max_area:
                j+=1
            if area < min_area or area > max_area:
                del contours_test[i]
            else:
                i+=1
        if len(contours_test)==n_inds:
            final, contours, meas_last, meas_now, cova_last, cova_now, bigblobs= tr.detect_and_draw_contours(frame,thresh, meas_last, meas_now, cova_last, cova_now, min_area, max_area, seuil, max_area_real)
            
            cost = cdist(meas_last, meas_now,metric='euclidean')
            min_index=int(np.argmin(cost))
            #meas_now=meas_now[min_index]
            # if len(meas_now) != n_inds:
                
            #     dif_pos=np.zeros(len(meas_last))
            #     cost = cdist(meas_last, meas_now,metric='euclidean')
            #     for i in range(len(meas_last)):
            #         dif_pos[i]=cost[i,:].min()
                    
            #     nclust=n_inds-len(meas_now)
            #     ml=np.array(meas_last)
            #     cova=np.array(cova_last)
            #     vi=ml[dif_pos.argsort()[-nclust:][::-1]]
            #     cova2=cova[dif_pos.argsort()[-nclust:][::-1]]
            #     contours_bigblobs=tr.detect_bigblobs(frame, thresh,max_area_real)
            #     #meas_now=tr.apply_k_means_bigblobs(thresh,contours_bigblobs, nclust, vi,meas_now)
            #     meas_now,gm_messy,m_messy2=tr.apply_gm_messy_bigblobs(final,contours_bigblobs,nclust, meas_now,thresh,vi,'full',np.linalg.inv(cova2))
            #     for i in range(nclust):
            #         cova_now.append(gm_messy.covariances_[i])
            # if len(meas_now)<n_inds:
            #     break
    
            row_ind, col_ind = tr.hungarian_algorithm(meas_last, meas_now,metric)
            final, meas_now, df = tr.reorder_and_draw(final, colours,n_inds, col_ind,meas_now, df, mot, this,t_id)
          
            # Create output dataframe
            for i in range(len(meas_now)):
                df.append([this, meas_now[i][0], meas_now[i][1], t_id[i]])
            
            # Display the resulting frame
            out.write(final)
            cv2.imshow('frame', final)
            if cv2.waitKey(1) == 27:
                break
                
    if last >= this:
        break
    if any(meas_now[:][0])==0:
        break
    last=this
    #input()
## Write positions to file
df = pd.DataFrame(np.matrix(df), columns = ['frame','pos_x','pos_y', 'id'])
df.to_csv(output_filepath, sep=',')

## When everything done, release the capture
cap.release()
out.release()
cv2.waitKey()
cv2.destroyAllWindows()

#%%




