
#%% 1) DEPENDENCIES
from __future__ import print_function
import numpy as np
import pandas as pd
import cv2
import sys
import argparse
from scipy.spatial.distance import cdist
import statistics as st
import string
from numpy.random import default_rng

#essai import file
sys.path.append('/home/dessart/Téléchargements/TRACKER/Larva_tracker')
#print(sys.path)
import tracktor_pyN as tr

#%% 
n_inds = 10 #nbe individus
if n_inds<=26:
    t_id=list(string.ascii_uppercase)[0:n_inds]
else:
    t_id=list(string.ascii_uppercase)[0:26]+list(string.ascii_lowercase)[0:n_inds-26] #liste des individus
    #chaque ind a une lettre en commençant par A
colours= np.zeros((n_inds, 3))
for i in range(0,n_inds):
    colours[i]=tr.random_color()
scaling =1

mot = True

    # name of source video and paths
video = 'MX10R2'
input_vidpath = '/home/dessart/Vidéos/' + video + '.mp4'
output_filepath = '/home/dessart/Vidéos/' + video + '_tracked.csv'
codec = 'avc1'  # try other codecs if the default doesn't work ('DIVX', 'avc1', 'XVID') note: this list is non-exhaustive   




#%% 3) BACKGROUND PARAMETERS
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
# grab references to the global variables
    global refPt, cropping
 
    # if the left mouse button is clicked, record the starting
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
#a=dir(cv2)


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
# from the image and display it
if len(refPt) == 2:
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    cv2.imshow("ROI", roi)
    model = np.zeros((roi.shape[0], roi.shape[1]))
    cv2.waitKey(0)
 # close all open windows
cv2.destroyAllWindows()




#%% 4) BACKGROUND CALCULATION

# Background calculation
cap = cv2.VideoCapture(input_vidpath)

# Randomly select 10 frames

medianFrame_par=[]
count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(10) :
    rng = default_rng()
    frameIds=rng.choice(count, size=25, replace=False)
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
    
    medianFrame_par.append(np.max(frames, axis=0))  
    

    
    
    
    
medianFrame=np.max(medianFrame_par, axis=0).astype(dtype=np.uint8)
# Display median frame
normaFrame=(medianFrame-medianFrame.mean(keepdims=True))/medianFrame.std(keepdims=True)
print('\a')
cv2.imshow('frame', medianFrame)
cv2.waitKey(0)
cv2.destroyAllWindows()



#%% 5) IMAGE CHECK
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


#%% 6) Adaptive threshold

cap = cv2.VideoCapture(input_vidpath)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
meas_last = list(np.zeros((n_inds,3)))
meas_now = list(np.zeros((n_inds,3)))

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

    final, contours, _, _ ,_ = tr.detect_and_draw_contours(frame, thresh, meas_last, meas_now, min_area, max_area)
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
    
    
    
parser = argparse.ArgumentParser(description='Code for Basic Thresholding Operations tutorial.')
parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
args = parser.parse_args()


# Convert the image to Gray

cv2.namedWindow(window_name)
cv2.createTrackbar(trackbar_frcount, window_name ,0,count, Threshold_Demo)
cv2.createTrackbar(trackbar_maxarea, window_name , 700, 1000, Threshold_Demo)
cv2.createTrackbar(trackbar_minarea, window_name , 80, 200, Threshold_Demo)
cv2.createTrackbar(trackbar_blocksize, window_name , 51, max_bs, Threshold_Demo)
# Create Trackbar to choose Threshold value
cv2.createTrackbar(trackbar_offset, window_name , 8, max_offset, Threshold_Demo)


# Call the function to initialize
Threshold_Demo(0)
# Wait until user finishes program
cv2.waitKey()
# close all open windows
cap.release()
cv2.destroyAllWindows()


#%% 7) TRACKER
cap = cv2.VideoCapture(input_vidpath)
if cap.isOpened() == False:
    sys.exit('Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file')

## Individual location(s) measured in the last and current step
meas_last = list(np.zeros((n_inds,3)))
meas_now = list(np.zeros((n_inds,3)))

metric='euclidian'

df = []
last = 0
seuil=0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    this = cap.get(1)
    
    if ret == True:
        # Preprocess the image for background subtraction
        frame = cv2.resize(frame, None, fx = scaling, fy = scaling, interpolation = cv2.INTER_CUBIC)
        frame=frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

    
        # Apply mask to ignore area outside the area of interest    
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
   
            final, contours, meas_last, meas_now, bigblobs= tr.detect_and_draw_contours(frame,thresh, meas_last, meas_now, min_area, max_area, seuil, max_area_real)
            
            cost = cdist(meas_last, meas_now,metric='euclidean')
            min_index=int(np.argmin(cost))
   
            row_ind, col_ind = tr.hungarian_algorithm(meas_last, meas_now,metric)
            final, meas_now, df = tr.reorder_and_draw(final, colours,n_inds, col_ind,meas_now, df, mot, this,t_id)
          
            # Create output dataframe
            for i in range(len(meas_now)):
                df.append([this, meas_now[i][0], meas_now[i][1], meas_now[i][2], t_id[i]])
            
            if cv2.waitKey(1) == 27:
                break
                
    if last >= this:
        break
    last=this
   

## Write positions to file
df = pd.DataFrame(np.matrix(df), columns = ['frame','pos_x','pos_y', 'size','id'])
df.to_csv(output_filepath, sep=',')
print('\a')

## When everything done, release the capture
cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
#%%