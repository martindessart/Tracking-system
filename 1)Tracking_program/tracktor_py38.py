import numpy as np
import pandas as pd
import cv2
import random
import matplotlib as mp

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull

def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))


def colour_to_thresh(frame, block_size = 31, offset = 25):
    """
    This function retrieves a video frame and preprocesses it for object tracking.
    The code blurs image to reduce noise, converts it to greyscale and then returns a 
    thresholded version of the original image.
    
    Parameters
    ----------
    frame: ndarray, shape(n_rows, n_cols, 3)
        source image containing all three colour channels
    block_size: int(optional), default = 31
        block_size determines the width of the kernel used for adaptive thresholding.
        Note: block_size must be odd. If even integer is used, the programme will add
        1 to the block_size to make it odd.
    offset: int(optional), default = 25
        constant subtracted from the mean value within the block
        
    Returns
    -------
    thresh: ndarray, shape(n_rows, n_cols, 1)
        binarised(0,255) image
    """
    #blur = cv2.blur(frame, (5,5))
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, offset)
    thresh = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, offset)
    return thresh

def colour_to_thresh2(frame, value = 170):
    """
    This function retrieves a video frame and preprocesses it for object tracking.
    The code blurs image to reduce noise, converts it to greyscale and then returns a 
    thresholded version of the original image.
    
    Parameters
    ----------
    frame: ndarray, shape(n_rows, n_cols, 3)
        source image containing all three colour channels
    block_size: int(optional), default = 31
        block_size determines the width of the kernel used for adaptive thresholding.
        Note: block_size must be odd. If even integer is used, the programme will add
        1 to the block_size to make it odd.
    offset: int(optional), default = 25
        constant subtracted from the mean value within the block
        
    Returns
    -------
    thresh: ndarray, shape(n_rows, n_cols, 1)
        binarised(0,255) image
    """
    blur = cv2.blur(frame, (5,5))
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,value,255,cv2.THRESH_BINARY_INV)
    return thresh


def detect_and_draw_contours(frame, thresh, meas_last, meas_now, cova_last, cova_now, min_area = 0, max_area = 10000, seuil=1, max_area_real=10000):
    
    # Detect contours and draw them based on specified area thresholds
    contours_test,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    i=0
    j=0
    while i < len(contours_test):
        area = cv2.contourArea(contours_test[i])
        if area > max_area_real:
            j+=1
        if area < min_area or area > max_area_real:
            del contours_test[i]
        else:
            i+=1
    final = frame.copy()   
    if len(contours_test)>=seuil:
        contours,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

        

        i = 0
        meas_last = meas_now.copy()
        cova_last=cova_now.copy() 
        del meas_now[:]
        del cova_now[:]
        if np.asarray(meas_last).shape[1]==3:
            while i < len(contours):
                area = cv2.contourArea(contours[i])
                if area < min_area or area > max_area_real:
                    del contours[i]
                else:
                    cv2.drawContours(final, contours, i, (0,0,255), 1)
                    M = cv2.moments(contours[i])
                    if M['m00'] != 0:
                    	cx = M['m10']/M['m00']
                    	cy = M['m01']/M['m00']
                    else:
                    	cx = 0
                    	cy = 0
                    meas_now.append([cx,cy,area])
                    i += 1
        else:
            while i < len(contours):
                area = cv2.contourArea(contours[i])
                if area < min_area or area > max_area_real:
                    del contours[i]
                   
                else:
                    cv2.drawContours(final, contours, i, (0,0,255), 1)
                    M = cv2.moments(contours[i])
                    if M['m00'] != 0:
                    	cx = M['m10']/M['m00']
                    	cy = M['m01']/M['m00']
                    else:
                    	cx = 0
                    	cy = 0
                    meas_now.append([cx,cy])                    
                    mask2 = np.zeros(thresh.shape,np.uint8)
                    cv2.drawContours(mask2,contours[i],-1,255,-1)
                    pixelpoints = cv2.findNonZero(mask2)
                    pixelpoints=pixelpoints.reshape((pixelpoints.shape[0], pixelpoints.shape[2]))
                    cova_now.append(np.cov(np.transpose(pixelpoints)))
                    i += 1
    return final, contours_test, meas_last, meas_now, cova_last, cova_now, j

def detect_bigblobs(frame, thresh,max_area_real=10000):
    
    
    contours,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    i=0
    while i < len(contours):
        area = cv2.contourArea(contours[i])
        if area < max_area_real:
            del contours[i]                
        else:
            i+=1
    return contours

def apply_k_means(contours, n_inds, meas_now,meas_last):
    """
    This function applies the k-means clustering algorithm to separate merged
    contours. The algorithm is applied when detected contours are fewer than
    expected objects(number of animals) in the scene.
    
    Parameters
    ----------
    contours: list
        a list of all detected contours that pass the area based threhold criterion
    n_inds: int
        total number of individuals being tracked
    meas_now: array_like, dtype=float
        individual's location on current frame
        
    Returns
    -------
    contours: list
        a list of all detected contours that pass the area based threhold criterion
    meas_now: array_like, dtype=float
        individual's location on current frame
    """
    del meas_now[:]
    # Clustering contours to separate individuals
    myarray = np.vstack(contours)
    myarray = myarray.reshape(myarray.shape[0], myarray.shape[2])
    #kmeans = KMeans(n_clusters=n_inds, random_state=0, n_init = 50).fit(myarray)

    kmeans = KMeans(n_clusters=n_inds, init=meas_last,random_state=0, n_init = 50).fit(myarray)
    
    l = len(kmeans.cluster_centers_)

    for i in range(l):
        x = int(tuple(kmeans.cluster_centers_[i])[0])
        y = int(tuple(kmeans.cluster_centers_[i])[1])
        meas_now.append([x,y])
        
    return contours, meas_now




def apply_k_means_mod(thresh,contours, n_inds, meas_now,meas_last):
    
    del meas_now[:]
    # Clustering contours to separate individuals
    
    mask2 = np.zeros(thresh.shape,np.uint8)
    cv2.drawContours(mask2,contours,-1,255,-1)
    pixelpoints = cv2.findNonZero(mask2)
    pixelpoints=pixelpoints.reshape((pixelpoints.shape[0], pixelpoints.shape[2]))
    
    #kmeans = KMeans(n_clusters=n_inds, random_state=0, n_init = 50).fit(myarray)

    kmeans = KMeans(n_clusters=n_inds, init=meas_last[:,0:2],max_iter=1000,random_state=42, n_init = 1).fit(pixelpoints)
    pixpoints_ind=kmeans.labels_
    unique, new_area = np.unique(pixpoints_ind, return_counts=True)
    l = len(kmeans.cluster_centers_)

    if meas_last.shape[1]==3:
        for i in range(l):
            x = int(tuple(kmeans.cluster_centers_[i])[0])
            y = int(tuple(kmeans.cluster_centers_[i])[1])
            meas_now.append([x,y,new_area[i]])
    else:
         for i in range(l):
            x = int(tuple(kmeans.cluster_centers_[i])[0])
            y = int(tuple(kmeans.cluster_centers_[i])[1])
            meas_now.append([x,y])
        
    return contours, meas_now

def apply_k_means_bigblobs(thresh,contours, n_clust, vi, meas_now):
    
    #del meas_now[:]
    # Clustering contours to separate individuals
    
    mask2 = np.zeros(thresh.shape,np.uint8)
    cv2.drawContours(mask2,contours,-1,255,-1)
    pixelpoints = cv2.findNonZero(mask2)
    pixelpoints=pixelpoints.reshape((pixelpoints.shape[0], pixelpoints.shape[2]))
    
    #kmeans = KMeans(n_clusters=n_inds, random_state=0, n_init = 50).fit(myarray)

    kmeans = KMeans(n_clusters=n_clust,init=vi,max_iter=1000,random_state=0, n_init = 1).fit(pixelpoints)
    pixpoints_ind=kmeans.labels_
    unique, new_area = np.unique(pixpoints_ind, return_counts=True)
    l = len(kmeans.cluster_centers_)
    meas=np.asarray(meas_now)


    if meas.shape[1]==3:
        for i in range(l):
            x = int(tuple(kmeans.cluster_centers_[i])[0])
            y = int(tuple(kmeans.cluster_centers_[i])[1])
            meas_now.append([x,y,new_area[i]])
    else:
         for i in range(l):
            x = int(tuple(kmeans.cluster_centers_[i])[0])
            y = int(tuple(kmeans.cluster_centers_[i])[1])
            meas_now.append([x,y])
        
    return meas_now



def apply_gm_messy(contours,n_inds, meas_now,meas_last,thresh):
    
#    area= list(np.zeros((len(meas_now),1)))
#    
#    for i in range(len(meas_now)):
#        area[i] = cv2.contourArea(contours[i])
#    
#    idx=area.index(max(area))
    del meas_now[:]
    mask2 = np.zeros(thresh.shape,np.uint8)
    cv2.drawContours(mask2,contours,-1,255,-1)
    pixelpoints = cv2.findNonZero(mask2)
    pixelpoints=pixelpoints.reshape((pixelpoints.shape[0], pixelpoints.shape[2]))
    gm_messy = GaussianMixture(n_components=n_inds,covariance_type='full',random_state=42).fit(pixelpoints)
    gm_messy2=gm_messy.predict(pixelpoints)
    
    unique, new_area = np.unique(gm_messy2, return_counts=True)
    l = len(gm_messy.means_)

    for i in range(l):
        x = int(tuple(gm_messy.means_[i])[0])
        y = int(tuple(gm_messy.means_[i])[1])
        meas_now.append([x,y,new_area[i]])
        
    return contours, meas_now

def apply_gm_messy_bigblobs(final,contours,n_inds, meas_now,thresh,vi,cov_type,prec=None):
    
#    area= list(np.zeros((len(meas_now),1)))
#    
#    for i in range(len(meas_now)):
#        area[i] = cv2.contourArea(contours[i])
#    
#    idx=area.index(max(area))
    #del meas_now[:]
#    for i  in range(len(contours)):
#        area = cv2.contourArea(contours[i])
#        n_i=round(area/max_area_real)
    mask2 = np.zeros(thresh.shape,np.uint8)
    cv2.drawContours(mask2,contours,-1,255,-1)
    pixelpoints = cv2.findNonZero(mask2)
    pixelpoints=pixelpoints.reshape((pixelpoints.shape[0], pixelpoints.shape[2]))
    gm_messy = GaussianMixture(n_components=n_inds,covariance_type=cov_type,random_state=42,means_init=vi,precisions_init=prec).fit(pixelpoints)
    gm_messy2=gm_messy.predict(pixelpoints)
        
    unique, new_area = np.unique(gm_messy2, return_counts=True)
    l = len(gm_messy.means_)
    
    for j in range(l):
        x = int(tuple(gm_messy.means_[j])[0])
        y = int(tuple(gm_messy.means_[j])[1])
        meas_now.append([x,y])
    
        
    return meas_now, gm_messy, gm_messy2 


def hungarian_algorithm(meas_last, meas_now,metric):
    """
    The hungarian algorithm is a combinatorial optimisation algorithm used
    to solve assignment problems. Here, we use the algorithm to reduce noise
    due to ripples and to maintain individual identity. This is accomplished
    by minimising a cost function; in this case, euclidean distances between 
    points measured in previous and current step. The algorithm here is written
    to be flexible as the number of contours detected between successive frames
    changes. However, an error will be returned if zero contours are detected.
   
    Parameters
    ----------
    meas_last: array_like, dtype=float
        individual's location on previous frame
    meas_now: array_like, dtype=float
        individual's location on current frame
        
    Returns
    -------
    row_ind: array, dtype=int64
        individual identites arranged according to input ``meas_last``
    col_ind: array, dtype=int64
        individual identities rearranged based on matching locations from 
        ``meas_last`` to ``meas_now`` by minimising the cost function
    """
    meas_last = np.array(meas_last)
    meas_now = np.array(meas_now)
    if meas_now.shape != meas_last.shape:
        if meas_now.shape[0] < meas_last.shape[0]:
            while meas_now.shape[0] != meas_last.shape[0]:
                meas_last = np.delete(meas_last, meas_last.shape[0]-1, 0)
        else:
            result = np.zeros(meas_now.shape)
            result[:meas_last.shape[0],:meas_last.shape[1]] = meas_last
            meas_last = result

    meas_last = list(meas_last)
    meas_now = list(meas_now)
    cost = cdist(meas_last, meas_now,metric='seuclidean')
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind

def reorder_and_draw(final, colours, n_inds, col_ind, meas_now, df, mot, fr_no,t_id):
    """
    This function reorders the measurements in the current frame to match
    identity from previous frame. This is done by using the results of the
    hungarian algorithm from the array col_inds.
    
    Parameters
    ----------
    final: ndarray, shape(n_rows, n_cols, 3)
        final output image composed of the input frame with object contours 
        and centroids overlaid on it
    colours: list, tuple
        list of tuples that represent colours used to assign individual identities
    n_inds: int
        total number of individuals being tracked
    col_ind: array, dtype=int64
        individual identities rearranged based on matching locations from 
        ``meas_last`` to ``meas_now`` by minimising the cost function
    meas_now: array_like, dtype=float
        individual's location on current frame
    df: pandas.core.frame.DataFrame
        this dataframe holds tracked coordinates i.e. the tracking results
    mot: bool
        this boolean determines if we apply the alogrithm to a multi-object
        tracking problem
        
    Returns
    -------
    final: ndarray, shape(n_rows, n_cols, 3)
        final output image composed of the input frame with object contours 
        and centroids overlaid on it
    meas_now: array_like, dtype=float
        individual's location on current frame
    df: pandas.DataFrame
        this dataframe holds tracked coordinates i.e. the tracking results
    """
    # Reorder contours based on results of the hungarian algorithm
#    equal = np.array_equal(col_ind, list(range(len(col_ind))))
#    
#    if equal == False:
#        current_ids = col_ind.copy()
#        reordered = [i[0] for i in sorted(enumerate(current_ids), key=lambda x:x[1])]
#        meas_now = [x for (y,x) in sorted(zip(reordered,meas_now))]
#        coords=np.array(meas_now)[:,0:2]
#    # Draw centroids
#    if mot == False:
#        for i in range(len(meas_now)):
#            if colours[i%4] == (0,0,255):
#                cv2.circle(final, tuple([int(x) for x in coords[i]]), 5, colours[i%4], -1, cv2.LINE_AA)
#    else:
#        for i in range(n_inds):
#            #cv2.circle(final, tuple([int(x) for x in meas_now[i]]), 5, colours[i%n_inds], -1, cv2.LINE_AA)
#            cv2.putText(final, t_id[i], tuple([int(x) for x in coords[i]]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,255,255), 2)
#    # add frame number
#    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
#    cv2.putText(final, str(int(fr_no)), (5,30), font, 1, (255,255,255), 2)
#        
#    return final, meas_now, df

    equal = np.array_equal(col_ind, list(range(len(col_ind))))
    coords=np.array(meas_now)[:,0:2]
    if equal == False:
        current_ids = col_ind.copy()
        reordered = [i[0] for i in sorted(enumerate(current_ids), key=lambda x:x[1])]
        meas_now = [x for (y,x) in sorted(zip(reordered,meas_now))]
        coords=np.array(meas_now)[:,0:2]
    # Draw centroids
    if mot == False:
        for i in range(len(meas_now)):
            if colours[i%4] == (0,0,255):
                cv2.circle(final, tuple([int(x) for x in coords[i]]), 5, colours[i%4], -1, cv2.LINE_AA)
    else:
        for i in range(len(meas_now)):
            #cv2.circle(final, tuple([int(x) for x in meas_now[i]]), 5, colours[i%n_inds], -1, cv2.LINE_AA)
            cv2.putText(final, t_id[i], tuple([int(x) for x in coords[i]]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,0,0), 2)
    # add frame number
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    cv2.putText(final, str(int(fr_no)), (5,30), font, 1, (255,255,255), 2)
        
    return final, meas_now, df



def reject_outliers(data, m):
    """
    This function removes any outliers from presented data.
    
    Parameters
    ----------
    data: pandas.Series
        a column from a pandas dataframe that needs smoothing
    m: float
        standard deviation cutoff beyond which, datapoint is considered as an outlier
        
    Returns
    -------
    index: ndarray
        an array of indices of points that are not outliers
    """
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d/mdev if mdev else 0.
    return np.where(s < m)







