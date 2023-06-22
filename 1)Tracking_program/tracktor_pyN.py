import numpy as np
import cv2
import random


from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))


def detect_and_draw_contours(frame, thresh, meas_last, meas_now, min_area = 0, max_area = 10000, seuil=1, max_area_real=10000):
    
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
        del meas_now[:]
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
                    i += 1
    return final, contours_test, meas_last, meas_now, j

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