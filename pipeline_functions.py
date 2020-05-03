import os
import pickle as pl
import cv2
import numpy as np
import matplotlib.image as mpimg

#import all necessary params from params directory
for file in os.listdir("params/"):
        if file[-3:]=='.pl':
            paramname = file[:-3]

            with open ("params/"+file, 'rb') as f:
                globals()[paramname] = pl.load(f)

def extract_binary_mask(img, roi=vertices):

    #convert the image to hsv color space and extract the channels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    colorthresh = np.zeros_like(v)
    colorthresh[(v>vmin1)|(s>smin1)] = 1

    colorthresh2 = np.zeros_like(v)
    colorthresh2[v>vmin2] = 1

    # Sobel x
    sobelx = cv2.Sobel(v, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    #very soft margin on the gradient ensuring, that detected pixels are at least slightly part of an vertical structure
    # suppressing shadows withot vertical preferential direction
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sobelx1_min) & (scaled_sobel <= sobelx1_max)] = 1

    sobelx2 = cv2.Sobel(s, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx2 = np.absolute(sobelx2) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel2 = np.uint8(255*abs_sobelx2/np.max(abs_sobelx2))
    
    sxbinary2 = np.zeros_like(scaled_sobel2)
    sxbinary2[(scaled_sobel2 >= sobelx2_min) & (scaled_sobel2 <= sobelx2_max)] = 1

    #defining a blank mask to start with
    mask = np.zeros_like(colorthresh)   
    
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    #cv2.fillPoly(mask, vertices, 1)
    cv2.fillPoly(mask, roi, 1)

    combined_binary = np.zeros_like(colorthresh)
    combined_binary[(((colorthresh == 1) & (sxbinary == 1))|((colorthresh2 == 1)&(sxbinary2 == 1))) & (mask == 1)] = 1


    #blurring suppresses noise
    blurred = cv2.GaussianBlur(combined_binary, (5, 5), 0)

    kernel = np.ones((5,5),np.uint8)

    #dilatation fills pixels within structures
    dilation = cv2.dilate(blurred,kernel,iterations = 2)
    #erosion sharpens the edges of structures
    erosion = cv2.erode(dilation,kernel,iterations = 1)
    
    return erosion

def unwarp(img, M):
    warped = cv2.warpPerspective(img, M, (xsize, ysize), flags=cv2.INTER_LINEAR)

    return warped


def find_lane_pixels(binary_warped):
    #determine a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    #find the peak of the left and right halves of the histogram
    #these will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #define the number of sliding windows
    nwindows = 9
    #set the width of the windows +/- margin
    margin = 100
    #set minimum number of pixels found to recenter window
    minpix = 50

    #set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    #identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    #current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    #create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    #create empty lists to receive lower and upper boundaries of the estimated windows
    returnWindows_low = []
    returnWindows_high = []
    
    #step through the windows one by one
    for window in range(nwindows):
        #identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        #determine the boundaries of the window
        win_xleft_low = leftx_current-margin  # Update this
        win_xleft_high = leftx_current+margin   # Update this
        win_xright_low = rightx_current-margin  # Update this
        win_xright_high = rightx_current+margin  # Update this
        
        #append the windows to the lists collecting them to be returned
        returnWindows_low.append((win_xleft_low,win_y_low))
        returnWindows_high.append((win_xleft_high,win_y_high))
        
        returnWindows_low.append((win_xright_low,win_y_low))
        returnWindows_high.append((win_xright_high,win_y_high))
        
        #identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        #append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        #if you found > minpix pixels, recenter next window
        #(`right` or `leftx_current`) on their mean position
        if len(good_left_inds)>minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds)>minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    #concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    #extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return (leftx, lefty), (rightx, righty), returnWindows_low, returnWindows_high


def fit_polynomial(left, right):
    #fit 2nd order polynomial to both lane markings
    try:
        left_fit = np.polyfit(left[1], left[0], 2)
    except:
        left_fit = None
    try:
        right_fit = np.polyfit(right[1], right[0], 2)
    except:
        right_fit = None
        
    return left_fit, right_fit


def calc_poly_vals(fit, y):
    x = fit[0]*y**2 + fit[1]*y + fit[2]
    warped = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), M_rev)
    return x, y, warped[0][0][0], warped[0][0][1]

    
def measure_curvature(y_eval, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in pixels or with the correct input in meters
    '''
    
    left_curverad = ((1+(2*left_fit[0]*y_eval+left_fit[1])**2)**(3/2))/np.abs(2*left_fit[0])
    right_curverad = ((1+(2*right_fit[0]*y_eval+right_fit[1])**2)**(3/2))/np.abs(2*right_fit[0]) 
    
    return left_curverad, right_curverad


def visualize_polyfit(binary_warped, colorize_markings=True, show_windows=True,\
                      suppress_noise=False, colorize_lane=False, rewarp=False, show_crv=False, verbose=False, history=None):

    #initialize replace flags and last reliable lane width in pixels
    replaced_l = False
    replaced_r = False

    if history is not None:
        lastReliableWidth_px = history[4]
    else:
        lastReliableWidth_px = 750
        
    #define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7/820 # meters per pixel in x dimension
    ym_per_pix = 30/720 # meters per pixel in y dimension    
    
    #determine the pixels belonging to the left and right marking
    left, right, returnWindows_low, returnWindows_high = find_lane_pixels(binary_warped)

    #calculate reliability
    reliability_l = np.min([1.0, len(left[0])/80000])
    reliability_r = np.min([1.0, len(right[0])/80000])
   
    #init output image
    if suppress_noise:
        #without the pixels which were identified as noise
        out_img = np.dstack((np.zeros_like(binary_warped), np.zeros_like(binary_warped), np.zeros_like(binary_warped)))
    else:
        #without all pixels
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    #fit 2nd ordder polynomial to both lane markings
    left_fit_ret, right_fit_ret = fit_polynomial(left, right)

    #if only one of the lane marking polynomials couldn't be fitted it is replaced with the other one and shifted
    if left_fit_ret is None and right_fit_ret is not None:
        left_fit_ret = np.copy(right_fit_ret)
        left_fit_ret[-1] -= lastReliableWidth_px
        reliability_l = 0
        replaced_l = True
    if left_fit_ret is not None and right_fit_ret is None:
        right_fit_ret = np.copy(left_fit_ret)
        right_fit_ret[-1] += lastReliableWidth_px
        reliability_r = 0
        replaced_r = True
        
    #applying smoothing with history
    if history is not None:
        #extract reliabilities and polynomial coeeficients from history
        Reliability_l = list(np.copy(history[0]))
        Reliability_r = list(np.copy(history[1]))
        Leftfit = list(np.copy(history[2]))
        Rightfit = list(np.copy(history[3]))
        
        #localy append current fits to history
        #a replaced value is here treated with a reliability of 0.15 instead of 0 to account for 
        #situations with empty histories (all reliabilities are 0 as initial)
        if replaced_l is False:
            Reliability_l.append(reliability_l)
        else:
            Reliability_l.append(0.15)
        if replaced_r is False:
            Reliability_r.append(reliability_r)
        else:
            Reliability_r.append(0.15)
        
        Leftfit.append(left_fit_ret)
        Rightfit.append(right_fit_ret)
        
        Reliability_l_copy = list(np.copy(Reliability_l))
        Reliability_r_copy = list(np.copy(Reliability_r))
        
        #calculate relative weights
        weights_l_copy = Reliability_l_copy / np.sum(Reliability_l_copy)
        weights_r_copy = Reliability_r_copy / np.sum(Reliability_r_copy)
        
        #multiply reliabilty values to forget values which are calculated longer ago
        Reliability_l[0]*=0.1
        Reliability_l[1]*=0.2
        Reliability_l[2]*=0.3
        Reliability_l[3]*=0.4
        Reliability_l[4]*=0.5
        Reliability_l[5]*=0.6
        Reliability_l[6]*=0.7
        Reliability_l[7]*=0.8
        Reliability_l[8]*=0.9
        Reliability_l[9]*=1.0
        
        Reliability_r[0]*=0.1
        Reliability_r[1]*=0.2
        Reliability_r[2]*=0.3
        Reliability_r[3]*=0.4
        Reliability_r[4]*=0.5
        Reliability_r[5]*=0.6
        Reliability_r[6]*=0.7
        Reliability_r[7]*=0.8
        Reliability_r[8]*=0.9
        Reliability_r[9]*=1.0
        
        #calculate relative weights
        weights_l = Reliability_l / np.sum(Reliability_l)
        weights_r = Reliability_r / np.sum(Reliability_r)

        #calculate the new fits as weighted averages
        #for the offset value weights without forgetting information used, 
        #where for the more dynamic coeeficients forgetting weights are used
        Leftfit = np.array(Leftfit)
        left_fit = [np.average(Leftfit[:, 0], weights=weights_l),\
                     np.average(Leftfit[:, 1], weights=weights_l),\
                     np.average(Leftfit[:, 2], weights=weights_l_copy)]
        
        
        Rightfit = np.array(Rightfit)
        right_fit = [np.average(Rightfit[:, 0], weights=weights_r),\
                     np.average(Rightfit[:, 1], weights=weights_r),\
                     np.average(Rightfit[:, 2], weights=weights_r_copy)]
        
    else:
        left_fit = left_fit_ret
        right_fit = right_fit_ret  

    if (replaced_l is False) and (replaced_r is False):
        #check if the intersection of the two polynomials is within the plotting range
        a = left_fit[0]-right_fit[0]
        b = left_fit[1]-right_fit[1]
        c = left_fit[2]-right_fit[2]
        #calculate the two numerical possible solutions
        sol1y = (-b+np.sqrt((b**2)-(4*a*c)))/(2*a)
        sol2y = (-b-np.sqrt((b**2)-(4*a*c)))/(2*a)

        if not np.isnan(sol1y):
            #convert first solution to integer (full pixels) and obtain the position of the intersection in x
            sol1y = np.int(sol1y)
            sol1x, sol1y, sol1x_imgspace, sol1y_imgspace = calc_poly_vals(left_fit, sol1y)

            #if the intersection lies in the plotting range something is wrong and the less reliable marking is replaced
            if sol1y_imgspace>440 and sol1y_imgspace<720 and sol1x_imgspace>180 and sol1x_imgspace<1200:
                if reliability_l < reliability_r:
                    reliability_l = 0.0
                    left_fit = np.copy(right_fit)
                    left_fit[-1] -= lastReliableWidth_px
                else:
                    reliability_r = 0.0
                    right_fit = np.copy(left_fit)
                    right_fit[-1] += lastReliableWidth_px

        if not np.isnan(sol2y):
            #convert second solution to integer (full pixels) and obtain the position of the intersection in x
            sol2y = np.int(sol2y)
            sol2x, sol2y, sol2x_imgspace, sol2y_imgspace = calc_poly_vals(left_fit, sol2y)

            #if the intersection lies in the plotting range something is wrong and the less reliable marking is replaced
            if sol2y_imgspace>440 and sol2y_imgspace<720 and sol2x_imgspace>180 and sol2x_imgspace<1200:
                if reliability_l < reliability_r:
                    reliability_l = 0.0
                    left_fit = np.copy(right_fit)
                    left_fit[-1] -= lastReliableWidth_px
                else:
                    reliability_r = 0.0
                    right_fit = np.copy(left_fit)
                    right_fit[-1] += lastReliableWidth_px
    
        #calculate position of top and bottom points of both lane markings
        left_x_top, left_y_top, left_x_top_imgspace, left_y_top_imgspace = calc_poly_vals(left_fit, 100)
        right_x_top, right_y_top, right_x_top_imgspace, right_y_top_imgspace = calc_poly_vals(right_fit, 100)

        left_x_bottom, left_y_bottom, left_x_bottom_imgspace, left_y_bottom_imgspace = calc_poly_vals(left_fit, 690)
        right_x_bottom, right_y_bottom, right_x_bottom_imgspace, right_y_bottom_imgspace = calc_poly_vals(right_fit, 690)
    
        #check if the top corners are within a certain range
        if (left_x_top_imgspace<450 or left_x_top_imgspace>650\
           or left_x_bottom_imgspace<200 or left_x_bottom_imgspace>430) and reliability_l<reliability_r:
            reliability_l = 0.0
            left_fit = np.copy(right_fit)
            left_fit[-1] -= lastReliableWidth_px

        if (right_x_top_imgspace<660 or right_x_top_imgspace>860\
            or right_x_bottom_imgspace<900 or right_x_bottom_imgspace>1200) and reliability_r<reliability_l:
            reliability_r = 0.0
            right_fit = np.copy(left_fit)
            right_fit[-1] += lastReliableWidth_px

        if (right_x_bottom_imgspace-left_x_bottom_imgspace)*xm_per_pix<2.2:
            if reliability_l<reliability_r:
                reliability_l = 0.0
                left_fit = np.copy(right_fit)
                left_fit[-1] -= lastReliableWidth_px
            else:
                reliability_r = 0.0
                right_fit = np.copy(left_fit)
                right_fit[-1] += lastReliableWidth_px
    
    #set the y value for the evaluation to the maximum value and convert it to meters
    y_eval = 690*ym_per_pix

    #obtain polynom in metric space
    left_fit_cr = [(xm_per_pix/(ym_per_pix**2))*left_fit[0], (xm_per_pix/ym_per_pix)*left_fit[1], left_fit[2]]
    right_fit_cr = [(xm_per_pix/(ym_per_pix**2))*right_fit[0], (xm_per_pix/ym_per_pix)*right_fit[1], right_fit[2]]

    #calculate the curvatures in meters
    crv = np.round(measure_curvature(y_eval, left_fit_cr, right_fit_cr),2)
    
    #if the two calculated curvatures are inconsistent the less reliable estimation is replaced
    if (crv[0]/crv[1])>2 or (crv[0]/crv[1])<0.5:
        if reliability_l<reliability_r:
            reliability_l = 0.0
            left_fit = np.copy(right_fit)
            left_fit[-1] -= lastReliableWidth_px
        else:
            reliability_r = 0.0
            right_fit = np.copy(left_fit)
            right_fit[-1] += lastReliableWidth_px
            
        #recalculate curvature values
        left_fit_cr = [(xm_per_pix/(ym_per_pix**2))*left_fit[0], (xm_per_pix/ym_per_pix)*left_fit[1], left_fit[2]]
        right_fit_cr = [(xm_per_pix/(ym_per_pix**2))*right_fit[0], (xm_per_pix/ym_per_pix)*right_fit[1], right_fit[2]]

        #calculate the curvatures in meters
        crv = np.round(measure_curvature(y_eval, left_fit_cr, right_fit_cr),2)
    
    #calculate position of top and bottom points of both lane markings
    #has to be recalculated as the fits could have been adapted above
    left_x_top, left_y_top, left_x_top_imgspace, left_y_top_imgspace = calc_poly_vals(left_fit, 100)
    right_x_top, right_y_top, right_x_top_imgspace, right_y_top_imgspace = calc_poly_vals(right_fit, 100)

    left_x_bottom, left_y_bottom, left_x_bottom_imgspace, left_y_bottom_imgspace = calc_poly_vals(left_fit, 690)
    right_x_bottom, right_y_bottom, right_x_bottom_imgspace, right_y_bottom_imgspace = calc_poly_vals(right_fit, 690)

    width = (right_x_bottom_imgspace-left_x_bottom_imgspace)*xm_per_pix
    lanecenter = (width/2)+left_x_bottom_imgspace*xm_per_pix
    imgcenter = (xsize/2)*xm_per_pix
    dist_cl = lanecenter - imgcenter

    if colorize_markings:
        #colorize pixels belonging to the markings
        out_img[left[1], left[0]] = [255, 0, 0]
        out_img[right[1], right[0]] = [0, 0, 255]
    
    if show_windows:
        #add windows on top of the output_images
        for wind_low, wind_high in zip(returnWindows_low, returnWindows_high):
            cv2.rectangle(out_img, wind_low, wind_high, (0,255,0), 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #change format of x and y points
    pts_l = []
    pts_r = []
    for x_l, x_r, y in zip(left_fitx, right_fitx, ploty):
        pts_l.append([x_l, y])
        pts_r.append([x_r, y])

    pts_l=np.int32(np.array([pts_l]))
    pts_r=np.int32(np.array([pts_r]))
    
    #plot fitted polynomials
    cv2.polylines(out_img, pts_l, False, (255, 255, 0), thickness=2)
    cv2.polylines(out_img, pts_r, False, (255, 255, 0), thickness=2)

    #colorize pixels between the lane amrkings
    if colorize_lane:
        lane_mask = np.zeros_like(out_img)

        arr = []
        for el in pts_l[0]:
            arr.append(el)
        for el in np.flip(pts_r[0], 0):
            arr.append(el)
        
        arr = np.array([arr])
        
        cv2.fillPoly(lane_mask, arr, (0, 255, 255))
        cv2.addWeighted(out_img, 0.5, lane_mask, 0.2, 0.0, out_img)

    if rewarp:
        #rewarp the output image
        out_img = cv2.warpPerspective(out_img, M_rev, (xsize, ysize), flags=cv2.INTER_LINEAR)

    if show_crv:
        #put the generated information as text overlay on the image
        cv2.putText(out_img,\
                    'Width of the Lane = '+str(np.round(width,2))+'m',\
                    (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        cv2.putText(out_img,\
                    'Radius of Curvature = '+str(crv[np.argmax([reliability_l, reliability_r])])+'m',\
                    (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        cv2.putText(out_img,\
                    'Distance to Lane Center = '+str(np.round(dist_cl, 2))+'m',\
                    (100, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        
        left_x_bottom = left_fit[0]*690**2 + left_fit[1]*690 + left_fit[2]
        right_x_bottom = right_fit[0]*690**2 + right_fit[1]*690 + right_fit[2]

    if verbose is False:
        return out_img
    else:
        return out_img, reliability_l, reliability_r, left_fit_ret, right_fit_ret, width, crv[np.argmax([reliability_l, reliability_r])]


def extract_original_pixels(img):
    #correct image distortion
    img = cv2.undistort(img, mtx, dist, None, None)
    
    #extract binary mask
    mask = extract_binary_mask(img)
    
    #return the pixels of the original image that are highly likely to belong to the lane markings
    ret = np.copy(img)
    ret[mask!=1]=0

    return ret


def show_rewarped_fitted_poly(img):
    #correct image distortion
    img = cv2.undistort(img, mtx, dist, None, None)
    
    #extract binary mask
    mask = extract_binary_mask(img)
    
    #unwarp the binary mask
    unwarped = unwarp(mask, M)

    #calculate image with fitted polynomials
    rewarped_out_img = visualize_polyfit(unwarped, colorize_markings=True, show_windows=False,\
                                        suppress_noise=True, colorize_lane=True, rewarp=True, show_crv=False)
    
    return rewarped_out_img


def process_image(img):
    #correct image distortion
    img = cv2.undistort(img, mtx, dist, None, None)
    
    #extract binary mask
    mask = extract_binary_mask(img)
    
    #unwarp the binary mask
    unwarped = unwarp(mask, M)

    #calculate image with fitted polynomials
    rewarped_out_img = visualize_polyfit(unwarped, colorize_markings=True, show_windows=False,\
                                        suppress_noise=True, colorize_lane=True, rewarp=True, show_crv=True)

    final = np.copy(img)
    cv2.addWeighted(final, 0.5, rewarped_out_img, 1.0, 0.0, final)

    return final


