################################################################################
#   Date:    07.07.2017
#   Author:  Nathan Greco (Nathan.Greco@gmail.com)
#
#   Project:
#       CarND-Advanced-Lane-Lines: Fourth project in Term 1 of Self-Driving
#           Car Nanodegree curriculum.
#
#   Module:
#       lane_detect_processor: Contains tools necessary for detecting road
#           lanes in a given image.
#
#   Repository:
#       http://github.com/NateGreco/CarND-Advanced-Lane-Lines.git
#
#   License:
#       Part of Udacity Self-Driving Car Nanodegree curriculum.
#
#   Notes:
#       Following google style guide here:
#       https://google.github.io/styleguide/pyguide.html
#      
################################################################################

#System imports

#3rd party imports
import cv2
import numpy as np

#Local project imports

#Constants
ROI_SF = np.array([[(0.00, 1.00), \
                    (0.00, 0.95), \
                    (0.45, 0.60), \
                    (0.55, 0.60), \
                    (1.00, 0.95), \
                    (1.00, 1.00)]], \
                  dtype=np.float64)
WHITE_LOWER_THRESH = (0, 40, 80)
WHITE_UPPER_THRESH = (180, 255, 255)
YELLOW_LOWER_THRESH = (0, 0, 80)
YELLOW_UPPER_THRESH = (90, 255, 255)
SOBEL_X_LOWER_THRESH = 0
SOBEL_X_UPPER_THRESH = 255
SOBEL_Y_LOWER_THRESH = 0
SOBEL_Y_UPPER_THRESH = 255

#Classes
class Line(): #TODO - Just copied from Udacity course
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

#Functions
def extract_roi(image): #TODO - Incomplete
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #Create blank mask
    mask = np.zeros_like(image)   
    #Fill based on number of channels
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #Scale ROI vertices
    vertices = np.copy(ROI_SF)
    for i in range(0, len(vertices[0])):
        vertices[0][i][0] *= float(image.shape[1])
        vertices[0][i][1] *= float(image.shape[0])
    vertices = vertices.astype(int)
    #Fill polygon created by roi vertices
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def average_lines(lines, new_line, num_to_keep): #TODO - Finish line class
    lines.append(new_line)
    #Check size of list
    if len(lines) > num_to_keep:
        del lines[0]
    #Average
    detected = 0
    for line in lines:
        if line.detected:
            detected += 1
            #TODO - Some sort of sum here
    #TODO - Divide by detected
    return

def calibrate_camera(images, x_pts, y_pts):
    #Assert all images are same shape
    assert(all(i.shape == images[0].shape) for i in images)
    #Prepare variables
    objp = np.zeros((y_pts * x_pts, 3), np.float32)
    objp[:,:2] = np.mgrid[0:x_pts, 0:y_pts].T.reshape(-1,2)
    objpoints = []
    imgpoints = []
    #Iterate through each image
    success_count = 0
    for image in images:
        #Change color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Find corners
        ret, corners = cv2.findChessboardCorners(gray, (x_pts, y_pts), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            success_count += 1
    #Check for succesful checkerboard detections
    assert(success_count != 0)
    print(success_count, " out of ", len(images), \
        " checkerboards detected, Calibration complete!")
    #Get matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera( objpoints, \
        imgpoints, \
        gray.shape[::-1], None, None )
    return mtx, dist

def hls_threshold(image): #TODO - Incomplete
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    #White threshold
    white_binary = cv2.inRange(hls, WHITE_LOWER_THRESH, WHITE_UPPER_THRESH)
    #Yellow_threshold
    yellow_binary = cv2.inRange(hls, YELLOW_LOWER_THRESH, YELLOW_UPPER_THRESH)
    #Combine masks
    mask = np.zeros_like(white_binary)
    mask[(white_binary == 1) | (yellow_binary == 1)] = 1
    return mask

def gradient_threshold(image): #TODO - Incomplete
    #Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #Get derivative with respect to x
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel_x = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))
    #Threshold
    sxbinary = np.zeros_like(scaled_sobel_x)
    sxbinary[(scaled_sobel_x >= SOBEL_X_LOWER_THRESH) & \
        (scaled_sobel_x <= SOBEL_X_UPPER_THRESH)] = 1
    #Get derivative with respect to y - maybe unecessary?
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel_y = np.absolute(sobel_y)
    scaled_sobel_y = np.uint8(255*abs_sobel_y/np.max(abs_sobel_y))
    #Threshold
    sybinary = np.zeros_like(scaled_sobel_y)
    sybinary[(scaled_sobel_y >= SOBEL_Y_LOWER_THRESH) & \
        (scaled_sobel_y <= SOBEL_Y_UPPER_THRESH)] = 1
    #Combine masks
    mask = np.zeros_like(sxbinary)
    mask[(sxbinary == 1) | (sybinary == 1)] = 1
    return mask


def detect_lines(image, mtx, dist): #TODO - Incomplete
    #Extract ROI - TODO
    image = extract_roi(image)
    #Undistort image
    true_image = cv2.undistort(image, mtx, dist, None, mtx)
    #Color threshold
    color_mask = hls_threshold(true_image)
    #Sobel threshold
    gradient_mask = gradient_threshold(true_image)
    #Combine masks
    binary = np.zeros_like(color_mask)
    binary[(color_mask == 1) & (gradient_mask == 1)] = 1
    #Create lines
    left_line = Line()
    right_line = Line()
    return left_line, right_line