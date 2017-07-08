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
import numpy as np
import matplotlib.pyplot as plt

#3rd party imports
import cv2

#Local project imports

#Constants
ROI_SF = np.array([[(0.00, 0.95), \
                    #(0.10, 0.80), \
                    (0.43, 0.62), \
                    (0.57, 0.62), \
                    #(0.90, 0.80), \
                    (1.00, 0.95)]], \
                  dtype=np.float64)
WHITE_LOWER_THRESH_1 = np.array([0, 10, 6])
WHITE_UPPER_THRESH_1 = np.array([30, 255, 255])
WHITE_LOWER_THRESH_2 = np.array([150, 10, 6]) #Should be same as 1 except Hue
WHITE_UPPER_THRESH_2 = np.array([180, 255, 255]) #Should be same as 1 except Hue
YELLOW_LOWER_THRESH = np.array([5, 10, 45])
YELLOW_UPPER_THRESH = np.array([55, 255, 255])
SOBEL_X_LOWER_THRESH = 20
SOBEL_X_UPPER_THRESH = 255
SOBEL_Y_LOWER_THRESH = 255 #Not in use when equal to upper
SOBEL_Y_UPPER_THRESH = 255 #Not in use when equal to lower
#Calculate the birds eye view matrix transformation
SRC = np.float32([[580, 500], \
                  [757, 500], \
                  [309, 700], \
                  [1094, 700]])
DST = np.float32([[280, 0], \
                  [1094, 0], \
                  [280, 720], \
                  [1094, 720]])
BEV_MATRIX = cv2.getPerspectiveTransform(SRC, DST)
INV_MATRIX = cv2.getPerspectiveTransform(DST, SRC)
NUM_WINDOWS = 4
WINDOW_WIDTH = 150
MIN_PIXELS = 50
LINE_THICKNESS = 10.0

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

#Global variables
left_line = Line()
right_line = Line()

#Functions
def extract_roi(image):
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

def trim_roi(image, pixels):
    """
    This function trims the image's ROI to reduce any false gradients
    created by the edge of the ROI.
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
    #Erode the mask by pixel count
    assert(pixels % 2 != 0) #Check for odd kernel size
    kernel = np.ones((pixels,pixels), np.uint8)
    mask = cv2.erode(mask, kernel)
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

def hls_threshold(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    #White threshold
    white_binary_1 = cv2.inRange(hls, WHITE_LOWER_THRESH_1, WHITE_UPPER_THRESH_1)
    white_binary_2 = cv2.inRange(hls, WHITE_LOWER_THRESH_2, WHITE_UPPER_THRESH_2)
    white_binary = cv2.bitwise_or(white_binary_1, white_binary_2)
    #Yellow_threshold
    yellow_binary = cv2.inRange(hls, YELLOW_LOWER_THRESH, YELLOW_UPPER_THRESH)
    #Combine masks
    mask = cv2.bitwise_or(white_binary, yellow_binary)
    return mask

def gradient_threshold(image):
    #Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Get derivative with respect to x
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel_x = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))
    #Threshold
    sxbinary = cv2.inRange(scaled_sobel_x, SOBEL_X_LOWER_THRESH, SOBEL_X_UPPER_THRESH)
    #Get derivative with respect to y - maybe unecessary?
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel_y = np.absolute(sobel_y)
    scaled_sobel_y = np.uint8(255 * abs_sobel_y / np.max(abs_sobel_y))
    #Threshold
    sybinary = cv2.inRange(scaled_sobel_y, SOBEL_Y_LOWER_THRESH, SOBEL_Y_UPPER_THRESH)
    #Combine masks
    mask = cv2.bitwise_or(sxbinary, sybinary)
    return mask

def fit_lines(image):
    #Take a histogram of the bottom half of the image
    histogram = np.sum(image[image.shape[0] // 2:,:], axis=0)
    #Sliding window methodology
    #Create an output image to draw on and  visualize the result
    output_image = np.dstack((image, image, image)) * 255
    window_height = np.int(image.shape[0] / NUM_WINDOWS)
    #Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    #Create startpoints
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_current = np.argmax(histogram[:midpoint])
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint
    #Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    #Step through the windows one by one
    for window in range(NUM_WINDOWS):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window + 1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_xleft_low = int(leftx_current - WINDOW_WIDTH / 2)
        win_xleft_high = int(leftx_current + WINDOW_WIDTH / 2)
        win_xright_low = int(rightx_current - WINDOW_WIDTH / 2)
        win_xright_high = int(rightx_current + WINDOW_WIDTH / 2)
        #Draw the windows on the visualization image
        cv2.rectangle(output_image, \
                      (win_xleft_low, win_y_low), \
                      (win_xleft_high, win_y_high), \
                      (0, 255, 0), \
                      2)
        cv2.rectangle(output_image, \
                      (win_xright_low, win_y_low), \
                      (win_xright_high, win_y_high),\
                      (0, 255, 0), \
                      2)
        #Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & \
                          (nonzeroy < win_y_high) & \
                          (nonzerox >= win_xleft_low) \
                          & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & \
                           (nonzeroy < win_y_high) & \
                           (nonzerox >= win_xright_low) & \
                           (nonzerox < win_xright_high)).nonzero()[0]
        #Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        #If you found > MIN_PIXELS, recenter next window on their mean position
        if len(good_left_inds) > MIN_PIXELS:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > MIN_PIXELS:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    #Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    #Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    #Fit a second order polynomial to each
    left_poly = np.polyfit(lefty, leftx, 2)
    right_poly = np.polyfit(righty, rightx, 2)
    return left_poly, right_poly, output_image

def plot_lines(image, left_line, right_line):
    #Check image resolution evenly divides by dpi
    dpi = 80
    assert((image.shape[0] % dpi == 0) & (image.shape[1] % dpi == 0))
    #Generate a figure with matplotlib
    figure = plt.figure(num=None, \
                        figsize=(int(image.shape[1] / dpi), int(image.shape[0] / dpi)), \
                        dpi=dpi)
    plot = figure.add_subplot(111)
    #Place image on plot
    if len(image.shape) != 3: #Check if gray or color image!
        image = np.dstack((image, image, image)) * 255
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plot.imshow(image)
    plot.axis('off')
    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    #Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    left_fitx = left_line.current_fit[0] * ploty**2 + \
                left_line.current_fit[1] * ploty + \
                left_line.current_fit[2]
    right_fitx = right_line.current_fit[0] * ploty**2 + \
                 right_line.current_fit[1] * ploty + \
                 right_line.current_fit[2]    
    plot.plot(left_fitx, ploty, color='blue', linewidth=LINE_THICKNESS)
    plot.plot(right_fitx, ploty, color='blue', linewidth=LINE_THICKNESS)
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    #Draw the renderer
    figure.canvas.draw()
    #Get the RGB buffer from the figure
    output_image = np.fromstring (figure.canvas.tostring_rgb(), dtype=np.uint8)
    output_image.shape = (image.shape[0], image.shape[1], 3)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
	#Transform back to origianl perspective
    output_image = cv2.warpPerspective(output_image, \
                                       INV_MATRIX, \
                                       (image.shape[1], image.shape[0]))
    return output_image

def combine_images(bottom, top):
    assert(bottom.shape == top.shape)
    #Overlay method
    #output_image = cv2.addWeighted(bottom, 1.0, top, 1.0, gamma=0.0)
    #Non-zero pixel method
    nonzero = top.nonzero()
    output_image = bottom
    for i in range(len(nonzero[0])):
        output_image[nonzero[0][i]][nonzero[1][i]] = \
            top[nonzero[0][i]][nonzero[1][i]]
    return output_image

def shade_lines(image, left_line, right_line):
    #Check that two valid lines exist
    #if !(left_line.detected & right_line.detected): return image
    shade_image = np.zeros_like(image)
    line_image = plot_lines(shade_image, left_line, right_line)
    shade_color = [0, 255, 0]
    left_switch = False
    right_switch = False
    for i in range(0, image.shape[0]):
        left_switch = False
        right_switch = False
        for j in range(0, image.shape[1]):
            if left_switch: shade_image[i][j] = shade_color
            if (not left_switch) & (not right_switch) & np.any(line_image[i][j] != 0):
                left_switch = True
            elif left_switch & (not right_switch) & np.all(line_image[i][j] == 0):
                right_switch = True
            elif left_switch & right_switch & np.any(line_image[i][j] != 0):
                break
            elif left_switch & right_switch & (j == (image.shape[1] - 1)):
                 shade_image[i].fill(0)
    #Shade lane area
    output_image = cv2.addWeighted(image, 1.0, shade_image, 1.0, gamma=0.0)
    #Draw lines
    output_image = combine_images(output_image, line_image)
    return output_image


def detect_lines(image, mtx, dist): #TODO - Incomplete
    #Work with working copy
    working_image = image.copy()
    #Normalize imaeg
    cv2.normalize(working_image, working_image, 0, 255, cv2.NORM_MINMAX)
    #Undistort image
    true_image = cv2.undistort(working_image, mtx, dist, None, mtx)
    #Extract ROI
    roi_image = extract_roi(true_image)
    #Color threshold
    color_mask = hls_threshold(roi_image)
    #Sobel threshold
    gradient_mask = gradient_threshold(roi_image)
    #Trim gradient mask to remove false edges
    gradient_mask = trim_roi(gradient_mask, 3)
    #Combine masks
    binary = cv2.bitwise_and(color_mask, gradient_mask)
    #Warp to birds-eye-view perspective
    bev_image = cv2.warpPerspective(binary, \
                                    BEV_MATRIX, \
                                    (binary.shape[1], binary.shape[0]))
    #Find lines
    left_poly, right_poly, visual_image = fit_lines(bev_image)
    left_line.current_fit = left_poly
    right_line.current_fit = right_poly
    #Create output image
    output_image = shade_lines(true_image, left_line, right_line)
    #Temporary test image
    #output_image = cv2.bitwise_and(working_image, cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR))
    #output_image = cv2.bitwise_and(working_image, cv2.cvtColor(gradient_mask, cv2.COLOR_GRAY2BGR))
    #output_image = cv2.bitwise_and(working_image, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
    #output_image = binary
    #output_image = bev_image
    #output_image = plot_lines(visual_image, left_line, right_line)
    return left_line, right_line, output_image