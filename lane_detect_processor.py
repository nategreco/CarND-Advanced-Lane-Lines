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

#Local project imports

#Code

#TODO
#Create lane class
#Create lane averaging function
#Create camera calibration function
#Create process image function
	#Remove distortion from camera
	#Apply HLS thresholding
	#Apply Sobel thresholding
	#Combine masks
	#Alter perspective
	#Fit polygons