################################################################################
#   Date:    07.07.2017
#   Author:  Nathan Greco (Nathan.Greco@gmail.com)
#
#   Project:
#       CarND-Advanced-Lane-Lines: Fourth project in Term 1 of Self-Driving
#           Car Nanodegree curriculum.
#
#   Module:
#       video_processor.py: Main module which handles video arguments, detects
#           lane lines, shades area, and displays road radius.
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
import sys
import os

#3rd party imports
import cv2

#Local project imports
import video_processor_tools as vidtools
import lane_detect_processor as lanetools

#Constants
CAL_PATH = '.\\camera_cal'
CAL_PTS_X = 9
CAL_PTS_Y = 6
LINES_TO_AVERAGE = 5

#Code
#Check for arguments
if len(sys.argv) < 2:
    input("No arguments passed, press ENTER to exit...")
    quit()

#Perform camera calibration
image_files = [f for f in os.listdir(CAL_PATH) \
    if os.path.isfile(os.path.join(CAL_PATH, f))]
images = []
for file in image_files:
    image = cv2.imread(CAL_PATH + '\\' + file)
    images.append(image)
cal_mtx, cal_dist = lanetools.calibrate_camera(images, CAL_PTS_X, CAL_PTS_Y)

#Iterate through video files
for i in range(1, len(sys.argv)):
    video_in = cv2.VideoCapture(sys.argv[i])   
    video_out = cv2.VideoWriter(vidtools.rename_output_file(sys.argv[i]), \
        int(video_in.get(cv2.CAP_PROP_FOURCC)), \
        video_in.get(cv2.CAP_PROP_FPS), \
        (int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH)), \
         int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    #Create empty line arrays
    left_lines = []
    right_lines = []
    while(video_in.isOpened()):
        result, frame = video_in.read()
        if result==False:
            print("File completed: " + sys.argv[i])
            break

        #Process frame to find lines
        left_line, right_line, test_image = \
            lanetools.detect_lines(frame, \
            cal_mtx, \
            cal_dist)
        #Average lines for smoothing
        left_line = lanetools.average_lines(left_lines, \
            left_line, \
            LINES_TO_AVERAGE)
        right_line = lanetools.average_lines(right_lines, \
            right_line, \
            LINES_TO_AVERAGE)
        #Draw averaged lines on original image

        #Write new frame
        video_out.write(frame)
        #Display new frame
        cv2.imshow('Results', test_image)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_in.release()
    video_out.release()

cv2.destroyAllWindows()