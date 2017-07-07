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
import time

#3rd party imports
import cv2

#Local project imports
import video_processor_tools as tools
import lane_detect_processor

#Code
#Check for arguments
if len(sys.argv) < 2:
    input("No arguments passed, press ENTER to exit...")
    quit()

for i in range(1, len(sys.argv)):
    video_in = cv2.VideoCapture(sys.argv[i])   
    video_out = cv2.VideoWriter(tools.rename_output_file(sys.argv[i]), \
                                int(video_in.get(cv2.CAP_PROP_FOURCC)), \
                                video_in.get(cv2.CAP_PROP_FPS), \
                                (int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                                 int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while(video_in.isOpened()):
        result, frame = video_in.read()
        if result==False:
            print("File completed: " + sys.argv[i])
            break
        #Do some processing! - ToDo!

        #Write new frame
        video_out.write(frame)
        #Display new frame
        cv2.imshow('Results', frame)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_in.release()
    video_out.release()

cv2.destroyAllWindows()