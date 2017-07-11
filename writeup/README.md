# **Advanced Lane Detection**

### Lane detection utilizing combined approaches

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./bev.JPG "Birds-eye-view"
[image2]: ./combined_threshold.JPG "Combined threshold"
[image3]: ./gradient_threshold.JPG "Gradient threshold"
[image4]: ./hls_threshold.JPG "HLS threshold"
[image5]: ./result.JPG "Result"
[image6]: ./windows.JPG "Windows"
[video1]: ../project_video_edit.mp4 "Video"
[video2]: ../challenge_video_edit.mp4 "Video"
[video3]: ../harder_challenge_video_edit.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

My project includes the following files:
* [video_processor.py](https://github.com/nategreco/CarND-Advanced-Lane-Lines/blob/master/video_processor.py) containing main function which handles video file paths as arguements to process
* [video_processor_tools.py](https://github.com/nategreco/CarND-Advanced-Lane-Lines/blob/master/video_processor_tools.py) containing various functions for the editing of video files
* [lane_detect_processor.py](https://github.com/nategreco/CarND-Advanced-Lane-Lines/blob/master/lane_detect_processor.py) containing all classes and functions for lane detection and image manipulation
* [project_video_edit.mp4](https://github.com/nategreco/CarND-Advanced-Lane-Lines/blob/master/project_video_edit.mp4) original video after processing
* [challenge_video_edit.mp4](https://github.com/nategreco/CarND-Advanced-Lane-Lines/blob/master/challenge_video_edit.mp4) challenge video after processing
* [harder_challenge_video_edit.mp4](https://github.com/nategreco/CarND-Advanced-Lane-Lines/blob/master/harder_challenge_video_edit.mp4) harder challenge video after processing
* [README.md](https://github.com/nategreco/CarND-Advanced-Lane-Lines/blob/master/writeup/README.md) the report you are reading now


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Upon initally running the video_processor.py script, all the images in "./camera_cal" directory are loaded into a list and passed to calibrate_camera() in lane_detect_processor.py with the the number of x and y points.

In calibrate_camera(), each image in the list is iterated through and all of the object points are detected wit h cv2.findChessboardCorners().  Both the image points and the corner numbers are appended to lists and then passed to  cv2.calibrateCamera(), which returns a transformation matrix which is later used by cv2.undistort() to undistort the original and present a true image.  This is done to every image in process_image() prior to any processing.

Here you can see an original and undistorted image, pay close attention near the edges of the image where the distortion is most significant:


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

As shown above on the checkboard image, each road image was also undistorted like shown prior to any image processing:


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Two functions were used to generate binary masks which where then combined together with a cv2.bitwise_and().

The first of these functions was hls_threshold().  In this function this image was converted to HLS colorspace, then three thresholds applied.  The first two were both used to detect white road markings. White was split into two thresholds because the cv2.inRange() function had to be called twice because neutral colors wrapped around the 0 value (i.e. a Hue of 178 is nearly the same as a Hue of 2).  Additionally, yellow was detected with another cv2.inRange() call and the two masks were combined with a cv2.bitwise_or().

![HLS Threshold][image4]

The second function was gradient_threshold().  This function first converts the image to grayscale and then performs both a sobel gradient in the X and Y directions.  Afterwards, cv2.inRange() is used to apply a threshold and create a mask, and a bitwise_or is then applied.

![Gradient Threshold][image3]

See below an example of the combined mask of the hls threshold and gradient threshold when applied to the undistorted image:

![Combined Threshold][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform was done in the process_image() to the combined binary masks of the hls_threshold and gradient_threshold results.  The result image appears to be a 'birds-eye' or 'top-down' view, making the road lines appear parallel on a straight road:

![Birds-eye-view][image1]

Additionally, a perspective transform was also necessary to perform the opposite transform in the shade_lines() function, changing the plotted and shadded road line images back to the same perspective as the undistorted image:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![Result][image5]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's my results:
[Original Video](../project_video_edit.mp4)
[Challenge 1](../challenge_video_edit.mp4)
[Challenge 2](../harder_challenge_video_edit.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
