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
[image7]: ./original.JPG "Original"
[image8]: ./undistorted.JPG "Undistorted"
[image9]: ./check_original.JPG "Original"
[image10]: ./check_undistorted.JPG "Undistorted"
[video1]: ../project_video_edit.mp4 "Video"
[video2]: ../challenge_video_edit.mp4 "Video"
[video3]: ../harder_challenge_video_edit.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

My project includes the following files:
* [video_processor.py](../video_processor.py) containing main function which handles video file paths as arguements to process
* [video_processor_tools.py](../video_processor_tools.py) containing various functions for the editing of video files
* [lane_detect_processor.py](../lane_detect_processor.py) containing all classes and functions for lane detection and image manipulation
* [project_video_edit.mp4](../project_video_edit.mp4) original video after processing
* [challenge_video_edit.mp4](../challenge_video_edit.mp4) challenge video after processing
* [harder_challenge_video_edit.mp4](../harder_challenge_video_edit.mp4) harder challenge video after processing
* [README.md](../writeup/README.md) the report you are reading now


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Upon initally running the video_processor.py script, all the images in "./camera_cal" directory are loaded into a list and passed to calibrate_camera() in lane_detect_processor.py with the the number of x and y points.

In [calibrate_camera()](../lane_detect_processor.py#L165), each image in the list is iterated through and all of the object points are detected wit h cv2.findChessboardCorners().  Both the image points and the corner numbers are appended to lists and then passed to cv2.calibrateCamera(), which returns a transformation matrix which is later used by cv2.undistort() to undistort the original and present a true image.  This is done to every image in [process_image()](../lane_detect_processor.py#L551) prior to any processing.

Here you can see an original and undistorted image, pay close attention to the straightness of the lines created by the checkerboard edges:

Original:

![Original][image9]

Undistorted:

![Undistorted][image10]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

As shown above on the checkboard image, each road image was also undistorted like shown prior to any image processing.

Original:

![Original][image7]

Undistorted:

![Undistorted][image8]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Two functions were used to generate binary masks which where then combined together with a cv2.bitwise_and().

The first of these functions was [hls_threshold()](../lane_detect_processor.py#L198).  In this function this image was converted to HLS colorspace, then three thresholds applied.  The first two were both used to detect white road markings. White was split into two thresholds because the cv2.inRange() function had to be called twice because neutral colors wrapped around the 0 value (i.e. a Hue of 178 is nearly the same as a Hue of 2).  Additionally, yellow was detected with another cv2.inRange() call and the two masks were combined with a cv2.bitwise_or().

![HLS Threshold][image4]

The second function was [gradient_threshold()](../lane_detect_processor.py#L215).  This function first converts the image to grayscale and then performs both a sobel gradient in the X and Y directions.  Afterwards, cv2.inRange() is used to apply a threshold and create a mask, and a bitwise_or is then applied.

![Gradient Threshold][image3]

See below an example of the combined mask of the hls threshold and gradient threshold when applied to the undistorted image:

![Combined Threshold][image2]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform was done in the [process_image()](../lane_detect_processor.py#L551) to the combined binary masks of the hls_threshold and gradient_threshold results.  The result image appears to be a 'birds-eye' or 'top-down' view, making the road lines appear parallel on a straight road:

![Birds-eye-view][image1]
The implemenation of the transform itself was easy, however, obtaining a proper transformation matrix was diffuclt.  In the [constants](../lane_detect_processor.py#L34) area of my code I implemented selected points in the source and destination arrays and generated both a "birds-eye-view" matrix and a matrix to warp back to the original perspective.  This was needed for drawing and shading the lanes in the original image.  The points were chosen by choosing top and bottom points from both lanes taken from a straight road segment (SRC) and then translating them so they would extend to the complete top and bottom of the image in paralell (DST), so all X points were maintained constant from the bottom of the original image and Y points were top and bottom of the image size


```python
SRC = np.float32([[582, 460], \
                  [698, 460], \
                  [185, 720], \
                  [1095, 720]])
DST = np.float32([[185, 0], \
                  [1095, 0], \
                  [185, 720], \
                  [1095, 720]])
BEV_MATRIX = cv2.getPerspectiveTransform(SRC, DST)
INV_MATRIX = cv2.getPerspectiveTransform(DST, SRC)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 582, 460      | 185, 0        | 
| 698, 460      | 1095, 0       |
| 185, 720      | 185, 720      |
| 1095, 720     | 1095, 720     |


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Fitting of the polynomial for the road lines was done in two places.  First, the [detect_lines_basic()](../lane_detect_processor.py#L243) function extracted all the 'good' points using the sliding window methodology.  They were then pased to the [update()](../lane_detect_processor.py#L100) method in each [Line()](../lane_detect_processor.py#L77) class, which pushed the then points into an FIFO array and best fit a polynomial to all of those points.  By implmenting the polynomial fit in this way the polynomial fit was always an average of the last x frames.

See the result below:

![Windows][image6]

The sliding window methodology was improved upon from the classroom exercises in the following ways:

* If a current best fit polynomial was already defined for the line, the start points were solved for by the equation instead of a histogram.
* Limits to road width were implemented, that way the function was discouraged to follow noise on the inside or outside of the road lines.
* If the left or right lane window did not detect the minium amount of pixels, it would adjust its position in the next window by the same amount the other lane moved.  This was especially helpful when one line was solid and another was segmented, as it would continue the windows in the direction of the line even if the window did not capture a segment.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The function [get_radius()](../lane_detect_processor.py#L486) was developed to find the radius of a line regardless of physical units, and it was called from [draw_status()](../lane_detect_processor.py#L496	) after plotted points from the original polynomial were scalled to real world units.

Additionally the calculation of road offset was done in the [draw_status()](../lane_detect_processor.py#L504) function, which was a simply the difference between the average of the two lines and the image center, multiplied by the real world unit scale factor.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Two functions applied the result of the lane detection to the image.  First was the [shade_lines()](../lane_detect_processor.py#L431) function, which plotted each of the lane polynomials into an array of points.  If there was a valid fit for both lines, the function shades the area, if only one of the lines were detected, it drew only that line.  Additionally for this function it was key to have the inverse matrix transform for the 'birds-eye' view perspective so the lines could be plotted then warped to origianl perspective.

The second function used was [draw_status()](../lane_detect_processor.py#L496), which calculated the road width, offset, and average radius of both lines combined.  This information was then drawn in the top left of the original image plus line shading.

See the result below:

![Result][image5]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.

Here's my results:

[Original Video](../project_video_edit.mp4) - Very succesful and tolerates the the changing road pavement well.

[Challenge 1](../challenge_video_edit.mp4) - Succesful, handles the changing contrast under the overpass with little distrubance and also ignores the seam in the center of the lane.

[Challenge 2](../harder_challenge_video_edit.mp4) - Not as succesful.  The sharp radiuses of the road warp it outside of the road image after the perspective transformation, so the pixels are never considered in the polyfit.  Additionally windshield glare and other factors necessitate some image pre-processing to reduce the effect of washout in the image.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.

The two largest challenges of this project were getting a corrrect warp matrix and determining good thresholding values.  The warp matrix was particularly tricky due to the sensitivity of the point values.  I believe this is likely due to the 'farther' into the distance you wish to detect, the closer the top left/right points became making only a pixels difference make radical changes in the radius and direction of the road line after warped.

The second challenge was tresholding, specifically in HLS space for the white lanes due to the difficulty of differentiating between the grey road surface and white road lines, as both had a near neutral hue and inconsistent lightness.  Saturation was the critcal value to distinguish in the end.