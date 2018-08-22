
[//]: # (Image References)

[preview]: ./test_images_output/solidWhiteCurveExtended.png "Preview"
[stage1]: ./pipeline_stages/1_grayscale.png "Grayscale"
[stage2]: ./pipeline_stages/2_smoothed.png "Smoothed"
[stage3]: ./pipeline_stages/3_canny.png "Canny"
[stage4]: ./pipeline_stages/4_masked.png "Masked"
[stage5]: ./pipeline_stages/5_hough.png "Hough"
[stage6]: ./pipeline_stages/6_weighted.png "Weighted"
[stage7]: ./pipeline_stages/7_extended.png "Extended"
[challenge_initial]: ./pipeline_stages/challenge_initial.png "Challenge Video Initial Attempt"
[challenge_modified]: ./pipeline_stages/challenge_modified.png "Challenge Video Modified"

### **Finding Lane Lines on the Road**


![preview]

The goals of this project is to setup a pipeline that finds lane lines on the road.

---

### Reflection

##### My initial pipeline consisted of the following stages:

First, the image is converted to grayscale. This makes it easier to use the same parameters to find yellow and white lanes alike.

![Gray][stage1]

Second, the image is smoothed out using an 11x11 Gaussian mask. This helps get rid of noise and small details and makes it easier to identify edges.

![Smoothed][stage2]

Then Canny algorithm is used for edge detection with a low threshold of 30, and a high threshold of 60. It is recommended to have a low to high threshold ratio of 1:2 or 1:3.

![Canny][stage3]

Then a mask is applied to define the region of interest. This allows for more accurate results and faster processing since it restricts the search to only the region where we know the lane lines should exist. I used a mask with a triangular shape, with its top vertex just above the vertical center of the photo where the lane lines meet by a safe margin.

![Masked][stage4]

Finally, hough line transform is used for line detection.

![Hough][stage5]

Then the lines are added on top of the original picture for comparison.

![Weighted][stage6]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by having it calculate the slopes and y-intercepts for all lines generated by the Hough transform within the region of interest, then separating the lines by slope; lines with slope > 0 belonging to the left lane, and lines with slope < 0 belonging to the right lane.

Then I find the median slope and range for each lane, and use those values to draw a single straight line that extends from the bottom of the image to the top of the mask area (region of interest).

![Extended][stage7]

When tested on the challenge.mp4 video, the results didn't look very promising, specially at 0:04 where the lighting changes.

![Initial Challenge Attempt][challenge_initial]

To tackle this problem, I modified the draw_lines() function to include a threshold for acceptable slope values, so that hough lines are compared to those slope values and if they're not within the acceptable range the line is rejected, and in the case of failure of finding any acceptable lines on a frame, the last acceptable lane line that had been detected is redrawn instead. This approach showed a significant improvement upon testing with the same video.

![Challenge Attempt after modification][challenge_modified]

### 2. Potential Shortcomings
One of the potential shortcomings of the current pipeline is that it may not work on different camera angles or placements since stuff like the region of interest are sort of hard coded.

Another shortcoming could be the lane lines appearing to be shaky sometimes.

Moreover, the line drawing function doesn't support curved lines.

### 3. Possible improvements to the pipeline

A possible improvement would be to calculate the region of interest in a more dynamic way.

Another potential improvement could be to reject sudden changes in lane line positions to make the lines look smoother while transitioning from frame to frame and get rid of the shakiness. This would likely achieve higher accuracy due to rejecting unrealistic sudden change.

Finally, the line drawing function can be modified to draw curved lines where the lanes are curved instead of one straight line.
