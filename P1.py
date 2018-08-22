
# coding: utf-8

# [//]: # (Image References)
# 
# [preview]: ./test_images_output/solidWhiteCurveExtended.png "Preview"
# 
# # **Finding Lane Lines on the Road**
# 
# ![preview]
# 
# ## The goals of this project is to setup a pipeline that finds lane lines on the road.
# 

# ## Import Packages

# In[4]:


import os
import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

get_ipython().magic('matplotlib inline')


# ## Original helper functions

# In[5]:


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# ## Function for calculating region of interest on an image

# In[6]:


def calculate_region_of_interest(image):
    bottom_left = (0,image.shape[0])
    top = (image.shape[1]/2,(image.shape[0]/2)+50)
    bottom_right = (image.shape[1],image.shape[0])
    
    return bottom_left, bottom_right, top


# ## Lane detection pipeline

# [stage1]: ./pipeline_stages/1_grayscale.png "Grayscale"
# [stage2]: ./pipeline_stages/2_smoothed.png "Smoothed"
# [stage3]: ./pipeline_stages/3_canny.png "Canny"
# [stage4]: ./pipeline_stages/4_masked.png "Masked"
# [stage5]: ./pipeline_stages/5_hough.png "Hough"
# [stage6]: ./pipeline_stages/6_weighted.png "Weighted"
# 
# ##### My initial pipeline consisted of the following stages:
# 
# First, the image is converted to grayscale. This makes it easier to use the same parameters to find yellow and white lanes alike.
# 
# ![Gray][stage1]
# 
# Second, the image is smoothed out using an 11x11 Gaussian mask. This helps get rid of noise and small details and makes it easier to identify edges.
# 
# ![Smoothed][stage2]
# 
# Then Canny algorithm is used for edge detection with a low threshold of 30, and a high threshold of 60. It is recommended to have a low to high threshold ratio of 1:2 or 1:3.
# 
# ![Canny][stage3]
# 
# Then a mask is applied to define the region of interest. This allows for more accurate results and faster processing since it restricts the search to only the region where we know the lane lines should exist. I used a mask with a triangular shape, with its top vertex just above the vertical center of the photo where the lane lines meet by a safe margin.
# 
# ![Masked][stage4]
# 
# Finally, hough line transform is used for line detection.
# 
# ![Hough][stage5]
# 
# Then the lines are added on top of the original picture for comparison.
# 
# ![Weighted][stage6]

# In[7]:


def draw_lane_lines(image):
    
    path = './pipeline_stages/'
    
    #transform image to grayscale
    image_gray = grayscale(image)
    
#     plt.imsave(path+'1_grayscale.png', image_gray, cmap='gray')
    
    #smooth image out using gaussian blur with mask size of 11x11
    image_smoothed = gaussian_blur(image_gray,11)

#     plt.imsave(path+'2_smoothed.png', image_smoothed, cmap='gray')
    
    #define parameters for canny edge detection
    low_threshold = 30
    high_threshold = 60
    
    #perform canny edge detection
    image_canny = canny(image_smoothed, low_threshold, high_threshold)
    
#     plt.imsave(path+'3_canny.png', image_canny, cmap='gray')
    
    #define vertices for region of interest triangle mask
    bottom_left, bottom_right, top = calculate_region_of_interest(image)
    mask_vertices = np.int32([[bottom_left, top, bottom_right]])
    
    #apply region of interest mask
    image_masked = region_of_interest(image_canny, mask_vertices)
    
#     plt.imsave(path+'4_masked.png', image_masked, cmap='gray')
    
    #define parameters for hough lines algorithm
    rho = 1
    theta = np.pi/180
    threshold = 10
    min_line_len = 12
    max_line_gap = 5  
    
    #perform hough lines transformation
    image_hough = hough_lines(image_masked, rho, theta, threshold, min_line_len, max_line_gap)
    
#     plt.imsave(path+'5_hough.png', image_hough, cmap='gray')
    
    #draw lines on the original image
    image_weighted = weighted_img(image_hough, image)
    
#     plt.imsave(path+'6_weighted.png', image_weighted, cmap='gray')
#     plt.imsave(path+'7_extended.png', image_weighted, cmap='gray')
    
    return image_weighted

def draw_lane_lines_and_save(image, path):
    
    #call pipeline function
    image_out = draw_lane_lines(image)
    
    #save image
    plt.imsave(path, image_out)
    
    return image_out


# ## Run lane detection pipeline on test images

# In[8]:


fig = plt.figure(figsize=(12,12))

for i,test_image in enumerate(os.listdir("test_images/")):
    im = plt.imread('./test_images/'+test_image)
    im_w_lines = draw_lane_lines_and_save(im, './test_images_output/'+test_image[:-4]+'.png')
    fig.add_subplot(3,2,i+1)
    plt.imshow(im_w_lines)


# ## Run lane detection pipeline on test videos

# In[9]:


white_output = 'test_videos_output/solidWhiteRight.mp4'
yellow_output = 'test_videos_output/solidYellowLeft.mp4'

white_clip = VideoFileClip("test_videos/solidWhiteRight.mp4").fl_image(draw_lane_lines)
yellow_clip = VideoFileClip("test_videos/solidYellowLeft.mp4").fl_image(draw_lane_lines)

get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')


# ## Play lane detection output videos inline

# In[10]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# In[11]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Rewrite the draw_lines() function to draw a single solid line over each lane

# [stage7]: ./pipeline_stages/7_extended.png "Extended"
# 
# In order to draw a single line on the left and right lanes, I modified the draw_lines() function by having it calculate the slopes and y-intercepts for all lines generated by the Hough transform within the region of interest, then separating the lines by slope; lines with slope > 0 belonging to the left lane, and lines with slope < 0 belonging to the right lane.
# 
# Then I find the median slope and range for each lane, and use those values to draw a single straight line that extends from the bottom of the image to the top of the mask area (region of interest).
# 
# ![Extended][stage7]

# In[12]:


class Lane:
    
    """
    Lane class defines a lane by its slope (m) and y-intercept (c)
    Points (x1,y1) and (x2,y2) are 
    calculated based on the region of interest passed to the constructor
    """
    
    def __init__(self, m, c, bottom, top):
        self.m = m
        self.c = c
        
        self.x1 = int((bottom-c)/m)
        self.y1 = int(bottom)
        
        self.x2 = int((top-c)/m)        
        self.y2 = int(top)
        
    def draw(self, img, color, thickness):
        cv2.line(img, (self.x1, self.y1), (self.x2, self.y2), color, thickness)
        

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    
    right_lane_slopes = []
    left_lane_slopes = []
    
    right_lane_intercepts = []
    left_lane_intercepts = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            
            m = (y2-y1)/(x2-x1) #slope
            c = y1-(m*x1) #y-intercept
            
            if(m>0): #if the line belongs to the right lane
                right_lane_slopes.append(m)
                right_lane_intercepts.append(c)
                
            elif(m<0): #if the line belongs to the left lane
                left_lane_slopes.append(m)
                left_lane_intercepts.append(c)
                
    #find median slope and y-intercept for right lane
    right_lane_m = np.median(right_lane_slopes) 
    right_lane_c = int(np.median(right_lane_intercepts))

    #find median slope and y-intercept for left lane
    left_lane_m = np.median(left_lane_slopes)
    left_lane_c = int(np.median(left_lane_intercepts))
    
    #calculate region of interest
    bottom_left, bottom_right, top = calculate_region_of_interest(img)
    
    #initialize lanes with slope, y-intercept, bottom y, and top y
    left_lane = Lane(left_lane_m, left_lane_c, bottom_left[1], top[1])
    right_lane = Lane(right_lane_m, right_lane_c, bottom_right[1], top[1])
    
    #draw lanes
    left_lane.draw(img, color, thickness)
    right_lane.draw(img, color, thickness)


# ## Run extended line detection function on test images

# In[13]:


fig = plt.figure(figsize=(12,12))

for i,test_image in enumerate(os.listdir("test_images/")):
    im = plt.imread('./test_images/'+test_image)
    im_w_lines = draw_lane_lines_and_save(im, './test_images_output/'+test_image[:-4]+'Extended.png')
    fig.add_subplot(3,2,i+1)
    plt.imshow(im_w_lines)


# ## Run extended line detection function on test videos

# In[14]:


white_extended_output = 'test_videos_output/solidWhiteRightExtended.mp4'
yellow_extended_output = 'test_videos_output/solidYellowLeftExtended.mp4'

clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
clip2 = VideoFileClip("test_videos/solidYellowLeft.mp4")

white_extended_clip = clip1.fl_image(draw_lane_lines)
yellow_extended_clip = clip2.fl_image(draw_lane_lines)

get_ipython().magic('time white_extended_clip.write_videofile(white_extended_output, audio=False)')
get_ipython().magic('time yellow_extended_clip.write_videofile(yellow_extended_output, audio=False)')


# ## Play extended lane detection output videos inline

# In[15]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_extended_output))


# In[16]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_extended_output))


# ## Test current pipeline on challenge video and examine results

# [challenge_initial]: ./pipeline_stages/challenge_initial.png "Challenge Video Initial Attempt"
# 
# When tested on the challenge.mp4 video, the results didn't look very promising, specially at 0:04 where the lighting changes.
# 
# ![Initial Challenge Attempt][challenge_initial]

# In[17]:


challenge_output = 'test_videos_output/challenge.mp4'

challenge_clip = VideoFileClip('test_videos/challenge.mp4').resize(height=540).fl_image(draw_lane_lines)

get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')


# In[18]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))


# ## Modify draw_lines() function to improve results with Challenge video

# [challenge_modified]: ./pipeline_stages/challenge_modified.png "Challenge Video Modified"
# 
# To tackle this problem, I modified the draw_lines() function to include a threshold for acceptable slope values, so that hough lines are compared to those slope values and if they're not within the acceptable range the line is rejected, and in the case of failure of finding any acceptable lines on a frame, the last acceptable lane line that had been detected is redrawn instead. This approach showed a significant improvement upon testing with the same video.
# 
# ![Challenge Attempt after modification][challenge_modified]

# In[19]:


'''
The follwing two variables hold the last correct lane variable found 
They are used in case of failing to find correct lanes in the a new frame
They're both initialized with average values for lanes on the challenge video
'''
prev_left_lane = Lane(-0.7, 875, 720, 410)
prev_right_lane = Lane(0.58, 49, 720, 410)

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    
    global prev_left_lane
    global prev_right_lane
    
    #threshold for acceptable slope in the left lane
    left_slope_max = -0.45
    left_slope_min = -0.75

    #threshold for acceptable slop in the right lane
    right_slope_max = 0.75
    right_slope_min = 0.45

    right_lane_slopes = []
    left_lane_slopes = []
    
    right_lane_intercepts = []
    left_lane_intercepts = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:                
            m = (y2-y1)/(x2-x1) #slope
            c = y1-(m*x1) #y-intercept
            
            if(m>right_slope_min and m<right_slope_max): #compare with right lane acceptance thresholds
                right_lane_slopes.append(m)
                right_lane_intercepts.append(c)
                
            elif(m>left_slope_min and m<left_slope_max): #compare with left lane acceptance thresholds
                left_lane_slopes.append(m)
                left_lane_intercepts.append(c)

    #calculate region of interest
    bottom_left, bottom_right, top = calculate_region_of_interest(img)
               
    if(right_lane_intercepts): #if new acceptable lanes were found
        right_lane_m = np.median(right_lane_slopes)
        right_lane_c = int(np.median(right_lane_intercepts))
        right_lane = Lane(right_lane_m, right_lane_c, bottom_right[1], top[1])
        right_lane.draw(img, color, thickness)
        prev_right_lane = right_lane #update prev_lane variable
    
    else: #if no new acceptable lanes were found redraw the last lane
        prev_right_lane.draw(img,color,thickness)

    if(left_lane_intercepts): #if new acceptable lanes were found
        left_lane_m = np.median(left_lane_slopes)
        left_lane_c = int(np.median(left_lane_intercepts))
        left_lane = Lane(left_lane_m, left_lane_c, bottom_left[1], top[1])
        left_lane.draw(img, color, thickness)
        prev_left_lane = left_lane #update prev_lane variable
    
    else: #if no new acceptable lanes were found redraw the last lane
        prev_left_lane.draw(img,color,thickness)


# ## Test modified draw_lines() on challenge video again

# In[20]:


challenge_output = 'test_videos_output/challenge_modified.mp4'

challenge_clip = VideoFileClip('test_videos/challenge.mp4').resize(height=540).fl_image(draw_lane_lines)

get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')


# In[21]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))


# ### 2. Potential Shortcomings
# One of the potential shortcomings of the current pipeline is that it may not work on different camera angles or placements since stuff like the region of interest are sort of hard coded.
# 
# Another shortcoming could be the lane lines appearing to be shaky sometimes.
# 
# Moreover, the line drawing function doesn't support curved l

# ### 3. Possible improvements to the pipeline
# 
# A possible improvement would be to calculate the region of interest in a more dynamic way.
# 
# Another potential improvement could be to reject sudden changes in lane line positions to make the lines look smoother while transitioning from frame to frame and get rid of the shakiness. This would likely achieve higher accuracy due to rejecting unrealistic sudden change.
# 
# Finally, the line drawing function can be modified to draw curved lines where the lanes are curved instead of one straight line.
