**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/writeup/car_not_car.png
[image2]: ./output_images/writeup/HOG_example.png
[image3]: ./output_images/writeup/sliding_windows.png
[image4]: ./output_images/writeup/sliding_window1.png
[image5]: ./output_images/writeup/sliding_window2.png
[image6]: ./output_images/writeup/sliding_window3.png
[image7]: ./output_images/writeup/sliding_window4.png
[image8]: ./output_images/writeup/sliding_window5.png
[image9]: ./output_images/writeup/sliding_window6.png
[image10]: ./output_images/writeup/bboxes_and_heat.png
[image11]: ./output_images/writeup/labels_map.png
[image12]: ./output_images/writeup/output_bboxes.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  
This document is the project writeup. You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  This is in 2nd and 3rd code cells.
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  This is in 5th code cell. I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like and how different the car and not-car images are. I applied the `skimage.hog()` functions to different channels to get a feel for which channel are most distinct with HOG from that of the other class.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

I also tried spatial and color features. I extract them from the training images and apply them as training feature as well.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried  various combinations of color spaces and HOG parameters. This is in 6th, 7th and 8th code cell.
For YCrCb color space the Y-channel appears to have the biggest effect. The other 2 channels can make some difference as well.
As for HLS, S and L channels can work well, however, the H channel does not seem to take effect.
I do not use RGB color space, because it depends much on RGB colors and may not recognize the shape well.
I tried different parameters to make the feature images well distinguished from that of the other class. I finally settled with YCrCb color space and `pixels_per_cell=(8,8)`,  `cells_per_block=(2, 2)` and `orientations=9`. The `orientations` with smaller values do not work well to make the image clear, while the larger value get more features but no obvious improvement.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with HOG, spatially binned color and historgram of colors as features. This is in 9th code cell.

First, I conver the image into YCrCb color space and extract all the above features with the above parameters for all 3 channels to stack into one features array.
Then I standardize and normalize the features with `StandardScaler()` from `sklearn` library.
I stack the training vectors and add a label vector together. Then I use `train_test_split()` to randomly split all the data into training set and test set, with 80% for training and 20% for testing.

Finally I fit the LinearSVC with the training set an test it with the test set. the accuracy is about 0.983. I think that means the training classfier works.

The classifier is saved and restored with pickle in 10th and 11th code cell.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

HOG features are extracted for each of the 3-channesl for the whole test image. This is in 12th code cell.
 Then a 8x8 window (with scale 1) of features in the subimage are taken to the classifer for it to predict as car or not-car. Then the window is moved to the next area with the step of `cells_per_step` parameters. It is 2 in my project, which means 75% overlap.

I checked the car size in the test images. The height of the car in the image is about 128 pixcels high, which is about scale of 2. Thus I use different window scales of 1, 1.5, 2, 2.5 and 3 to search, while the smaller scaling are for farther cars and the larger are for the closer ones.
The Y start point are all 400.
For the scaling of 1, 1.5 and 2, I do not search the whole bottom half image. Instead I make them search partially in Y direction, because the small scaling, which is for the farther cars, should not be in lower part(closer) in the test image.
![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 5 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9]

I do not search the windows with small scaling in all lower area to reduce the searching times. For the scaling of 1, 1.5 and 2, I do not search the whole bottom half image. Instead I make them search partially in Y direction, because the small scaling, which is for the farther cars, should not be in lower part(closer) in the test image.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

All the rest parts are in code cells from 13 to 18.
I recorded the positions of positive detections in latest 10 frames of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of scipy.ndimage.measurements.label() and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image10]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image11]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image12]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* To reject outltiers in the video, I tuned the heat threshold. However, in the mean time, the positive windows are reduced as well. I will try other classifiers to reduce false positives and more smooth thresholds to reject outliers.
In addition, the highlighted boxes are not always stable in the video, should apply a smooth threshold for the accumulated heat map.

* I will try deep learning method such as YOLO to do this project and have a comparison with Linear SVC.

* The FPS is quite low. I need to try more effient feature extraction methods and other classifiers.
