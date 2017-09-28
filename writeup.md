# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_and_notcar.png "Car and Not Car Images"
[image2]: ./output_images/HOG_car_and_notcar.png "HOG Car and Not Car Images"
[image3]: ./output_images/DrawBoxes.png "Draw Sliding Window Detection"
[image4]: ./output_images/detection.png "Boxes Drawn on Cars"
[image5]: ./output_images/heatmap_with_false.png "Heatmap with False"
[image6]: ./output_images/heatmap_with_withoutfalse.png "Heatmap without False"
[video1]: ./output_videos/project_video.mp4 "Final Video"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained under the heading of 'Feature Extraction' of the IPython notebook (file called `Vehicle-Detection.ipynb`)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the bases for chosing the parameters was the one which provided me the best SVM Test Accuracy, which was 0.9834 out of 1.

| Parameters        | Value   |
|-------------------|---------|
|Color Space        |YCrCb    |
|Orientation        |9        |
|Pixels per Cell    |16       |
|Cells per Block    |2        |
|HOG Channel        |ALL      |
|Spatial Size       |(32, 32) |
|Histogram Bins     |32       |
|Spatial Features   |True     |
|Histogram Features |True     |
|HOG Features       |True     |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the default classifier parameters and using 'ALL' HOG features as well as spatial and histogram features. The trained classifier could be found under the header "Classifier" in the IPython notebook (`Vehicle-Detection.ipynb`), the final test Accuracy for SVM was 0.9834.

### Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search I adapted to was the one in Udacity section called "Method for Using Classifier to Detect Cars in an Image", the method that was taken and implemented in the Python notebook is the `find_cars`. Why I choose this method over windows search over the whole frame is because this combines HOG feature extraction with the sliding window search. Where HOG features are extracted for a part of the image which was classified and then the sliding windows search applies to just that area, that will speed up computations. The function at the end results in drawing rectangle objects corresponding to the windows that generated a positive prediction meaning a car.

The image below shows a attemp in using find_cars on multiple test images, using a single window size, as show it also picks up false positives:

![alt text][image3]

I tried different scale sizes and the final configuration for the search window were 1, 1.5, 2.5, this gave reasonable window sizes where the car is close to the camera and as it moves further as well, overlap to windows was at 50%. And a unusual approach I took for this project was that the car was driving in the left lane, where no car came infront of it, and predicting the oncoming traffic wasnt useful, there for the window search area was later confined to a smaller region where it was greater than 500 on the X-axis.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

##### Here is a test image with its corresponding heatmap:

![alt text][image5]

Another way to stop false detection being picked up from the oncoming traffic was removed by not searching in the left side of the images, this increased the chances of succesful detection of the cars

##### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

Classifiers accuracy is very crucial in this case and as close as it can get to 100% the better without overfitting it, for example if we are driving a car, and it see 1000 frames, using my accuracy of 98% it will mean it has misclassified 20 frames.

Also the sliding windows search is computational very expensive to search a certain area again, a normal approach might be to cut half of the image vertical to ignore the sky, however on a approaching a hill it might not detect for a car as the sliding windows is not searching in that area.

The project also had to be fine-tuned with the colorspace it will be using as well as many other factors, this gives me a real appreciation with neural network approaches.

I would love to come back to this project and fine-tune it alot more as well as apply a deep learning approach to it.
