## Vehicle Detection and Tracking Project
### David del Río Medina

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/dataset.png
[image2]: ./output_images/hog.png
[image3]: ./output_images/spatial.png
[image4]: ./output_images/windows.png
[image5]: ./output_images/vehicle_windows.png
[image6]: ./output_images/heatmap.png
[video1]: ./processed.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation. My code is implemented in the accompanying Jupyter notebook `p5_vehicle_detection.ipynb` as a step-by-step guide through all the process, and then all those steps are used in a final function that applies the whole pipeline to each frame of a given video.

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained through cells 1-8 of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images and converting them to from BGR to RGB for convenience.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

For the HOG features, I settled for the parameter values suggested in the lesson: `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. The image is converted to the HLS color space, that worked very well for project 4, and HOG features are extracted from every individual channel. 

Here is an example using the `L` channel of the HLS color space. The shape of the back of the car is captured by the HOG features.


![alt text][image2]

A spatial binning of the whole three channels of the HLS image, resized to 16x16, is added to the feature vector:

![alt text][image3]

####2. Explain how you settled on your final choice of HOG parameters.

The RGB color space seemed to work, but it looked like all the three channels provided more or less the same features, which is expected, since in this color space hue, luminosity and other properties are mixed across all the channels. The HLS color space separates this properties, which helps extracting features of a higher quality.

For the parameters of the hog function, I could not find a combination other than the one suggested by Udacity that resulted in a better performance of the classifier.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Cells 9 to 11. After combining all the features in a single array, all the feature vectors are normalized using the `StandardScaler` provided by `scikit-learn`. Then, the labels are created and both features and labels (x and y) arrays are shuffled and split in training and test sets (80/20).

I found that a support vector machine classifier with an RBF kernel worked better than a linear one, although much slower. The RBF kernel shows better accuracy with the test set, but we know is not entirely reliable since the images come from the same video source. Anyway, testing both classifiers with some test images it is clear that the linear kernel produces lots of false positives, where as the RBF kernel might have a slightly higher false negatives rate.  

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Cells 12 to 16. I applied the suggested sliding windows implementation in the bottom half of the images. I chose squared windows of sizes 64, 96, 128, 160, 192 and 224, trying to cover all the possibilities:

![alt text][image4]

The test I performed showed that the classifier was able to detect the vehicles in most images, without adding many false positives:

![alt text][image5]
 

####2. What did you do to optimize the performance of your classifier?

In order to keep false negatives to a minimum, the suggested implementation based on thresholded heat maps and labels was applied. Apart from filtering out possible false negatives, it is also used to obtain a final window for every identified vehicle from all the overlapping windows:

![alt text][image6]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://www.dropbox.com/s/966a3i9lvxx1hvg/processed.mp4?dl=0)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Cell 17. Besides adding a threshold filter to the heat map of every individual frame, the final windows detected in the last five frames (including the current one) are saved. 

Then, the same procedure (add heat map, apply threshold, labeling, get resulting windows) is applied to this list of windows from the last frames, this time with a bigger threshold. The final image at the end of the pipeline is the original one with the resulting windows of this last process drawn in it.


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found out that it is very hard to identify the vehicles all or most of the time while having a small number of false positives. The processed video shows that the classifier have problems identifying the sides of a vehicle, probably because most pictures in the original dataset just show the back of the cars, and sometimes the windows go missing for a few frames. There are a few false positives, but most of them are cars coming from the other direction.

I think a better result could have been achieved with a better dataset, but I did not try Udacity's due to lack of time. Also, several combinations of color spaces and filters could help extracting better features. As for the classifier, it is possible that a well trained CNN could be more robust than a classic machine learning model and spare some parts of the feature engineering process, but it may be too slow. Finally, the sequential nature of the problem is not fully exploited with the actual pipeline: estimating the speed and position of the detected vehicles can prevent missing them in some frames.
