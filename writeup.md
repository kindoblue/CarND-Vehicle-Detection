## Writeup Template

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/lousy.png
[image5]: ./examples/toomany.png
[image6]: ./examples/unrecognized.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 15 through 140 of the file called `hog_classifier.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example from a greyscale image and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the first thing I notice is how big the feature vector could easily get. So I decided to have 16 `pixels_per_cell` and to balance I increased the bins for `orientations` from 9 to 11. In this way the length of the feature vector become 396 bytes for a single channel. For the classifier the colorspace was 'YUV' and I used all the channels (396*3 feature vectors)  

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a `GridSearchCV` so I could explore multiple combination for the kernel and C parameter. The best combination was `kernel=rbf` and `C=3`.

### Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search from top to bottom, using different scales. For each scale I chose different starting and ending y's, like this:

```python
    windows = [(1.0, 400, 400 + img_size),
               (1.0, 416, 416 + img_size),
               (1.5, 400, 400 + int(img_size * 1.5)),
               (1.5, 432, 432 + int(img_size * 1.5)),
               (2.0, 400, 400 + int(img_size * 2.0)),
               (2.0, 432, 432 + int(img_size * 2.0)),
               (3.5, 400, 400 + int(img_size * 3.5)),
               (3.5, 464, 464 + int(img_size * 3.5))]
```

And then I simply cycle to find the cars. 


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales using YUV 3-channel HOG features, which provided a decent result. To see the result of the classifier and the heatmap, I used some debug code drawing yellow rectangles for the hog classifier match, and blue rectangle the result from the thresholding of the heat map, as in the following picture (taken while experimenting):

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem was with the classifier. At the beginning I also forgot to use the full training set. The reduced training set, on top of that, was comprised of jpeg images, so with the full set I spent some time troubleshooting due to range mismatch (0..1 vs 0..255). Only after solving this problem I realised they warned about that in the lesson.
Another problem is that the code doesn't take into account the frame sequences: a rectangle in frame 0 could be used to infer the corresponding rectangle in frame 1 and so on. The code definitely could be improved. 
    

