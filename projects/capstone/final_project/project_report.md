# Machine Learning Engineer Nanodegree
## Capstone Project
Harshvardhan Aggarwal  
January 14th, 2018



[//]: # (Image References)

[image1]: ./images/testing_data_stats.png "Testing Data Stats"
[image2]: ./images/testing_data.png "Testing Data"
[image3]: ./images/training_data_stats.png "Training Data Stats"
[image4]: ./images/training_data_business_mapping.png "Training data business mapping"
[image5]: ./images/training_data_distribution.png "Training Data Distribution"
[image6]: ./images/training_labels.png "Training labels"

## I. Definition

### Project Overview
This project focuses on analyzing and predicting various types of labels a Restaurant may have. Many of the problems today lie in correctly classifying the images, and restaurant classification is one of them. With growing number of restaurants, it becomes difficult to filter out restaurants based on specific category.

With users wanting a good recommendation for trying out new restaurants, it becomes increasingly necessary to develop a robust machine learning algorithm that can correctly classify restaurants based on various parameters like the quality of food, average wait times, kid friendly etc. This problem is just a stepping stone in classifying different types of business based on user uploaded pictures of restaurants and their qualities in order to provide better services than manual task are able to achieve.


### Problem Statement

The main goal of this project is to look at various pictures of a restaurant business, identify its core features from the images and classify them based on their features. There are thousands of photos uploaded by users everyday in different lighting conditions, shapes, angle that it becomes difficult to properly tag them without the help of automated system.

The problem can be considered as a classification problem, where images can have multiple class labels assigned to them. A user searching for a particular restaurant, say kid friendly, will help deliver results which are useful to them and in the end saving time.



### Metrics
The following evaluation metrics can be used in this case:
1. F(beta) score (from Kaggle evaluation method)
   The F(beta) score can be calculated as a measure of precision and recall.

   **_Precision_** = Sum (True Positives) / Sum (True Positives + False Positives)  

   **_Recall_** = Sum (True Positives) / Sum (True Positives + False Negatives)

   ###### F(beta) =  
              (1 + beta^^2) * (precision * recall) /
              (beta^^2 * precision) + recall

  The benchmark model for this problem has a F-beta score of 0.64597 with random guessing algorithm [3]. A solution model having F-beta score _more_ than the benchmark model can be considered as good model.


## II. Analysis

### Data Exploration
The dataset for this problem is obtained from Kaggle Competition "Yelp Restaurant Photo Classification" (Refer [1] and [2]). This dataset contains thousands of user uploaded images for restaurants. These images contain images of food, drinks of various cuisines, restaurant interiors, exteriors etc belonging to a particular restaurant. Similarly for all other restaurants, several types of images are present.


Following are some specifications of the dataset

| Feature |     Total count	|
|:---------------------:|:---------------------------------------------:|
| Total number of training images | 234,842 |
| Total number of testing images | 237,152  |
| Total number of businesses in training set   |  1,996 |
| Total number of businesses in testing set   | 10,000  |

The dataset contains mapping of photos to a particular business in both train and test set. The images are of high quality and mostly have different height and width.

The below image shows how the training photos id's are mapped to each business id in the dataset.

![alt text][image4]

As shown above, each photo is mapped to a particular business (restaurant). Each business owner can have multiple photos which can then be assigned multiple labels depending on the image.

Similarly, for testing dataset also there is a mapping between each test photo and businesses. These businesses are different from the training data.

![alt text][image2]

The dataset also has 9 categories of labels to be predicted upon. They are:

| Category Label |     Category Name	|
|:---------------------:|:---------------------------------------------:|
| 0 | good_for_lunch |
| 1 | good_for_dinner  |
| 2 | takes_reservations  |
| 3 | outdoor_seating  |
| 4 | restaurant_is_expensive  |
| 5 | has_alcohol  |
| 6 | has_table_service  |
| 7 | ambience_is_classy  |
| 8 | good_for_kids  |


Hence, for each business there is clearly defined multiple output labels which identifies the type or category of restaurant.

So, for example, below is the mapping of training business with their labels.

![alt text][image6]

The numbers corresponding in label are mapped to the label name in table above. So business id `1001` has labels `(0, 1, 6, 8)`. This means business 1001 is identified as `good_for_lunch`, `good_for_dinner`, `has_table_service` and `good_for_kids`.

The goal is to find the label for testing businesses.

The training dataset did have some abnormalities, which were identified while working with the data. Four of the businesses in the dataset don't have any labels associated with them. Out of total 2000 businesses present in the dataset, only 1996 were considered for training.



### Exploratory Visualization
The plot below shows the number of photos present for each class of label for restaurant.

![alt text][image5]

From the above plot, it is evident that each class of labels that we are trying to predict upon has sufficient number of photos to extract features and make a prediction. This ensures that no particular class of label will be biased in training.


Also, the below images show the distribution of images between different businesses.

For training set, out of total 234,842, the average number of photos per business is 117. Also, the minimum number of photos for a restaurant is at least 2 images.

![alt text][image3]


Similarly, the number of images per business for testing set are in line with the training images as shown below:

![alt text][image1]



### Algorithms and Techniques
The algorithm and techniques used in this project includes use of convolution neural networks to extract the features from images and then using these features as an attribute in a supervised learning algorithm like linear regression or support vector machines.

During training process, a CNN with many layers would be used to extract the high-level and low-level features from the images. I incorporated Caffe to make use of multiple core of GPU's at a single time to reduce time extracting features. Caffe is built to support parallel processing on multiple cores of GPU.

The tunable parameters for this stage can range from specifying the shape of image to be used to the number of filters at each layer of CNN.

The CNN was chosen to represent an image in terms of numerical features along with business (restaurant) and its labels. The business and combination of multiple image features can be used as input to a supervised learning classifier while labels will act as the output to be predicted upon.

The support vector machines are used as supervised learning classifier to assign a label for each business. The tunable parameters for SVM can be the type of kernel used, the initial random state to shuffle data and the gamma variable to determine the variance.


The classification of restaurants problem falls into the multi-label classification problems. The multi-label classification problems are the ones where each instance or row of data can be assigned to multiple classes (also called labels) at once. They differ from binary classifiers or multi-class classifiers in way that binary or multi-class classifiers have all classes as mutually exclusive. This means  an instance of data cannot be assigned to more than one label.

This project uses OneVsRestClassifier algorithm, which is an implementation of multi-label classification provided by sklearn using a base classifier of user's choice (e.g. Decision Tree, SVM, Linear Regression). This classifier fits a classifier for each class of label to be predicted upon. In this problem, it will generate 9 types of classifier for each label to output a confidence measure if the particular instance can belong to one of the classes. In the end it outputs all the labels which have a confidence measure as compared to others.


### Benchmark
The benchmark metrics for this project is obtained from Kaggle leaderboard page. The project has a F-beta score of 0.64597 with random guessing algorithm as shown in Kaggle leaderboard page[3]. This will be considered as the benchmark model for this problem.

A solution model having F-beta score _more_ than the benchmark model can be considered as good model.


## III. Methodology

### Data Preprocessing
The dataset for this project was given in form of csv files. The csv files contains the mapping of each image (image name) with a particular restaurant id.

For pre-processing the images, I used Caffe framework to modify the input images in the form Caffe expects. Caffe is a deep-learning framework designed for speed and works well for image classification.

The steps performed in pre-processing the data are as follows:
1. The pandas library is used to load all the image file names from the csv file.
2. Once the file name of images are present in an array, each of the image is read using Caffe library.
3. Caffe framework provides Caffe transformer module, which can transform the input image in the form Caffe can process.
4. The first step in Caffe transformation is to convert input image size channel from `(Height x Width x Number of color channels)` to `(Number of color channels x Height x Width)`.
5. Next step is to calculate the mean and subtract the mean BGR pixel values from image. This is done to even out the image properties (like contrast etc).
6. The input image is now scaled to [0, 255] pixel values as Caffe operates on images in this range.
7. Next, the RGB color channel is swapped into BGR format so that Caffe can process the image correctly.


At the end of these steps, all the images were of same size with same order of color channels to provide consistent results.


### Implementation
The implementation step consisted of below main steps:

1. Load the training images from CSV file.
2. Compute features from these training images and store them.
3. Load the businesses and their labels from CSV file.
4. Compute mean of all image features associated with one business and add it as a feature along with label.
5. Use the business and features as training data and labels as training output to train a multi-label classifier.
6. Extract features from testing images.
7. Compute mean of all images associated with a testing business.
8. Predict the labels for testing set using the classifier trained earlier.


Now lets take a detailed look at each of the steps described above.

**1.** Firstly I used pandas DataFrame to load `train_photo_to_biz_ids.csv` file to create a list of all the images being used in training dataset. The training dataset looked like below:

![alt text][image4]


**2.** Next, to extract features from each image I thought of using a custom CNN to extract low and high-level features with multiple layers using different filter size and strides and padding. Since, the training image dataset consisted of 234,842 images with similar number for testing set, it was essential that all important features were extracted within a reasonable amount of time. A custom-built CNN turned out to be very time consuming taking hours to extract features from all the images. I then started to explore to use Caffe within my project to extract image features. As it turns out, Caffe proved helpful in extracting the features as it reduced the time to couple of hours as compared earlier.

However, using Caffe was no small feat as setting up of Caffe took a long time on AWS and made sure all the dependencies were up to date. Once the Caffe was up and running, I used the Caffe provided model called `BVLC Reference CaffeNet` which is Caffe's implementation of ImageNet optimized for Caffe framework. The BVLC reference model has 8 layers as shown below:

I used the BVLC CaffeNet to provide image as input and perform computation on the images for all layers. The layer-8 is very specific to CaffeNet as it outputs features based on the number of classes for this model. I used the output of layer-7 as the image features that can be used to classify each business later. The layer-7, also called fc7, provides good representation of all the features for image. Some of the features extracted from images looks like below:



**3.** Next, I loaded the business and their output labels from CSV file. This looked like below:

![alt text][image6]

**4.** The earlier features calculated from images were then averaged out and added as an extra column to above businesses. This created a new attribute to be used by classifier to train and learn which labels are associated with image features. The combination of businesses, features and labels is shown as below:


**5.** The next step was to train the classifier. Before classifying, I used the image features as input to the classifier and labels as the expected output (labels). This is a multi-label classification problem so it required to one hot encode the labels from (0, 1, 2...) to something that can be represented in binary form. I used `sklearn.MultiLabelBinarizer()` to transform the training labels into a 2-D matrix where point in (row x column) having value 1 represents that this data instance (image feature) can be represented by this label column. The output of MultiLabelBinarizer looks like below:

I then used `sklearn.train_test_split()` method to split the data into training and validation set and then used the `sklearn.OneVsRestClassifier()` with SVM as base classifier as explained earlier to be used as classifier algorithm.


**6.** I then used the same Caffe model `BVLC Reference CaffeNet` to calculate the image features for testing dataset. I again used the second-to-last 'fc7' layer to get the same level of features as in training set.

**7.** As described in step-4 above, the mean feature was calculated from the 'fc7' layer features.

**8.** Using these mean testing features, I used the classifier which was trained in step-5 to predict multiple labels for each image feature.


At the end of the above step-8, we get the predicted labels for all the testing business which can be scored again the Kaggle submission test suite.



### Refinement
The first step in refinement was to choose between a custom Convolution Neural network built to extract image features versus using transfer learning by using an existing algorithm and reduce training time.

The main tradeoff between choosing the two was the training time for computing features for 234,842 training images and 237,152 testing images. Using a custom built CNN was taking long time to process all the images (sometimes more than 6 hours). The accuracy of finding the correct features was also a concern as if enough different features are not present, the classifier may not produce good results. Due to processing time and accuracy, I used the existing Caffe model to extract the features.


The second refinement I made was adjusting the parameters of Support Vector Machine used in the OneVsRestClassifier algorithm.

The results for the linear kernel SVM on testing dataset had an accuracy of 0.75. Given the dataset (images) and the features extracted from it, it seems that this data is not linearly separable as many image features were conflicting with the labeled outputs. So, I used the 'Radial basis function (rbf)' kernel to create hyperplanes so as to differentiate between different features. The results for using rbf kernel on testing dataset resulted in improved accuracy of 0.79.




## IV. Results

### Model Evaluation and Validation

The final model consisted of the following algorithms and techniques:

1. A CaffeNet model called `BVLC Reference CaffeNet` was used to extract image features from both training and testing set. The output of the second last 'fc7' layer was used as input features to classifier.
2. The output labels in training set was one hot encoded using sklearn.MultiLabelBinarizer().
3. The image features extracted above for both training and testing were used as input to OneVsRestClassifier().
4. The OneVsRestClassifier() used SVM as base classifier with `rbf` kernel to classify training images and predict on testing images.

The choice of using CaffeNet reference model assures that any changes in input images will provide a similar fingerprint of image features if image input dataset was changed.

The choice of OneVsRestClassifier() seems appropriate for this multi-label classifying problem as this algorithm will try to fit a classifier for each class of labels in order to predict the correct labels for testing set.

This model choice is confirmed by the fact that it was able to achieve a high accuracy of 0.79 on 237,152 testing images alone.

Also, the model achieved a high accuracy score of 0.829 on validation dataset (25% of training dataset was reserved to be used as validation data).


### Justification

The benchmark model earlier mentioned had an accuracy rate of 0.64597 with random guessing as the algorithm chosen.

The model implemented above with an SVM classifier using `Linear` kernel had an accuracy of 0.75.

The same model with SVM classifier using `RBF` kernel has an even higher accuracy of 0.79. This accuracy result is from the Kaggle website and affirms that this model is stronger than the benchmark model and does a good job in classifying of nearly 80% of images into correct categories.



## V. Conclusion

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
