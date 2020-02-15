# Bigdata-SOEN-691-Winter2020
This repository holds the code for "Human Activity Recgonition" system developed as the project component of SOEN 691 ("Big Data Analytics" - Winter 2020) course held at Concordia University, Montreal, Canada.

## Abstract
We aim at recognizing human activity being carried out based on the information collected from the smartphones. Human activity recognition has been an active field of research since quite a time owing to the wide array of its applications. Applications of human activity recognition involve active and assisted living systems for smart homes, healthcare monitoring applications, monitoring and surveillance systems for indoor and outdoor activities, and tele-immersion applications. We will be aiming to recognize the human activity based on the data collected from accelerometer and gyroscope embedded in smartphones devices. We will be using the data collected from a group of 30 volunteers within an age bracket of 19-48 years performing six activities namely walking, walking upstairs, walking downstairs, sitting, standing, and laying. We will be using this multivariate timeseries data to classify the performed activity amongst the aformenetioned six activities. This task will be performed utilizing the Apache Spark framework, dask, and scikit-learn library. The technology stack used might alter as the project proceeds.

## Introduction
The advent of a variety of wearable devices with high-functioning and multipurpose sensors provides an opportunity to utilize the data generated from it to build exciting applications upon it. At the stem of this lies an interesting challenge of Human Activity Recognition (HAR). HAR aims at recognizing the activity being performed based on a myriad of data collected continuously through a variety of sensors. Various kinds of sensors are used for this purpose like accelerometer and gyroscope, and these sensors capture three-dimensional data at a constant rate providing a temporal picture of movements. HAR marks as an interesting problem to solve due to its extensive applications in active and assisted living systems for smart homes, healthcare monitoring applications, monitoring and surveillance systems for indoor and outdoor activities, and tele-immersion applications.

For our project, we aim at utilizing the well-curated dataset hosted by the ‘UCI Machine learning repository’ and holds the data collected from smartphone devices. The primary task will be to correctly classify the data collected from accelerometer and gyroscopes amongst the following six categories {Walking, Walking upstairs, Walking downstairs, Standing, Laying, Sitting}. Correctly recognizing the human activity from the data collected through these sensors will give a boost to a variety of applications ranging from easing human life to fitness maintenance. Moreover, the results derived from developing such a HAR system could further be generalized or act as a step to work upon for developing more complex applications.We aim at utilizing Apache Spark data processing engine and Scikit-learn library to meet our goals.

### Related work
Human activity recognition has been a topic of research for a long time now and multitude of exciting and fruitful research has been conducted already. Few interesting work that we explored are links as follow:

- [Human Activity Recognition using Smartphone devices dataset](http://cs229.stanford.edu/proj2016/report/CanovaShemaj-HumanActivityRecognitionUsingSmartphoneData-report.pdf)
- [An explanatory Jupyter Notebook](https://rstudio-pubs-static.s3.amazonaws.com/291850_859937539fb14c37b0a311db344a6016.html)

## Materials and methods
###### Dataset
The dataset that we are using is Human Activity Recognition using a smartphone from UCI Machine Learning. The dataset is collected from a group of 30 persons with an age group of 18-50 years. It captures daily activities like walking, standing, laying, sitting, walking_upstair, walking_downstairs. It is collected with Samsung Galaxy II which has an accelerometer and gyroscope and captured the 3-axial linear acceleration and 3-axial angular velocity with a 50Hz rate.

This dataset is subject independent as 70% of the people’s data were taken as a training dataset and the other 30% of the people’s data were taken as a testing dataset. This is a time-series multivariate time-series dataset. It is sampled with an overlapping time window of 2.56 sec.  

After having a look at the dataset we found that it is already pre-processed a little with no missing values for the features and normalized the values between -1 to 1. 

- [Human Activity Recognition using Smartphone devices dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

###### Attribute information
Each record in dataset consists of the following:
1. Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
2. Triaxial Angular velocity from the gyroscope.
3. A 561-feature vector with time and frequency domain variables.
4. Its activity label.
5. An identifier of the subject who carried out the experiment.

###### Feature Selection:
Principal Component Analysis (PCA) might be used to select the important features that are highly uncorrelated. 

###### Technologies and Algorithms
We'd be utilizing Apache Spark to facilitate our data processing. For the purpose of classification, we will aim to explore three main techniques k-Nearest Neighbours, Random Forests, and Support Vector Machine. We will utilize scikit-learn library to achieve this tasks. We will aim to use different kernels with SVM like linear, polynomial, radial basis function. Additionally, in order to increase the performance of the model, we might try using k-fold cross-validation.

###### Model Evaluation
The model will be evaluated based on precision, recall, accuracy, and F1-score measures since relying on only one of the metrics would hardly suffice for our project.
