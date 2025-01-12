# Bigdata-SOEN-691-Winter2020
This repository holds the code for "Human Activity Recgonition" system developed as the project component of SOEN 691 ("Big Data Analytics" - Winter 2020) course held at Concordia University, Montreal, Canada.

## Abstract
Human activity recognition has been an active field of research since quite a time owing to the wide array of its applications. Applications of human activity recognition involve active and assisted living systems for smart homes, healthcare monitoring applications, monitoring and surveillance systems for indoor and outdoor activities, and tele-immersion applications. We recognize human activity based on the data collected from a group of 30 volunteers performing six activities namely walking, walking upstairs, walking downstairs, standing, sitting, and laying. This information was collected using accelerometers and gyroscopes embedded in smartphone devices. Using technologies like Apache Spark, Scikit-learn, and Pandas we employed three algorithms namely k-Nearest Neighbours, Random Forests, and Support Vector Machines to recognize human activities. While SVM outperformed the other two on the test set with F1-Score of 94.2%, Random Forests showed the most consistent performance in subject cross validation with an average F1-Score of 92.8%.

## Introduction
The advent of a variety of wearable devices with high-functioning and multipurpose sensors provides an opportunity to utilize the data generated from it to build exciting applications upon it. At the stem of this lies an interesting challenge of Human Activity Recognition (HAR). HAR aims at recognizing the activity being performed based on a myriad of data collected continuously through a variety of sensors. Various kinds of sensors are used for this purpose like accelerometer and gyroscope, and these sensors capture three-dimensional data at a constant rate providing a temporal picture of movements. HAR marks as an interesting problem to solve due to its extensive applications in active and assisted living systems for smart homes, healthcare monitoring applications, monitoring and surveillance systems for indoor and outdoor activities, and tele-immersion applications.

For our project, we utilized the well-curated dataset hosted by the ‘UCI Machine learning repository’ and holds the data collected from smartphone devices. The primary task was to  correctly classify the data collected from accelerometer and gyroscopes amongst the following six categories: Walking, Walking upstairs, Walking downstairs, Sitting, Standing, Laying. Correctly recognizing the human activity from the data collected through these sensors will give a boost to a variety of applications ranging from easing human life to fitness maintenance. Moreover, the results derived from developing such a HAR system could further be generalized or act as a step to work upon for developing more complex applications. 


### Related work
Human activity recognition has been a topic of research for a long time now and multitude of exciting and fruitful research has been conducted already. We majorly relied on the work mentioned below. One of the following works employed SVM with linear, radial basis, and polynomial kernels, gradient boosted trees, linear discriminant analysis, and multinomial models. Multinomial model gave the best results with an error rate of 3.33% and the polynomial kernel worked best for SVM with an error rate of 3.97%. Whereas, the other work investigates the effect of subject cross validation and highlights the overestimation done by k-fold cross validation, specifically for overlapping windows, for human activity recognition.

- [Human Activity Recognition using Smartphone devices dataset](http://cs229.stanford.edu/proj2016/report/CanovaShemaj-HumanActivityRecognitionUsingSmartphoneData-report.pdf)
- [An explanatory Jupyter Notebook](https://rstudio-pubs-static.s3.amazonaws.com/291850_859937539fb14c37b0a311db344a6016.html)
- [Subject Cross Validation in Human Activity Recognition](https://arxiv.org/pdf/1904.02666.pdf)

## Materials and methods
###### Dataset
The dataset that we are using is 'Human Activity Recognition using a smartphone' from UCI Machine Learning repository. 

###### Dataset collection
- The dataset is collected from a group of 30 persons with an age group of 18-50 years.
- Collected with Samsung Galaxy II which has an accelerometer and gyroscope and captured the 3-axial linear acceleration and 3-axial angular velocity with a 50Hz rate.
- Captures daily activities such as:
    - Walking
    - Walking upstairs
    - Walking downstairs
    - Standing
    - Sitting
    - Laying

###### Dataset information
- It is a multivariate time-series dataset sampled with an overlapping window of 2.56 seconds.
- Train/test split: 70% training (7352 records) and 30% testing (2947 records).
- Each record consists of:
    - Triaxial acceleration of the accelerometer and the body acceleration.
    - Triaxial angular velocity from gyroscope.
    - 561-feature vector of time and frequency domain variables.
    - Activity label.
    - Identifier of the subject.
  
###### Dataset preprocessing
- The dataset is already preprocessed with no missing values.
- Each instance is normalized between -1 to 1.
- Fairly balanced dataset.


[Human Activity Recognition using Smartphone devices dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

###### Technologies
- PySpark (Apache Spark)
- Scikit-learn
- Pandas

###### Algorithms
- k-Nearest Neighbours
- Random Forests
- Support Vector Machines

###### Method
Models were developed for each of the three algorithms using two feature sets as mentioned below and further evaluated using subject cross validation.

- Feature set 1: Features representing ‘mean’, ‘standard deviation’, ‘maximum’, and ‘minimum’ value along 3-axial linear acceleration and 3-axial angular velocity.
- Feature set 2: Using all the available 561 features.

As both, precision and recall, are important measures for this task, we evaluated all our model using F1-score as it is a harmonic mean of precision and recall.

Moreover, to test the robustness of all the three algorithms for this task we performed subject cross validation, i.e. leave out one participant's activities as the test set and use data of the rest of participants as the training set. The overall performance for subject cross validation is represented as average F1-score for all 30 participants.

###### k-Nearest Neighbours
k-NN is implemented using Scikit-learn library. The performance of the k-NN model is evaluated thrice. Initially, k-NN is applied on the test data set. Furthermore, to better evaluate the performance of k-NN on unseen data 10-fold validation is applied. Since we are dealing with time-series data we also checked the performance with TimeSeriesSplit.

###### Random Forests
Initially, we did classification with the default parameters of the random forest classifier of Scikit-learn library. To better test the performance of the trained model, we applied 10-fold cross validation with TimeSeriesSplit. To improve the results and for parameter tuning, we applied a random search implementation of Scikit-learn that uses cross validation with TimeSeriesSplit. 

###### Support Vector Machines
Support vector machine with linear kernel was implemented using MLlib of PySpark. PySpark only supports linear kernel with SVM and because PySpark's implementation of SVM only supports binary classification, we implemented it using 'one-vs-rest' approach. However, expectedly that ran us into the problem of class imbalance. So to solve that problem we undersampled the majority class which improved the result by 3.5%.

## Results
The results obtained with k-NN are as follows:
Model setting | Feature set 1   | Feature set 2
------------- | --------------  | --------------
k-NN model without cross validation |   0.891   |   0.902
k-NN model with k-fold validation  |   0.910   |0.916
k-NN model with TimeSeriesSplit |   0.864   |   0.884

The results obtained with Random Forests are as follows:
Model setting | Feature set 1   | Feature set 2
------------- | --------------  | --------------
RF model without cross validation |   0.909   |   0.925
RF model with 10-fold validation with TimeSeriesSplit   |   0.887   |0.905
RF model with random Search hyperparameter tuning |   0.903   |   0.922

The results obtained with Support Vector Machines are as follows:
Model setting | Feature set 1   | Feature set 2
------------- | --------------  | --------------
SVM model without undersampling majority class  | 0.879  |   0.911
SVM model with undersampling of majority class  | 0.910  |  0.942

Performance comparison of the three selected algorithms is as follows:
Algorithm | F1-Score on test data set using feature set 1 | F1-Score on test data set using feature set 2 | Avg. F1-Score in subject cross validation
--------- | --------------------------------------------- | --------------------------------------------- | ------------------------------------------
k-Nearest Neighbours      |       0.891                                         |       0.902                                         |       0.839
Random Forests |        0.909                                   |       0.925                                         |           **0.928**
Support Vector Machine  |       0.910 |       **0.942** |       0.918

While the SVM with linear kernel outperforms the other two algorithm on test set when using all the 561 features, it fails to generalize the same way with different participants in subject cross validation. Whereas, the Random Forests maintained a consistent performance with different participants in subject cross validation.

Performing subject cross validation for all the three algorithms reveal an important insight that all these algorithms perform particularly bad for participant number 14. This goes to highlight that peculiar type of movements by different participants affects the results and hence, dataset with wider range of participants can help better accomadate these variance.

The graphical results of subject cross validation for the three algorithms are as follows:
![kNN SCV](Results/k-NN-subject-cross-validation.png)
![RF SCV](Results/RF-subject-cross-validation.png)
![SVM SCV](Results/SVM-subject-cross-validation.png)

Moreover, the confusion matrix generated by the three models on test data set for the best parameters and feature sets reveals interesting insight. All the three model lacks the capability of clearly distinguishing between the activities of 'standing' and 'sitting'. From our empirical results till this point, we suspect that there is an intrinsic confusion/similarity in the feature vectors for both activities. Hence, more sophisticated methods such as SVM with polynomial kernel or neural networks are required to handle this problem or perhaps more data can help solve this problem. 

The confusion matrix for the test set using feature set 2, i.e. all the features, by the three models are as follows:
![kNN CM](Results/kNN-confusion-matrix.png)
![RF CM](Results/RF-confusion-matrix.png)
![SVM CM](Results/SVM-confusion-matrix.png)

## Discussion
The key derivations from our solution of 'Human Activity Recognition' for 'Human Activity Recognition with Smarthphones' dataset hosted by 'UCI Machine Learning' repository are as follows: 
1) There is need for wider range of participants to incorporate the pecularities of performing the said activities to avoid getting particularly bad result for some participants like for the participant number 14.
2) K-fold cross validation fails to capture the true picture of the performance of k-NN as it fails to generate similar results for subject cross validation. K-fold cross validation with k-NN overestimated the performance of the model by 8.4%.
3) Empirical results from our model indicate an intrinsic similarity/correlation in the feature vectors of the activities namely 'sitting' and 'standing'. Hence, more sophisticated methods such as SVM with polynomial kernel or neural networks are required to handle this problem or perhaps more data can help solve this problem. 
4) Amongst the three chosen methods, Random forests tuned with random search operation maintains the most consistent performance with both, test data set and subject cross validation. 

The results generated by our solution are competitive and falls behind by 2-3% compared with the benchmark solution quoted in the related work. Potentially, using similar techniques as quoted in the related work or using more complex kernels like radial basis or polynomial kernel with SVM might help achieve similar results.

For future work, investigation into the improvement of the recognition system by incorporating wider range of participants is required. Moreover, either more data is required for better distinguishing between 'sitting' and 'standing' activities or attempting to use more complex methods for classification might help improve the performance.
