from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import StructType, StructField, FloatType
import numpy as np

# Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Human activity classification model") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def string_split(x):
    return x['value'].split()

def change_label(i,x):
    return LabeledPoint(1 if x.label==i else 0, x.features)

def model_per_class(i, labelled_training_data):
    one_against_rest_data = labelled_training_data.map(lambda x: change_label(i, x))
    model = SVMWithSGD.train(one_against_rest_data, iterations=10000)
    model.clearThreshold()
    return model

if __name__ == '__main__':
    #Create a spark session
    spark = init_spark()

    #Read training data, testing data, and headers
    training_data = spark.read.text("./data/train/X_train.txt").rdd
    training_label = spark.read.text("./data/train/y_train.txt").rdd

    testing_data = spark.read.text("./data/test/X_test.txt").rdd
    testing_label = spark.read.text("./data/test/y_test.txt").rdd

    data_headers = open("./data/features.txt", "r").read().split("\n")[:-1]
    data_headers = [i.split()[1] for i in data_headers]

    print("Preparing training and testing data")
    #create dataframe with apt. headers
    training_data = training_data.map(lambda x: string_split(x))
    training_df = training_data.toDF(data_headers)

    testing_data = testing_data.map(lambda x: string_split(x))
    testing_df = testing_data.toDF(data_headers)



    # Feature Set 1: fetch only features representing 'mean'
    print("FEATURE SET 1: FETCHING ONLY FEATURES REPRESENTING 'MEAN'")
    mean_headers = []
    for i in data_headers:
        if "mean" in i:
            mean_headers.append(i)

    training_mean_df = training_df.select(mean_headers)
    testing_mean_df = testing_df.select(mean_headers)
    #print((training_df.count(), len(training_df.columns))) #shape of the selected dataframe

    #Create labelled training and testing points points
    training_data = training_mean_df.rdd.zipWithIndex().map(lambda x:(x[1], x[0]))
    training_label = training_label.zipWithIndex().map(lambda x:(x[1],x[0]))
    joined = training_data.join(training_label)
    labelled_training_data = joined.map(lambda x: LabeledPoint(float(x[1][1]['value']), list(x[1][0])))

    #testing_data = testing_df.rdd
    testing_data = testing_mean_df.rdd.zipWithIndex().map(lambda x:(x[1], x[0]))
    testing_label = testing_label.zipWithIndex().map(lambda x:(x[1], x[0]))
    joined = testing_data.join(testing_label)
    labelled_testing_data = joined.map(lambda x: LabeledPoint(float(x[1][1]['value']), list(x[1][0])))

    print("Data prepared.\n")
    print("Preparing models\n")
    #create models for one-vs-rest SVM binary classifiers
    models = [model_per_class(i, labelled_training_data) for i in range(1, 7)]

    print("Models prepared.\n")
    print("Making predictions.\n")
    #make predictions for testing data
    predictions = labelled_testing_data.map(lambda x: (float(np.argmax([model.predict(x.features) for model in models]) + 1), x.label))
    print("Predictions completed.\n")

    print("Calculating evaluation metrics.\n")
    #calculate precision, recall, and f-measure
    metrics = MulticlassMetrics(predictions)

    accuracy = metrics.accuracy
    print("Accuracy: ", accuracy)
    print("Confusion matrix\n\n")
    print(metrics.confusionMatrix())

    for i in range(1,7):
        print("Precision for ", i, " is ", metrics.precision(i))
        print("Recall for ", i, " is ", metrics.recall(i))
        print("f-measure for ", i, " is ", metrics.fMeasure(float(i)), "\n")



    # Feature Set 2: fetch only features representing 'mean'
    print("FEATURE SET 2: FETCHING ONLY FEATURES REPRESENTING MULTIPLE FEATURES")
    feature_headers = []
    for i in data_headers:
        if "mean" in i or "std" in i or "max" in i or "min" in i:
            feature_headers.append(i)

    training_multiple_df = training_df.select(feature_headers)
    testing_multiple_df = testing_df.select(feature_headers)
    # print((training_df.count(), len(training_df.columns))) #shape of the selected dataframe

    # Create labelled training and testing points points
    training_data = training_multiple_df.rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
    joined = training_data.join(training_label)
    labelled_training_data = joined.map(lambda x: LabeledPoint(float(x[1][1]['value']), list(x[1][0])))

    # testing_data = testing_df.rdd
    testing_data = testing_multiple_df.rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
    joined = testing_data.join(testing_label)
    labelled_testing_data = joined.map(lambda x: LabeledPoint(float(x[1][1]['value']), list(x[1][0])))

    print("Data prepared.\n")
    print("Preparing models\n")
    # create models for one-vs-rest SVM binary classifiers
    models = [model_per_class(i, labelled_training_data) for i in range(1, 7)]

    print("Models prepared.\n")
    print("Making predictions.\n")
    # make predictions for testing data
    predictions = labelled_testing_data.map(lambda x: (float(np.argmax([model.predict(x.features) for model in models]) + 1), x.label))
    print("Predictions completed.\n")

    print("Calculating evaluation metrics.\n")
    # calculate precision, recall, and f-measure
    metrics = MulticlassMetrics(predictions)

    accuracy = metrics.accuracy
    print("Accuracy: ", accuracy)
    print("Confusion matrix\n\n")
    print(metrics.confusionMatrix())

    for i in range(1, 7):
        print("Precision for ", i, " is ", metrics.precision(i))
        print("Recall for ", i, " is ", metrics.recall(i))
        print("f-measure for ", i, " is ", metrics.fMeasure(float(i)), "\n")



    # Feature Set 3: fetch all features
    print("FEATURE SET 3: FETCHING All THE FEATURES")

    # Create labelled training and testing points points
    training_data = training_df.rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
    joined = training_data.join(training_label)
    labelled_training_data = joined.map(lambda x: LabeledPoint(float(x[1][1]['value']), list(x[1][0])))

    # testing_data = testing_df.rdd
    testing_data = testing_df.rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
    joined = testing_data.join(testing_label)
    labelled_testing_data = joined.map(lambda x: LabeledPoint(float(x[1][1]['value']), list(x[1][0])))

    print("Data prepared.\n")
    print("Preparing models\n")
    # create models for one-vs-rest SVM binary classifiers
    models = [model_per_class(i, labelled_training_data) for i in range(1, 7)]

    print("Models prepared.\n")
    print("Making predictions.\n")
    # make predictions for testing data
    predictions = labelled_testing_data.map(
        lambda x: (float(np.argmax([model.predict(x.features) for model in models]) + 1), x.label))
    print("Predictions completed.\n")

    print("Calculating evaluation metrics.\n")
    # calculate precision, recall, and f-measure
    metrics = MulticlassMetrics(predictions)

    accuracy = metrics.accuracy
    print("Accuracy: ", accuracy)
    print("Confusion matrix\n\n")
    print(metrics.confusionMatrix())

    for i in range(1, 7):
        print("Precision for ", i, " is ", metrics.precision(i))
        print("Recall for ", i, " is ", metrics.recall(i))
        print("f-measure for ", i, " is ", metrics.fMeasure(float(i)), "\n")