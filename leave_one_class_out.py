from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import StructType, StructField, FloatType
import numpy as np
import random
import plot


# Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Human activity classification model") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

# Helper function to read data
def string_split(x):
    return x['value'].split()

# Helper function for creating LabeledPoint
def change_label(i, x):
    return LabeledPoint(1 if x.label == i else 0, x.features)

# Helper function for training the SVM model
def model_per_class(i, labelled_training_data):
    one_against_rest_data = labelled_training_data.map(lambda x: change_label(i, x))
    ones = one_against_rest_data.filter(lambda x: x.label == 1)
    zeros = one_against_rest_data.filter(lambda x: x.label == 0)
    lis = random.sample(range(zeros.count()), ones.count())
    zeros = zeros.zipWithIndex().filter(lambda x: x[1] in lis).map(lambda x: x[0])
    one_against_rest_data = ones.union(zeros)
    model = SVMWithSGD.train(one_against_rest_data, iterations=10000)
    model.clearThreshold()
    return model


if __name__ == '__main__':
    # Create a spark session
    spark = init_spark()

    # Read training data, testing data, and headers
    training_data = spark.read.text("./data/train/X_train.txt").rdd
    training_label = spark.read.text("./data/train/y_train.txt").rdd
    training_subject = spark.read.text("./data/train/subject_train.txt").rdd

    testing_data = spark.read.text("./data/test/X_test.txt").rdd
    testing_label = spark.read.text("./data/test/y_test.txt").rdd
    testing_subject = spark.read.text("./data/test/subject_test.txt").rdd

    data_headers = open("./data/features.txt", "r").read().split("\n")[:-1]
    data_headers = [i.split()[1] for i in data_headers]

    # create dataframe with apt. headers
    print("Preparing training and testing data")
    training_data = training_data.map(lambda x: string_split(x))
    training_df = training_data.toDF(data_headers)

    testing_data = testing_data.map(lambda x: string_split(x))
    testing_df = testing_data.toDF(data_headers)

    training_label = training_label.zipWithIndex().map(lambda x: (x[1], x[0]))
    testing_label = testing_label.zipWithIndex().map(lambda x: (x[1], x[0]))

    training_subject = training_subject.zipWithIndex().map(lambda x: (x[1], x[0]))
    testing_subject = testing_subject.zipWithIndex().map(lambda x: (x[1], x[0]))

    # precision, recall, and fmeasure metric containing list
    precision = []
    recall = []
    fmeasure = []

    # Create labelled training and testing points points
    training_data = training_df.rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
    joined = training_data.join(training_label).join(training_subject)
    labelled_training_data = joined.map(
        lambda x: (LabeledPoint(float(x[1][0][1]['value']), list(x[1][0][0])), int(x[1][1]['value'])))

    testing_data = testing_df.rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
    joined = testing_data.join(testing_label).join(testing_subject)
    labelled_testing_data = joined.map(
        lambda x: (LabeledPoint(float(x[1][0][1]['value']), list(x[1][0][0])), int(x[1][1]['value'])))

    dataset = labelled_training_data.union(labelled_testing_data)
    results_dictionary = {}

    for participant in range(1, 31):
        print("Leaving out participant: ", participant)
        labelled_testing_data = dataset.filter(lambda x: x[1] == participant).map(lambda x: x[0])
        labelled_training_data = dataset.filter(lambda x: x[1] != participant).map(lambda x: x[0])

        # create models for one-vs-rest SVM binary classifiers
        print("Preparing models\n")
        models = [model_per_class(i, labelled_training_data) for i in range(1, 7)]
        print("Models prepared.\n")

        # make predictions for testing data
        print("Making predictions.\n")
        predictions = labelled_testing_data.map(
            lambda x: (float(np.argmax([model.predict(x.features) for model in models]) + 1), x.label))
        print("Predictions completed.\n")

        # calculate precision, recall, and f-measure
        print("Calculating evaluation metrics.\n")
        metrics = MulticlassMetrics(predictions)

        print("F-Measure: ", metrics.fMeasure())
        results_dictionary[participant] = metrics.fMeasure()
        print("Confusion matrix\n\n")
        confusion_matrix_filename = str(participant) + "_" + "cm_refactored.png"
        plot.plot_confusion_matrix(metrics.confusionMatrix().toArray(), confusion_matrix_filename)

        # print Precision and Recall for all the activities
        for i in range(1, 7):
            print("Precision for ", i, " is ", metrics.precision(i))
            print("Recall for ", i, " is ", metrics.recall(i))
            print("f-measure for ", i, " is ", metrics.fMeasure(float(i)), "\n")
            precision.append(metrics.precision(i))
            recall.append(metrics.recall(i))
            fmeasure.append(metrics.fMeasure(float(i)))
        fscore_filename = str(participant) + "_" + "fs_refactored.png"
        plot.plot_per_activity_metric(precision, recall, fmeasure, fscore_filename)
        precision = []
        recall = []
        fmeasure = []

    print("Results dictionary\n")
    print(results_dictionary)