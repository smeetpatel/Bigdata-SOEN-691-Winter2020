from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType

# Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Human activity classification model") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def string_split(x):
    return str(x).split()[1:]

if __name__ == '__main__':
    #Create a spark session
    spark = init_spark()

    #Read training data and headers
    training_data = spark.read.text("./data/train/X_train.txt").rdd
    training_label = spark.read.text("./data/train/y_train.txt").rdd
    data_headers = open("./data/features.txt", "r").read().split("\n")[:-1]
    data_headers = [i.split()[1] for i in data_headers]

    #create dataframe with apt. headers
    training_data = training_data.map(lambda x: string_split(x))
    tdf = training_data.toDF(data_headers)

    #fetch only features representing 'mean'
    mean_headers = []
    for i in data_headers:
        if "mean" in i:
            mean_headers.append(i)
    tdf = tdf.select(mean_headers)
    print((tdf.count(), len(tdf.columns))) #shape of the selected dataframe