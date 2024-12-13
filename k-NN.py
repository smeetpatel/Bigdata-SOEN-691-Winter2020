import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.decomposition import PCA

# h1 = pd.read_csv('E:/CONCORDIA/SOEN 691 (Big Data)/Project/UCI HAR Dataset/features.txt', sep='\t', names=['index'], header=None)
# header = h1['index'].str.split(" ", n=1, expand=True)
# h1['features'] = header[1]
# h1.drop(columns=['index'], inplace=True)
# header_list = h1['features'].values.tolist()

# Spliting features.txt to get headers for the data
h1 = open('E:/CONCORDIA/SOEN 691 (Big Data)/Project/UCI HAR Dataset/features.txt','r').read().split('\n')[:-1]
header_list = [i.split()[1] for i in h1]

# Reading training dataset with headers column
X_train = pd.read_csv('E:/CONCORDIA/SOEN 691 (Big Data)/Project/UCI HAR Dataset/train/X_train.txt', header=None, names=['column1'], sep='\t')
X_train = X_train['column1'].str.split("\s{1,}", expand=True)
X_train.drop(X_train.columns[0], inplace=True, axis=1)
X_train.columns = header_list

# Reading training target label
y_train = pd.read_csv('E:/CONCORDIA/SOEN 691 (Big Data)/Project/UCI HAR Dataset/train/y_train.txt', header=None)


# Reading testing dataset with headers column
X_test = pd.read_csv('E:/CONCORDIA/SOEN 691 (Big Data)/Project/UCI HAR Dataset/test/X_test.txt', header=None, names=['column2'], sep='\t')
X_test = X_test['column2'].str.split("\s{1,}", expand=True)
X_test.drop(X_test.columns[0], inplace=True, axis=1)
X_test.columns = header_list

# Reading testing target label
y_test = pd.read_csv('E:/CONCORDIA/SOEN 691 (Big Data)/Project/UCI HAR Dataset/test/y_test.txt', header=None)


# FEATURE SET 1
# Creating first feature set with only mean()-columns for training and testing dataset
X_train_fs1 = X_train.loc[:, [column_name for column_name in list(X_train.columns) if 'mean()' in column_name]]
X_test_fs1 = X_test.loc[:, [column_name for column_name in list(X_test.columns) if 'mean()' in column_name]]

# Training k-NN model and predicting for Feature Set 1
knn_fs1 = KNeighborsClassifier(7)
knn_fs1.fit(X_train_fs1, y_train)
predicted_knn_fs1 = knn_fs1.predict(X_test_fs1)

print("FS1 with 7 nearest neighbours")
print(metrics.f1_score(y_test, predicted_knn_fs1, average=None))
print(metrics.confusion_matrix(y_test, predicted_knn_fs1, normalize='all'))



# FEATURE SET 2
# First creating  feature set with mean, standard_deviation, maximum, minimum columns for training and testing dataset
X_train_fs2 = X_train.loc[:, [column_name for column_name in list(X_train.columns) if 'mean()' in column_name or 'std()' in column_name or 'max()' in column_name or 'min()' in column_name]]
X_test_fs2 = X_test.loc[:, [column_name for column_name in list(X_test.columns) if 'mean()' in column_name or 'std()' in column_name or 'max()' in column_name or 'min()' in column_name]]

# Training k-NN model and predicting for Feature Set 2
knn_fs2 = KNeighborsClassifier(7)
knn_fs2.fit(X_train_fs2, y_train)
predicted_knn_fs2 = knn_fs2.predict(X_test_fs2)

print("FS2 with 7 nearest neighbours")
print(metrics.f1_score(y_test, predicted_knn_fs2, average=None))
print(metrics.confusion_matrix(y_test,predicted_knn_fs2, normalize='all'))




# FEATURE SET 3
# Applying Principal Component Analysis to create feature set 3 for training and testing dataset
pca = PCA(.99)
pca.fit(X_train)
print('\n')
print(pca.n_components_)
X_train_fs3 = pca.transform(X_train)
X_test_fs3 = pca.transform(X_test)

# Training k-NN model and predicting for feature set 3
knn_fs3 = KNeighborsClassifier(7)
knn_fs3.fit(X_train_fs3, y_train)
predicted_knn_fs3 = knn_fs3.predict(X_test_fs3)

#Evaluting using F1_score and confusion matrix
print("FS3 with 7 nearest neighbours")
print(metrics.f1_score(y_test, predicted_knn_fs3, average=None))
print(metrics.confusion_matrix(y_test, predicted_knn_fs3))






