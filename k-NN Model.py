#!/usr/bin/env python
# coding: utf-8

from math import sqrt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score


# Spliting features.txt to get headers for the data
h1 = open('./data/features.txt','r').read().split('\n')[:-1]
header_list = [i.split()[1] for i in h1]

# Reading training dataset with headers column
X_train = pd.read_csv('./data/train/X_train.txt', header=None, names=['column1'], sep='\t')
X_train = X_train['column1'].str.split("\s{1,}", expand=True)
X_train.drop(X_train.columns[0], inplace=True, axis=1)
X_train.columns = header_list

# Reading training target label
y_train = pd.read_csv('./data/train/y_train.txt', header=None)


# Reading testing dataset with headers column
X_test = pd.read_csv('./data/test/X_test.txt', header=None, names=['column2'], sep='\t')
X_test = X_test['column2'].str.split("\s{1,}", expand=True)
X_test.drop(X_test.columns[0], inplace=True, axis=1)
X_test.columns = header_list

# Reading testing target label
y_test = pd.read_csv('./data/test/y_test.txt', header=None)


# FEATURE SET 1
# First creating  feature set with mean, standard_deviation, maximum, minimum columns for training and testing dataset
X_train_fs1 = X_train.loc[:, [column_name for column_name in list(X_train.columns) if 'mean()' in column_name or 'std()' in column_name or 'max()' in column_name or 'min()' in column_name]]
X_test_fs1 = X_test.loc[:, [column_name for column_name in list(X_test.columns) if 'mean()' in column_name or 'std()' in column_name or 'max()' in column_name or 'min()' in column_name]]


# k-NN model without CV for FS1
knn_fs1 = KNeighborsClassifier(7)
knn_fs1.fit(X_train_fs1, y_train)
y_pred_fs1_knn = knn_fs1.predict(X_test_fs1)
print(metrics.f1_score(y_test, y_pred_fs1_knn, average='weighted'))
print(metrics.confusion_matrix(y_test, y_pred_fs1_knn, normalize=None))


# k-NN model without CV for FS2
knn_fs2 = KNeighborsClassifier(7)
knn_fs2.fit(X_train, y_train)
y_pred_fs2_knn = knn_fs2.predict(X_test)
print(metrics.f1_score(y_test, y_pred_fs2_knn, average='weighted'))
print(metrics.confusion_matrix(y_test, y_pred_fs2_knn))


# Combining the training and testing set
X_dataset = pd.concat([X_train, X_test])
Y_dataset = pd.concat([y_train,y_test])

# FEATURE SET 1
# Creating FS1 again from combined dataset for cross validation
X_data_fs1 = X_dataset.loc[:, [column_name for column_name in list(X_dataset.columns) if 'mean()' in column_name or 'std()' in column_name or 'max()' in column_name or 'min()' in column_name]]


# k-NN model with TimeSeriesSplit with FS1
knn_fs1_cv = KNeighborsClassifier(7)
tscv = TimeSeriesSplit(n_splits=10)
scores_fs1_cv = cross_val_score(knn_fs1_cv, X_data_fs1, np.ravel(Y_dataset, order='C'), cv=tscv, scoring='f1_weighted')
print(scores_fs1_cv.mean())


# k-NN model with TimeSeriesSplit CV for FS2
knn_fs2_cv = KNeighborsClassifier(7)
scores_fs2_cv = cross_val_score(knn_fs2_cv, X_dataset, np.ravel(Y_dataset, order='C'), cv=tscv, scoring='f1_weighted')
print(scores_fs2_cv.mean())


# k-NN model with k-fold for FS1
knn_fs1_k = KNeighborsClassifier(7)
scores_fs1_k = cross_val_score(knn_fs1_k, X_data_fs1, np.ravel(Y_dataset, order='C'), cv=10, scoring='f1_weighted')
print(scores_fs1_k.mean())


# k-NN moddel with k-fold for FS2
knn_fs2_k = KNeighborsClassifier(7)
scores_fs2_k = cross_val_score(knn_fs2_k, X_dataset, np.ravel(Y_dataset, order='C'), cv=10, scoring='f1_weighted')
print(scores_fs2_k.mean())


# Reading subject file to create subjectCV dataset
subj_train = pd.read_csv('./data/train/subject_train.txt', header=None, names=['subject'])
subj_test = pd.read_csv('./data/test/subject_test.txt', header=None, names=['subject'])
subj = pd.concat([subj_train, subj_test])
X_dataset['subject'] = subj.values
Y_dataset['subject'] = subj.values


# Subject CV with k-NN 
scores_subjCV_knn = {}
knn_subj = KNeighborsClassifier(8)
for i in range(1,31):
  X_test1 = X_dataset[X_dataset['subject'] == i]
  X_train1 = X_dataset[X_dataset['subject'] != i]

  y_train1 = Y_dataset[Y_dataset['subject'] != i]
  y_test1 = Y_dataset[Y_dataset['subject'] == i]
  y_test1 = y_test1.drop(['subject'], axis=1)
  y_train1 = y_train1.drop(['subject'], axis=1)
  
  knn_subj.fit(X_train1,np.ravel(y_train1, order='C'))
  y_pred_i = knn_subj.predict(X_test1)
  f1_measure = metrics.f1_score(y_test1, y_pred_i, average='weighted')
  scores_subjCV_knn[i] = f1_measure
print(scores_subjCV_knn)

