#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit


# Spliting features.txt to get headers for the data
h1 = open('/features.txt','r').read().split('\n')[:-1]
header_list = [i.split()[1] for i in h1]

# Reading training dataset with headers column
X_train = pd.read_csv('/X_train.txt', header=None, names=['column1'], sep='\t')
X_train = X_train['column1'].str.split("\s{1,}", expand=True)
X_train.drop(X_train.columns[0], inplace=True, axis=1)
X_train.columns = header_list

# Reading training target label
y_train = pd.read_csv('/y_train.txt', header=None)


# Reading testing dataset with headers column
X_test = pd.read_csv('/X_test.txt', header=None, names=['column2'], sep='\t')
X_test = X_test['column2'].str.split("\s{1,}", expand=True)
X_test.drop(X_test.columns[0], inplace=True, axis=1)
X_test.columns = header_list

# Reading testing target label
y_test = pd.read_csv('/y_test.txt', header=None)

X_dataset = pd.concat([X_train, X_test])
Y_dataset = pd.concat([y_train,y_test])

# Reading the subject file for subject CV
subj_train = pd.read_csv('/subject_train.txt', header=None, names=['subject'])
subj_test = pd.read_csv('/subject_test.txt', header=None, names=['subject'])
subj = pd.concat([subj_train, subj_test])
X_dataset['subject'] = subj.values
Y_dataset['subject'] = subj.values


# Feature Set 1
# First creating  feature set with mean, standard_deviation, maximum, minimum columns for training and testing dataset
X_train_fs1 = X_train.loc[:, [column_name for column_name in list(X_train.columns) if 'mean()' in column_name or 'std()' in column_name or 'max()' in column_name or 'min()' in column_name]]
X_test_fs1 = X_test.loc[:, [column_name for column_name in list(X_test.columns) if 'mean()' in column_name or 'std()' in column_name or 'max()' in column_name or 'min()' in column_name]]


# RFC with default parameters for FS1
rfc_model_fs1 = RandomForestClassifier(random_state=42)
rfc_model_fs1.fit(X_train_fs1, np.ravel(y_train))
y_pred_fs1_model = rfc_model_fs1.predict(X_test_fs1)

print('RFC without CV for FS1')
print(metrics.f1_score(y_test, y_pred_fs1_model, average='weighted'))


# RFC with default parameters for FS2
rfc_model_all = RandomForestClassifier(random_state=42)
rfc_model_all.fit(X_train, np.ravel(y_train))
y_pred_all_model = rfc_model_all.predict(X_test)

print('RFC without CV for all dataset')
print(metrics.f1_score(y_test, y_pred_all_model, average='weighted'))




# Hyperparameter tuning with Random Search 

n_estimators = [10,50,100,500,1000,1500,2000]
max_features = ['auto', 'sqrt']
max_depth = [10,20,40,50,70,90,100,None]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rfc = RandomForestClassifier()
tscv = TimeSeriesSplit(n_splits=3)
rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = tscv, random_state=42, n_jobs=-1)


# Hyperparameter tunned RFC model for FS1
rf_random.fit(X_train_fs1, np.ravel(y_train, order='C'))
y_pred_fs1 = rf_random.predict(X_test_fs1)
print(metrics.f1_score(y_test, y_pred_fs1, average='weighted'))
print(metrics.f1_score(y_test, y_pred_fs1, average=None))


# Hyperparameter tunned RFC model for FS2
rfc_model = rf_random.fit(X_train, np.ravel(y_train, order='C'))
y_pred_rfc = rfc_model.predict(X_test)
print(rfc_model.best_params_)
print(metrics.f1_score(y_test, y_pred_rfc, average='weighted'))
print(metrics.f1_score(y_test, y_pred_rfc, average=None))
print(metrics.confusion_matrix(y_test, y_pred_rfc))




# Subject CV with RFC model
scores_subjectCV = {}
rfc_subj = RandomForestClassifier(n_estimators=50, min_samples_split=2, min_samples_leaf=4, max_features='sqrt', max_depth=70, bootstrap=False, random_state=42, verbose=2, n_jobs=-1)
for i in range(1,31):
  X_test1 = X_dataset[X_dataset['subject'] == i]
  X_train1 = X_dataset[X_dataset['subject'] != i]

  y_train1 = Y_dataset[Y_dataset['subject'] != i]
  y_test1 = Y_dataset[Y_dataset['subject'] == i]
  y_test1 = y_test1.drop(['subject'], axis=1)
  y_train1 = y_train1.drop(['subject'], axis=1)

  rfc_subj.fit(X_train1, np.ravel(y_train1))
  y_pred_one = rfc_subj.predict(X_test1)
  f1_measures = metrics.f1_score(y_test1, y_pred_one, average='weighted')
  scores_subjectCV[i] = f1_measures
print(scores_subjectCV)




# Creating FS1 again for cross validation
X_data_fs1 = X_dataset.loc[:, [column_name for column_name in list(X_dataset.columns) if 'mean()' in column_name or 'std()' in column_name or 'max()' in column_name or 'min()' in column_name]]

# RFC model with CV for FS1
rfc_fs2 = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True)
tscv_data = TimeSeriesSplit(10)
scores_rfc_fs2 = cross_val_score(rfc_fs2, X_data_fs1, np.ravel(Y_dataset, order='C'), cv=tscv_data, scoring='f1_weighted')
print(scores_rfc_fs2.mean()) 



# RFC model with CV for FS2
rfc_fs4 = RandomForestClassifier(random_state=42)
scores_rfc_fs4 = cross_val_score(rfc_fs4, X_dataset, np.ravel(Y_dataset, order='C'), cv=tscv_data, scoring='f1_weighted')
print(scores_rfc_fs4.mean())

