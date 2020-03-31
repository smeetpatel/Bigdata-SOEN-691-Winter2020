import pandas as pd
from pandas import read_csv
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


#FilePaths of training and test dataSet
x_filepath="C:\\Users\\Sahana\\Desktop\\BigData\\HARDataset\\train\\X_train.txt"
y_filepath="C:\\Users\\Sahana\\Desktop\\BigData\\HARDataset\\train\\y_train.txt"
X_test_path="C:\\Users\\Sahana\\Desktop\\BigData\\HARDataset\\test\\X_test.txt"
y_test_path="C:\\Users\\Sahana\\Desktop\\BigData\\HARDataset\\test\\y_test.txt"


#training
X_train = pd.read_csv(x_filepath, header=None,names=['columheader'], sep='\t' )
y_train = pd.read_csv(y_filepath, header=None, delim_whitespace=True)
#print(X_train.head())

#print(y_train)
x_test=pd.read_csv(X_test_path, header=None,names = ['columnheader1'],sep ='\t')
y_test=read_csv(y_test_path, header=None, delim_whitespace=True)
#print(x_test)

h1 = open('C:/Users/Sahana/Desktop/BigData/HARDataset/features.txt','r').read().split('\n')[:-1]
header_list = [i.split()[1] for i in h1]
#print(header_list)

#print(X_train.columns[0])
X_train = X_train['columheader'].str.split("\s{1,}", expand=True)
X_train.drop(X_train.columns[0], inplace=True, axis=1)
X_train.columns = header_list


x_test= x_test['columnheader1'].str.split("\s{1,}", expand=True)
x_test.drop(x_test.columns[0], inplace=True, axis=1)
x_test.columns = header_list

X_train_fil = X_train.filter(like='-mean()', axis=1)
#print(X_train_fil.head())
x_test_fil = x_test.filter(like='-mean()', axis=1)
#print(x_test_fil.head())

rfClassifier = RandomForestClassifier(n_estimators=20, random_state=0)
rfClassifier_fit = rfClassifier.fit(X_train_fil, y_train.values.ravel())
predict_fs1=rfClassifier_fit.predict(x_test_fil)
print(predict_fs1)
acc = accuracy_score(y_test, predict_fs1)*100
print("acc:", acc)

print("FS1 with RFC")
print(metrics.f1_score(y_test, predict_fs1, average=None))
print(metrics.confusion_matrix(y_test, predict_fs1, normalize='all'))



X_train_fs2 = X_train.loc[:, [column_name for column_name in list(X_train.columns) if 'mean()' in column_name or 'std()' in column_name or 'max()' in column_name or 'min()' in column_name]]
X_test_fs2 = x_test.loc[:, [column_name for column_name in list(x_test.columns) if 'mean()' in column_name or 'std()' in column_name or 'max()' in column_name or 'min()' in column_name]]
#print(X_test_fs2)
rfClassifier_2 = RandomForestClassifier(n_estimators=20, random_state=0)
rfClassifier_fit_2 = rfClassifier_2.fit(X_train_fs2, y_train.values.ravel())
predict_fs2=rfClassifier_fit_2.predict(X_test_fs2)
#print(predict_fs2)

acc = accuracy_score(y_test, predict_fs2)*100
print("acc:", acc)
print("FS2 with RFC")
print(metrics.f1_score(y_test, predict_fs2, average=None))
print(metrics.confusion_matrix(y_test, predict_fs2, normalize='all'))


pca = PCA(.99)
pca.fit(X_train)
#print('\n')
#print(pca.n_components_)
X_train_fs3 = pca.transform(X_train)
X_fs3_test = pca.transform(x_test)


rfClassifier_3 = RandomForestClassifier(n_estimators=20, random_state=0)
rfClassifier_fit_3 = rfClassifier_3.fit(X_train_fs3, y_train.values.ravel())
predict_fs3=rfClassifier_fit_3.predict(X_fs3_test)
#print(predict_fs2)

acc = accuracy_score(y_test, predict_fs3)*100
print("acc:", acc)
print("FS3 with RFC")
print(metrics.f1_score(y_test, predict_fs3, average=None))
print(metrics.confusion_matrix(y_test, predict_fs3, normalize='all'))




