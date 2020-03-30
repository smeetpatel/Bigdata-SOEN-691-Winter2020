import pandas as pd
from pandas import read_csv
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

x_filepath="C:\\Users\\Sahana\\Desktop\\BigData\\HARDataset\\train\\X_train.txt"
y_filepath="C:\\Users\\Sahana\\Desktop\\BigData\\HARDataset\\train\\y_train.txt"

X_test_path="C:\\Users\\Sahana\\Desktop\\BigData\\HARDataset\\test\\X_test.txt"
y_test_path="C:\\Users\\Sahana\\Desktop\\BigData\\HARDataset\\test\\y_test.txt"


X_train = pd.read_csv(x_filepath, header=None,names=['columheader'], sep='\t' )


y_train = pd.read_csv(y_filepath, header=None, delim_whitespace=True)
#print(X_train.head())

#print(y_train)
x_test=pd.read_csv(X_test_path, header=None,names = ['columnheader1'],sep ='\t')
y_test=read_csv(y_test_path, header=None, delim_whitespace=True)
#print(x_test)
'''
rfClassifier = RandomForestClassifier(n_estimators=20, random_state=0)
rfClassifier.fit(X_train, y_train.values.ravel())
y_pred = rfClassifier.predict(x_test)
acc=accuracy_score(y_test, y_pred)*100
#print("acc:", acc)
rec=recall_score(y_test, y_pred,average='weighted')*100
#print("rec:", rec)

pre=precision_score(y_test, y_pred, average='weighted')*100
#print("pre:", pre)

'''


df1_mean=X_train.filter([1, 2, 3, 41, 42, 43, 81, 82, 83, 121, 122, 123, 161, 162, 163, 201, 214, 227, 240, 253, 266, 267, 268, 345, 346, 347, 424, 425, 426, 503, 516, 529, 542])
#print(df1_mean)



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

dfColm_path="C:\\Users\\Sahana\\Desktop\\BigData\\HARDataset\\features.txt"
af_coln = pd.read_csv(dfColm_path, header=None, delim_whitespace=True)
value1 = af_coln.filter([1]).transpose()


#df=pd.merge(value1, X_train)
#print(type(value1))
#print(value1)


#print(X_train)
#print(value1)
