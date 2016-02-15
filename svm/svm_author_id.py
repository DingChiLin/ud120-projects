#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
#sys.path.append("../choose_your_own/")
#from class_vis import prettyPicture
#import matplotlib.pyplot as plt

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

#########################################################

#speed up!
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=10000)
#clf = SVC(kernel='linear')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print(clf.score(features_test, labels_test))
print(len(filter(lambda x:x==1, pred)))

#prettyPicture(clf, features_test, labels_test)
#plt.show()
