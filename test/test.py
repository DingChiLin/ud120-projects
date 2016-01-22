import sys
from time import time
import numpy

sys.path.append("../choose_your_own/")
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
import matplotlib.pyplot as plt

features_train, labels_train, features_test, labels_test = makeTerrainData()
from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=100, gamma=1)
clf.fit(features_train, labels_train)

print(clf.score(features_test, labels_test))

prettyPicture(clf, features_test, labels_test)
plt.show()
