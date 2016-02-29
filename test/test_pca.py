#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL')
features = ["bonus", 'long_term_incentive']
data = featureFormat(data_dict, features)

clean_data = filter(lambda x:x[0]!=0 and x[1]!=0,data)

# PLOT
#for point in clean_data:
    #x = point[0]
    #y = point[1]
    #matplotlib.pyplot.scatter( x, y )

#matplotlib.pyplot.xlabel(features[0])
#matplotlib.pyplot.ylabel(features[1])
#matplotlib.pyplot.show()


### PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(data)
print('explained_variance_ratio')
print(pca.explained_variance_ratio_)

first_pc = pca.components_[0]
second_pc = pca.components_[1]
#third_pc = pca.components_[2]
print('components')
print(first_pc)
print(second_pc)
#print(third_pc)

transformed_data = pca.transform(data)

mean_value = np.mean(clean_data, axis=0)

for c_point, t_point in zip(clean_data, transformed_data):
    t_x1 = t_point[0] * first_pc[0] + mean_value[0]
    t_y1 = t_point[0] * first_pc[1] + mean_value[1]
    t_x2 = t_point[1] * second_pc[0] + mean_value[0]
    t_y2 = t_point[1] * second_pc[1] + mean_value[1]

    matplotlib.pyplot.scatter( c_point[0], c_point[1])
    matplotlib.pyplot.scatter( t_x1, t_y1, color='g' )
    matplotlib.pyplot.scatter( t_x2, t_y2, color='r' )


matplotlib.pyplot.xlabel(features[0])
matplotlib.pyplot.ylabel(features[1])
matplotlib.pyplot.show()


for t_point in transformed_data:
    matplotlib.pyplot.scatter( t_point[0], t_point[1])

matplotlib.pyplot.xlabel('transformed_' + features[0])
matplotlib.pyplot.ylabel('transformed_' + features[1])
matplotlib.pyplot.show()


