#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL')
features = ["bonus", "long_term_incentive"]
data = featureFormat(data_dict, features)

clean_data = filter(lambda x:x[0]!=0 and x[1]!=0,data)

## PLOT
for point in clean_data:
    x = point[0]
    y = point[1]
    matplotlib.pyplot.scatter( x, y )

matplotlib.pyplot.xlabel(features[0])
matplotlib.pyplot.ylabel(features[1])
matplotlib.pyplot.show()

