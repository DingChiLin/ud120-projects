#!/usr/bin/pytadd_feature_by_scalar_and_pcahon

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from helper import *

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary','to_messages','deferral_payments','total_payments',\
                 #'exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi',\
                 #'restricted_stock_deferred','total_stock_value','expenses','loan_advances',\
                 #'from_messages','other','from_this_person_to_poi','director_fees',\
                 #'deferred_income','long_term_incentive','from_poi_to_this_person']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 1: Remove outliers : only 122 records remain after doing this

# remove TOTAL : 1
del data_dict['TOTAL']

financial_values = data_dict.values()
names = data_dict.keys()
remove_name_list = []
for idx, financial_value in enumerate(financial_values):
    values = financial_value.values()
    nan_count = 0
    for value in values:
        if(value == 'NaN'):
            nan_count += 1

    if(nan_count >= 16 and not financial_value['poi']):
        remove_name_list.append(names[idx])

# remove data with too many 'NaN's : 23
for name in remove_name_list:
    del data_dict[name]

### Task 2: Create new feature(s) : Scaler all data and create two new features by PCA
from sklearn.preprocessing import StandardScaler

#Transform by scaler
scaler_data_dict = transform_by_scaler(data_dict, StandardScaler())

#Add New Feature by PCA
scaler_pca_feature_data_dict = add_feature_by_pca(scaler_data_dict)

### Store to my_dataset for easy export below.
my_dataset = scaler_pca_feature_data_dict


### Task 3: Select my feature: Using selectKBest to find the three best features
from sklearn.feature_selection import SelectKBest, f_classif

all_features_list = my_dataset.values()[0].keys()
all_features_list.remove('poi') #poi will be label, not feature
all_features_list.remove('email_address') #email is not a numerical feature

all_data = featureFormat(my_dataset, ['poi']+all_features_list, sort_keys = False)
all_labels, all_features = targetFeatureSplit(all_data)

sel = SelectKBest(f_classif, k=3)
sel.fit(all_features, all_labels)

#Find The Three Best Component: exercised_stock_options, total_stock_value, pca_component1
features_list = ['poi'] + map(lambda x:x[1], \
                          sorted(zip(sel.scores_, all_features_list), key= lambda x:x[0])[-3:])

print(features_list)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
print(len(labels))

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, recall_score, make_scorer
recall_scorer = make_scorer(recall_score)

from sklearn.naive_bayes import GaussianNB
param_grid1 = {} # GaussianNB have no parameter
clf1 = GridSearchCV(GaussianNB(), param_grid1, scoring=recall_scorer)

from sklearn.svm import SVC
param_grid2 = {
          'kernel' : ['rbf', 'poly'],
          'C': [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }

clf2 = GridSearchCV(SVC(), param_grid2, scoring=recall_scorer)

from sklearn.ensemble import RandomForestClassifier
param_grid3 = {
          'min_samples_split' : [2,5,8,10],
          'max_depth': [1,2,3,None],
          }

clf3 = GridSearchCV(RandomForestClassifier(), param_grid3, scoring=recall_scorer)

from sklearn.ensemble import AdaBoostClassifier
param_grid4 = {
          'n_estimators' : [10, 30, 50, 70, 100],
          'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
          }

clf4 = GridSearchCV(AdaBoostClassifier(), param_grid4, scoring=recall_scorer)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

for clf in [clf1, clf2, clf3, clf4]:

    clf.fit(features_train, labels_train)
    print(clf.best_estimator_.score(features_test, labels_test))
    labels_pred = clf.best_estimator_.predict(features_test)

    print(clf.best_params_)

    ######################
    #            Pred    #
    #           0    1   #
    #        0  TN   FP  #
    # ACTUAL             #
    #        1  FN   TP  #
    ######################
    print(confusion_matrix(labels_test, labels_pred))

    ###########################
    # Recall     = TP/(TP+FN) #
    # Precession = TP/(TP+FP) #
    ###########################
    print(classification_report(labels_test, labels_pred))





### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

