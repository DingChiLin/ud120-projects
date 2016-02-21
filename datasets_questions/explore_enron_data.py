#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

import sys
sys.path.append("../tools/")
import feature_format as ff
sys.path.append("../final_project/")
import poi_email_addresses as pea

def printData(data):
    for key, value in data.iteritems():
        print('---------')
        print(key)
        print(value)
        print(len(value))

    print(len(data))

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print(len(enron_data))

poi_data = {k:v for (k,v) in enron_data.iteritems() if v['poi']}
print(len(poi_data))

print(enron_data['SKILLING JEFFREY K']['total_payments'])
print(enron_data['LAY KENNETH L']['total_payments'])
print(enron_data['FASTOW ANDREW S']['total_payments'])

salary_data = {k:v for (k,v) in enron_data.iteritems() if v['salary'] != 'NaN'}
print(len(salary_data))

email_data = {k:v for (k,v) in enron_data.iteritems() if v['email_address'] != 'NaN'}
print(len(email_data))

no_total_payment_data = {k:v for (k,v) in enron_data.iteritems() if v['total_payments'] == 'NaN'}
print(len(no_total_payment_data))
print(len(no_total_payment_data) / float(len(enron_data)))

##########

#features = ['poi', 'salary', 'from_this_person_to_poi']
#enron_list = ff.featureFormat(enron_data, features, remove_any_zeroes=True)
#print(enron_list)
#print(len(enron_list))

###########

#print(pea.poiEmails())


