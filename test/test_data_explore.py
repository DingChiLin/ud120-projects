import sys
import pickle
import csv

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

############
# Raw Data
############

with open("../final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print(data_dict)

###########
# Prepare
###########

name_list = data_dict.keys()
print(name_list)

all_features_list = data_dict.values()[0].keys()
all_features_list.remove('email_address')

all_value_list = data_dict.values()


print(all_features_list)
all_data = featureFormat(data_dict, all_features_list, sort_keys = True)
print(all_data)

explore_data = []
explore_data.append(['name']+all_features_list)
for idx, val in enumerate(all_value_list):
    values = zip(val.keys(), val.values())
    cleaned_value = []
    for val in values:
        if val[0] != 'email_address':
            if val[0] == 'poi':
                if val[1] == True:
                    cleaned_value.append(1)
                else:
                    cleaned_value.append(0)
            else:
                if val[1] == 'NaN':
                    cleaned_value.append(0)
                else:
                    cleaned_value.append(val[1])

    explore_data.append([name_list[idx]] + cleaned_value)

with open('explore_data.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(explore_data)

###########
# Explore
###########
import pandas

# Load the data into a DataFrame
data = pandas.read_csv('explore_data.csv')
print(data.describe())
