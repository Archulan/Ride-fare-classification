import pandas as pd
from xgboost import XGBClassifier

column_names = ['tripid', 'additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
                'meter_waiting_till_pickup', 'pickup_time', 'drop_time', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon',
                'fare', 'label']

# loading dataset
training_data = pd.read_csv('train.csv')
training_data = training_data.dropna()
testing_data = pd.read_csv('test.csv')

# split training dataset into feature and target variables
feature_columns = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
                'meter_waiting_till_pickup', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon',
                'fare']

x_train = training_data[feature_columns]

labelToBinary = {'correct': 1, 'incorrect': 0}
training_data.label = [labelToBinary[item] for item in training_data.label]
y_train = training_data.label

# x_test = testing_data[feature_columns]

# create classifier
clf = XGBClassifier(learning_rate=0.25, min_child_weight=10, gamma=2, seed=1)

# train classifier
clf = clf.fit(x_train, y_train)

# classify the ride fare
y_predict = clf.predict(x_test)


df = pd.DataFrame(y_predict, columns=['prediction'], index=testing_data['tripid'])
df.index.name = 'tripid'

df.to_csv('output/160040d_submission_1')
