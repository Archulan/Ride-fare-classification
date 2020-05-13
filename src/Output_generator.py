import pandas as pd
import csv
from xgboost import XGBClassifier

def compute_column(csv_file):
    with open(csv_file, newline='') as f:
        with open('refined_test.csv', 'w', newline='') as f2:
            writer = csv.writer(f2)
            rows = csv.reader(f)
            for row in rows:
                delta_lat = [abs(float(row[8]) - float(row[10]))]
                delta_lon = [abs(float(row[9]) - float(row[11]))]
                writer.writerow(row + delta_lat + delta_lon)

column_names = ['tripid', 'additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
                'meter_waiting_till_pickup', 'pickup_time', 'drop_time', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon',
                'fare', 'label']

# compute_column('test.csv')
# loading dataset
training_data = pd.read_csv('refined_train.csv')
training_data = training_data.dropna()
testing_data = pd.read_csv('refined_test.csv')

# split training dataset into feature and target variables
training_feature_columns = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
                'meter_waiting_till_pickup', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon',
                'fare', 'delta_lat', 'delta_lon']

testing_feature_columns = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
                'meter_waiting_till_pickup', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon',
                'fare', 'delta_lat', 'delta_lon']

x_train = training_data[training_feature_columns]

labelToBinary = {'correct': 1, 'incorrect': 0}
training_data.label = [labelToBinary[item] for item in training_data.label]
y_train = training_data.label

x_test = testing_data[testing_feature_columns]

# create classifier
clf = XGBClassifier(learning_rate=0.25, min_child_weight=10, gamma=2, seed=1)

# train classifier
clf = clf.fit(x_train, y_train)

# classify the ride fare
y_predict = clf.predict(x_test)

df = pd.DataFrame(y_predict, columns=['prediction'], index=testing_data['tripid'])
df.index.name = 'tripid'

df.to_csv('160040d_submission_3')
