import math
import csv
import numpy as np
import pandas as pd
from sklearn import metrics
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

column_names = ['tripid', 'additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
                'meter_waiting_till_pickup', 'pickup_time', 'drop_time', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon',
                'fare', 'label']


def compute_column(csv_file):
    with open(csv_file, newline='') as f:
        with open('refined_train.csv', 'w', newline='') as f2:
            writer = csv.writer(f2)
            rows = csv.reader(f)
            r = 6373.0
            for row in rows:
                lat1 = math.radians(float(row[8]))
                lat2 = math.radians(float(row[10]))
                delta_lat = abs(lat1 - lat2)
                delta_lon = abs(math.radians(float(row[9])) - math.radians(float(row[11])))
                a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                distance = r * c

                pickup_time = datetime.strptime(row[6], '%m/%d/%Y %H:%M')
                drop_time = datetime.strptime(row[7], '%m/%d/%Y %H:%M')
                delta_time = abs((drop_time - pickup_time).seconds)

                writer.writerow(row[1:6] + row[8:] + [delta_lat] + [delta_lon] + [distance] + [delta_time])


# compute_column('train.csv')
# loading dataset
training_data = pd.read_csv('refined_train.csv')
training_data = training_data.dropna()
# print(training_data.dtypes)
testing_data = pd.read_csv('test.csv')

# split training dataset into feature and target variables
feature_columns = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
                   'meter_waiting_till_pickup', 'pick_lat', 'pick_lon', 'drop_lat',
                   'drop_lon', 'fare', 'delta_lat', 'delta_lon', 'distance', 'delta_time']

x_train = training_data[feature_columns]

labelToBinary = {'correct': 1, 'incorrect': 0}
training_data.label = [labelToBinary[item] for item in training_data.label]
y_train = training_data.label

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

# create classifier
clf = XGBClassifier(learning_rate=0.04,
                    n_estimators=1725,
                    max_depth=12,
                    subsample=0.8,
                    colsample_bytree=1,
                    gamma=1)

# train classifier
clf = clf.fit(x_train, y_train)

# classify the ride fare
y_predict = clf.predict(x_test)

print('Accu:', metrics.accuracy_score(y_test, y_predict))
