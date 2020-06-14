import csv
import math
import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier


def preprocess_train(csv_file):
    with open(csv_file, newline='') as f:
        with open('refined_train.csv', 'w', newline='') as f2:
            writer = csv.writer(f2)
            rows = csv.reader(f)
            r = 6373.0
            count = 0
            for row in rows:
                if count == 0:
                    writer.writerow(row[1:6] + row[8:] + ['trip_day'] + ['distance'] + ['travel_time'] + ['travel_hour'] + ['trip_fare'])
                elif count > 0:
                    lat1 = math.radians(float(row[8]))
                    lat2 = math.radians(float(row[10]))
                    delta_lat = abs(lat1 - lat2)
                    delta_lon = abs(math.radians(float(row[9])) - math.radians(float(row[11])))
                    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                    distance = r * c

                    if row[2] == '':
                        duration = 0
                    else:
                        duration = float(row[2])
                    if row[3] == '':
                        meter_waiting = 0
                    else:
                        meter_waiting = float(row[3])
                    if row[1] == '':
                        additional_fare = 0
                    else:
                        additional_fare = float(row[1])

                    travel_time = duration - meter_waiting - additional_fare

                    pickup_time = datetime.strptime(row[6], '%m/%d/%Y %H:%M')  # drop day also available
                    trip_day = pickup_time.strftime('%w')

                    travel_hour = pickup_time.strftime('%H')

                    if row[12] == '':
                        fare = 0
                    else:
                        fare = float(row[12])
                    if row[4] == '':
                        waiting_fare = 0
                    else:
                        waiting_fare = float(row[4])

                    trip_fare = fare - waiting_fare

                    writer.writerow(row[1:6] + row[8:] + [trip_day] + [distance] + [travel_time] + [travel_hour] + [trip_fare])
                count += 1

def preprocess_test(csv_file):
    with open(csv_file, newline='') as f:
        with open('refined_test.csv', 'w', newline='') as f2:
            writer = csv.writer(f2)
            rows = csv.reader(f)
            r = 6373.0
            count = 0
            for row in rows:
                if count == 0:
                    writer.writerow(row[:6] + row[8:] + ['trip_day'] + ['distance'] + ['travel_time'] + ['travel_hour'] +['trip_fare'])
                elif count > 0:
                    lat1 = math.radians(float(row[8]))
                    lat2 = math.radians(float(row[10]))
                    delta_lat = abs(lat1 - lat2)
                    delta_lon = abs(math.radians(float(row[9])) - math.radians(float(row[11])))
                    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                    distance = r * c

                    if row[2] == '':
                        duration = 0
                    else:
                        duration = float(row[2])
                    if row[3] == '':
                        meter_waiting = 0
                    else:
                        meter_waiting = float(row[3])
                    if row[1] == '':
                        additional_fare = 0
                    else:
                        additional_fare = float(row[1])

                    travel_time = duration - meter_waiting - additional_fare

                    pickup_time = datetime.strptime(row[6], '%m/%d/%Y %H:%M')  # drop day also available
                    trip_day = pickup_time.strftime('%w')

                    travel_hour = pickup_time.strftime('%H')

                    if row[12] == '':
                        fare = 0
                    else:
                        fare = float(row[12])
                    if row[4] == '':
                        waiting_fare = 0
                    else:
                        waiting_fare = float(row[4])

                    trip_fare = fare - waiting_fare

                    writer.writerow(row[:6] + row[8:] + [trip_day] + [distance] + [travel_time] + [travel_hour] + [trip_fare])
                count += 1


preprocess_train('train.csv')
preprocess_test('test.csv')
# loading dataset
dataset = pd.read_csv('refined_train.csv')
training_data = pd.DataFrame(dataset).fillna(dataset.mean())
testing_data = pd.read_csv('refined_test.csv')

# split training dataset into feature and target variables
training_feature_columns = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
                            'meter_waiting_till_pickup', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon',
                            'fare', 'trip_day', 'distance', 'travel_time', 'travel_hour', 'trip_fare']

testing_feature_columns = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
                           'meter_waiting_till_pickup', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon',
                           'fare', 'trip_day', 'distance', 'travel_time', 'travel_hour', 'trip_fare']

x_train = training_data[training_feature_columns]

labelToBinary = {'correct': 1, 'incorrect': 0}
training_data.label = [labelToBinary[item] for item in training_data.label]
y_train = training_data.label

x_test = testing_data[testing_feature_columns]

# create classifier
# clf = XGBClassifier(learning_rate=0.04,
#                     n_estimators=1725,
#                     max_depth=12,
#                     subsample=0.8,
#                     colsample_bytree=1,
#                     gamma=1, base_score=0.5)
clf = XGBClassifier(booster='gbtree', learning_rate=0.25000008, gamma=0, max_depth=25,
                    min_child_weight=0, max_delta_step=0, subsample=1, colsample_bytree=1,
                    colsample_bylevel=1, colsample_bynode=1, reg_lambda=2.0005)

# train classifier
clf = clf.fit(x_train, y_train)

# classify the ride fare
y_predict = clf.predict(x_test)

df = pd.DataFrame(y_predict, columns=['prediction'], index=testing_data['tripid'])
df.index.name = 'tripid'

df.to_csv('160040d_submission_21')
