import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
import numpy as np

column_names = ['tripid', 'additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
                'meter_waiting_till_pickup', 'pickup_time', 'drop_time', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon',
                'fare', 'label']

# load dataset
training_data = pd.read_csv('train.csv')
training_data = training_data.dropna()
testing_data = pd.read_csv('test.csv')

print(training_data.head())

# split training dataset into feature and target variables
feature_columns = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
                'meter_waiting_till_pickup', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon',
                'fare']

x_train = training_data[feature_columns]
y_train = training_data.label

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
# x_test = testing_data[feature_columns]

# shuffle_index = np.random.permutation(13574)
# print(len(x_train), 'length', len(y_train))
# x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

# create classifier
clf = KNeighborsClassifier(n_neighbors=15)

# train classifier
clf = clf.fit(x_train, y_train)

# classify the ride fare
y_predict = clf.predict(x_test)

print('Accu:', metrics.accuracy_score(y_test, y_predict))
