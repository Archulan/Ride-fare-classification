import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

column_names = ['tripid', 'additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
                'meter_waiting_till_pickup', 'pickup_time', 'drop_time', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon',
                'fare', 'label']

# load dataset
training_data = pd.read_csv('train.csv')
testing_data = pd.read_csv('test.csv')

print(training_data.head())

# split training dataset into feature and target variables
feature_columns = ['tripid', 'additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
                'meter_waiting_till_pickup', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon',
                'fare']

x_train = training_data[feature_columns]
y_train = training_data.label

x_test = testing_data[feature_columns]

# create decision tree classifier
dtc = DecisionTreeClassifier()

# train decision tree classifier
dtc = dtc.fit(x_train, y_train)
print(testing_data.dtypes)

