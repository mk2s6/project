from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import subprocess
import sys

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


FLAGS = None
labels = None

def map_target(data, fieldName):
    data_modified = data.copy()
    targets = data[fieldName].unique()
    labels = targets
    mapper = {
        name : n for n,
        name in enumerate(targets)
    }
    data_modified[fieldName] = data_modified[fieldName].replace(mapper)

    return (data_modified, targets)

def visualize_tree(tree, feature_names, dotName):
    with open(dotName, 'w') as f:
        export_graphviz(tree, out_file=f, feature_names = feature_names)

    command = ["dot", "-Tpng", dotName, "-o", dotName + ".png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
        "produce visualization")

# Function to split the dataset 
def splitdataset(data, label): 
  
    # Seperating the target variable 
    X = data
    Y = label
  
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 100) 
      
    return X_train, X_test, y_train, y_test 
  
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Prediction on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, targets): 
      
    print("Confusion Matrix: \n", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : \n", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : \n", 
     classification_report(y_test, y_pred, target_names=targets))

def integrated(dataset, actual):
    print(actual)
  
def main(_):
    # print(FLAGS.data)
    data_set = pd.read_csv(FLAGS.data)
    # data_set.head()
    # print(data_set.head())
    # print(data_set["label"].unique())
    data_set_processed, targets = map_target(data_set, "label")
    # print(data_set_processed.head())
    # print(data_set_processed.tail())
    print(targets)
    features = list(data_set_processed.columns[:20])
    print(features)
    y = data_set_processed["label"]
    # print(y.head())
    x = data_set_processed[features]
    # print(x)
    x_train, x_test, y_train, y_test = splitdataset(x, y)
    # print("Training values")
    # print(x_train)
    # print("Training labels")
    # print(y_train)
    # print("Testing values")
    # print(x_test)
    # print("Testing lables")
    # print(y_test)

    print("\n\n CART\n\n")
    CART_train = DecisionTreeClassifier(min_samples_split=500, random_state=99)
    CART_train.fit(x_train, y_train)
    CART_pred = CART_train.predict(x_test)
    # print(CART_pred)
    CART_train_pred =  CART_train.predict(x_train)
    # visualize_tree(CART_train, features, 'CART.dot')
    print("Training Measures")
    cal_accuracy(y_train, CART_train_pred, targets)
    print("Testing Measures")
    cal_accuracy(y_test, CART_pred, targets)

    print('\n\n SVM \n\n')
    SVM_train = svm.SVC(gamma='auto')
    SVM_train.fit(x_train, y_train)
    SVM_train_pred = SVM_train.predict(x_train)
    SVM_pred = SVM_train.predict(x_test)
    # print(SVM_pred)
    # visualize_tree(SVM_train, features, 'SVM.dot')
    print("Training Measures")
    cal_accuracy( y_train , SVM_train_pred, targets)
    print("Testing Measures")
    cal_accuracy(y_test, SVM_pred, targets)


    print('\n\n RFC \n\n')
    RFC_train = RandomForestClassifier(n_estimators=10)
    RFC_train.fit(x_train, y_train)
    RFC_train_pred = RFC_train.predict(x_train)
    RFC_pred = RFC_train.predict(x_test)
    # print(RFC_pred)
    # visualize_tree(RFC_train, features, 'RFC.dot')
    print("Training Measures")
    cal_accuracy( y_train , RFC_train_pred, targets)
    print("Testing Measures")
    cal_accuracy(y_test, RFC_pred, targets)


    print('\n\n XGB \n\n')
    original_params = {'n_estimators': 1000, 'max_leaf_nodes': 2, 'max_depth': None, 'random_state': 2,
                    'min_samples_split': 5}

    for label, color, setting in [(1, 'orange',
                                {'learning_rate': 0.01, 'subsample': 1.0}),
                                (0, 'turquoise',
                                {'learning_rate': 0.01, 'subsample': 1.0})]:
        params = dict(original_params)
        params.update(setting)
    XGB_train = GradientBoostingClassifier(**params)
    XGB_train = XGB_train.fit(x_train, y_train)
    XGB_train_pred = XGB_train.predict(x_train)
    XGB_pred = XGB_train.predict(x_test)
    # print(XGB_pred)
    print("Training Measures")
    cal_accuracy( y_train , XGB_train_pred, targets)
    print("Testing Measures")
    cal_accuracy(y_test, XGB_pred, targets)

    print('\n\n ADA \n\n')
    ADA_train =  AdaBoostClassifier(n_estimators=50, learning_rate=1)

    ADA_train.fit(x_train, y_train)
    ADA_train_pred = ADA_train.predict(x_train)
    ADA_pred = ADA_train.predict(x_test)
    # print(ADA_pred)
    print("Training Measures")
    cal_accuracy( y_train , ADA_train_pred, targets)
    print("Testing Measures")
    cal_accuracy(y_test, ADA_pred, targets)

    int_train = [ CART_train_pred, SVM_train_pred, RFC_train_pred, XGB_train_pred, ADA_train_pred, y_train ]
    int_test = [ CART_pred, SVM_pred, RFC_pred, XGB_pred, ADA_pred, y_test ]
    int_data_columns = ['CART', 'SVM', 'RFC', 'XGB', 'ADA', 'label']

    train_data = { 'CART' : CART_train_pred, 'SVM' : SVM_train_pred, 'RFC' : RFC_train_pred, 'XGB' : XGB_train_pred, 'ADA' : ADA_train_pred}
    test_data = { 'CART' : CART_pred, 'SVM' : SVM_pred, 'RFC' : RFC_pred, 'XGB' : XGB_pred, 'ADA' : ADA_pred }
    int_train_data = pd.DataFrame(train_data)
    int_test_data = pd.DataFrame(test_data)

    # print(int_train_data)
    # print(int_test_data)

    
    print('\n\n INT \n\n')
    INT_train = RandomForestClassifier(n_estimators=10)
    INT_train.fit(int_train_data, y_train)
    INT_train_pred = INT_train.predict(int_train_data)
    INT_pred = INT_train.predict(int_test_data)
    # print(INT_pred)
    # visualize_tree(INT_train, features, 'INT.dot')
    print("Training Measures")
    cal_accuracy( y_train , INT_train_pred, targets)
    print("Testing Measures")
    cal_accuracy(y_test, INT_pred, targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        default='./voice.csv',
        help='Location of speech training data archive on the web.')

FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)