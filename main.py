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
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import ex as acousticGenerator

act_Data = acousticGenerator.acoustics


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
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0, random_state = 100) 
      
    return X_train, X_test, y_train, y_test 
  
      
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
    data_set = pd.read_csv(FLAGS.data)
    data_set_processed, targets = map_target(data_set, "label")
    # print(targets)
    features = list(data_set_processed.columns[:20])
    # print(features)
    print(act_Data[features])

    y = data_set_processed["label"]
    x = data_set_processed[features]
    x_train, x_test, y_train, y_test = splitdataset(x, y)

    CART_train = DecisionTreeClassifier(min_samples_split=500, random_state=99)
    CART_train.fit(x_train, y_train)
    CART_train_pred =  CART_train.predict(x_train)
    # CART_pred = CART_train.predict(x_test)
    CART_act_pred = CART_train.predict(act_Data)
    # print(CART_act_pred)

    SVM_train = svm.SVC(gamma='auto', kernel='linear')
    SVM_train.fit(x_train, y_train)
    SVM_train_pred = SVM_train.predict(x_train)
    # SVM_pred = SVM_train.predict(x_test)
    SVM_act_pred = SVM_train.predict(act_Data)
    # print(SVM_act_pred)


    RFC_train = RandomForestClassifier(n_estimators=2)
    RFC_train.fit(x_train, y_train)
    RFC_train_pred = RFC_train.predict(x_train)
    # RFC_pred = RFC_train.predict(x_test)
    RFC_act_pred = RFC_train.predict(act_Data)
    # print(RFC_act_pred)

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
    # XGB_pred = XGB_train.predict(x_test)
    XGB_act_pred = XGB_train.predict(act_Data)
    # print(XGB_act_pred)

    ADA_train =  AdaBoostClassifier(n_estimators=50, learning_rate=1)

    ADA_train.fit(x_train, y_train)
    ADA_train_pred = ADA_train.predict(x_train)
    # ADA_pred = ADA_train.predict(x_test)
    ADA_act_pred = ADA_train.predict(act_Data)


    train_data = { 'CART' : CART_train_pred, 'SVM' : SVM_train_pred, 'RFC' : RFC_train_pred, 'XGB' : XGB_train_pred, 'ADA' : ADA_train_pred}
    # test_data = { 'CART' : CART_pred, 'SVM' : SVM_pred, 'RFC' : RFC_pred, 'XGB' : XGB_pred, 'ADA' : ADA_pred }
    act_pred_data = {'CART' : CART_act_pred, 'SVM' : SVM_act_pred, 'RFC' : RFC_act_pred, 'XGB' : XGB_act_pred, 'ADA' : ADA_act_pred}
    int_train_data = pd.DataFrame(train_data)
    # int_test_data = pd.DataFrame(test_data)
    int_act_data = pd.DataFrame(act_pred_data)

    # print(int_train_data)
    # print(int_test_data)
    print(int_act_data)

    INT_train = svm.SVC(gamma='auto', kernel='linear')
    INT_train.fit(int_train_data, y_train)
    # INT_train_pred = INT_train.predict(int_train_data)
    # INT_pred = INT_train.predict(int_test_data)
    INT_act_pred = INT_train.predict(int_act_data)
    # print(INT_act_pred)
    # visualize_tree(INT_train, features, 'INT.dot')

    for data in INT_act_pred:
        if data == 1:
            print('\n\nOutput: The given voice belongs to Female')
        else:
            print('\n\nOutput: The given voice belongs to Male')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        default='./voice.csv',
        help='Location of speech training data archive on the web.')

FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)