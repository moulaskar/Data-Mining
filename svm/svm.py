# CSE572
# Assignment 5
# Aditi Baraskar, James Smith, Moumita Laraskar, Tejas Ruikar
# Spring 2019

import pandas as pd
import enum
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")

KERNEL = ['linear','poly','rbf']

# for determining parameters for SVM
class TASK(enum.Enum):
    TASK_2 = 1
    TASK_4 = 2


# load the training and test data
# @param    trainData   file name for training data
# @param    testData    file name for testing data
def process_and_load_data(trainData, testData):
    columns = ['height', 'age', 'weight', 'gender']
    attr_names = ['height', 'age', 'weight']
    train_dataset = pd.read_csv(trainData, names=columns)
    test_dataset = pd.read_csv(testData, names=columns)
    
    # train dataset : attributes and labels
    train_attr = train_dataset[attr_names]
    train_label = train_dataset.gender

    # test_dataset, attributes and labels
    test_attr = test_dataset[attr_names]
    test_label = test_dataset.gender

    return train_attr, train_label, test_attr, test_label


# train a naive bayes classifier given the training data and evaluate on the test data
# @param    trainData   file name for training data
# @param    testData    file name for testing data
def naive_bayes_classifier(trainData, testData):
    train_attr, train_label, test_attr, test_label = process_and_load_data(trainData, testData)

    features = zip(train_attr['height'], train_attr['age'], train_attr['weight'])
    model = GaussianNB()
    model.fit(train_attr, train_label)
    predicted = model.predict(test_attr)
    acc = accuracy_score(test_label, predicted) * 100
    print("accuracy: ", acc)
    print(predicted)
    

# train a support vector machine given the training data and evaluate on the test data
# @param    trainData   file name for training data
# @param    testData    file name for testing data
# @param    task        the specific task to configure the SVM, [TASK.TASK_2, TASK.TASK_4]
def support_vector_machine_classifier(trainData, testData,task):
    train_attr, train_label, test_attr, test_label = process_and_load_data(trainData, testData)

    for i,ker in enumerate(KERNEL):
        if ker == 'linear':
            svm_classifier = SVC(kernel=ker)
        elif ker == 'poly':
            if task == TASK.TASK_2:
                # with gamma = 'scale' and C is default, accuracy = 35%
                # with gamma='auto',max_iter=50, accuracy = 70%, but there is a warning
                # "ConvergenceWarning: Solver terminated early (max_iter=50)", Currenly warnings are disabled.
                
                svm_classifier = SVC(kernel=ker,degree = 5,gamma = 'auto',max_iter=50)
                
            elif task == TASK.TASK_4:
                #with gamma = 'scale' and C is default, accuracy = 36.66%
                #with gamma='auto',max_iter=50, accuracy = 60%, but there is a warning
                #"ConvergenceWarning: Solver terminated early (max_iter=50)", Currenly warnings are disabled.
                
                svm_classifier = SVC(kernel=ker,degree = 7,gamma = 'auto',max_iter=50)
                
        else:
            #with gamma = 'scale', Task2 accuracy =43.33% and Task4 accuracy =36.66%
            #with gamma = 'auto', Task2 accuracy =100% and Task4 accuracy =36.66%
            
            svm_classifier = SVC(kernel='rbf',gamma = 'auto')
            
        
        svm_classifier.fit(train_attr,train_label)
        predict_val = svm_classifier.predict(test_attr)
        acc = accuracy_score(test_label, predict_val) * 100
        print("Kernel: ", ker)
        print("predicted value:  ",predict_val)
        print("accuracy: ", acc, "%")
        print("\n")


if __name__ == "__main__":
    print("Task 1")
    naive_bayes_classifier('PB1_train.csv','PB1_test.csv')
    print("\n")
    
    print("TASK 2")
    support_vector_machine_classifier('PB1_train.csv','PB1_test.csv',TASK.TASK_2)
    print("\n")
    
    print("Task 3")
    naive_bayes_classifier('PB2_train.csv','PB2_test.csv')
    print("\n")
    
    print("TASK 4")
    support_vector_machine_classifier('PB2_train.csv','PB2_test.csv',TASK.TASK_4)
