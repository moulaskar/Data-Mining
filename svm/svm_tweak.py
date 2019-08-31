import pandas as pd
import enum
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

KERNEL = ['linear','poly','rbf']

class TASK(enum.Enum):
    TASK_2 = 1
    TASK_4 = 2
    
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



def support_vector_machine_classifier(trainData, testData,task):
    train_attr, train_label, test_attr, test_label = process_and_load_data(trainData, testData)

    for i,ker in enumerate(KERNEL):
        if ker == 'linear':
            svm_classifier = SVC(kernel=ker)
        elif ker == 'poly':
            if task == TASK.TASK_2:
                svm_classifier = SVC(C=10,kernel=ker,degree = 5,gamma = 'auto',max_iter=50)
            elif task == TASK.TASK_4:
                svm_classifier = SVC(kernel=ker,degree = 7,gamma = 'auto',max_iter=50)
        else:
            svm_classifier = SVC(kernel='rbf',gamma = 'auto')
            
        
        svm_classifier.fit(train_attr,train_label)
        predict_val = svm_classifier.predict(test_attr)
        acc = accuracy_score(test_label, predict_val) * 100
        print("Kernel: ", ker)
        print("predicted value:  ",predict_val)
        print("accuracy: ", acc)
        print("\n")
        
        
        


if __name__ == "__main__":
    print("TASK 2")
    support_vector_machine_classifier('PB1_train.csv','PB1_test.csv',TASK.TASK_2)
    print("\n")
    print("TASK 4")
    support_vector_machine_classifier('PB2_train.csv','PB2_test.csv',TASK.TASK_4)
