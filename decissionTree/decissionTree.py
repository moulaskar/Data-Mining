import enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from mpl_toolkits.mplot3d import Axes3D


class TASK(enum.Enum):
    TASK_3 = 1
    TASK_4 = 2


# load data from CSV to a numpy array
def load_dataset(file_name):
    data_frame = pd.read_csv(file_name, delimiter=',', header=None)
    return np.array(data_frame)


def process_and_load_data(trainData, testData):
    columns = ['height', 'weight', 'age', 'gender']
    attr_names = ['height', 'weight', 'age']
    train_dataset = pd.read_csv(trainData, names=columns)
    test_dataset = pd.read_csv(testData, names=columns)
    # train dataset : attributes and labels
    train_attr = train_dataset[attr_names]
    train_label = train_dataset.gender

    # test_dataset, attributes and labels
    test_attr = test_dataset[attr_names]
    test_label = test_dataset.gender

    return train_attr, train_label, test_attr, test_label


def get_coordinates(theta, x):
    return np.matmul(x, theta)


def plot_lr_plane(test_data, theta, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, -1], c='g', marker='o')
    x_grid, y_grid = np.meshgrid(np.linspace(test_data[:, 0].min(), test_data[:, 0].max(), 100),
                                 np.linspace(test_data[:, 0].min(), test_data[:, 1].max(), 100))
    data_grid = np.column_stack((np.ravel(x_grid), np.ravel(y_grid), np.ones(x_grid.shape[0] ** 2)))
    z_plane = get_coordinates(theta, data_grid)
    z_grid = z_plane.reshape(x_grid.shape)
    ax.plot_surface(x_grid, y_grid, z_grid)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig(filename + ".png")
    plt.show()


def linear_regression(train_file, test_file, point):
    train_data = load_dataset(train_file)
    test_data = load_dataset(test_file)

    lr_model = LinearRegression()
    lr_model.fit(train_data[:, 0:2], train_data[:, -1])
    predictions = lr_model.predict(test_data[:, 0:2])

    mse = mean_squared_error(predictions, test_data[:, -1])
    theta = np.array([lr_model.coef_.item(0), lr_model.coef_.item(1), lr_model.intercept_])

    plot_lr_plane(test_data, theta, test_file)
    point.append(1)
    pred_point = get_coordinates(theta, point)
    # print(pred_point)

    return theta, predictions, mse, pred_point


def get_decision_tree(task):
    """
    DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
    min_impurity_split=None, class_weight=None, presort=False)

    By default DecisionTreeClassifier uses Gini Index.
    """
    if task == TASK.TASK_3:
        return DecisionTreeClassifier()
    elif task == TASK.TASK_4:
        # Decreasing the impurity by 10, gives a better accuracy
        # default give 23%
        return DecisionTreeClassifier(min_impurity_decrease=10)


def decision_tree_classifier(trainData, testData, task):
    train_attr, train_label, test_attr, test_label = process_and_load_data(trainData, testData)

    # create DecissionTreeClassifier
    dtc = get_decision_tree(task)

    # Train Decision Tree Classifier
    dtc.fit(train_attr, train_label)

    # predict the test label
    predict_val = dtc.predict(test_attr)

    # calculate accuracy
    acc = accuracy_score(test_label, predict_val) * 100
    return predict_val, acc

    
if __name__ == "__main__":
    pb1_point = [46, 53]
    pb2_point = [19, 76]
    print("TASK 1")
    params1, pred1, mse1, pred_point1 = linear_regression("PB1_train.csv", "PB1_test.csv", pb1_point)
    print("Parameters: {}\n predictions: {}\n mse: {}\n prediction for {}:{}".format(params1, pred1, mse1, pb1_point,
                                                                                  pred_point1))
    print("TASK 2")
    params2, pred2, mse2, pred_point2 = linear_regression("PB2_train.csv", "PB2_test.csv", pb2_point)
    print("Parameters: {}\n predictions: {}\n mse: {}\n prediction for {}:{}".format(params2, pred2, mse2, pb2_point,
                                                                                  pred_point2))
    print("TASK 3")
    predictions3, accuracy3 = decision_tree_classifier('PB3_train.csv', 'PB3_test.csv', TASK.TASK_3)
    print("Predictions: {}, accuracy: {}".format(predictions3, accuracy3))
    print("TASK 4")
    predictions4, accuracy4 = decision_tree_classifier('PB4_train.csv', 'PB4_test.csv', TASK.TASK_4)
    print("Predictions: {}, accuracy: {}".format(predictions4, accuracy4))
