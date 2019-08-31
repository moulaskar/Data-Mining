import random

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
import matplotlib.pyplot as plt


def process_and_load_data(file_name):
    data_frame = pd.read_csv(file_name)
    data_frame['Population'] = [el.replace(',', '') for el in data_frame['Population']]
    data_frame['Population'] = data_frame['Population'].astype(int)
    data_frame['Deaths'] = [el.replace(',', '') for el in data_frame['Deaths']]
    data_frame['Deaths'] = data_frame['Deaths'].astype(int)

    extracted_data = np.empty([2, 50], int)
    extracted_data[0] = data_frame['Population']
    extracted_data[1] = data_frame['Deaths']
    extracted_data = np.transpose(extracted_data)

    data_labels = data_frame['Abbrev']
    return extracted_data, data_labels


def distance_l2(x, y):
    sum = 0
    for i in range(0, x.shape[0]):
       sum += (x[i] - y[i])**2

    return sum**0.5 


def k_means_clustering(data, k, iterations=500, epsilon=0.0):
    # initial setup for k-means clustering
    pool = np.empty([50], int)  # pool of indexes for random selection
    table = np.empty([50, 2], int)  # table of row index and cluster index

    for i in range(0, pool.shape[0]):
        table[i][0] = i
        pool[i] = i

    centers = np.zeros([k, data.shape[1]])

    # randomly select cluster centers from data points
    for i in range(0, centers.shape[0]):
        index = random.randint(0, pool.shape[0] - 1)
        row = pool[index]
        for j in range(0, centers.shape[1]):
            centers[i][j] = data[row][j]
        table[row][1] = i
        pool = np.delete(pool, index)

    for _ in range(0, iterations):
        # assign all data points to its closest center
        for i in range(0, table.shape[0]):
            row = table[i][0]

            center = 0
            least_distance = distance_l2(data[row], centers[0])

            for j in range(1, centers.shape[0]):
                distance = distance_l2(data[row], centers[j])
                if distance < least_distance:
                    least_distance = distance
                    center = j

            table[i][1] = center

        # find the new centers
        # data.shape[1] + 1 adds 1 for the count of elements in the cluster
        sums = np.zeros([centers.shape[0], data.shape[1] + 1])
        for i in range(0, table.shape[0]):
            row = table[i][0]
            cluster = table[i][1]
            for j in range(0, data.shape[1]):
                sums[cluster][j] += data[row][j]
            sums[cluster][data.shape[1]] += 1

        max_change = 0
        for i in range(0, centers.shape[0]):
            new_center = np.empty(centers.shape[1])
            for j in range(0, centers.shape[1]):
                new_center[j] = sums[i][j] / sums[i][centers.shape[1]]
            
            difference = distance_l2(new_center, centers[i])

            if (difference > max_change):
                max_change = difference

            centers[i] = new_center

        if max_change == 0:
            break

    return table, centers


def calculate_objective(data, table, centers):
    objective = 0
    for i in range(len(data)):
        objective = objective + (distance_l2(data[i], centers[table[i][1]]))**2
    return objective


def calculate_costs(data):
    costs = []
    for i in range(2, 16):
        table, centers = k_means_clustering(data, i)
        costs.append(calculate_objective(data, table, centers))
    return costs


def plot_k_vs_cost(k, costs, filename):
    plt.title("Cost vs Cluster Size")
    plt.xlabel("Cluster Size")
    plt.ylabel("Objective Function Cost")
    plt.plot(k, costs, marker='o')
    plt.show()
    plt.savefig(filename)


if __name__ == "__main__":
    costs = []
    data, labels = process_and_load_data("overdoses.csv")

    costs = calculate_costs(data)

    plot_k_vs_cost([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], costs, "task1_cost_vs_k.png")

    sim_matrix = cosine_similarity(data)
    costs2 = calculate_costs(sim_matrix)

    plot_k_vs_cost([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], costs2, "task2_cost_vs_k.png")

    table, centers = k_means_clustering(data, 5)
    np.savetxt("task1_data.csv", data, delimiter=",")
    np.savetxt("task1_k5_table.csv", table, delimiter=",")
    np.savetxt("task2_sim_matrix.csv", sim_matrix, delimiter=",")
