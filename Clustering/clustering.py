# CSE572 - Assignment 3
# Spring 2019


import random

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse.linalg as linalg
from collections import Counter


# load data from CSV to a numpy array
def load_dataset(file_name):
    data_frame = pd.read_csv(file_name, delimiter=',', header=None)
    return np.array(data_frame)


# initialize centers for the farthest centers algorithm
def initialize_centers(data, k):
    dataset = np.copy(data)

    # min_distances stores the distance of the point to the closest center
    min_distances = np.empty(len(dataset))
    min_distances[:] = np.inf
    centers = []

    for i in range(0, k):
        # Select the first center randomly. Else select the maximum distance from the array min_distances
        if i == 0:
            curr_center = random.randint(0, len(dataset) - 1)
        else:
            curr_center = np.argmax(min_distances)

        centers.append(dataset[curr_center])

        # Delete the center selected from the dataset and min_distances
        dataset = np.delete(dataset, curr_center, axis=0)
        min_distances = np.delete(min_distances, curr_center, axis=0)

        # Update the min distances fom the new center. The distance will only get updated if the point is closer to the
        # current center as compared to the previous one. This saves computational time as you would not recalculate the
        # distance of all the points from all the centers.
        for idx in range(len(dataset)):
            min_distances[idx] = min(min_distances[idx], np.linalg.norm(centers[len(centers) - 1] - dataset[idx]))
    return centers


# assign points to its closest cluster center
def assign_centers(dataset, k=2):
    centers = initialize_centers(dataset, k)
    table = np.empty([len(dataset), 2], int)

    for i in range(0, table.shape[0]):
        table[i][0] = i

    for i in range(len(dataset)):
        center = 0
        least_distance = np.linalg.norm(dataset[i] - centers[0])

        for j in range(1, len(centers)):
            distance = np.linalg.norm(dataset[i] - centers[j])
            if distance < least_distance:
                least_distance = distance
                center = j

        table[i][1] = center

    return table, centers


# create k clusters for farthest centers clustering
def farthest_centers_clustering(dataset, k=2):
    min_diameter = np.inf

    for i in range(0, 1000):
        table, centers = assign_centers(dataset, k)
        largest_cluster_label = Counter(el[1] for el in table).most_common(1)[0][0]

        largest_cluster = []
        for i in range(len(table)):
            if table[i][1] == largest_cluster_label:
                largest_cluster.append(table[i])

        diameter = 0
        for i in range(len(largest_cluster)):
            for j in range(i+1, len(largest_cluster)):
                dist = np.linalg.norm(dataset[largest_cluster[i]] - dataset[largest_cluster[j]])
                if dist > diameter:
                    diameter = dist

        if diameter < min_diameter:
            min_diameter = diameter
            final_table = table
            final_centers = centers

    return final_table, final_centers


# compute the L2 distance between 2 vectors
def distance_l2(x, y):
    sum = 0
    for i in range(0, x.shape[0]):
        sum += (x[i] - y[i]) ** 2

    return sum ** 0.5


# create a table of point indices and their assigned cluster
def k_means_clustering(data, k, iterations=500, epsilon=0.0):
    # initial setup for k-means clustering
    pool = np.empty(data.shape[0], int)  # pool of indexes for random selection
    table = np.empty([data.shape[0], 2], int)  # table of row index and cluster index

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


# create a 2D matrix for the affinity scores between points
def get_affinity_score(data):
    dim = np.shape(data)
    affinity_matrix = np.zeros([dim[0], dim[0]])
    for i in range(dim[0]):
        for j in range(dim[0]):
            if i != j:
                pdst = np.linalg.norm(data[i] - data[j])
                affinity_matrix[i][j] = np.exp(- (pdst * pdst) / (2))
    return affinity_matrix


# create a diagonal matrix of the sum of all weights in each row
def compute_degree_matrix(data):
    dim = data.shape
    degree_matrix = np.zeros((dim[0], dim[0]))
    for i in range(dim[0]):
        degree_matrix[i][i] = sum(data[i])
    return degree_matrix


# create a k-NN matrix from the input matrix
def get_KNN_matrix(vec):
    knn_matrix = np.zeros(vec.shape)

    for i in range(0, knn_matrix.shape[0]):
        neighbors = np.argsort(vec[i])

        for j in range(neighbors.shape[0] - 6, neighbors.shape[0] - 1):
            idx = neighbors[j]
            knn_matrix[i][idx] = vec[i][idx]
            knn_matrix[idx][i] = vec[i][idx]

    return knn_matrix


# compute an unnormalized Laplacian matrix
def get_Lapliacian_matrix(degree_matrix, affinity_matrix):
    laplacian_matrix = degree_matrix - affinity_matrix
    return laplacian_matrix


# get the first k eigenvectors ordered by the smallest real part
def get_eigen(data, k):
    eig_values, eig_vectors = linalg.eigs(data, k, which='SR')
    return eig_vectors.real[:, :k]


# create k clusters using spectral clustering
def spectral_clustering(dataset, k):
    affinity_matrix = get_affinity_score(dataset)
    adjacency_matrix = get_KNN_matrix(affinity_matrix)
    degree_matrix = compute_degree_matrix(adjacency_matrix)
    laplacian_matrix = get_Lapliacian_matrix(degree_matrix, adjacency_matrix)
    result_vectors = get_eigen(laplacian_matrix, k)
    labels = np.transpose(k_means_clustering(result_vectors, k)[0][:, 1:])[0]
    return labels


# create a scatter plot
def create_scatter_plot(title, x, y, labels):
    colors = ['red', 'green', 'blue']
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title(title)
    plt.scatter(x, y, c=labels, cmap=clr.ListedColormap(colors))
    # plt.savefig(title + '.png')
    plt.show()


# run all 3 algorithms for the specific dataset with k clusters
def task(filename, k):
    dataset = load_dataset(filename)
    create_scatter_plot('Data scatter plot for ' + filename, dataset[:, 0], dataset[:, 1], dataset[:, -1])

    # algorithmn1
    table, centers = k_means_clustering(dataset[:, :2], k)
    create_scatter_plot('K means clustering for ' + filename, dataset[:, 0], dataset[:, 1], table[:, -1])

    # algorithmn2
    table, centers = farthest_centers_clustering(dataset[:, :2], k)
    create_scatter_plot('Farthest center clustering for ' + filename, dataset[:, 0], dataset[:, 1], table[:, -1])

    # algorithm3
    labels = spectral_clustering(dataset[:, :2], k)
    create_scatter_plot('Spectral clustering for ' + filename, dataset[:, 0], dataset[:, 1], labels)


if __name__ == "__main__":
    task("Dataset_1.csv", 2)
    task("Dataset_2.csv", 2)
    task("Dataset_3.csv", 3)
