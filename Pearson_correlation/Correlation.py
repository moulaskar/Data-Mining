# Assignment 1
# CSE 572 Data Mining
# Spring 2019

# Required libraries:
#   matplotlib
#   numpy
#   openpyxl
#   pandas
#   scipy


import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

def process_and_load_data(file_name):
    data_frame = pd.read_csv(file_name)
    data_frame['Population'] = [el.replace(',','') for el in data_frame['Population']]
    data_frame['Population'] = data_frame['Population'].astype(int)
    data_frame['Deaths'] = [el.replace(',','') for el in data_frame['Deaths']]
    data_frame['Deaths'] = data_frame['Deaths'].astype(int)
    return data_frame

#Removing the redundant columns for States and Abbrev
#and putting the Abbreviation of States in State Header

def data_cleanup(data_frame):
    #create an array and store the Abrrev list
    tempArray = np.array(data_frame['Abbrev'])
    data_frame_clean = data_frame.insert(0,"States",tempArray)
    data_frame_clean = data_frame.drop(['State','Abbrev'],1)
    return data_frame_clean

def calculate_pearson_correlation(feature_one, feature_two):
    return pearsonr(feature_one, feature_two)

def create_bar_graph(labels, values):
    values = values * 1000000
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Opiod Deaths Per Million People')
    plt.xlabel('States')
    plt.title('Opiod Death Density Per State')
    plt.savefig('ODD_bargraph.png', bbox_inches='tight')
    plt.show()

def calculate_opiod_death_density(population, deaths):
    return np.divide(deaths, population)

#Calculate the positive difference between two points in ODD values

def create_similarity_matrix(data,odd,label):
    #Euclidean distance to get the similarity matrix

    similarity_matrix = np.empty([len(odd),len(odd)])

    max_odd = max(odd)
    min_odd = min(odd)
    max_distance = distance.euclidean(max_odd,min_odd)

    for i in range(len(odd)):
        for j in range(len(odd)):
            similarity_matrix[i][j] = (max_distance - distance.euclidean(odd[i],odd[j])) / max_distance

    dfNew = pd.DataFrame(similarity_matrix,index= label, columns=label).round(4)
       
    dfNew.to_excel('task3.xlsx')            
   

if __name__ == "__main__":
    data = process_and_load_data('overdoses.csv')
    
    data = data_cleanup(data)
    correlation, p_value = calculate_pearson_correlation(data['Population'], data['Deaths'])
    print("Correlation: {} P-value: {}".format(correlation, p_value))

    odd = calculate_opiod_death_density(data['Population'], data['Deaths'])
    state_labels = np.array(data['States'])

    create_bar_graph(state_labels, odd)
    create_similarity_matrix(data, odd, state_labels)
