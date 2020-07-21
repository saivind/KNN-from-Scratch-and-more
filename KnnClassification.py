#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(1,len(row1)):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a classification prediction with neighbors
def knn_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# Importing data     
mc = pd.read_csv("C:/Users/saivi/Documents/Machine Learning/Assignment 1/A1_datasets/A1_datasets/microchips.csv", header = None)

#Adjusting the data into an array
dataset = np.asarray(mc)

# Records for prediction
row = [[-0.3,1.0],[-0.5,-0.1],[0.6,0.0]]

# predict the chipset knn-classification
num_neighbors = 1
for num_neighbors in range(1,8,2):
    print('K =',num_neighbors)
    for i in range(len(row)):
        label = knn_classification(dataset, row[i], num_neighbors)
        if label == 1:
            print('chip=%s, Predicted: %s ==> OK' % (row[i], label))
        else:
            print('Data=%s, Predicted: %s ==> Fail' % (row[i], label))
    
    

    
    


# In[15]:


#K=3
num_neighbors = 3
print('K=3')
for i in range(len(row)):
    label = knn_classification(dataset, row[i], num_neighbors)
    if label == 1:
        print('Data=%s, Predicted: %s ==> OK' % (row[i], label))
    else:
        print('Data=%s, Predicted: %s ==> Fail' % (row[i], label))
    
#K=5
num_neighbors = 5
print('K=5')
for i in range(len(row)):
    label = knn_classification(dataset, row[i], num_neighbors)
    if label == 1:
        print('Data=%s, Predicted: %s ==> OK' % (row[i], label))
    else:
        print('Data=%s, Predicted: %s ==> Fail' % (row[i], label))
    
#K=7
num_neighbors = 7
print('K=7')
for i in range(len(row)):
    label = knn_classification(dataset, row[i], num_neighbors)
    if label == 1:
        print('Data=%s, Predicted: %s ==> OK' % (row[i], label))
    else:
        print('Data=%s, Predicted: %s ==> Fail' % (row[i], label))


# In[ ]:




