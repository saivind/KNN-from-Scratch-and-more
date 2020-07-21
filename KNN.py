#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import operator
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from random import randrange


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2, length):
    distance = 0.0
    for i in range(length):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(dataset, testInstance, k):
    distances = []
    #dist = 0.0
    length = len(testInstance) - 1
    for x in range(len(dataset)):
        dist = euclidean_distance(dataset[x], testInstance, length)
        distances.append((dataset[x], dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

    return sortedVotes[0][0]

def getAccuracy(dataset, predictions):
    trainingErrors = 0
    for x in range(len(dataset)):
        if np.not_equal(dataset[x][2], predictions[x]):
            trainingErrors += 1
    return trainingErrors


def mse(predictions, dataset):
    error=np.mean((np.array(dataset[:, 1:2]) - predictions) ** 2)
    return error


