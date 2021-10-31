# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:38:26 2021

@author: julio
"""

from Perceptron import Perceptron
import csv
import numpy as np
from matplotlib import pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

def loadIris(examples,classes):
    with open("iris.data", newline = '') as csvFile:
        rows = csv.reader(csvFile,delimiter=',')

        for row in rows:
            tempList = [float(i) for i in row[:4]]
            examples.append(tempList)
            if row[4] == "Iris-setosa":
                classes.append(1)
            else:
                classes.append(0)
    
def loadHappy(examples,classes):
    with open("SomervilleHappinessSurvey2015.csv", newline = '') as csvFile:
        rows = csv.reader(csvFile,delimiter=',')

        for row in rows:
            tempList = [float(i) for i in row[1:]]
            examples.append(tempList)
            classes.append(row[0])

def loadHouse(examples,classes):
    with open("house-votes-84.data", newline = '') as csvFile:
        rows = csv.reader(csvFile,delimiter=',')

        for row in rows:
            tempList = []
            j = 0
            for i in row[1:]:
                if i == 'y':
                    tempList.append(1)
                elif i == 'n':
                    tempList.append(-1)
                else:
                    tempList.append(0)
                j += 1
            examples.append(tempList)
            if row[0] == "republican":
                classes.append(1)
            else:
                classes.append(0)

def loadHaberman(examples,classes):
    with open("haberman.data", newline = '') as csvFile:
        rows = csv.reader(csvFile,delimiter=',')

        for row in rows:
            tempList = [float(i) for i in row[:3]]
            examples.append(tempList)
            classes.append(float(row[3]) - 1)

def loadTTT(examples,classes):
    with open("tic-tac-toe.data", newline = '') as csvFile:
        rows = csv.reader(csvFile,delimiter=',')

        for row in rows:
            tempList = []
            for i in row[:9]:
                if i == 'o':
                    tempList.append(0)
                else:
                    tempList.append(1)
            examples.append(tempList)
            if row[9] == "positive":
                classes.append(1)
            else:
                classes.append(0)

def loadBank(examples,classes):
    with open("data_banknote_authentication.txt", newline = '') as csvFile:
        rows = csv.reader(csvFile,delimiter=',')

        for row in rows:
            tempList = [float(i) for i in row[:4]]
            examples.append(tempList)
            classes.append(float(row[4]))

def loadDryBean(examples,classes):
    with open("Dry_Beans_Dataset.csv", newline='') as csvFile:
        rows = csv.reader(csvFile,delimiter=',')

        for row in rows:
            temp = [float(i) for i in row[:16]]
            if row[16] == "BOMBAY":
                classes.append(1)
                examples.append(temp)
            if row[16] == "SIRA":
                classes.append(0)
                examples.append(temp)


if __name__ == '__main__':
    examples = []
    classes = []

    #loadIris(examples,classes)
    #loadHappy(examples,classes)
    #loadHouse(examples,classes)
    #loadHaberman(examples,classes)
    #loadTTT(examples, classes)
    loadBank(examples, classes)
    #loadDryBean(examples,classes)

    # training = np.split(np.array(examples))



    n_pos = 0
    n_neg = 0
    for c in classes:
        if c == 1:
            n_pos += 1
        else:
            n_neg += 1

    x_train, x_test, y_train, y_test = train_test_split(examples,classes,test_size=0.3)

    undersample = RandomUnderSampler(sampling_strategy='majority')
    x_train_under, y_train_under = undersample.fit_resample(examples, classes)

    per = Perceptron(examples=x_train_under,classes=y_train_under,epochs=10,learningRate=0.01,verbose="low")

    per.RunModel(epochs=10,learning_rate=0.01)