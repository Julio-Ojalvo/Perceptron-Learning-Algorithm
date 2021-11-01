# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:38:26 2021

@author: julio
"""
from datetime import time

import random as rand

from Perceptron import Perceptron
import csv
import numpy as np
from matplotlib import pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as SKPerceptron
from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)
from collections import Counter
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", category=ConvergenceWarning)


def loadIris(examples, classes):
    with open("iris.data", newline='') as csvFile:
        rows = csv.reader(csvFile, delimiter=',')

        for row in rows:
            tempList = [float(i) for i in row[:4]]
            examples.append(tempList)
            if row[4] == "Iris-setosa":
                classes.append(1)
            else:
                classes.append(0)


def loadHappy(examples, classes):
    with open("SomervilleHappinessSurvey2015.csv", newline='') as csvFile:
        rows = csv.reader(csvFile, delimiter=',')

        for row in rows:
            tempList = [float(i) for i in row[1:]]
            examples.append(tempList)
            classes.append(row[0])


def loadHouse(examples, classes):
    with open("house-votes-84.data", newline='') as csvFile:
        rows = csv.reader(csvFile, delimiter=',')

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


def loadHaberman(examples, classes):
    with open("haberman.data", newline='') as csvFile:
        rows = csv.reader(csvFile, delimiter=',')

        for row in rows:
            tempList = [float(i) for i in row[:3]]
            examples.append(tempList)
            classes.append(float(row[3]) - 1)


def loadTTT(examples, classes):
    with open("tic-tac-toe.data", newline='') as csvFile:
        rows = csv.reader(csvFile, delimiter=',')

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


def loadBank(examples, classes):
    with open("data_banknote_authentication.txt", newline='') as csvFile:
        rows = csv.reader(csvFile, delimiter=',')

        for row in rows:
            tempList = [float(i) for i in row[:4]]
            examples.append(tempList)
            classes.append(float(row[4]))


def loadDryBean(examples, classes):
    with open("Dry_Beans_Dataset.csv", newline='') as csvFile:
        rows = csv.reader(csvFile, delimiter=',')

        for row in rows:
            temp = [float(i) for i in row[:16]]
            if row[16] == "BOMBAY":
                classes.append(1)
                examples.append(temp)
            if row[16] == "SIRA":
                classes.append(0)
                examples.append(temp)


def joinData(examples, classes):
    res = []
    for x in range(len(examples)):
        res.append(examples[x] + [classes[x]])
    return res


def arrangeBankData(data, sType):
    pos = []
    neg = []
    for x in data:
        if x[4] == 1:
            pos.append(x)
        else:
            neg.append(x)
    if sType == -1:
        for x in range(692):
            #pos.remove(pos[rand.randint(0, len(pos) - 1)])
            neg.remove(neg[rand.randint(0, len(neg) - 1)])
    elif sType == 0:
        for x in range(692):
            neg.remove(neg[rand.randint(0, len(neg) - 1)])
    else:
        for x in range(0):
            neg.remove(neg[rand.randint(0, len(neg) - 1)])
    return pos + neg


def splitBankData(data, examples, classes):
    examples.clear()
    classes.clear()
    for row in data:
        tempList = [float(i) for i in row[:4]]
        examples.append(tempList)
        if float(row[4]) == 1:
            classes.append(1)
        else:
            classes.append(0)


if __name__ == '__main__':
    examples = []
    classes = []
    mExamples = []
    mClasses = []

    # loadIris(examples,classes)
    # loadHappy(examples,classes)
    # loadHouse(examples,classes)
    # loadHaberman(examples,classes)
    # loadTTT(examples, classes)
    loadBank(examples, classes)
    # loadDryBean(examples,classes)

    x = joinData(examples, classes)
    y = arrangeBankData(x, -1)
    splitBankData(y, mExamples, mClasses)

    # training = np.split(np.array(examples))

    x_train, x_test, y_train, y_test = train_test_split(examples, classes, test_size=.3, shuffle=True)
    # x_train = examples
    # y_train = classes
    # x_test = examples
    # y_test = classes

    tomek = TomekLinks(sampling_strategy='majority')
    undersample = RandomUnderSampler(sampling_strategy='majority')

    #x_train_under, y_train_under = undersample.fit_resample(x_train, y_train)

    print("Class balance before undersampling: " + str(Counter(y_train)))
    #plt.bar(Counter(y_train).keys(), Counter(y_train).values())
    #plt.show()
    per_imbalanced = Perceptron(examples=x_train, classes=y_train, x_test=mExamples, y_test=mClasses, epochs=10,learningRate=0.01,verbose="none")
    per_imbalanced.RunModel(epochs=10,learning_rate=0.01)
    print("Imbalanced training set performance: ")
    per_imbalanced.EvaluatePerformance()

    # print("")

    # print("Class balance after undersampling: " + str(Counter(y_train_under)))
    # plt.bar(Counter(y_train_under).keys(), Counter(y_train_under).values())
    # plt.show()
    # per_balanced = Perceptron(examples=x_train_under,classes=y_train_under,x_test=x_test, y_test=y_test,epochs=10,learningRate=0.01,verbose="none")
    # per_balanced.RunModel(epochs=10,learning_rate=0.01)
    # print("Balanced training set performance: ")
    # per_balanced.EvaluatePerformance()
    # print("")

    accuracy_all = []
    precision_all = []
    recall_all = []

    # print("Class balance before undersampling: " + str(Counter(y_train)))
    # print("Class balance before undersampling: " + str(Counter(y_test)))
    # clf = SKPerceptron(max_iter=10, eta0=0.01)
    # clf.fit(x_train, y_train)
    # accuracy_b = clf.score(x_test, y_test)
    # print("Accuracy: " + str(accuracy_b))
    # predictions = clf.predict(x_test)
    # precision_b, recall_b, _ = precision_recall_curve(y_test, predictions)
    # print("Precision: " + str(precision_b[0]) + " \nRecall:" + str(recall_b[0]))
    # # print(" ")
    # accuracy_all.append(accuracy_b)
    # precision_all.append(precision_b[0])
    # recall_all.append(recall_b[1])

    n_accuracy_above_bench = 0
    n_precision_above_bench = 0
    n_recall_above_bench = 0

    # for iteration in range(0,999):
    #     seed = np.random.seed(seed=iteration)
    #     #x_train, x_test, y_train, y_test = train_test_split(examples, classes, test_size=.30, shuffle=true, random_state=seed)
    #     undersample = RandomUnderSampler(sampling_strategy='majority',random_state=seed)
    #
    #     x_train_under, y_train_under = undersample.fit_resample(x_train, y_train)
    #
    #     #print("Class balance after undersampling: " + str(Counter(y_train_under)))
    #     clf.fit(x_train_under, y_train_under)
    #     accuracy = clf.score(x_test, y_test)
    #     #print("Accuracy: " + str(clf.score(x_test, y_test)))
    #     predictions = clf.predict(x_test)
    #     precision, recall, _ = precision_recall_curve(y_test, predictions)
    #     #print("Precision: " + str(precision[0]) + " \nRecall:" + str(recall[1]))
    #
    #     accuracy_all.append(accuracy)
    #     precision_all.append(precision[0])
    #     recall_all.append(recall[1])
    #
    #     if accuracy > accuracy_b:
    #         n_accuracy_above_bench += 1
    #     if precision[0] > precision_b[0]:
    #         n_precision_above_bench += 1
    #     if recall[1] > recall_b[1]:
    #         n_recall_above_bench += 1

    pct_accuracy_above_bench = (n_accuracy_above_bench / 999) * 100
    pct_precision_above_bench = (n_precision_above_bench / 999) * 100
    pct_recall_above_bench = (n_recall_above_bench / 999) * 100

    # pd.set_option("display.max_rows", 100, "display.max_columns", 4)
    df = pd.DataFrame({"Accuracy": accuracy_all, "Precision": precision_all, "Recall": recall_all})
    # print(df)
    #df.to_csv('bank_under.csv', sep=',')
    # df.plot.hist()
    # plt.show()
    # print(accuracy_all)

    #print("n_accuracy_above_bench: " + str(n_accuracy_above_bench) + " PCT: " + str(pct_accuracy_above_bench))
    #print("n_precision_above_bench: " + str(n_precision_above_bench) + " PCT: " + str(pct_precision_above_bench))
    #print("n_recall_above_bench: " + str(n_recall_above_bench) + " PCT: " + str(pct_recall_above_bench))

    # predictions_imbalanced = clf.predict(x_test)
    # precision_imbalanced, recall_imbalanced, _ = precision_recall_curve(y_test, predictions_imbalanced)

    # x_train_imbalanced = np.split(np.array(examples),2)[0]
    # x_test_imbalanced = np.split(np.array(examples),2)[1]
    # y_train_imbalanced = np.split(np.array(classes),2)[0]
    # y_test_imbalanced = np.split(np.array(classes),2)[1]
