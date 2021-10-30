# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:36:19 2021

@author: julio
"""

class Perceptron:
    def __init__(self,examples,classes,epochs,learningRate,bias=0,verbose="none"):
        verboseD = {"none":0,"low":1,"medium":2,"high":3}
        self.examples = examples
        self.classes = classes
        self.epochs = epochs
        self.learningRate = learningRate
        self.weights = []
        self.bias = bias
        self.successRate = 0
        self.epochCounter = 1
        self.verbose = verboseD[verbose]

        self.n_fp = 0   # false pos
        self.n_fn = 0   # false neg
        self.n_tp = 0   # true pos
        self.n_tn = 0   # true neg

        for x in range(len(examples[0])):
            self.weights.append(0)
        
    def Hypothesis(self,example):
        if self.DotProduct(example,self.weights) + self.bias > 0:
            return 1
        else:
            return 0
        
    def UpdateWeights(self):
        self.epochCounter = 1
        for epoch in range(0,self.epochs):
            num_false = 0 # number of false examples
            c = 0
            if self.verbose >= 2: print("Beginning epoch " + str(self.epochCounter))
            for ex in self.examples:
                hyp = self.Hypothesis(ex)
                if self.verbose >= 3: print("\n" + "Hypothesis was " + str(hyp))
                if self.verbose >= 3: print("Correct class was " + str(self.classes[c]))


                error = self.classes[c] - hyp
                c += 1
                if error != 0:
                    num_false += 1
                    for att in range(0,len(ex)):
                        self.weights[att] += (self.learningRate*ex[att]) * error
                    if self.verbose >= 3: print("Weight values were updated to " + str(self.weights)) 
            self.successRate = 1 - (num_false/len(self.examples))
            if self.successRate == 1:
                break
            if self.verbose >= 1: print("\n" + "Success rate after " + str(self.epochCounter) + " epochs is " + str(self.successRate))
            if self.verbose >= 1: print("Weight values after " + str(self.epochCounter) + " epochs are " + str(self.weights) + "\n")
            if epoch < self.epochs-1: self.epochCounter += 1
            
    def CompleteTest(self):
        self.UpdateWeights()
        # self.Print()
    
    def DotProduct(self,attributes,weights):
        res = 0
        for x in range(len(attributes)):
            res += attributes[x]*weights[x]
        return res

    def RunModel(self):
        counter = 0
        for example in self.examples:
            ex_hyp = self.Hypothesis(example)
            ex_class = self.classes[counter]
            print("Hypothesis: " + str(ex_hyp) + " | Correct class: " + str(ex_class))
            counter += 1

            # count false pos
            if ex_hyp != ex_class and ex_class == 1:
                self.n_fp += 1

            # count false neg
            if ex_hyp != ex_class and ex_class == 0:
                self.n_tp += 1

            # count true pos
            if ex_hyp == ex_class and ex_class == 1:
                self.n_tp += 1

            # count true neg
            if ex_hyp == ex_class and ex_class == 0:
                self.n_tp += 1

        # calculate metrics
        precision = self.n_tp / (self.n_tp + self.n_fp)
        recall = self.n_tp / (self.n_tp + self.n_fn)
        sensitivity = self.n_tp / (self.n_tp + self.n_fn)
        #specificity = self.n_tn / (self.n_tn + self.n_fp)

        print("n_fp: " + str(self.n_fp))
        print("n_fn: " + str(self.n_fn))
        print("n_tp: " + str(self.n_tp))
        print("n_fn: " + str(self.n_fn))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("Sensitivity: " + str(sensitivity))
       # print("Specificity: " + str(specificity))

    def Print(self):
        print("\nSuccess rate reached " + str(self.successRate*100) + "% in " + str(self.epochCounter) + " epochs")
        print("Learning rate: " + str(self.learningRate))
        print("Current weight values: " + str(self.weights))