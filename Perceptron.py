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
            numWrong = 0
            c = 0
            if self.verbose >= 2: print("Beginning epoch " + str(self.epochCounter))
            for ex in self.examples:
                hyp = self.Hypothesis(ex)
                if self.verbose >= 3: print("\n" + "Hypothesis was " + str(hyp))
                if self.verbose >= 3: print("Correct class was " + str(self.classes[c]))
                error = self.classes[c] - hyp
                c += 1
                if error != 0:
                    numWrong += 1
                    for att in range(0,len(ex)):
                        self.weights[att] += (self.learningRate*ex[att]) * error
                    if self.verbose >= 3: print("Weight values were updated to " + str(self.weights)) 
            self.successRate = 1 - (numWrong/len(self.examples))
            if self.successRate == 1:
                break
            if self.verbose >= 1: print("\n" + "Success rate after " + str(self.epochCounter) + " epochs is " + str(self.successRate)) 
            if self.verbose >= 1: print("Weight values after " + str(self.epochCounter) + " epochs are " + str(self.weights) + "\n") 
            if epoch < self.epochs-1: self.epochCounter += 1
            
    def CompleteTest(self):
        self.UpdateWeights()
        self.Print()
    
    def DotProduct(self,attributes,weights):
        res = 0
        for x in range(len(attributes)):
            res += attributes[x]*weights[x]
        return res
    
    def Print(self):
        print("\nSuccess rate reached " + str(self.successRate*100) + "% in " + str(self.epochCounter) + " epochs")
        print("Learning rate of " + str(self.learningRate))
        print("Current weight values are " + str(self.weights))