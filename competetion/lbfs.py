#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: lbfs.py
#Author: yuxuan
#Created Time: 2016-12-26 11:40:36
############################

import datetime
import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib import pyplot as plt
import random
import time

classify_dict = np.load("classify.dict.npz")['arr_0'][()]

class SGD():

    def __init__(self, extend, parmes):
        self.extend = extend
        if parmes != None:
            self.parmes = parmes
        else:
            #self.parmes = (np.random.random(size=(15,15))-0.5)*2000
            self.parmes = np.zeros(shape=(15,15))
        self.new_temp = []
        for item in self.extend:
            self.new_temp.append(classify_dict[item[0]])
        self.new_temp = np.array(self.new_temp)
        print self.new_temp.shape
    
    def solveall(self, maxIter, theta0):
        m = len(self.extend)
        item = np.array(self.extend, dtype=float)
        for i in range(maxIter):
            theta = theta0*(maxIter-i)/maxIter+0.05
            pred_new = item[:,1] + np.sum(np.dot(self.new_temp, self.parmes)*self.new_temp, axis=1)
            time.sleep(1)
            print "iter "+str(i)+" error：" + str(np.sqrt(np.sum((item[:,2]-pred_new)**2)/m))
            self.parmes += theta/m*np.sum(item[:,2]-pred_new)*np.dot(self.new_temp.T, self.new_temp)
        return self.parmes

    def solve(self, maxIter, theta0, theta1):
        error1 = 100
        error0 = 0
        theta = 0.5
        for i in range(maxIter):
            #theta = theta0*(maxIter-i)/maxIter+0.1
            for item in self.extend:
                new_temp = classify_dict[item[0]].reshape(1, 15)
                pred_new = item[1] + np.dot(np.dot(new_temp, self.parmes), new_temp.T)
                self.parmes += (theta*(item[2]-pred_new[0][0])*np.dot(new_temp.T, new_temp))
            bestsum = 0 
            for item in self.extend:
                new_temp = classify_dict[item[0]].reshape(1, 15)
                pred_new = item[1] + np.dot(np.dot(new_temp, self.parmes), new_temp.T)
                bestsum +=  (item[2]-pred_new[0][0])**2
            error1 = np.sqrt(bestsum/len(self.extend))
            if i%1000 == 0:
                print "iter "+str(i)+" error：" + str(error1)
            #if abs(error1-error0) < 0.1:
            if error1 < 800:
                break
            error0 = error1
        return self.parmes
