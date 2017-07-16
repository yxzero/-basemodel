#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: pso.py
#Author: yuxuan
#Created Time: 2016-12-11 17:07
############################

import datetime
import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib import pyplot as plt
import random

classify_dict = np.load("classify.dict.npz")['arr_0'][()]
train_day15 = []

def fitFunc(position, allexample):
    '''
    bestsum = 0
    for i in allexample:
        new_temp = classify_dict[i[0]].reshape(1, 15)
        pred_new = i[1] + np.dot(np.dot(new_temp, position), new_temp.T)
        bestsum +=  (i[2]-pred_new[0][0])**2
    return np.sqrt(bestsum/len(allexample))
    '''
    basesum = 0.0
    for i in allexample:
        new_temp = classify_dict[i[0]].reshape(1, 15)
        pred_new = i[1] + np.dot(np.dot(new_temp, position), new_temp.T)
        basesum += abs(pred_new[0][0] - i[2]) / i[2]
    return basesum/len(allexample)

class bird:
    """
    speed:速度
    position:位置
    fit:适应度
    lbestposition:经历的最佳位置
    lbestfit:经历的最佳的适应度值
    """
    def __init__(self, speed, position, fit, lBestPosition, lBestFit):
        self.speed = speed
        self.position = position
        self.fit = fit
        self.lBestFit = lBestFit
        self.lBestPosition = lBestPosition

class PSO:
    """
    fitFunc:适应度函数
    birdNum:种群规模
    w:惯性权重 一般设置0.4左右
    c1,c2:个体学习因子，社会学习因子 一般都取2
    solutionSpace:解空间，/2
    """
    def __init__(self, fitFunc, birdNum, c1, c2, solutionSpace, extend, birds):
        self.fitFunc = fitFunc
        self.c1 = c1
        self.c2 = c2
        self.w = 0.9
        self.extend = extend
        self.solutionmax = 100
        self.solutionmin = -100
        if birds == None:
            self.birds, self.best = self.initbirds(birdNum, solutionSpace)
        else:
            self.best = birds[0]
            for birdi in birds:
                if birdi.fit < self.best.fit:
                    self.best = birdi
            self.birds = birds

    def initbirds(self, size, solutionSpace):
        birds = []
        for i in range(size):
            # position = random.uniform(solutionSpace[0], solutionSpace[1])
            position = (np.random.random(size=(15,15))-0.5)*solutionSpace
            speed = np.zeros(shape=(15, 15))
            fit = self.fitFunc(position, self.extend)
            birds.append(bird(speed=speed, position=position, fit=fit, lBestPosition=position, lBestFit=fit))
        best = birds[0]
        for birdi in birds:
            if birdi.fit < best.fit:
                best = birdi
        return birds,best

    def updateBirds(self):
        for birdi in self.birds:
            # 更新速度
            birdi.speed = self.w * birdi.speed + self.c1 * random.random() * (birdi.lBestPosition - birdi.position) + self.c2 * random.random() * (self.best.lBestPosition - birdi.position)
            # 更新位置
            birdi.position = birdi.position + birdi.speed
            for i in range(birdi.position.shape[0]):
                for j in range(birdi.position.shape[1]):
                    if birdi.position[i][j] > self.solutionmax:
                        birdi.position[i][j] = self.solutionmax*0.5 + self.solutionmax*random.random()*0.5
                    if birdi.position[i][j] < self.solutionmin:
                        birdi.position[i][j] = self.solutionmin*0.5 + self.solutionmin*random.random()*0.5
            # 跟新适应度
            birdi.fit = self.fitFunc(birdi.position, self.extend) 
            # 查看是否需要更新经验最优
            if birdi.fit < birdi.lBestFit:
                birdi.lBestFit = birdi.fit
                birdi.lBestPosition = birdi.position

    def solve(self, maxIter):
        # 只考虑了最大迭代次数，如需考虑阈值，添加判断语句就好
        for i in range(maxIter):
            # 更新粒子
            self.w = 0.5*(maxIter-i)/maxIter + 0.4
            self.updateBirds()
            for birdi in self.birds:
                # 查看是否需要更新全局最优
                if birdi.fit < self.best.lBestFit:
                    self.best = birdi
            '''
            if i%10 == 0:
                print "第"+str(i)+"次:"+str(self.best.lBestFit)
            '''
            if self.best.lBestFit < 0.05:
                break
        return self.best.lBestPosition, self.birds, self.best.lBestFit
