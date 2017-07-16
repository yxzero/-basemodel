#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: train_data.py
#Author: yuxuan
#Created Time: 2016-12-22 10:02:31
############################

from draw_data import draw_data
from pso import *
import numpy as np
import matplotlib.pyplot as plt

classify_dict = np.load("classify.dict.npz")['arr_0'][()]
dict_id_pred = np.load("dict_id_pred.npz")['arr_0'][()]

def targetIn(bitem, titem):
    temp = titem * 0.2
    less_line = titem - temp
    up_line = titem + temp
    if bitem >= less_line and bitem <= up_line:
        return 1
    else:
        return 0

def train_data(queue, birds, level=3000):
    allexample = []
    bestsum = 0.0
    for listi in queue:
        for idnew in listi:
            if dict_id_pred[idnew][0] > level:
                allexample.append([idnew, dict_id_pred[idnew][0], dict_id_pred[idnew][1]])
                bestsum += (abs(dict_id_pred[idnew][0] - dict_id_pred[idnew][1])/dict_id_pred[idnew][1])
    if len(allexample) > 0:
        print "基础模型mape:"+str(bestsum/len(allexample))
        pso = PSO(fitFunc=fitFunc, birdNum=1500, c1=2, c2=2, solutionSpace=100, extend=allexample, birds=birds) 
        lBestPosition, birds, lBestFit = pso.solve(100)
        return lBestPosition, birds, lBestFit, bestsum/len(allexample)
    return None, birds

def predict_data(position, predict_example, level=3000, true_predict = []):
    #print position
    basesum = 0.0
    predsum = 0.0
    for i in predict_example:
        if dict_id_pred[i][0] > level:
            new_temp = classify_dict[i].reshape(1, 15)
            pred_new = dict_id_pred[i][0] + np.dot(np.dot(new_temp, position), new_temp.T)
            true_predict.append([pred_new[0][0], dict_id_pred[i][0], dict_id_pred[i][1]])
            predsum += targetIn(pred_new[0][0], dict_id_pred[i][1])
        else:
            true_predict.append([dict_id_pred[i][0], dict_id_pred[i][0], dict_id_pred[i][1]])
            predsum +=  targetIn(dict_id_pred[i][0], dict_id_pred[i][1])
        basesum +=  targetIn(dict_id_pred[i][0], dict_id_pred[i][1])
    if len(predict_example)>0:
        print "基础预测模型:"+str(basesum/len(predict_example))
        print "竞争预测模型:"+str(predsum/len(predict_example))
    return true_predict

def drawPosition(dayposition):
    topic_name = [u'汽车',u'财经',u'科技',u'健康',u'体育',
                  u'旅游',u'教育',u'文化',u'军事',u'社会',
                  u'国内',u'国际',u'房产',u'娱乐',u'时尚']
    plt.title("interaction", fontsize=18)
    plt.matshow(dayposition)
    plt.xticks(np.arange(15), topic_name, fontsize=16, rotation=30)
    plt.yticks(np.arange(15), topic_name, fontsize=16, rotation=30)
    plt.colorbar()
    plt.show()

def get_data():
    draw_mysql = draw_data()
    rmses = []
    for level in range(1000, 4200, 200):
        begin_time = datetime.datetime.strptime('2015-09-25', '%Y-%m-%d')
        end_time = datetime.datetime.strptime('2015-09-26', '%Y-%m-%d')
        queue = []
        for i in range(7):
            print begin_time
            temp = draw_mysql.get_title_data(str(begin_time), str(end_time), 0) 
            today_day = []
            for ti in temp:
                if ti['_id'] in dict_id_pred and ti['_id'] in classify_dict:
                    today_day.append(ti['_id'])
            queue.append(today_day)
            begin_time = end_time
            end_time = end_time + datetime.timedelta(days = 1)
        birds = None
        pred_position = None 
        true_predict = []
        lbestFitList = []
        baseFitList = []
        for i in range(80):
            print str(begin_time) + " " + str(level)
            temp = draw_mysql.get_title_data(str(begin_time), str(end_time), 0)
            today_day = []
            for ti in temp:
                if ti['_id'] in dict_id_pred and ti['_id'] in classify_dict:
                    today_day.append(ti['_id'])
            temp_position, birds, lbestfit, basefit = train_data(queue, birds, level)
            lbestFitList.append(lbestfit)
            baseFitList.append(basefit)
            if temp_position != None:
                pred_position = temp_position
                drawPosition(pred_position)
            true_predict = predict_data(pred_position, today_day, level, true_predict)
            queue.pop(0)
            queue.append(today_day)
            begin_time = end_time
            end_time = end_time + datetime.timedelta(days = 1)
        # evaluation
        mape = 0.0
        for i in true_predict:
            mape += targetIn(i[0], i[2])
        mape /= len(true_predict)
        rmses.append(mape)
        print mape
    basemape = 0.0
    for i in true_predict:
        basemape += targetIn(i[1], i[2])
    basemape /= len(true_predict)
    plt.plot(range(1000, 4200, 200), rmses, 'b-')
    plt.plot([1000, 3900], [basemape, basemape], 'r-')
    plt.show()

if __name__ == "__main__":
    get_data()
