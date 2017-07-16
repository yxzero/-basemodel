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
        pso = PSO(fitFunc=fitFunc, birdNum=1500, c1=3, c2=2, solutionSpace=100, extend=allexample, birds=birds) 
        lBestPosition, birds, lBestFit = pso.solve(300)
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

def drawPosition(dayposition, daytime, level, k):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates   
    topic_name = [u'汽车',u'财经',u'科技',u'健康',u'体育',
                  u'旅游',u'教育',u'文化',u'军事',u'社会',
                  u'国内',u'国际',u'房产',u'娱乐',u'时尚']
    plt.matshow(dayposition)
    plt.title("interaction "+str(daytime+datetime.timedelta(days = (-k+1)))+" to "+str(daytime+datetime.timedelta(days = 1))+" level="+str(level), fontsize=18)
    plt.xticks(np.arange(15), topic_name, fontsize=16, rotation=30)
    plt.yticks(np.arange(15), topic_name, fontsize=16, rotation=30)
    plt.colorbar()
    plt.show()

def plot_3d(dateX, travel):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import datetime
    import matplotlib.dates as mdates
    from scipy import interpolate
    from scipy.interpolate import spline

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    X = mdates.date2num(np.array(dateX))
    Y = np.arange(0, 15, 1)

    #xnew = np.linspace(X.min(), X.max(), 2*len(X))
    #ynew = np.linspace(Y.min(), Y.max(), 2*len(Y))
    xnew = np.linspace(X.min(), X.max(), 6*len(X))
    ynew = np.linspace(Y.min(), Y.max(), 3*len(Y))

    z = travel.T
    Z = np.zeros(shape=(0,6*len(X)))
    for i in range(z.shape[0]):
        z_smooth = spline(X, z[i], xnew)
        Z = np.insert(Z, Z.shape[0], values=z_smooth, axis=0)
    znew = np.zeros(shape=(3*len(Y),0))
    for i in range(Z.shape[1]):
        z_smooth = spline(Y, Z[:,i], ynew)
        znew = np.insert(znew, znew.shape[1], values=z_smooth, axis=1)
    xnew, ynew = np.meshgrid(xnew, ynew)


    #Y, X = np.meshgrid(Y, X)
    #xnew, ynew = np.meshgrid(xnew, ynew)

    #tck = interpolate.bisplrep(X, Y, travel, s=0)
    #znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)
    #print X.shape
    #print Y.shape
    #print travel.shape
    topic_name = [u'汽车',u'财经',u'科技',u'健康',u'体育',
                      u'旅游',u'教育',u'文化',u'军事',u'社会',
                      u'国内',u'国际',u'房产',u'娱乐',u'时尚']
    yearsFmt = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(yearsFmt)
    surf = ax.plot_surface(xnew, ynew, znew, rstride=1, cstride=1, cmap=plt.cm.jet)
    #surf = ax.plot_surface(X, Y, travel, rstride=1, cstride=1, cmap=plt.cm.YlGnBu_r)
    plt.yticks(np.arange(15), topic_name, fontsize=15, rotation=40)
    #ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
    #ax.set_zlim(-2,2)

    fig.colorbar(surf)
    # savefig('../figures/plot3d_ex.png',dpi=48)
    plt.show()


def get_data():
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    draw_mysql = draw_data()
    rmses = []
    for level in range(1000, 4200, 200):
        allposition = np.zeros(shape=(15, 15))
        begin_time = datetime.datetime.strptime('2015-11-10', '%Y-%m-%d')
        end_time = datetime.datetime.strptime('2015-11-11', '%Y-%m-%d')
        queue = []
        for i in range(3):
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
        xdate = []
        travel = np.zeros(shape=(0, 15))
        for i in range(9):#87
            xdate.append(begin_time+datetime.timedelta(days = -1))
            print str(begin_time) + " " + str(level)
            temp = draw_mysql.get_title_data(str(begin_time), str(end_time), 0)
            today_day = []
            for ti in temp:
                if ti['_id'] in dict_id_pred and ti['_id'] in classify_dict:
                    today_day.append(ti['_id'])
            birds = None
            temp_position, birds, lbestfit, basefit = train_data(queue, birds, level)
            lbestFitList.append(lbestfit)
            baseFitList.append(basefit)
            if temp_position != None:
                pred_position = temp_position
                travel = np.insert(travel, travel.shape[0], values=temp_position[8], axis=0)
                allposition = allposition + pred_position
                #drawPosition(pred_position, begin_time+datetime.timedelta(days = -1), level, 3)
            #true_predict = predict_data(pred_position, today_day, level, true_predict)
            queue.pop(0)
            queue.append(today_day)
            begin_time = end_time
            end_time = end_time + datetime.timedelta(days = 1)
        plot_3d(xdate, travel)
        # evaluation
        print allposition/len(xdate)
        with open("matrix.txt", 'w') as f:
            f.write(str(allposition/len(xdate)))
        yearsFmt = mdates.DateFormatter('%Y-%m-%d')
        fig, ax = plt.subplots()
        ax.set_title(u"腾讯新闻2个月竞争预测图 level="+str(level), fontsize=18)
        ax.set_xlabel("per day", fontsize=18)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.plot(xdate, lbestFitList, label=u"加入后")
        ax.plot(xdate, baseFitList, label=u"原始")
        ax.grid(True)
        plt.xticks(fontsize=18) 
        plt.yticks(fontsize=18)
        plt.legend()
        plt.show()
        drawPosition(allposition/len(xdate), begin_time+datetime.timedelta(days = -1), level, 0)

if __name__ == "__main__":
    get_data()
