#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: fans_model.py
#Author: yuxuan
#Created Time: 2016-12-09 17:07
############################
import datetime
import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib import pyplot as plt
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def mape(p, t):
    return np.mean(abs(p-t)/t)*100


def regression():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.externals import joblib
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn import linear_model
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import mean_squared_error
    from sklearn import linear_model
    #each_item = np.load("each_item.npz")['arr_0'][()]
    each_item = np.load("each_itemSH.npz")['arr_0'][()]
    X_data = []
    Y_data = []
    save_id = each_item.keys();
    for i in save_id:
        #if len(each_item[i]) == 20:
        if len(each_item[i]) == 4:
            X_data.append(each_item[i][0:-1])
            print each_item[i][0:-1]
            Y_data.append(each_item[i][-1])
    n_sample = len(X_data)
    logging.info(u"符合的数据有"+str(n_sample)+u"条。")
    sidx = np.random.permutation(n_sample)
    n_train = int(np.round(n_sample * 0.3))
    #test_x_set = np.array([X_data[s] for s in sidx[n_train:]])
    #test_y_set = np.array([Y_data[s] for s in sidx[n_train:]])
    #test_x_set = np.array([X_data[s] for s in sidx[:n_train]])
    test_x_set = np.log10(np.array([X_data[s] for s in sidx[:n_train]])+1)
    test_y_set = np.array([Y_data[s] for s in sidx[:n_train]])
    test_id = [save_id[s] for s in sidx[:n_train]]

    #train_x_set = np.array([X_data[s] for s in sidx])
    train_x_set = np.log10(np.array([X_data[s] for s in sidx])+1)
    #train_y_set = np.array([Y_data[s] for s in sidx])
    train_y_set = np.log10(np.array([Y_data[s] for s in sidx]))
    train_id = [save_id[s] for s in sidx]

    #rg = DecisionTreeRegressor(max_depth=5, random_state=0)
    rg = linear_model.LinearRegression()
    rg.fit(train_x_set, train_y_set)

    # = rg.predict(test_x_set)
    predict_y = np.power(10, rg.predict(test_x_set))
    train_y_pred = rg.predict(train_x_set)
    evs = explained_variance_score(test_y_set, predict_y)
    rmse = np.sqrt(mean_squared_error(test_y_set, predict_y))
    logging.info("rmse:"+str(rmse)+" evs:"+str(evs))
    logging.info("mape:"+str(mape(predict_y, test_y_set))+"%")
    '''
    for i in range(len(predict_y)):
        print str(predict_y[i]) + "," + str(test_y_set[i])
    '''
    draw_same(test_y_set, predict_y)
    dict_id_pred = dict()
    for i in range(len(train_id)):
        dict_id_pred[train_id[i]] = [train_y_pred[i], train_y_set[i]]
    for i in range(len(test_id)):
        dict_id_pred[test_id[i]] = [predict_y[i], test_y_set[i]]
    np.savez("dict_id_pred", dict_id_pred)
    logging.info(u"存储新闻id预测与真实文件, done!")

def draw_same(y_true, y_pred):
    plt.plot(y_true, y_pred, 'r.')
    plt.plot([0, 50000], [0, 50000], 'b-')
    plt.show()

if __name__ == "__main__":
    regression()
