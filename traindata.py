# -*-coding:utf-8-*-
'''
Created on 2016年12月19日

@author: yx
'''
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
import logging
import numpy as np
from draw_data import draw_data
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read_sourse(each_item, begin_time=u"2015-09-25", last_time = u"2015-09-27"):
    # 取出时间
    getdata = draw_data()
    title_item = getdata.get_title_data(begin_time, last_time, 100)
    for i in title_item:
        each_item[i['_id']] = [float(i['title_time'].split()[1].split(':')[0])] #取出时间作为特征
    logging.info(u"取出3个月的新闻,done")

    idfile = open("paragraph_name.txt", 'r')
    id_x = []
    for i in idfile.readlines():
        id_x.append(i.strip().split('\t')[1])
    idfile.close()

    vectorfile = open("sentence_vectors.txt", 'r')
    vector_list = []
    for lines in vectorfile.readlines():
        line = lines.strip().split(' ')
        tv = []
        for i in range(1,len(line)):
            tv.append(float(line[i]))
        vector_list.append(tv)
    vectorfile.close()
    logging.info(u"取出新闻的vector,done")

    for i in range(len(id_x)):
        if id_x[i] in each_item:
            each_item[id_x[i]] += vector_list[i]

    hours_file = open("2.5hour.txt", 'r')
    for i in hours_file.readlines():
        line = i.strip().split('\t')
        if float(line[3]) < 5:
            continue
        if line[0] in each_item:
            each_item[line[0]] += [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
            #each_item[line[0]].append([a1, a2, a3, np.log10(float(line[4]))])
    hours_file.close()
    # each_item 每个：0是时间；1-15是doc vector；16-18是1,1.5,2小时的评论量；19是最终热度
    np.savez("each_item", each_item)
    logging.info(u"写入文件each_item.npz,done")

if __name__ == "__main__":
    each_item = dict()
    read_sourse(each_item, "2015-09-25", "2015-12-26")
