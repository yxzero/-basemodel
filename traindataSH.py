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

    hours_file = open("2.5hour.txt", 'r')
    for i in hours_file.readlines():
        line = i.strip().split('\t')
        if float(line[3]) < 5:
            continue
        each_item[line[0]] = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
        #each_item[line[0]].append([a1, a2, a3, np.log10(float(line[4]))])
    hours_file.close()
    # each_item 每个：0是时间；1-15是doc vector；16-18是1,1.5,2小时的评论量；19是最终热度
    np.savez("each_itemSH", each_item)
    logging.info(u"写入文件each_itemSH.npz,done")

if __name__ == "__main__":
    each_item = dict()
    read_sourse(each_item, "2015-09-25", "2015-12-26")
