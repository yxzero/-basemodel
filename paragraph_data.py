# -*- coding:utf-8
'''
    created at 2016-03-31
'''
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
import logging
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

title_name = []
title_id = []

# 得到 paragraph vector
def get_vector():
    temp_vector = []
    vector_file = open('sentence_vectors.txt', 'r')
    for lines in vector_file.readlines():
        line = lines.strip().split(' ')
        tv = []
        for i in range(1,len(line)):
            tv.append(float(line[i]))
        temp_vector.append(tv)
    vector_file.close()
    name_file = open('paragraph_name.txt', 'r')
    for lines in name_file.readlines():
        line = lines.strip().split('\t')
        title_name.append(line[0])
        title_id.append(line[1])
    name_file.close()
    return np.mat(temp_vector)

def cluster(methor = 'DBSCAN'):
    from sklearn.cluster import DBSCAN
    p_vector = get_vector()
    id2cluster = dict()
    if methor == 'DBSCAN':
        db = DBSCAN(eps=1.3, min_samples=7).fit(p_vector)
        labels = db.labels_
        print labels
        cluster_num = max(labels) + 1
        label_name = [[] for i in range(cluster_num)]
        for i in range(len(labels)):
            if labels[i] >= 0:
                label_name[labels[i]].append(title_name[i])
                id2cluster[title_id[i]] = labels[i]
    for i in range(len(label_name)):
        print i
        for j in label_name[i]:
            print j
    np.savez("id2cluster", id2cluster)
    logging.info("id2cluster save, done!")

if __name__ == '__main__':
    import getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], "fc", ["getdata", "cluster="])
        for op, value in opts:
            if op == "--getdata":
                process_data()
            elif op == "--cluster":
                cluster(value)
            elif op == '-c':
                logging.info(u"默认聚类为dbscan...")
                cluster()
            elif op == "-f":
                findsimirlar()        
    except getopt.GetoptError:
        logging.info(u'参数错误..')
        system.exit()
