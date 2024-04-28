import os
from graphDataset import UdGraphDataset


def loadTree(dataname):
    treePath = "C:\\Users\\priya\\Documents\\ProjectData\\" + dataname + "\\data.TD_RvNN.vol_5000.txt"
    print("reading twitter tree")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
    print('tree no:', len(treeDic))

    return treeDic

def loadUdData(dataname, treeDic, fold_x_train, fold_x_test, droprate):

    print("loading train set", )
    traindata_list = UdGraphDataset(fold_x_train, treeDic, dataname=dataname, use_word2vec=False)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = UdGraphDataset(fold_x_test, treeDic, dataname=dataname, use_word2vec=False)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

