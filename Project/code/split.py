import random

def loadSplitData(obj, test_ratio=0.1):
    labelPath = "C:\\Users\\priya\\Documents\\ProjectData\\" + obj + "\\" + obj + "_label_All.txt"
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    print("loading tree label")

    NR, F, T, U = [], [], [], []
    labelDic = {}

    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        labelDic[eid] = label.lower()
        if label in labelset_nonR:
            NR.append(eid)
        if labelDic[eid] in labelset_f:
            F.append(eid)
        if labelDic[eid] in labelset_t:
            T.append(eid)

    print(len(labelDic))

    # Shuffle the data
    random.shuffle(NR)
    random.shuffle(F)
    random.shuffle(T)
    
    # Select instances randomly from each label dataset
    test_NR = random.sample(NR, 41)
    test_F = random.sample(F, 41)
    test_T = random.sample(T, 42)

    # Combine the test sets
    test_set = test_NR + test_F + test_T

    # Remove the selected instances from the datasets
    NR = [x for x in NR if x not in test_NR]
    F = [x for x in F if x not in test_F]
    T = [x for x in T if x not in test_T]

    # Remaining instances will be part of the training set
    train_set = NR + F + T

    return test_set, train_set
