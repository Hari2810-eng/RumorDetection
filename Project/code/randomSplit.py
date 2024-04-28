import random

random.seed()

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
        #if labelDic[eid] in labelset_u:
         #   U.append(eid)

    print(len(labelDic))

    # Shuffle the data
    random.shuffle(NR + F + T)
    
    # Calculate the number of instances for testing based on the ratio
    len_data = len(NR + F + T)
    len_test = int(len_data * test_ratio)

    # Split the data into training and testing sets
    test_set = NR[:len_test] + F[:len_test] + T[:len_test]
    train_set = NR[len_test:] + F[len_test:] + T[len_test:]

    return test_set, train_set
