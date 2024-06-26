# -*- coding: utf-8 -*-
"""RumorDetection

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GSy7NluJtXi5XqvaY89fDk8ixoVYsM-s

**DATA FOLDING AND SPLITTING**

**Data Source:** Reads Twitter data from a specified file containing tweet information, separating tweets into their types (e.g., non-rumor, news, false rumor, true rumor, unverified) and respective IDs.

**Train-Test Split:** Divides the tweet data into train and test sets using a folding technique.

**Balanced Representation:** Ensures an equitable distribution of different tweet types in both training and testing sets across multiple folds.

**Output Details:** Provides the count and specific tweet IDs for each fold's training and testing datasets, aiding in comprehensive analysis and model development.
"""

import random

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split('\t')
            tweet_type = values[0]
            tweet_id = values[2]
            data.append((tweet_type, tweet_id))
    return data

def split_data_by_type(data):
    data_by_type = {}
    for tweet_type, tweet_id in data:
        if tweet_type not in data_by_type:
            data_by_type[tweet_type] = []
        data_by_type[tweet_type].append(tweet_id)
    return data_by_type

def split_train_test_data(data_by_type, folds=5):
    train_test_splits = []
    for _ in range(folds):
        train_set = []
        test_set = []
        for tweet_type, tweets in data_by_type.items():
            random.shuffle(tweets)
            split_index = len(tweets) // folds
            test_split = tweets[:split_index]
            train_split = tweets[split_index:]
            train_set.extend(train_split)
            test_set.extend(test_split)
        random.shuffle(train_set)
        random.shuffle(test_set)
        train_test_splits.append((train_set, test_set))
    return train_test_splits

# File path to your data file
file_path = '/content/drive/MyDrive/ProjectData/Twitter15_label_All.txt'

# Load data from file
data = load_data(file_path)

# Split data by tweet type
data_by_type = split_data_by_type(data)

# Split data into train and test for each fold
folds_count = 5  # Modify as needed
train_test_splits = split_train_test_data(data_by_type, folds=folds_count)

# Print train and test sets for each fold
for i, (train_set, test_set) in enumerate(train_test_splits):
    print(f"Fold {i + 1} - Train Data:", len(train_set))
    print(f"Fold {i + 1} - Test Data:", len(test_set))

for i, (train_set, test_set) in enumerate(train_test_splits):
    print(f"Fold {i + 1} - Train Data:")
    print(train_set)
    print(f"Fold {i + 1} - Test Data:")
    print(test_set)

def read_twitter_tree(file_path):
    print("Reading Twitter tree...")
    tree_data = {}

    file = open(file_path, 'r')
    for line in file:
        eid, index_p, index_c, max_degree, max_l, vec = line.rstrip().split('\t')[:6]
        index_c = int(index_c)
        max_degree, max_l = int(max_degree), int(max_l)

        tree_data.setdefault(eid, {})[index_c] = {'parent': index_p, 'max_degree': max_degree, 'maxL': max_l, 'vec': vec}

    file.close()

    print(f"Number of trees loaded: {len(tree_data)}")
    return tree_data

# Provide the path to your file
file_path = '/content/drive/MyDrive/ProjectData/data.TD_RvNN.vol_5000.txt'

# Call the function with the file path
tree_data = read_twitter_tree(file_path)

import pandas as pd
import numpy as np
import datetime
from dateutil import rrule

def get_months(create_at):
    create_at = create_at.split(' ')[0]  # Extracting only the date part
    date_i = datetime.datetime.strptime(create_at, '%Y-%m-%d').date()  # Converting to date object

    # Define the current date
    today = datetime.datetime.now().date()

    # Calculate the number of months between the account creation date and the current date
    month_sep = rrule.rrule(rrule.MONTHLY, dtstart=date_i, until=today)
    return month_sep.count()

def get_edge_index(length):
    edge_index = []
    for i in range(length):
        if i == 0:
            continue
        edge_index.append([0, i])
        edge_index.append([i, 0])
    return edge_index

def extract_user_features(data):
    user_features = []
    for index, row in data.iterrows():
        user_id = row['user_id']
        user_status = row['user_status']
        if user_status == 0:  # Non-existent accounts
            features = [user_id] + [-1] * 9 + [user_status]
        else:
            features = [row['user_id'], row['url'], row['protected'], row['verified'],
                        row['followers_count'], row['friends_count'],
                        row['listed_count'], row['favourites_count'],
                        row['statuses_count'], get_months(row['created_at']),
                        row['user_status']]
        user_features.append(features)
    return user_features

def extract_friend_features(data):
    friend_features = []
    friends_list = data[data['reason'].isna()]['friend_id'].unique().tolist()
    for friend_id in friends_list:
        friend_data = data[data['friend_id'] == friend_id].iloc[0, :]
        friend_status = friend_data['user_status']
        if friend_status == 0:  # Non-existent accounts
            features = [friend_id] + [-1] * 9 + [friend_status]
        else:
            features = [friend_data['friend_id'], friend_data['url'], friend_data['protected'], friend_data['verified'],
                        friend_data['followers_count'], friend_data['friends_count'],
                        friend_data['listed_count'], friend_data['favourites_count'],
                        friend_data['statuses_count'], get_months(friend_data['created_at']),
                        friend_data['user_status']]
        friend_features.append(features)
    return friend_features

def normalize_features(data):
    cols = ['url', 'protected', 'verified', 'followers_count', 'friends_count',
            'listed_count', 'favourites_count', 'statuses_count', 'created_at', 'user_status']

    for col in cols:
        mean = np.mean(data[col])
        std = np.std(data[col])
        if std:
            data[col] = data[col].apply(lambda x: (x - mean) / std)
    return data

def main(obj):
    user_info = pd.read_csv('/content/drive/MyDrive/ProjectData/' + obj + '_User_Information.csv', sep='\t')
    user_friends = pd.read_csv('/content/drive/MyDrive/ProjectData/' + obj + '_User_Friends.csv', sep='\t')
    ego_relation = pd.read_csv('/content/drive/MyDrive/ProjectData/' + obj + '_Ego_Relationships.csv', sep='\t')

    user_features = extract_user_features(user_info)
    friend_features = extract_friend_features(user_friends)

    total_feature = user_features + friend_features
    data_total = pd.DataFrame(total_feature,
                              columns=['user_id', 'url', 'protected', 'verified', 'followers_count', 'friends_count',
                                       'listed_count', 'favourites_count', 'statuses_count',
                                       'created_at', 'user_status'])

    data_total = normalize_features(data_total)
    # Initialize a list to store dictionaries containing features
    standard_list = []

    # Assuming data_3['user_id'] is used to create user_id_friend
    user_id_friend = ego_relation['user_id'].unique().tolist()

    for index, row in user_info.iterrows():
        twitter_id = row['twitter_id']
        user_id = row['user_id']
        tree_feature = []

        # Construct root feature
        user_data = data_total[data_total['user_id'] == user_id].iloc[0]
        root_feature = [
            float(user_data['url']), float(user_data['protected']), float(user_data['verified']),
            float(user_data['followers_count']), float(user_data['friends_count']),
            float(user_data['listed_count']), float(user_data['favourites_count']),
            float(user_data['statuses_count']), float(user_data['created_at']),
            float(user_data['user_status'])
        ]
        tree_feature.append(root_feature)

        # Checking if user_id is in user_id_friend
        if user_id in user_id_friend:
            friend_ids = user_friends[(user_friends['user_id'] == user_id) & (user_friends['user_status'] == 1)]['friend_id'].tolist()
            for friend_id in friend_ids:
                friend_data = data_total[data_total['user_id'] == friend_id].iloc[0]
                temp_feature = [
                    float(friend_data['url']), float(friend_data['protected']), float(friend_data['verified']),
                    float(friend_data['followers_count']), float(friend_data['friends_count']),
                    float(friend_data['listed_count']), float(friend_data['favourites_count']),
                    float(friend_data['statuses_count']), float(friend_data['created_at']),
                    float(friend_data['user_status'])
                ]
                tree_feature.append(temp_feature)

            # Create dictionary to store features for this user
            temp_dict = {
                'twitter_id': twitter_id,
                'user_id': str(user_id),
                'root_feature': np.array(root_feature),
                'tree_feature': np.array(tree_feature),
                'edge_index': np.array(get_edge_index(len(tree_feature))),
                'root_index': 0,
                'tree_len': len(tree_feature),
                'status': True
            }
            standard_list.append(temp_dict)
        else:
            # User_id not found in user_id_friend
            temp_dict = {
                'twitter_id': twitter_id,
                'user_id': str(user_id),
                'root_feature': np.array(root_feature),
                'tree_feature': np.array(tree_feature),
                'edge_index': np.array(get_edge_index(len(tree_feature))),
                'root_index': 0,
                'tree_len': len(tree_feature),
                'status': False
            }
            standard_list.append(temp_dict)
    return standard_list

total_graph = main('Twitter15')

for i, dic_sample in enumerate(total_graph):
            file_path = f'/content/drive/MyDrive/ProjectData/Ego_graph/{dic_sample["twitter_id"]}.npz'  # Replace with your desired folder path
            np.savez(file_path,
                    user_id=dic_sample['user_id'],
                    root_feature=dic_sample['root_feature'],
                    tree_feature=dic_sample['tree_feature'],
                    edge_index=dic_sample['edge_index'],
                    root_index=dic_sample['root_index'],
                    tree_len=dic_sample['tree_len'],
                    status=dic_sample['status'])

len(total_graph)

import os
import numpy as np
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

class Node_tweet:
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

def str2matrix(Str):
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq, index = map(float, pair.split(':'))
        if index <= 5000:
            wordFreq.append(freq)
            wordIndex.append(int(index))
    return wordFreq, wordIndex

def constructMat(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        if indexP != 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        else:
            root_index = nodeC.index
            root_word = nodeC.word
    rootfeat = np.zeros([1, 5000])
    if root_index:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    matrix = np.zeros([len(index2node), len(index2node)])
    row, col = [], []
    x_word, x_index = [], []
    for index_i, node_i in enumerate(index2node.values()):
        for index_j, node_j in enumerate(index2node.values()):
            if node_i.children and node_j in node_i.children:
                matrix[index_i][index_j] = 1
                row.append(index_i)
                col.append(index_j)
        x_word.append(node_i.word)
        x_index.append(node_i.index)
    edgematrix = [row, col]
    return x_word, x_index, edgematrix, rootfeat

def getfeature(x_word, x_index):
    x = np.zeros([len(x_index), 5000])
    for i, index in enumerate(x_index):
        if index:
            x[i, np.array(index)] = np.array(x_word[i])
    return x

def loadEid(tree, id, label):
    if tree is None or len(tree) < 2:
        return None
    x_word, x_index, tree, rootfeat = constructMat(tree)
    x_x = getfeature(x_word, x_index)
    rootfeat, tree, x_x = np.array(rootfeat), np.array(tree), np.array(x_x)
    np.savez(f'/content/drive/MyDrive/ProjectData/Interaction_graph/{id}.npz', x=x_x, root=rootfeat, edgeindex=tree, y=label)

def main(obj):
    treePath = f'/content/drive/MyDrive/ProjectData/data.TD_RvNN.vol_5000.txt'
    labelPath = f'/content/drive/MyDrive/ProjectData/{obj}_label_All.txt'

    treeDic = {}
    for line in open(treePath):
        eid, indexP, indexC, max_degree, maxL, Vec = line.rstrip().split('\t')
        if eid not in treeDic:
            treeDic[eid] = {}
        treeDic[eid][int(indexC)] = {'parent': indexP, 'max_degree': int(max_degree), 'maxL': int(maxL), 'vec': Vec}

    event, labels = [], {}
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    for line in open(labelPath):
        label, eid = line.rstrip().split('\t')[0], line.rstrip().split('\t')[2].lower()
        event.append(eid)
        if label in labelset_nonR:
            labels[eid] = 0
        elif label in labelset_f:
            labels[eid] = 1
        elif label in labelset_t:
            labels[eid] = 2
        elif label in labelset_u:
            labels[eid] = 3

    Parallel(n_jobs=-1, backend='threading')(delayed(loadEid)(treeDic[eid], eid, labels[eid]) for eid in tqdm(event))

if __name__ == '__main__':
    obj = 'Twitter15'  # Update 'Your_Object_Name' with the specific dataset name
    main(obj)

pip install torch-scatter

pip install torch-geometric

pip install torch-spline-conv

pip install torch-cluster

pip install torch-sparse

pip install --upgrade torch torchvision torchaudio torchtext torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-$(echo $TORCH_VERSION | cut -f1-3 -d'.').html



import torch
from torch.utils.data import Dataset


def root_edge_enhance(row, col):
    c = set(row).union(set(col))
    sorted_list = sorted(c)

    if sorted_list[0] != 0:
        return row, col

    new_row = []
    new_col = []
    for element in sorted_list[1:]:
        new_row.append(0)
        new_col.append(element)

    indices_row = [index for index, value in enumerate(row) if value == 0]
    row = [row[i] for i in range(len(row)) if i not in indices_row]
    col = [col[i] for i in range(len(col)) if i not in indices_row]

    indices_col = [index for index, value in enumerate(col) if value == 0]
    row = [row[i] for i in range(len(row)) if i not in indices_col]
    col = [col[i] for i in range(len(col)) if i not in indices_col]

    row.extend(new_row)
    col.extend(new_col)
    return row, col

class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, dataname, lower=2, upper=100000, droprate=0):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.graph_path = '/content/drive/MyDrive/ProjectData/' + dataname + 'Interaction_graph/'
        self.ego_path = '/content/drive/MyDrive/ProjectData/' + dataname + 'Ego_graph/'
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]

        # Load graph data
        data = np.load(self.graph_path + str(id) + '.npz', allow_pickle=True)
        edgeindex = data['edgeindex']

        # Enhance edge connections
        row, col = root_edge_enhance(list(edgeindex[0]), list(edgeindex[1]))
        burow = list(col)
        bucol = list(row)
        row.extend(burow)
        col.extend(bucol)

        # Apply droprate if required
        if self.droprate > 0:
            length = len(row)
            poslist = np.random.choice(length, int(length * (1 - self.droprate)), replace=False)
            row = [row[i] for i in poslist]
            col = [col[i] for i in poslist]

        new_edgeindex = [row, col]

        # Load ego-centric network data
        user_ego = np.load(self.ego_path + str(id) + '.npz', allow_pickle=True)
        ego_twitter_id = id
        ego_root_feature = np.array(eval(str(user_ego['root_feature'])))
        ego_tree_feature = np.array(eval(str(user_ego['tree_feature'])))
        ego_edge_index = np.array(eval(str(user_ego['edge_index'])))
        ego_root_index = user_ego['root_index']

        # Prepare and return PyTorch geometric Data objects
        return (
            Data(
                x=torch.tensor(data['x'], dtype=torch.float32),
                edge_index=torch.LongTensor(new_edgeindex),
                y=torch.LongTensor([int(data['y'])]),
                root=torch.LongTensor(data['root']),
                rootindex=torch.LongTensor([int(data['rootindex'])])
            ),
            Data(
                x=torch.tensor(ego_tree_feature, dtype=torch.float32),
                edge_index=torch.LongTensor(ego_edge_index),
                y=torch.LongTensor([int(data['y'])]),
                root=torch.LongTensor([ego_root_feature]),
                rootindex=torch.LongTensor([int(ego_root_index)]),
                tree_text_id=torch.LongTensor([int(ego_twitter_id)])
            )
        )

def loadUdData(dataname, treeDic, fold_x_train, fold_x_test, droprate):
    print("Loading train set...")
    traindata_list = UdGraphDataset(fold_x_train, treeDic, dataname=dataname, droprate=droprate)
    print("Number of samples in the train set:", len(traindata_list))

    print("\nLoading test set...")
    testdata_list = UdGraphDataset(fold_x_test, treeDic, dataname=dataname, droprate=droprate)
    print("Number of samples in the test set:", len(testdata_list))

    return traindata_list, testdata_list

class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, accs, model, modelname, str):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
        elif score < self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.accs = accs
            self.counter = 0

    def save_checkpoint(self, val_loss, model, modelname, str):
        if val_loss < self.val_loss_min:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
            torch.save(model.state_dict(), modelname + str + '.m')
            self.val_loss_min = val_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, init_size, hidden_size, output_size):
        super(Attention, self).__init__()

        self.init_size = init_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.line_q = nn.Linear(self.init_size, self.hidden_size, bias=False)
        self.line_k = nn.Linear(self.init_size, self.hidden_size, bias=False)
        self.line_v = nn.Linear(self.init_size, self.output_size, bias=False)

    def forward(self, query, key, value, mask=None, dropout=None):
        query = self.line_q(query)
        key = self.line_k(key)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = F.dropout(p_attn, p=dropout, training=self.training)

        return self.line_v(torch.matmul(p_attn, value))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv
import copy

class Attention(nn.Module):
    def __init__(self, init_size, hidden_size, output_size):
        super(Attention, self).__init__()
        self.init_size = init_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.line_q = nn.Linear(self.init_size, self.hidden_size, bias=False)
        self.line_k = nn.Linear(self.init_size, self.hidden_size, bias=False)
        self.line_v = nn.Linear(self.init_size, self.output_size, bias=False)

    def forward(self, query, key, value, mask=None, dropout=None):
        query = self.line_q(query)
        key = self.line_k(key)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).to(query.device))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = F.dropout(p_attn, p=dropout, training=self.training)

        return self.line_v(torch.matmul(p_attn, value))


class EgoEncoder(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(EgoEncoder, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.dropout_rate = 0.2
        self.w1 = nn.Linear(hid_feats * 3, hid_feats * 3)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(x.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (data.batch == num_batch)
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(x.device)
        for num_batch in range(batch_size):
            index = (data.batch == num_batch)
            root_extend[index] = x2[rootindex[num_batch]]
        x = torch.cat((x, root_extend, x2), 1)
        x = scatter_mean(F.relu(self.w1(x)), data.batch, dim=0)
        return x


class InteractionEncoder(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(InteractionEncoder, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.w1 = nn.Linear(hid_feats * 3, hid_feats * 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(x.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (data.batch == num_batch)
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(x.device)
        for num_batch in range(batch_size):
            index = (data.batch == num_batch)
            root_extend[index] = x2[rootindex[num_batch]]
        x = torch.cat((x, root_extend, x2), 1)
        x = scatter_mean(F.relu(self.w1(x)), data.batch, dim=0)
        return x


class RDMSC(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, atten_out_dim):
        super(RDMSC, self).__init__()
        self.EgoEncoder = EgoEncoder(10, hid_feats, out_feats)
        self.InteractionEncoder = InteractionEncoder(in_feats, hid_feats, out_feats)
        self.Atten = Attention(hid_feats * 3, hid_feats * 3, atten_out_dim)
        self.fc = nn.Linear(hid_feats * 6 + atten_out_dim, 4)

    def forward(self, data, data2):
        EE_x = self.EgoEncoder(data2)
        IE_x = self.InteractionEncoder(data)
        query = copy.deepcopy(IE_x.detach())
        key = value = copy.deepcopy(EE_x.detach())
        Attn = self.Atten(query=query, key=key, value=value, dropout=0.5)
        x = torch.cat((IE_x, EE_x, Attn),1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

