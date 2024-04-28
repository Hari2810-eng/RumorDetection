import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pickle
import numpy as np
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from torch.nn import Module
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, dataname, lower=2, upper=100000, droprate=0, use_word2vec=False):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.graph_path = "C:\\Users\\priya\\Documents\\ProjectData\\" + dataname + "\\InteractionGraph\\"
        self.ego_path = "C:\\Users\\priya\\Documents\\ProjectData\\" + dataname + "\\EgoGraph\\"
        self.droprate = droprate
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.use_word2vec = use_word2vec
        #self.result_dataset = torch.load("C:\\Users\\priya\\Documents\\ProjectData\\" + dataname + "\\result_list.pt")

        if self.use_word2vec:
            self.result_dataset = torch.load("C:\\Users\\priya\\Documents\\ProjectData\\" + dataname + "\\result_list_word2vec.pt")

        else:
            # Use BERT embeddings
            self.result_dataset = torch.load("C:\\Users\\priya\\Documents\\ProjectData\\" + dataname + "\\result.pt")
        
    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(self.graph_path + str(id) + '.npz', allow_pickle=True)
        edgeindex = data['edgeindex']

        row,col = root_edge_enhance(list(edgeindex[0]),list(edgeindex[1]))

        burow = list(col)
        bucol = list(row)
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]

        user_ego = np.load(self.ego_path + str(id) + '.npz', allow_pickle=True)
        ego_twitter_id = id
        ego_root_feature = np.array(eval(str(user_ego['root_feature'])))
        ego_tree_feature = np.array(eval(str(user_ego['tree_feature'])))
        ego_edge_index = np.array(eval(str(user_ego['edge_index'])))
        ego_root_index = user_ego['root_index']
        # ego_user_id = user_ego[5]

        embedding_entry = next((entry for entry in self.result_dataset if entry['tweet_id'] == str(id)), None)
        embedding = embedding_entry['embedding'] if embedding_entry else None
        
        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])])) , \
               Data(x=torch.tensor(ego_tree_feature, dtype=torch.float32),
                    edge_index=torch.LongTensor(ego_edge_index), y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor([ego_root_feature]),
                    rootindex=torch.LongTensor([int(ego_root_index)]), tree_text_id=torch.LongTensor([int(ego_twitter_id)])), torch.Tensor(embedding)

def root_edge_enhance(row,col): 
    c = set(row).union(set(col))
    sorted_list = sorted(c)

    if sorted_list[0] != 0:
        return row,col

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
    return row,col