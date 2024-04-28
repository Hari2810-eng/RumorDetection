import sys, os
import pandas as pd 
from LoadTreeGraph import *
import torch as th
import torch.nn.functional as F
import numpy as np
from EarlyStopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from split import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import random
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from RDMSC import RDMSC
import argparse
import warnings

warnings.filterwarnings("ignore")

def save_metrics_to_table(metrics_list, save_path):
    columns = ['Epoch','Precision', 'Recall', 'F1 Score', 'Accuracy','Loss']
    metrics_df = pd.DataFrame(metrics_list, columns=columns)
    metrics_df.to_csv(save_path, index=False)

def plot_metrics(train_metrics, val_metrics, metric_name, save_path):
    plt.figure()
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, label='Train')
    plt.plot(epochs, val_metrics, label='Validation')
    plt.title(f'{metric_name} Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def compute_metrics(true_labels, predicted_labels):
    precision, recall, f1score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    accuracy = accuracy_score(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    return precision, recall, f1score, accuracy, conf_matrix

def collate_fn(data_list):
    return data_list

def run_model(treeDic, x_test, x_train, droprate, lr, weight_decay, patience, n_epochs, batchsize, in_feats, hid_feats,
              out_feats, atten_out_dim,
              dataname):
    model_state = th.load("C:\\Users\\priya\\Documents\\ProjectData\\" + dataname + "\\" + "RDMSC_best_score.m")
    model = RDMSC(in_feats, hid_feats, out_feats, atten_out_dim)  # Create an instance of your model
    model.load_state_dict(model_state)  # Load the state dict into the model
    model.to(device) 
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    traindata_list, testdata_list = loadUdData(dataname, treeDic, x_train, x_test, droprate)
    true_labels = []
    predicted_labels = []
    feature_IE = [] 
    feature_EE = []
    feature_BERT = []
    feature_Attn = []

    train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=1, collate_fn=collate_fn)
    test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=1, collate_fn=collate_fn)
    avg_loss = []
    avg_acc = []
    batch_idx = 0
    for Batch_data, Batch_data2, Batch_data3 in tqdm(train_loader):
        out_labels, EE_xt, IE_xt, data3t, Attnt = model(Batch_data, Batch_data2, Batch_data3)
        finalloss = F.nll_loss(out_labels, Batch_data.y)
        loss = finalloss
        optimizer.zero_grad()
        loss.backward()
        avg_loss.append(loss.item())
        optimizer.step()
        _, pred = out_labels.max(dim=-1)
        correct = pred.eq(Batch_data.y).sum().item()
        train_acc = correct / len(Batch_data.y)
        avg_acc.append(train_acc)
        true_labels.append(Batch_data.y.cpu().numpy())
        predicted_labels.append(pred.cpu().numpy())
        feature_IE.append(IE_xt.cpu().detach().numpy())  # Store the Interaction Encoder feature vectors
        feature_EE.append(EE_xt.cpu().detach().numpy())  # Store the Ego Encoder feature vectors
        feature_BERT.append(data3t.cpu().detach().numpy())  # Store the data3 feature vectors
        feature_Attn.append(Attnt.cpu().detach().numpy())
        batch_idx = batch_idx + 1
    true_labels = np.concatenate(true_labels)
    predicted_labels = np.concatenate(predicted_labels)
    train_losses.append(np.mean(avg_loss))
    train_accs.append(np.mean(avg_acc))
    train_precision, train_recall, train_f1, train_accuracy,_ = compute_metrics(true_labels, predicted_labels)         
    #print("Train_Losses{:.4f} | Train_Accuracy{:.4f}".format(np.mean(avg_loss),np.mean(avg_acc)))
        
    temp_val_losses = []
    temp_val_accs = []
    val_true_labels = []
    val_predicted_labels = []
    val_features = []  # List to store validation features
    val_labels = [] 
    model.eval()
    for Batch_data, Batch_data2, Batch_data3 in tqdm(test_loader):
        val_out, EE_xv, IE_xv, data3v, Attnv = model(Batch_data, Batch_data2, Batch_data3)           
        val_loss = F.nll_loss(val_out, Batch_data.y)
        temp_val_losses.append(val_loss.item())
        _, val_pred = val_out.max(dim=1)
        correct = val_pred.eq(Batch_data.y).sum().item()
        val_acc = correct / len(Batch_data.y)
        temp_val_accs.append(val_acc)
        val_true_labels.append(Batch_data.y.cpu().numpy())
        val_predicted_labels.append(val_pred.cpu().numpy())
        feature_IE.append(IE_xv.cpu().detach().numpy())  # Store the Interaction Encoder feature vectors
        feature_EE.append(EE_xv.cpu().detach().numpy())  # Store the Ego Encoder feature vectors
        feature_BERT.append(data3v.cpu().detach().numpy())  # Store the data3 feature vectors
        feature_Attn.append(Attnv.cpu().detach().numpy())  
    val_losses.append(np.mean(temp_val_losses))
    val_accs.append(np.mean(temp_val_accs))
    val_true_labels = np.concatenate(val_true_labels)
    val_predicted_labels = np.concatenate(val_predicted_labels)
    feature_IE = np.concatenate(feature_IE)
    feature_EE = np.concatenate(feature_EE)
    feature_BERT = np.concatenate(feature_BERT)
    feature_Attn = np.concatenate(feature_Attn)
    true_label = np.concatenate([true_labels, val_true_labels])
    predicted_label = np.concatenate([predicted_labels, val_predicted_labels])
    precision, recall, fscore, support = precision_recall_fscore_support(
            val_true_labels, val_predicted_labels, labels=[0, 1, 2, 3], average=None)
    _,_,_,_,conf = compute_metrics(val_true_labels, val_predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
    print(" Val_Loss {:.4f}| Val_Accuracy {:.4f}".format( np.mean(temp_val_losses), np.mean(temp_val_accs)))
    print("\n")
    features = np.concatenate((feature_IE, feature_EE, feature_BERT, feature_Attn), axis=1)

# Define column names for the DataFrame
    '''columns = ['IE_' + str(i) for i in range(feature_IE.shape[1])] + \
          ['EE_' + str(i) for i in range(feature_EE.shape[1])] + \
          ['BERT_' + str(i) for i in range(feature_BERT.shape[1])] + \
          ['Attn_' + str(i) for i in range(feature_Attn.shape[1])] + \
          ['True_Label']

    data = np.concatenate((features, true_label.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(data, columns=columns)
    file_path =  "C:\\Users\\priya\\Documents\\ProjectData\\" + dataname + "\\" + "RDVSC_features"
    df.to_csv(file_path, index=False)'''

    precision, recall, fscore, support = precision_recall_fscore_support(
            val_true_labels, val_predicted_labels, labels=[0, 1, 2], average=None)

    class_labels = {0: 'Non-rumor', 1: 'True-rumor', 2: 'False-rumor'}

    for label in class_labels:
        print(f"Metrics for class '{class_labels[label]}':")
        print("Precision:", precision[label])
        print("Recall:", recall[label])
        print("F-score:", fscore[label])
        print("Support:", support[label])
        print()

    return train_losses, val_losses, train_accs, val_accs


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    #print("seed:", seed)


parser = argparse.ArgumentParser(description='RDMSC')
parser.add_argument('--lr', default=0.0005, type=float, help='Learning Rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay coefficient')
parser.add_argument('--patience', default=10, type=int, help='Early Stopping')
parser.add_argument('--n_epochs', default=200, type=int, help='Training Epochs')
parser.add_argument('--batchsize', default=128, type=int, help='Batch Size')
parser.add_argument('--droprate', default=0.2, type=float, help='Randomly invalidate some edges')
parser.add_argument('--seed', default=11, type=int)
parser.add_argument('--in_feats', default=5000, type=int)
parser.add_argument('--hid_feats', default=64, type=int)
parser.add_argument('--out_feats', default=64, type=int)
parser.add_argument('--atten_out_dim', default=4, type=int)

args = parser.parse_args()


if __name__ == '__main__':
    set_seed(args.seed)
    datasetname = "Twitter16"  # Twitter15 Twitter16
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    test_set, train_set = loadSplitData(datasetname)
    treeDic = loadTree(datasetname)
    train_losses, val_losses, train_accs, val_accs = run_model(treeDic, test_set, train_set, args.droprate, args.lr,
                                                               args.weight_decay, args.patience, args.n_epochs,
                                                               args.batchsize, args.in_feats, args.hid_feats,
                                                               args.out_feats, args.atten_out_dim, datasetname)

    print("Accuracy:{}".format(max(val_accs)))