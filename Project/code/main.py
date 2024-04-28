import sys, os
import pandas as pd 
from LoadTreeGraph import *
import torch as th
import torch.nn.functional as F
import numpy as np
from EarlyStopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from randomSplit import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import random
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
    return precision, recall, f1score, accuracy

def collate_fn(data_list):
    return data_list

def run_model(treeDic, x_test, x_train, droprate, lr, weight_decay, patience, n_epochs, batchsize, in_feats, hid_feats,
              out_feats, atten_out_dim,
              dataname):
    model = RDMSC(in_feats, hid_feats, out_feats, atten_out_dim).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    traindata_list, testdata_list = loadUdData(dataname, treeDic, x_train, x_test, droprate)

    train_metrics_df = pd.DataFrame(columns=['Epoch', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    test_metrics_df = pd.DataFrame(columns=['Epoch', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    val_metrics_df = pd.DataFrame(columns=['Epoch', 'Loss', 'Accuracy', 'Precision_0', 'Recall_0', 'F1 Score_0',
                                           'Precision_1', 'Recall_1', 'F1 Score_1',
                                           'Precision_2', 'Recall_2', 'F1 Score_2',
                                           'Precision_3', 'Recall_3', 'F1 Score_3'])


    

    for epoch in range(n_epochs):
        true_labels = []
        predicted_labels = []

        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=1, collate_fn=collate_fn)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=1, collate_fn=collate_fn)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        for Batch_data, Batch_data2, Batch_data3 in tqdm(train_loader):
            #Batch_data.to(device)
            #Batch_data2.to(device)
            out_labels = model(Batch_data, Batch_data2, Batch_data3)
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
            batch_idx = batch_idx + 1
        true_labels = np.concatenate(true_labels)
        predicted_labels = np.concatenate(predicted_labels)
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        train_precision, train_recall, train_f1, train_accuracy = compute_metrics(true_labels, predicted_labels)
        train_metrics_df = pd.concat([train_metrics_df, pd.DataFrame({'Epoch': epoch,
                                                             'Loss': np.mean(avg_loss),
                                                             'Accuracy': np.mean(avg_acc),
                                                             'Precision': train_precision,
                                                             'Recall': train_recall,
                                                             'F1 Score': train_f1}, index=[0])],
                             ignore_index=True)

        class_metrics = {
            'precision': np.zeros(4),
            'recall': np.zeros(4),
            'fscore': np.zeros(4)
        }
        print("Epoch {:05d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch,np.mean(avg_loss),np.mean(avg_acc)))
        
        temp_val_losses = []
        temp_val_accs = []
        val_true_labels = []
        val_predicted_labels = []
        model.eval()
        for Batch_data, Batch_data2, Batch_data3 in tqdm(test_loader):
            #Batch_data.to(device)
            #Batch_data2.to(device)
            val_out = model(Batch_data, Batch_data2, Batch_data3)
            
            val_loss = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            temp_val_accs.append(val_acc)
            val_true_labels.append(Batch_data.y.cpu().numpy())
            val_predicted_labels.append(val_pred.cpu().numpy())
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        val_true_labels = np.concatenate(val_true_labels)
        val_predicted_labels = np.concatenate(val_predicted_labels)

        precision, recall, fscore, support = precision_recall_fscore_support(
            val_true_labels, val_predicted_labels, labels=[0, 1, 2, 3], average=None)

        # Update class_metrics dictionary
        class_metrics['precision'] += precision
        class_metrics['recall'] += recall
        class_metrics['fscore'] += fscore

        test_precision, test_recall, test_f1, test_accuracy = compute_metrics(val_true_labels, val_predicted_labels)
        test_metrics_df = pd.concat([test_metrics_df, pd.DataFrame({'Epoch': epoch,
                                                             'Loss': np.mean(temp_val_losses),
                                                             'Accuracy': np.mean(temp_val_accs),
                                                             'Precision': test_precision,
                                                             'Recall': test_recall,
                                                             'F1 Score': test_f1}, index=[0])],
                             ignore_index=True)
        val_metrics_df = pd.concat([val_metrics_df, pd.DataFrame({
        'Epoch': epoch,
        'Loss': np.mean(temp_val_losses),
        'Accuracy': np.mean(temp_val_accs),
        'Precision_0': class_metrics['precision'][0],
        'Recall_0': class_metrics['recall'][0],
        'F1 Score_0': class_metrics['fscore'][0],
        'Precision_1': class_metrics['precision'][1],
        'Recall_1': class_metrics['recall'][1],
        'F1 Score_1': class_metrics['fscore'][1],
        'Precision_2': class_metrics['precision'][2],
        'Recall_2': class_metrics['recall'][2],
        'F1 Score_2': class_metrics['fscore'][2],
        'Precision_3': class_metrics['precision'][3],
        'Recall_3': class_metrics['recall'][3],
        'F1 Score_3': class_metrics['fscore'][3]
    }, index=[0])], ignore_index=True)

        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))
        print("\n")
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), model, 'RDMSC', dataname)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    show_val = list(val_accs)

    dt = datetime.now()
    save_time = dt.strftime('%Y_%m_%d_%H_%M_%S')

    save_metrics_to_table(train_metrics_df, "C:\\Users\\priya\\Documents\\ProjectData\\{}\\{}_train_metrics.csv".format(dataname, dataname))
    save_metrics_to_table(test_metrics_df, "C:\\Users\\priya\\Documents\\ProjectData\\{}\\{}_test_metrics.csv".format(dataname, dataname))
    val_metrics_df.to_csv("C:\\Users\\priya\\Documents\\ProjectData\\{}\\{}_val_metrics.csv".format(dataname, dataname), index=False)

    plot_metrics(train_metrics_df['Accuracy'], test_metrics_df['Accuracy'], 'Accuracy', "C:\\Users\\priya\\Documents\\ProjectData\\{}\\{}_accuracy_{}.png".format(dataname, dataname, save_time))

    plot_metrics(train_metrics_df['F1 Score'], test_metrics_df['F1 Score'], 'F1 Score', "C:\\Users\\priya\\Documents\\ProjectData\\{}\\{}_f1score_{}.png".format(dataname, dataname, save_time))

    plot_metrics(train_metrics_df['Loss'], test_metrics_df['Loss'], 'Loss', "C:\\Users\\priya\\Documents\\ProjectData\\{}\\{}_loss_{}.png".format(dataname, dataname, save_time))

    plot_metrics(train_metrics_df['Precision'], test_metrics_df['Precision'], 'Precision', "C:\\Users\\priya\\Documents\\ProjectData\\{}\\{}_precision_{}.png".format(dataname, dataname, save_time))

    plot_metrics(train_metrics_df['Recall'], test_metrics_df['Recall'], 'Recall', "C:\\Users\\priya\\Documents\\ProjectData\\{}\\{}_recall_{}.png".format(dataname, dataname, save_time))

    '''fig = plt.figure()
    plt.plot(range(1, len(train_accs) + 1), train_accs, color='b', label='train')
    plt.plot(range(1, len(show_val) + 1), show_val, color='r', label='dev')
    plt.grid()
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.xticks(np.arange(1, len(train_accs), step=4))
    fig.savefig("C:\\Users\\priya\\Documents\\ProjectData\\" + '{}\\{}_accuracy_{}.png'.format(dataname, dataname, save_time))

    fig = plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, color='b', label='train')
    plt.plot(range(1, len(val_losses) + 1), val_losses, color='r', label='dev')
    plt.grid()
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.xticks(np.arange(1, len(train_losses) + 1, step=4))
    fig.savefig("C:\\Users\\priya\\Documents\\ProjectData\\" + '{}_loss_{}.png'.format(dataname, save_time))'''

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

    print("Total_Best_Accuracy:{}".format(max(val_accs)))