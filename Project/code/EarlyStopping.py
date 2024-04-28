import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs=0
        self.val_loss_min = np.Inf
        self.best_model_weights = None

    def __call__(self, val_loss, accs,model,modelname,str):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
        elif score < self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_checkpoint(val_loss, model, modelname, str)
        else:
            self.best_score = score
            self.accs = accs
            self.counter = 0
            self.best_model_weights = model.state_dict()

    def save_checkpoint(self, val_loss, model,modelname,str):
        checkpoint_path = "C:\\Users\\priya\\Documents\\ProjectData\\" + str + "\\" + modelname + "_best_score" + ".m"
        torch.save(self.best_model_weights, checkpoint_path)        
        self.val_loss_min = val_loss