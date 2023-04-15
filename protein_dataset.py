from torch.utils.data import Dataset,DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class Protein_Dataset(Dataset):
    
    def __init__(self,split = "train"):
        self.data = np.load("protein_processed.npy",allow_pickle=True).item()
        
        self.int2clss = {v:k for k,v in self.data["class_dict"].items()}
        
        np.random.seed(42)
        n_data = self.data["seq"].shape[0]
        train_idx = np.random.choice(np.arange(n_data),size=int(0.8*n_data),replace=False)
        valtest_idx = np.array(list(set(np.arange(n_data))-set(train_idx)))
        val_idx = np.random.choice(valtest_idx,size=int(0.5*len(valtest_idx)),replace=False)
        test_idx = np.array(list(set(valtest_idx) - set(val_idx)))
        
        self.idx = {"train":train_idx,"val":val_idx,"test":test_idx}
        self.X = self.data["seq"][self.idx[split]]
        self.y = self.data["class"][self.idx[split]]
        
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        x = self.X[idx]
        y = self.y[idx]
        f = x!=0

        return x[f],y
    
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for x,y in batch:
        src_batch.append(torch.as_tensor(x,dtype=torch.int64).view(-1,1))
        tgt_batch.append(torch.as_tensor(y,dtype=torch.int64).view(-1,1))
    src_batch = pad_sequence(src_batch, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, padding_value=0)
    
    return src_batch.permute(1,0,2).squeeze(2), tgt_batch.squeeze()