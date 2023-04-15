from fast_transformers.attention import FullAttention,LinearAttention,CausalLinearAttention
import torch
from protein_dataset import Protein_Dataset,collate_fn
from torch.utils.data import DataLoader
from classifier import Protein_Classifier_LoRA
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch.nn.functional as F
from pynvml import *
from sklearn.metrics import average_precision_score
import numpy as np


EPOCHS = 30
N_BATCH = 128

if __name__=="__main__":
    

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    torch.manual_seed(0)
    
    dataset=Protein_Dataset(split="train")
    dataset_val=Protein_Dataset(split="val")
    
    int2clss = dataset.int2clss
    
    train_dataloader=DataLoader(dataset,batch_size=N_BATCH,collate_fn=collate_fn,shuffle=True)
    val_dataloader=DataLoader(dataset,batch_size=N_BATCH,collate_fn=collate_fn)
    
    writer = SummaryWriter(logdir = f"./tensorboard_logs/LoRA_32")


    model = Protein_Classifier_LoRA(layer=LinearAttention,dim=256,n_layers=6,n_heads=8,dim_feedfwd=512,causal=False,r=32)
    D = torch.load("./weights/full_model.pth")["params"]
    model.load_state_dict(D,strict=False)
    model.cuda()
    
    
    LoRA_Params = [param for n,param in model.named_parameters() if "LoRA_adapter" in n]
    for n,param in model.named_parameters():
        if ("LoRA_adapter" in n):
            param.requires_grad = True
        else:
            param.requires_grad = False

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(LoRA_Params, lr=1e-04 ,weight_decay=1e-04)

    # model.train()
    
    itr=0
    for e in tqdm(range(EPOCHS)):
        training_loss = 0
        training_acc = 0
        training_samples = 0
        model.train()
        for src, tgt in (train_dataloader):
            src=src.cuda()
            tgt=tgt.cuda()
            
            logits = model(src)
            y_oh = F.one_hot(tgt,33)

            optimizer.zero_grad()
            loss = sigmoid_focal_loss(logits,y_oh.float()).mean()
            loss.backward()
            optimizer.step()
            
            itr+=1
            with torch.no_grad():
                equals = (logits.sigmoid().argmax(1)==tgt).reshape(-1,1).detach().cpu()
                training_acc += torch.sum(equals.type(torch.FloatTensor)).item()
                training_loss += src.shape[0] * loss.item()
                training_samples += src.shape[0]
            if itr%50:
                train_loss = (training_loss/training_samples)
                train_acc = (training_acc/training_samples)
                writer.add_scalar("train/acc", train_acc, itr)
                writer.add_scalar("train/train_loss", train_loss, itr)


                info = nvmlDeviceGetMemoryInfo(handle)
                use = nvmlDeviceGetUtilizationRates(handle).gpu
                writer.add_scalar("GPU/mem_used",info.used/1e06,itr)
                writer.add_scalar("GPU/mem_utilization",info.used/info.total,itr)
                writer.add_scalar("GPU/gpu_utilization",use,itr)

        train_loss = (training_loss/training_samples)
        train_acc = (training_acc/training_samples)    
        writer.add_scalar("train/train_loss_e", train_loss, itr)   
        writer.add_scalar("train/train_acc_e", train_acc, itr)   
        
        val_loss = 0
        val_acc = 0
        val_samples = 0
        ys = []
        preds = []
        model.eval()

        for src, tgt in val_dataloader:
            with torch.no_grad():
                src=src.cuda()
                tgt=tgt.cuda()
                y_oh = F.one_hot(tgt,33)
                logits = model(src)
                loss = sigmoid_focal_loss(logits,y_oh.float()).mean()

                equals = (logits.sigmoid().argmax(1)==tgt).reshape(-1,1).detach().cpu()
                val_acc += torch.sum(equals.type(torch.FloatTensor)).item()
                val_loss += src.shape[0] * loss.item()
                val_samples += src.shape[0]
                
                ys.append(y_oh.cpu().numpy())
                preds.append(logits.softmax(dim=1).cpu().numpy())

        ys = np.vstack(ys)
        preds = np.vstack(preds)
        test_dict = {}
        
        aps = 0
        for i in range(ys.shape[1]):
            ap = average_precision_score(ys[:,i:i+1],preds[:,i:i+1])
            aps += ap
            test_dict[int2clss[i]] = ap
            
        writer.add_scalar(f"val_AP/mAP",aps/ys.shape[1],itr)    
        for k,v in test_dict.items():

            writer.add_scalar(f"val_AP/{k}_AP",v,itr)
        
        val_loss = (val_loss/val_samples)
        val_acc = (val_acc/val_samples)    
        writer.add_scalar("val/val_loss_e", val_loss, itr)   
        writer.add_scalar("val/val_acc_e", val_acc, itr)
        
        model_dict = {"params":model.state_dict(),"itr":itr,"epoch":e}
        torch.save(model_dict,"./weights/LoRA_32.pth")