## AI6103 Project: Linear Attention in Transformers: A Study on Backward Compatibility
- This repo is contains training scripts on our experiments with swapping full softmax attention with linear attention
## Requirements

- The Fast-Transformers library is required: ![Fast-Transformers](https://github.com/idiap/fast-transformers)

```
    git clone https://github.com/idiap/fast-transformers.git
    cd fast-transformers
    python setup.py install
```

## How to run:
1. Download the dataset from kaggle: https://www.kaggle.com/datasets/shahir/protein-data-set
2. Run data_explore.ipynb to convert the protein dataset into processed values named protein_processed.npy
3. Run the appropriate train_*.py script
4. tensorboard logs will be stored under "./tensorboard_logs" and weights will be stored under "./weights"
```
    python train_full.py
```
## Sample code on swapping attention layers after training:
- In the model that we trained, 6 encoder layers are used
- The code below shows how we can replace the last 3 encoders with linear attention
```python

    from fast_transformers.attention import FullAttention,LinearAttention
    import torch
    from protein_dataset import Protein_Dataset,collate_fn
    from torch.utils.data import DataLoader
    from classifier import Protein_Classifier2,Protein_Classifier_LoRA
    from torchvision.ops.focal_loss import sigmoid_focal_loss
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score
    import numpy as np

    N_BATCH =512

    torch.manual_seed(0)

    dataset_val=Protein_Dataset(split="val")

    int2clss = dataset.int2clss

    val_dataloader=DataLoader(dataset,batch_size=N_BATCH,collate_fn=collate_fn)

    layers = [FullAttention,FullAttention,FullAttention,LinearAttentionLinearAttention,LinearAttention]
    model = Protein_Classifier2(layers=layers,dim=256,n_layers=6,n_heads=8,dim_feedfwd=512,causal=False)
    D = torch.load("./weights/full_model.pth")["params"]
    model.load_state_dict(D,strict=False)
    model.cuda()
    
    for src, tgt in tqdm(val_dataloader):
        with torch.no_grad():
            src=src.cuda()
            tgt=tgt.cuda()
            y_oh = F.one_hot(tgt,33)
            logits = model(src)
            loss = sigmoid_focal_loss(logits,y_oh.float()).mean()
```