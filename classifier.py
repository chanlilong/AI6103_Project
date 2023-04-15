from f_transformers import Encoder,Encoder2,Encoder_LoRA
from fast_transformers.attention import FullAttention,LinearAttention,CausalLinearAttention
import torch

class Protein_Classifier_LoRA(torch.nn.Module):
    def __init__(self,layer=FullAttention,dim=256,n_layers=6,n_heads=8,dim_feedfwd=1024,causal=False,r=4):
        super().__init__()
        self.enc = Encoder_LoRA(layer,n_layers=n_layers,n_dim=dim,n_heads=n_heads,dim_feedfwd=dim_feedfwd,causal=causal,r=r)
        self.emb = torch.nn.Embedding(26,dim)
        self.class_head = torch.nn.Linear(dim,33)
    def forward(self,x):
        x = self.emb(x)
        x = self.enc(x)
        x =self.class_head(x.max(1).values)
        
        return x
    
class Protein_Classifier(torch.nn.Module):
    def __init__(self,layer=FullAttention,dim=256,n_layers=6,n_heads=8,dim_feedfwd=1024,causal=False):
        super().__init__()
        self.enc = Encoder(layer,n_layers=n_layers,n_dim=dim,n_heads=n_heads,dim_feedfwd=dim_feedfwd,causal=causal)
        self.emb = torch.nn.Embedding(26,dim)
        self.class_head = torch.nn.Linear(dim,33)
    def forward(self,x):
        x = self.emb(x)
        x = self.enc(x)
        x =self.class_head(x.max(1).values)
        
        return x
    
class Protein_Classifier2(torch.nn.Module):
    def __init__(self,layers=[FullAttention,FullAttention,FullAttention],dim=256,n_layers=3,n_heads=8,dim_feedfwd=512,causal=False):
        super().__init__()
        self.enc = Encoder2(layers,n_layers=n_layers,n_dim=dim,n_heads=n_heads,dim_feedfwd=dim_feedfwd,causal=causal)
        self.emb = torch.nn.Embedding(26,dim)
        self.class_head = torch.nn.Linear(dim,33)
    def forward(self,x):
        x = self.emb(x)
        x = self.enc(x)
        x =self.class_head(x.max(1).values)
        
        return x
    
    
# class Protein_Classifier(tf.keras.Model):
#     def __init__(self,n_class=33,n_vocab=26,dim=128):
#         super().__init__()
#         self.emb = Embedding(n_vocab,dim)
#         self.xformer = Linear_Transformer_Encoder(n_enc_layers=6,ffn_dim=512,dim=dim,return_inter = True,causal=False)
#         self.linear = tf.keras.layers.Dense(n_class)
        
#     def call(self,x,return_QK=False):
#         x = self.emb(x)
#         x,QK = self.xformer(x,return_QK)
        
#         clss = []
#         # print(x.shape)#(32, 512, 128)
#         for l in x:
#             clss.append(self.linear(tf.reduce_max(l,axis=1)))
        
#         return clss,QK