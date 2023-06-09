from torch.nn import Linear, Module, Sequential
from fast_transformers.events import EventDispatcher,QKVEvent
import torch

class LoRA_Adapter(Module):
    
    def __init__(self,dim = 256,r=4):
        super().__init__()
        self.q_adapt = Sequential(Linear(dim,r,bias=False),Linear(r,dim,bias=False))
        self.k_adapt = Sequential(Linear(dim,r,bias=False),Linear(r,dim,bias=False))
        self.v_adapt = Sequential(Linear(dim,r,bias=False),Linear(r,dim,bias=False))
        self.o_adapt = Sequential(Linear(dim,r,bias=False),Linear(r,dim,bias=False))
        self.init_weights()
        self.d = {"q":self.q_adapt,"k":self.k_adapt,"v":self.v_adapt,"o":self.o_adapt}
        
    def init_weights(self):
        
        for m in [self.q_adapt,self.k_adapt,self.v_adapt,self.o_adapt]:
            if isinstance(m, Sequential):
                torch.nn.init.normal_(m[0].weight)
                torch.nn.init.zeros_(m[1].weight)

    def forward(self,x,w_type="q"):

        return self.d[w_type](x)
        
        
    
class AttentionLayer_LoRA(Module):
    """Implement the attention layer with option for LoRA. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.
    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.
    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality for the queries source
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        d_model_keys: The input feature dimensionality for keys source
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, d_model_keys=None, event_dispatcher="",r=4):
        super(AttentionLayer_LoRA, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        d_model_keys = d_model_keys or d_model
        
        self.LoRA_adapter = LoRA_Adapter(d_model,r=r)
        self.inner_attention = attention
        self.query_projection = Linear(d_model, d_keys * n_heads)
        self.key_projection = Linear(d_model_keys, d_keys * n_heads)
        self.value_projection = Linear(d_model_keys, d_values * n_heads)
        self.out_projection = Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.
        In the argument description we make use of the following sizes
            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'
        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = (self.query_projection(queries)+self.LoRA_adapter(queries,"q")).view(N, L, H, -1)
        keys = (self.key_projection(keys)+self.LoRA_adapter(keys,"k")).view(N, S, H, -1)
        values = (self.value_projection(values)+self.LoRA_adapter(values,"v")).view(N, S, H, -1)

        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)+self.LoRA_adapter(new_values,"o")