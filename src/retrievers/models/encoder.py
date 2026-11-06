import os
import hydra
from itertools import chain
import transformers
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, log, inf
import tokenizers
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
import torch.optim as optim

from src.data_utils.corpus import Corpus
from src.data_utils.query import Query
from src.data_utils.dataset import QueryDocsDataset

TOKENIZER = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
VOCAB_SIZE = TOKENIZER.vocab_size

class Encoder1Block(nn.Module):
    def __init__(self, emb_dim, n_heads, feed_forward_dim, dropout = 0.1):
        super().__init__()
    
        self.multihead = MultiHeadedAttention(n_heads, emb_dim, dropout=0.1)

        self.linear = nn.Sequential(nn.Linear(emb_dim, feed_forward_dim), nn.ReLU(), nn.Linear(feed_forward_dim, emb_dim))

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        atten = self.multihead(x,x,x,mask)
        x = x + self.dropout((self.norm1(atten)))
        ff = self.linear(x)
        x = x + self.dropout(self.norm2(ff))
        return x

    
class TransformerEncoder(nn.Module):
    def __init__(self, cfg = None):
        super().__init__()        
        self.cfg = cfg
        self.n_layers = self.cfg.model.n_layers
        self.n_heads = self.cfg.model.n_heads
        self.emb_dim =  self.cfg.model.emb_dim
        self.hidden_dim = self.cfg.model.hidden_dim
        self.feed_forward_dim = self.cfg.model.feed_forward_dim
        self.vocab_size = VOCAB_SIZE
        self.setup_model()


    def setup_model(self):
        """
        Instantiate the model.
        """
        
        self.embed = nn.Embedding(self.vocab_size, self.emb_dim) 
        self.position_encode= PositionalEncoding(self.emb_dim, dropout = 0.1)

        self.layers = nn.ModuleList([Encoder1Block(self.emb_dim, self.n_heads, self.feed_forward_dim) for _ in range(self.n_layers)]) # get n_layers of weights
        self.linear_hidden = nn.Linear(self.emb_dim, self.hidden_dim)

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """
        Run a forward pass on the model
        """
        x = self.embed(x)
        x = self.position_encode(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.pool(x, mask)
        x = self.linear_hidden(x)

        return x

    def pool(self, x, mask):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.cfg.model.pooling_type == 'cls':
            return x[:,0,:]
        
        elif self.cfg.model.pooling_type == 'max':
            raise NotImplementedError()
        
        elif self.cfg.model.pooling_type == 'mean':
            mask = mask.unsqueeze(-1)
            return (x * mask).sum(dim=1) / mask.sum(dim=1)





class PositionalEncoding(nn.Module): #from lecture notebook #6
    "Implement the PE function." 
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -inf)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """A simple Multi-head attention layer."""
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None # store the attention maps
        self.dropout = nn.Dropout(p=dropout)
   
    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(1)


        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))] #here they apply the nn.Linear with dim (emb_dim x emb_dim) to the input matrices

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def loss_encoding(query_encoded, docs_encoded):
    num_matrix = (docs_encoded @ query_encoded.T)
    return -(num_matrix.diag() - torch.logsumexp(num_matrix, dim=1)).mean()
    



def collate_docs(batch):
    queries = [q for q, _ in batch]
    docs    = [d for _, d in batch]
        
    tok_query = TOKENIZER(
    queries,
    padding=True,
    truncation=True,     
    max_length=128,        
    return_tensors="pt",   
    )

    tok_docs = TOKENIZER(
    docs,
    padding=True,          
    truncation=True,      
    max_length=128,        
    return_tensors="pt",   
    )
    return tok_query, tok_docs


def train(cfg, corpus=None):
    """Train the query/doc encoders. Callable from main.py (pass its corpus to avoid reloading)."""
    if corpus is None:
        corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')

    batch_size = cfg.model.batch_size
    train_dataloader= DataLoader(QueryDocsDataset(corpus, "train"), batch_size = batch_size, shuffle = True, collate_fn=collate_docs)
    val_dataloader = DataLoader(QueryDocsDataset(corpus, "val"), batch_size = batch_size, shuffle = True, collate_fn=collate_docs)
    test_dataloader = DataLoader(QueryDocsDataset(corpus, "test"), batch_size = batch_size, shuffle = True, collate_fn=collate_docs)


    q_Encoder = TransformerEncoder(cfg)
    d_Encoder = TransformerEncoder(cfg)

    all_params = chain(q_Encoder.parameters(), d_Encoder.parameters())

    optimizer = optim.Adam(all_params, lr=0.0001, weight_decay = 0.00005)

    for epoch in range(cfg.model.epochs):
        i = 0
        for queries, docs in tqdm(train_dataloader):
            optimizer.zero_grad()

            queries_latent = q_Encoder(queries['input_ids'], queries['attention_mask'])

            docs_latent = d_Encoder(docs['input_ids'], docs['attention_mask'])

            batch_loss = loss_encoding(queries_latent, docs_latent)
            batch_loss.backward()
            optimizer.step()
            if i % 100 ==0:
                print(batch_loss)
            i += 1
        print(batch_loss)


@hydra.main(config_path='../../../config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    train(cfg)


if __name__ == "__main__":
    main()
    