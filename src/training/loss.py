import torch
import torch.nn as nn


class DPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, queries: torch.tensor, documents: torch.tensor):
        """
        @param queries: batch_size x embedding_dim
        @param documents: batch_size x embedding_dim

        Computes in-batch negative log-likelihood loss contrasting each query Q_i
        against all documents D_j in the batch:
            -log( exp(D_i^T Q_i) / sum_j exp(D_j^T Q_i) )

        Returns: A scalar torch tensor
        """
        # Row i corresponds to query Q_i, column j corresponds to document D_j
        scores = queries @ documents.T
        return -(scores.diag() - torch.logsumexp(scores, dim=1)).mean()
