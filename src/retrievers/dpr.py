import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.retrievers.base_retriever import Retriever
from src.retrievers.models.encoder import TransformerEncoder, TOKENIZER
from src.training.trainer import resolve_device


class DPR(Retriever):
    def __init__(
        self,
        cfg,
        documents: list[dict] = None,
        index_path: str = None,
        query_encoder: nn.Module = None,
        document_encoder: nn.Module = None,
    ) -> None:
        super(DPR, self).__init__(documents, index_path)
        self.cfg = cfg
        self.device = resolve_device(cfg.get("device", "cpu"))
        self.batch_size = cfg.model.get("batch_size", 32)
        self.max_length = cfg.model.get("max_length", 128)

        if query_encoder is not None and document_encoder is not None:
            self.query_encoder = query_encoder.to(self.device)
            self.document_encoder = document_encoder.to(self.device)
        else:
            self.__load_models()

        self.query_encoder.eval()
        self.document_encoder.eval()

        # Pre-compute dense vector index for all documents in the corpus
        self.doc_embeddings = self.EncodeDocuments()

    def __load_models(self):
        self.query_encoder = TransformerEncoder(self.cfg)
        self.document_encoder = TransformerEncoder(self.cfg)

        default_dir = os.path.join(self.cfg.get("base_path", "."), "checkpoints")
        dataset_name = self.cfg.dataset.name
        pooling_type = self.cfg.model.pooling_type

        query_weights_path = self.cfg.model.get("query_encoder_weights_path", None) or os.path.join(
            default_dir, f"{dataset_name}_{pooling_type}_query_encoder.pt"
        )
        document_weights_path = self.cfg.model.get("document_encoder_weights_path", None) or os.path.join(
            default_dir, f"{dataset_name}_{pooling_type}_document_encoder.pt"
        )

        if query_weights_path and os.path.exists(query_weights_path):
            state_dict = torch.load(query_weights_path, map_location=self.device, weights_only=True)
            self.query_encoder.load_state_dict(state_dict)

        if document_weights_path and os.path.exists(document_weights_path):
            state_dict = torch.load(document_weights_path, map_location=self.device, weights_only=True)
            self.document_encoder.load_state_dict(state_dict)

        self.query_encoder = self.query_encoder.to(self.device)
        self.document_encoder = self.document_encoder.to(self.device)

    @torch.no_grad()
    def _encode_texts(self, texts: list[str], encoder: nn.Module, desc: str = "Encoding") -> torch.Tensor:
        encoder.eval()
        embeddings = []
        for start_idx in tqdm(range(0, len(texts), self.batch_size), desc=desc, leave=False):
            batch_texts = texts[start_idx : start_idx + self.batch_size]
            tokens = TOKENIZER(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)
            batch_emb = encoder(input_ids, attention_mask)
            embeddings.append(batch_emb.cpu())
        return torch.cat(embeddings, dim=0)

    def EncodeDocuments(self) -> torch.Tensor:
        """Encode all corpus documents into a (num_documents, hidden_dim) tensor."""
        doc_texts = [doc.GetText() for doc in self.index.GetDocuments()]
        return self._encode_texts(doc_texts, self.document_encoder, desc="Building DPR Document Index")

    def CalculateScores(self, queries: list[str]) -> np.ndarray:
        """
        Calculate dot-product similarity scores between queries and all indexed documents.

        Returns:
            np.ndarray of shape (num_queries, num_documents)
        """
        query_embeddings = self._encode_texts(queries, self.query_encoder, desc="Encoding Queries")
        scores = query_embeddings @ self.doc_embeddings.T
        return scores.numpy()

