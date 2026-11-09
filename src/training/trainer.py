import os
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.loss import DPRLoss
from src.data_utils.corpus import Corpus
from src.data_utils.dataset import QueryDocsDataset


def resolve_device(device_str: str = "cpu") -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


class Trainer:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.criterion = DPRLoss()

    def train(
        self,
        query_encoder: nn.Module,
        document_encoder: nn.Module,
        corpus: Corpus = None,
        collate_fn=None,
    ):
        if self.cfg is None:
            raise ValueError("Trainer requires a configuration object `cfg`.")

        if corpus is None:
            corpus = Corpus(
                dataset_name=self.cfg.dataset.name,
                base_dir=f"{self.cfg.base_path}/retrieval_data",
            )

        if collate_fn is None:
            from src.retrievers.models.encoder import collate_docs
            collate_fn = collate_docs

        device = resolve_device(self.cfg.get("device", "cpu"))
        query_encoder = query_encoder.to(device)
        document_encoder = document_encoder.to(device)

        batch_size = self.cfg.model.batch_size
        train_dataloader = DataLoader(
            QueryDocsDataset(corpus, "train"),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_dataloader = (
            DataLoader(
                QueryDocsDataset(corpus, "val"),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            if corpus.val is not None
            else None
        )

        lr = self.cfg.model.get("lr", 1e-4)
        weight_decay = self.cfg.model.get("weight_decay", 5e-5)
        all_params = nn.ModuleList([query_encoder, document_encoder]).parameters()
        optimizer = optim.Adam(all_params, lr=lr, weight_decay=weight_decay)

        epochs = self.cfg.model.epochs
        for epoch in range(epochs):
            query_encoder.train()
            document_encoder.train()
            total_train_loss = 0.0

            for step, (queries, docs) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")):
                optimizer.zero_grad()

                q_input_ids = queries["input_ids"].to(device)
                q_mask = queries["attention_mask"].to(device)
                d_input_ids = docs["input_ids"].to(device)
                d_mask = docs["attention_mask"].to(device)

                queries_latent = query_encoder(q_input_ids, q_mask)
                docs_latent = document_encoder(d_input_ids, d_mask)

                batch_loss = self.criterion(queries_latent, docs_latent)
                batch_loss.backward()
                optimizer.step()

                total_train_loss += batch_loss.item()
                if step % 100 == 0:
                    print(f"Epoch {epoch + 1} | Step {step} | Batch Loss: {batch_loss.item():.4f}")

            avg_train_loss = total_train_loss / max(len(train_dataloader), 1)
            msg = f"Epoch {epoch + 1}/{epochs} | Avg Train Loss: {avg_train_loss:.4f}"

            if val_dataloader is not None:
                query_encoder.eval()
                document_encoder.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for queries, docs in val_dataloader:
                        q_input_ids = queries["input_ids"].to(device)
                        q_mask = queries["attention_mask"].to(device)
                        d_input_ids = docs["input_ids"].to(device)
                        d_mask = docs["attention_mask"].to(device)

                        queries_latent = query_encoder(q_input_ids, q_mask)
                        docs_latent = document_encoder(d_input_ids, d_mask)
                        val_loss = self.criterion(queries_latent, docs_latent)
                        total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / max(len(val_dataloader), 1)
                msg += f" | Avg Val Loss: {avg_val_loss:.4f}"

            print(msg)

        # Save model weights
        default_dir = os.path.join(self.cfg.get("base_path", "."), "checkpoints")
        dataset_name = self.cfg.dataset.name
        pooling_type = self.cfg.model.pooling_type

        q_weights_path = self.cfg.model.get("query_encoder_weights_path", None) or os.path.join(
            default_dir, f"{dataset_name}_{pooling_type}_query_encoder.pt"
        )
        d_weights_path = self.cfg.model.get("document_encoder_weights_path", None) or os.path.join(
            default_dir, f"{dataset_name}_{pooling_type}_document_encoder.pt"
        )

        os.makedirs(os.path.dirname(q_weights_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(d_weights_path) or ".", exist_ok=True)
        torch.save(query_encoder.state_dict(), q_weights_path)
        torch.save(document_encoder.state_dict(), d_weights_path)
        print(f"Saved trained encoders to {q_weights_path} and {d_weights_path}")

        return query_encoder, document_encoder
