from src.data_utils.document import Document
from itertools import combinations
from datasets import DatasetDict, load_from_disk
from torch.utils.data import Dataset
from src.data_utils.query import Query
import numpy as np 
import tokenizers
import random
class DocumentDataset:
    def __init__(self, documents: list[dict]) -> None:
        self.documents: list[Document] = self.__BuildDocuments(documents)
        
    def __BuildDocuments(self, documents):
        
        return [Document(text=text, _id=_id) for entries in documents 
                    for (text, _id) in zip(entries['text'], entries['document_id'])
                ]
        
    def GetDocuments(self) -> list[Document]:
        return self.documents

class QueryDocsDataset(Dataset):
    def __init__(self, corpus, split_type):
        documents = corpus.all_documents
        self.docs = {doc.get('document_id')[0]: doc.get('text')[0] for doc in documents}
        
        if split_type == "train":
            self.queries = [
            Query(q_text, id=q_id, relevant_document_ids=corpus.relevant_ids[q_id])
            for q_text, q_id in zip(corpus.train['question'], corpus.train['query_id'])
        ]

        elif split_type == "val":
            self.queries = [
                Query(q_text, id=q_id, relevant_document_ids=corpus.relevant_ids[q_id])
                for q_text, q_id in zip(corpus.val['question'], corpus.val['query_id'])
            ]

        elif split_type == "test":
            self.queries = [
            Query(q_text, id=q_id, relevant_document_ids=corpus.relevant_ids[q_id])
            for q_text, q_id in zip(corpus.test['question'], corpus.test['query_id'])
            ]

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        query_text = query.text
        doc_id = random.choice(query.relevant_document_ids) #just one doc per query - simpler and over time all will get picked for each query
        

        return query_text,  self.docs[doc_id]







