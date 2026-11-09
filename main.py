import os
import hydra
from src.data_utils.corpus import Corpus
from src.retrievers.tf_idf import TFIDF
from src.retrievers.bm25 import BM25
from src.retrievers.dpr import DPR
from src.data_utils.query import Query
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate
from src.retrievers.models.encoder import train as train_encoder


@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    ### Prepare queries
    train_queries = [
        Query(q_text, id=q_id, relevant_document_ids=corpus.relevant_ids[q_id])
        for q_text, q_id in zip(corpus.train['question'], corpus.train['query_id'])
    ]

    val_queries = [
        Query(q_text, id=q_id, relevant_document_ids=corpus.relevant_ids[q_id])
        for q_text, q_id in zip(corpus.val['question'], corpus.val['query_id'])
    ]
    
    if corpus.test is not None:
        test_queries = [
            Query(q_text, id=q_id, relevant_document_ids=corpus.relevant_ids[q_id])
            for q_text, q_id in zip(corpus.test['question'], corpus.test['query_id'])
        ]

    ### Initialize Retriever class
    method = cfg.retriever.method if hasattr(cfg, 'retriever') else "tf_idf"
    if method in ["tf_idf", "tfidf", "td-idf"]:
        retriever = TFIDF(documents=corpus.all_documents)
    elif method == "bm25":
        k1 = cfg.retriever.get("k1", 1.5)
        b = cfg.retriever.get("b", 0.75)
        retriever = BM25(documents=corpus.all_documents, k1=k1, b=b)
    elif method == "dpr":
        q_weights = cfg.model.get("query_encoder_weights_path", None)
        d_weights = cfg.model.get("document_encoder_weights_path", None)
        if q_weights and d_weights and os.path.exists(q_weights) and os.path.exists(d_weights) and not cfg.model.get("force_train", False):
            retriever = DPR(cfg, documents=corpus.all_documents)
        else:
            q_enc, d_enc = train_encoder(cfg, corpus)
            retriever = DPR(cfg, documents=corpus.all_documents, query_encoder=q_enc, document_encoder=d_enc)
    else:
        raise ValueError(f"Unknown retriever method: {method}")
    
    ### Retrieve top-k relevant queries
    train_relevancies = retrieveQueryAndGetRelevancies(retriever, train_queries, k=cfg.top_k)
    val_relevancies = retrieveQueryAndGetRelevancies(retriever, val_queries, k=cfg.top_k)
    if corpus.test is not None:
        test_relevancies = retrieveQueryAndGetRelevancies(retriever, test_queries, k=cfg.top_k)

    ### Evaluate
    train_scores = Evaluate(train_relevancies, train_queries)
    val_scores = Evaluate(val_relevancies, val_queries)
    if corpus.test is not None:
        test_scores = Evaluate(test_relevancies, test_queries)

    print(f'Train results: {train_scores}')
    print(f'Val results: {val_scores}')
    if corpus.test is not None:
        print(f'Test results: {test_scores}')

if __name__ == '__main__':
    main()

