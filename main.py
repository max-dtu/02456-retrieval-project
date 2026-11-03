import hydra
from src.data_utils.corpus import Corpus
from src.retrievers.tf_idf import TFIDF
from src.retrievers.bm25 import BM25
from src.data_utils.query import Query
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate

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
    if hasattr(cfg, 'retriever') and cfg.retriever.method == "bm25":
        k1 = cfg.retriever.get("k1", 1.5)
        b = cfg.retriever.get("b", 0.75)
        tf_idf = BM25(documents=corpus.all_documents, k1=k1, b=b)
    else:
        tf_idf = TFIDF(documents=corpus.all_documents)
    
    ### Retrieve top-k relevant queries
    tfidf_train_relevancies = retrieveQueryAndGetRelevancies(tf_idf, train_queries, k=cfg.top_k)
    tfidf_val_relevancies = retrieveQueryAndGetRelevancies(tf_idf, val_queries, k=cfg.top_k)
    if corpus.test is not None:
        tfidf_test_relevancies = retrieveQueryAndGetRelevancies(tf_idf, test_queries, k=cfg.top_k)

    ### Evaluate
    train_scores = Evaluate(tfidf_train_relevancies, train_queries)
    val_scores = Evaluate(tfidf_val_relevancies, val_queries)
    if corpus.test is not None:
        test_scores = Evaluate(tfidf_test_relevancies, test_queries)

    print(f'Train results: {train_scores}')
    print(f'Val results: {val_scores}')
    if corpus.test is not None:
        print(f'Test results: {test_scores}')

if __name__ == '__main__':
    main()
