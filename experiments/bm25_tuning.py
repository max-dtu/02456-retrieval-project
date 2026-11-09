"""
BM25 parameter tuning on FiQA.

Grid search over k1 and b on the validation set (selected by MRR@10),
then a final comparison of TF-IDF, default BM25 and tuned BM25 on validation and test.

Run from the project root:
    python experiments/bm25_tuning.py

Results are saved to results/bm25_tuning_results.json.
"""
import os
import sys
import time
import json
import itertools

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_utils.corpus import Corpus
from src.data_utils.query import Query
from src.retrievers.tf_idf import TFIDF
from src.retrievers.bm25 import BM25
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate

DATASET = 'fiqa'
TOP_K = 10
K1_GRID = [0.5, 0.9, 1.2, 1.5, 2.0]
B_GRID = [0.25, 0.5, 0.75, 1.0]
DEFAULT_K1, DEFAULT_B = 1.5, 0.75
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'bm25_tuning_results.json')


def make_queries(split, corpus):
    return [Query(text, id=q_id, relevant_document_ids=corpus.relevant_ids[q_id])
            for text, q_id in zip(split['question'], split['query_id'])]


def evaluate(retriever, queries):
    relevancies = retrieveQueryAndGetRelevancies(retriever, queries, k=TOP_K)
    return Evaluate(relevancies, queries)


def fmt(scores):
    return f"recall={scores['recall']:.4f} precision={scores['precision']:.4f} mrr={scores['mrr']:.4f}"


def save(results):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    corpus = Corpus(dataset_name=DATASET, base_dir=os.path.join(PROJECT_ROOT, 'retrieval_data'))
    val_queries = make_queries(corpus.val, corpus)
    test_queries = make_queries(corpus.test, corpus)
    documents = corpus.all_documents
    print(f"documents={len(documents)} val_queries={len(val_queries)} test_queries={len(test_queries)}", flush=True)

    results = {'dataset': DATASET, 'top_k': TOP_K, 'grid': [], 'final': {}}

    # TF-IDF baseline
    tfidf = TFIDF(documents=documents)
    results['final']['TF-IDF'] = {'val': evaluate(tfidf, val_queries), 'test': evaluate(tfidf, test_queries)}
    print("TF-IDF val :", fmt(results['final']['TF-IDF']['val']), flush=True)
    del tfidf

    # BM25 grid search on validation.
    # Vocabulary and IDF do not depend on k1/b, so we build BM25 once and only rebuild the matrix.
    bm25 = BM25(documents=documents, k1=DEFAULT_K1, b=DEFAULT_B)
    for k1, b in itertools.product(K1_GRID, B_GRID):
        start = time.time()
        bm25.k1, bm25.b = k1, b
        bm25.bm25_matrix = bm25.GetDocumentsBM25Vectors()
        scores = evaluate(bm25, val_queries)
        results['grid'].append({'k1': k1, 'b': b, **scores})
        print(f"k1={k1:<4} b={b:<4} val {fmt(scores)} ({time.time() - start:.1f}s)", flush=True)
        save(results)

    best = max(results['grid'], key=lambda r: r['mrr'])
    print(f"Best on validation (MRR@{TOP_K}): k1={best['k1']} b={best['b']}", flush=True)

    # Final comparison on validation and test
    settings = {
        f'BM25 default (k1={DEFAULT_K1}, b={DEFAULT_B})': (DEFAULT_K1, DEFAULT_B),
        f"BM25 tuned (k1={best['k1']}, b={best['b']})": (best['k1'], best['b']),
    }
    for name, (k1, b) in settings.items():
        bm25.k1, bm25.b = k1, b
        bm25.bm25_matrix = bm25.GetDocumentsBM25Vectors()
        results['final'][name] = {'val': evaluate(bm25, val_queries), 'test': evaluate(bm25, test_queries)}

    print("\nTest results:")
    for name, scores in results['final'].items():
        print(f"  {name:<32} {fmt(scores['test'])}")

    save(results)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
