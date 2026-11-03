from src.data_utils.query import Query

def calculateRecall(relevancies: list[bool], query: Query):
    denom = min(len(relevancies), query.getNumberOfRelevantDocuments())
    if denom == 0:
        return 0.0
    return sum(relevancies) / denom

def calculatePrecision(relevancies: list[bool], query: Query = None):
    if len(relevancies) == 0:
        return 0.0
    return sum(relevancies) / len(relevancies)

def calculateReciprocalRank(relevancies: list[bool], query: Query = None):
    for rank, is_relevant in enumerate(relevancies, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0

def Evaluate(relevancies: list[list[bool]], queries: list[Query]):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    n_queries = len(queries)
    if n_queries == 0:
        return {'recall': 0.0, 'precision': 0.0, 'mrr': 0.0}

    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0
    
    for i in range(n_queries):
        total_recall += calculateRecall(relevancies[i], queries[i])
        total_precision += calculatePrecision(relevancies[i], queries[i])
        total_mrr += calculateReciprocalRank(relevancies[i], queries[i])
    
    scores = {
        'recall': total_recall / n_queries,
        'precision': total_precision / n_queries,
        'mrr': total_mrr / n_queries
    }
    return scores
