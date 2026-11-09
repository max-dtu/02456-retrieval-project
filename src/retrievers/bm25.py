import numpy as np
from scipy.sparse import lil_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document

class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        self.bm25_matrix = self.GetDocumentsBM25Vectors()

    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        BM25 smoothed IDF: log((N + 1) / (df + 0.5))
        """
        dfs = self.GetDocumentFrequencies()
        N = len(self.index.GetDocuments())
        return {term: float(np.log((N + 1) / (df + 0.5))) for term, df in dfs.items()}

    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.PreprocessText(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        return document_lengths, average_document_length / len(self.index.GetDocuments())

    def GetDocumentsBM25Vectors(self):
        """
        Computes sparse CSR matrix of shape (n_documents, n_terms) with BM25 weights.
        """
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        bm25_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        document_lengths, avg_doc_len = self.GetDocumentLengths()

        for doc_idx, doc in enumerate(self.index.GetDocuments()):
            term_counts = self.GetDocumentTermCounts(doc)
            if not term_counts:
                continue

            doc_len = document_lengths[doc]
            len_norm = 1.0 - self.b + self.b * (doc_len / avg_doc_len) if avg_doc_len > 0 else 1.0
            denom_const = self.k1 * len_norm

            for term, count in term_counts.items():
                if term in self.term_to_index:
                    col_idx = self.term_to_index[term]
                    tf = (count * (self.k1 + 1.0)) / (count + denom_const)
                    bm25_matrix[doc_idx, col_idx] = tf * self.idf[term]

        return bm25_matrix.tocsr()

    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores
