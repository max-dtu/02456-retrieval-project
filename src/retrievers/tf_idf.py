from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
import numpy as np
from scipy.sparse import lil_matrix


class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()

        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted charatcters from text and lowercase
        """
        text=text.lower()
        return text.split()
        
    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """
        return list(set(self.PreprocessText(query)))

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """
        text=document.GetText()
        return self.GetQueryVocabulary(text)
    
    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        corpus_vocabulary = set().union(*processed_texts)
        return corpus_vocabulary

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """
        df = {}
        for document in self.index.GetDocuments() :
            unique_words=self.GetDocumentVocabulary(document)
            for word in unique_words : 
                df[word]=df.get(word,0)+1
        return df
    
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        idf = {}
        df = self.GetDocumentFrequencies()
        n_documents = len(self.index.GetDocuments())
        for word in df : 
            idf[word]=np.log(n_documents/df[word])
        return idf
    
    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """
        text=document.GetText()
        text=self.PreprocessText(text)
        tc={}
        for word in text : 
                tc[word]=tc.get(word,0)+1
        return tc

    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        documents = self.index.GetDocuments()
        n_documents = len(documents)
        terms = self.corpus_vocabulary
        n_terms = len(terms)
        tfidf_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)
        for i, document in enumerate(documents) :
            tc=self.GetDocumentTermCounts(document)
            if not tc:
                continue
            max_freq=max(tc.values())
            for term, count in tc.items():
                if term in self.term_to_index :
                    j=self.term_to_index[term]
                    tf=count/max_freq
                    idf=self.idf.get(term,0)
                    tfidf_matrix[i,j]=tf*idf
        return tfidf_matrix.tocsr()   # Convert to CSR format for efficient row slicing
    
    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.
        
        Args:
            query (str): The query to convert
        
        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)
        for k, query in enumerate(queries):
            words=self.PreprocessText(query)
            tc = {}
            for word in words :
                tc[word]=tc.get(word,0)+1
            for term, count in tc.items():
                if term in self.term_to_index:
                    j=self.term_to_index[term]
                    query_vector[k,j] = count
        return query_vector.tocsr()

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        return scores
    