"""
Semantic Search Engine with SBERT, FAISS, and Cross-Encoder Reranking
untuk Pencarian Dokumen Komentar Kasus Polio
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    print("Warning: PySastrawi not installed")
    SASTRAWI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SBERT_AVAILABLE = True
except ImportError:
    print("ERROR: sentence-transformers required!")
    SBERT_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: FAISS not installed")
    FAISS_AVAILABLE = False


class SemanticSearchEngine:
    
    def __init__(self, 
                 bi_encoder_model: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                 use_faiss: bool = True,
                 use_cross_encoder: bool = True):
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.use_cross_encoder = use_cross_encoder and SBERT_AVAILABLE
        
        self.documents_raw = []
        self.corpus = []
        self.doc_embeddings = None
        self.faiss_index = None
        
        if SBERT_AVAILABLE:
            print(f"Loading SBERT bi-encoder: {bi_encoder_model}")
            self.bi_encoder = SentenceTransformer(bi_encoder_model)
            print("Bi-encoder loaded!")
        else:
            raise ImportError("sentence-transformers is required!")
        
        if self.use_cross_encoder:
            print(f"Loading Cross-Encoder: {cross_encoder_model}")
            self.cross_encoder = CrossEncoder(cross_encoder_model)
            print("Cross-encoder loaded!")
        else:
            self.cross_encoder = None
            print("Cross-encoder disabled")
            
        if SASTRAWI_AVAILABLE:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
        else:
            self.stemmer = None
            
        self.stopwords = {
            'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk', 
            'adalah', 'ada', 'juga', 'tidak', 'sudah', 'saya', 'aku', 'kamu', 
            'dia', 'kami', 'kita', 'mereka', 'pada', 'dalam', 'ya', 'aja', 
            'nya', 'nih', 'tuh', 'gak', 'ga', 'gue', 'gw', 'lo', 'lah', 'deh', 
            'sih', 'kok', 'dong', 'kan', 'si', 'tapi', 'kalau', 'kalo', 'karena',
            'krn', 'jadi', 'jd', 'lebih', 'atau', 'bisa', 'buat', 'sama', 'mau',
            'baru', 'udah', 'sdh', 'apa', 'gimana', 'banget', 'bgt', 'banyak',
            'semua', 'saja', 'pun', 'lagi', 'terus', 'trus', 'memang', 'emang',
            'nggak', 'enggak', 'belum', 'blm', 'karna', 'dgn', 'utk', 'yg', 
            'dr', 'dlm', 'pd', 'tp', 'kl', 'sy', 'org', 'dg', 'dpt', 'jgn',
            'tsb', 'bs', 'klo', 'abis', 'habis', 'kayak', 'kayanya', 'kak', 
            'kk', 'ku', 'tau', 'tahu', 'bilang', 'kata', 'orang', 'pake', 
            'pakai', 'terus', 'udah', 'mah', 'dah'
        }
    
    def load_data_from_excel(self, filepath: str, sheet_name: int = 0) -> None:
        print(f"\nLoading data from: {filepath}")
        df = pd.read_excel(filepath, header=1)
        
        if len(df.columns) >= 2:
            df.columns = ['No', 'Dokumen']
        
        df = df.dropna(subset=['Dokumen'])
        self.documents_raw = df['Dokumen'].tolist()
        
        print(f"Loaded {len(self.documents_raw)} documents")
        print(f"Sample: {self.documents_raw[0][:80]}...")
        
    def load_data_from_list(self, documents: List[str]) -> None:
        self.documents_raw = documents
        print(f"Loaded {len(self.documents_raw)} documents")
    
    def preprocess(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def build_embeddings(self) -> None:
        print("\nBuilding document embeddings with SBERT...")
        
        self.corpus = [self.preprocess(doc) for doc in self.documents_raw]
        
        self.doc_embeddings = self.bi_encoder.encode(
            self.corpus,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        print(f"Embeddings created! Shape: {self.doc_embeddings.shape}")
        
    def build_faiss_index(self) -> None:
        if not self.use_faiss:
            print("FAISS disabled. Using numpy search instead.")
            return
            
        if self.doc_embeddings is None:
            raise ValueError("Build embeddings first!")
        
        print("\nBuilding FAISS index...")
        
        d = self.doc_embeddings.shape[1]
        
        self.faiss_index = faiss.IndexFlatIP(d)
        
        self.faiss_index.add(self.doc_embeddings.astype('float32'))
        
        print(f"FAISS index built! Total vectors: {self.faiss_index.ntotal}")
        print(f"Index type: Inner Product (cosine similarity)")
        
    def search_with_faiss(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        query_processed = self.preprocess(query)
        query_embedding = self.bi_encoder.encode(
            [query_processed],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), 
            min(top_k, len(self.documents_raw))
        )
        
        results = [
            (int(idx), float(score)) 
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]
        
        return results
    
    def search_with_numpy(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        query_processed = self.preprocess(query)
        query_embedding = self.bi_encoder.encode(
            [query_processed],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        scores = np.dot(self.doc_embeddings, query_embedding)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            (int(idx), float(scores[idx]))
            for idx in top_indices
        ]
        
        return results

    def rerank_with_cross_encoder(self, query: str, candidates: List[Tuple[int, float]], 
                                  top_k: int = 10) -> List[Tuple[int, float, Dict]]:
        if not self.use_cross_encoder or self.cross_encoder is None:
            results = [
                (doc_id, score, {
                    'bi_encoder': round(score, 5),
                    'cross_encoder': None,
                    'final': round(score, 5)
                })
                for doc_id, score in candidates[:top_k]
            ]
            return results
        
        print(f"Reranking {len(candidates)} candidates with Cross-Encoder...")
        
        pairs = [
            [query, self.documents_raw[doc_id]]
            for doc_id, _ in candidates
        ]
        
        cross_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        
        reranked = []
        for (doc_id, bi_score), cross_score in zip(candidates, cross_scores):
            score_details = {
                'bi_encoder': round(float(bi_score), 5),
                'cross_encoder': round(float(cross_score), 5),
                'final': round(float(cross_score), 5)
            }
            reranked.append((doc_id, float(cross_score), score_details))
        
        reranked = sorted(reranked, key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]
    
    def search(self, query: str, top_k: int = 10, 
              retrieval_k: int = 100) -> List[Tuple[int, str, float, Dict]]:
        print(f"\nSearching for: '{query}'")
        print(f"   Stage 1: Retrieving top-{retrieval_k} candidates")
        print(f"   Stage 2: Reranking to top-{top_k}")
        
        if self.use_faiss and self.faiss_index is not None:
            candidates = self.search_with_faiss(query, top_k=retrieval_k)
            print(f"Stage 1 (FAISS): Retrieved {len(candidates)} candidates")
        else:
            candidates = self.search_with_numpy(query, top_k=retrieval_k)
            print(f"Stage 1 (Numpy): Retrieved {len(candidates)} candidates")
        
        reranked = self.rerank_with_cross_encoder(query, candidates, top_k=top_k)
        print(f"Stage 2 (Cross-Encoder): Reranked to top-{len(reranked)}")
        
        results = [
            (doc_id, self.documents_raw[doc_id], score, details)
            for doc_id, score, details in reranked
        ]
        
        return results
    
    def get_statistics(self) -> Dict:
        stats = {
            'total_documents': len(self.documents_raw),
            'embedding_dimension': self.doc_embeddings.shape[1] if self.doc_embeddings is not None else 0,
            'faiss_enabled': self.use_faiss,
            'cross_encoder_enabled': self.use_cross_encoder,
            'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0,
        }
        return stats


def main():
    print("="*70)
    print("SEMANTIC SEARCH ENGINE with SBERT + FAISS + Cross-Encoder")
    print("="*70)
    
    print("\nInitializing Semantic Search Engine...")
    engine = SemanticSearchEngine(
        bi_encoder_model='paraphrase-multilingual-MiniLM-L12-v2',
        cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-6-v2',
        use_faiss=True,
        use_cross_encoder=True
    )
    
    print("\nLoading data...")
    import os
    data_file = os.path.join("versi lama", "Tugas_TKI _1_.xlsx")
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        return
    
    engine.load_data_from_excel(data_file)
    
    engine.build_embeddings()
    engine.build_faiss_index()
    
    stats = engine.get_statistics()
    print("\nEngine Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*70)
    print("Search Engine Ready!")
    print("="*70)
    
    queries = [
        "anak saya divaksin polio tapi malah sakit",
        "indonesia bebas polio"
    ]
    
    for query in queries:
        print("\n" + "="*70)
        results = engine.search(query, top_k=5, retrieval_k=50)
        
        print(f"\nTop Results:")
        print("-"*70)
        
        for rank, (doc_id, doc, score, details) in enumerate(results, 1):
            preview = doc[:120] + "..." if len(doc) > 120 else doc
            print(f"\n{rank}. D{doc_id+1} | Score: {score:.4f}")
            print(f"   Bi-encoder: {details['bi_encoder']:.4f} | Cross-encoder: {details['cross_encoder']:.4f}")
            print(f"   {preview}")


if __name__ == "__main__":
    main()
