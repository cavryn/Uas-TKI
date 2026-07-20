"""
Sentiment Classification and Information Retrieval System
Using SBERT + Scikit-learn Cosine Similarity

Architecture:
1. Documents labeled using SBERT + example sentences (no extra model)
2. Query classified DIRECTLY by comparing to positive/negative examples
3. Similar documents used for EXPLANATION only
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    print("ERROR: sentence-transformers required!")
    SBERT_AVAILABLE = False


class SentimentSearchEngine:
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        print("Initializing Sentiment Search Engine...")
        
        self.documents_raw = []
        self.doc_embeddings = None
        self.doc_sentiments = []
        
        if not SBERT_AVAILABLE:
            raise ImportError("sentence-transformers is required!")
        
        print(f"Loading SBERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("SBERT loaded successfully!")
        
        # Example sentences for sentiment comparison
        self._init_example_sentences()
        self._init_example_embeddings()
    
    def _init_example_sentences(self):
        self.positive_examples = [
            "anak saya sehat setelah vaksin polio",
            "vaksin polio sangat baik untuk anak",
            "alhamdulillah imunisasi berhasil",
            "saya senang anak saya divaksin",
            "vaksinasi melindungi anak dari penyakit",
            "program vaksinasi sangat efektif",
            "anak saya tumbuh sehat berkat imunisasi",
            "pemerintah berhasil menurunkan kasus polio",
            "saya mendukung program vaksinasi nasional",
            "vaksin polio aman dan efektif",
            "imunisasi lengkap mencegah penyakit berbahaya",
            "terima kasih atas vaksin gratisnya",
            "semua anak berhak mendapatkan imunisasi",
            "vaksinasi massal berjalan lancar",
            "saya bangga indonesia bebas polio",
            "vaksinasi polio penting untuk anak-anak",
            "kesehatan anak menjadi prioritas utama",
            "program imunisasi sangat membantu masyarakat",
            "terima kasih petugas vaksinasi yang berdedikasi",
            "vaksinasi membuat anak Indonesia sehat"
        ]
        
        self.negative_examples = [
            "anak saya sakit setelah vaksin polio",
            "efek samping vaksin sangat berbahaya",
            "saya trauma anak saya cacat",
            "vaksin polio menyebabkan kelumpuhan",
            "anak saya demam tinggi setelah imunisasi",
            "saya menyesal anak saya divaksin",
            "vaksinasi menimbulkan efek samping yang parah",
            "kasus polio justru meningkat setelah vaksinasi",
            "anak saya meninggal setelah vaksin",
            "saya takut efek samping vaksin",
            "vaksinasi membuat anak saya sakit parah",
            "saya tidak percaya dengan vaksin polio",
            "anak saya menderita karena imunisasi",
            "vaksin palsu menyebabkan kerugian",
            "imunisasi justru membahayakan anak",
            "saya trauma setelah anak saya divaksin",
            "saya marah dengan program vaksinasi",
            "vaksinasi adalah program yang gagal",
            "banyak anak menjadi cacat setelah vaksin",
            "saya khawatir efek samping vaksin polio"
        ]
    
    def _init_example_embeddings(self):
        print("Creating example embeddings for sentiment analysis...")
        all_examples = self.positive_examples + self.negative_examples
        self.example_embeddings = self.model.encode(
            all_examples,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        self.num_positive = len(self.positive_examples)
        print(f"Examples: {self.num_positive} positive, {len(self.negative_examples)} negative")
    
    def load_data_from_excel(self, filepath: str) -> None:
        print(f"\nLoading data from: {filepath}")
        df = pd.read_excel(filepath, header=1)
        
        if len(df.columns) >= 2:
            df.columns = ['No', 'Dokumen']
        
        df = df.dropna(subset=['Dokumen'])
        self.documents_raw = df['Dokumen'].tolist()
        
        print(f"Loaded {len(self.documents_raw)} documents")
    
    def preprocess_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def analyze_query_sentiment(self, text: str) -> Dict:
        query_embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        similarities = cosine_similarity(query_embedding, self.example_embeddings)[0]
        
        pos_similarities = similarities[:self.num_positive]
        neg_similarities = similarities[self.num_positive:]
        
        pos_avg = float(np.mean(pos_similarities))
        neg_avg = float(np.mean(neg_similarities))
        pos_max = float(np.max(pos_similarities))
        neg_max = float(np.max(neg_similarities))
        pos_median = float(np.median(pos_similarities))
        neg_median = float(np.median(neg_similarities))
        
        pos_top3 = float(np.mean(np.sort(pos_similarities)[-3:]))
        neg_top3 = float(np.mean(np.sort(neg_similarities)[-3:]))
        
        combined_pos = 0.3 * pos_avg + 0.3 * pos_top3 + 0.2 * pos_max + 0.2 * pos_median
        combined_neg = 0.3 * neg_avg + 0.3 * neg_top3 + 0.2 * neg_max + 0.2 * neg_median
        
        # Calculate confidence
        total = combined_pos + combined_neg
        if combined_pos > combined_neg:
            sentiment = 'baik'
            confidence = combined_pos / total if total > 0 else 0.5
        else:
            sentiment = 'buruk'
            confidence = combined_neg / total if total > 0 else 0.5
        
        top_pos_idx = int(np.argmax(pos_similarities))
        top_neg_idx = int(np.argmax(neg_similarities))
        
        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'scores': {
                'positive_avg': round(pos_avg, 4),
                'negative_avg': round(neg_avg, 4),
                'positive_top3': round(pos_top3, 4),
                'negative_top3': round(neg_top3, 4),
                'positive_max': round(pos_max, 4),
                'negative_max': round(neg_max, 4)
            },
            'best_positive_example': self.positive_examples[top_pos_idx],
            'best_negative_example': self.negative_examples[top_neg_idx],
            'best_positive_score': round(pos_max, 4),
            'best_negative_score': round(neg_max, 4)
        }
    
    def label_documents(self) -> None:
        print("\nLabeling documents with SBERT-based sentiment analysis...")
        self.doc_sentiments = []
        
        for i, doc in enumerate(self.documents_raw):
            analysis = self.analyze_query_sentiment(doc)
            self.doc_sentiments.append(analysis)
            if (i + 1) % 10 == 0:
                print(f"  Labeled {i+1}/{len(self.documents_raw)}...")
        
        counts = {
            'baik': sum(1 for s in self.doc_sentiments if s['sentiment'] == 'baik'),
            'buruk': sum(1 for s in self.doc_sentiments if s['sentiment'] == 'buruk')
        }
        print(f"\nDistribution: {counts['baik']} Baik, {counts['buruk']} Buruk")
    
    def build_embeddings(self) -> None:
        print("\nBuilding document embeddings...")
        self.doc_embeddings = self.model.encode(
            self.documents_raw,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        print(f"Embeddings created! Shape: {self.doc_embeddings.shape}")
    
    def search_similar_documents(self, query: str, top_k: int = 10) -> List[Tuple]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                self.documents_raw[idx],
                float(similarities[idx]),
                self.doc_sentiments[idx]
            ))
        return results
    
    def classify_query_sentiment(self, query: str, top_k: int = 10) -> Dict:
        print(f"\nClassifying: '{query}'")
        
        # Step 1: Classify query directly via example comparison
        query_analysis = self.analyze_query_sentiment(query)
        query_sentiment = query_analysis['sentiment']
        
        print(f"Direct: {query_sentiment.upper()} (confidence: {query_analysis['confidence']:.2%})")
        
        # Step 2: Find similar documents
        similar_docs = self.search_similar_documents(query, top_k=top_k)
        
        # Step 3: Organize docs by sentiment for explanation
        docs_by_sent = {'baik': [], 'buruk': []}
        for doc_id, doc, sim, sent_info in similar_docs:
            doc_sent = sent_info['sentiment']
            docs_by_sent[doc_sent].append({
                'doc_id': doc_id + 1,
                'document': doc,
                'similarity': round(sim, 4)
            })
        
        sentiment_votes = {
            'baik': len(docs_by_sent['baik']),
            'buruk': len(docs_by_sent['buruk'])
        }
        
        # Step 4: Generate explanation
        explanation = self._generate_explanation(
            query, query_sentiment, query_analysis,
            sentiment_votes, top_k,
            docs_by_sent[query_sentiment][:5]
        )
        
        result = {
            'query': query,
            'sentiment': query_sentiment,
            'sentiment_label': 'BAIK (Positif)' if query_sentiment == 'baik' else 'BURUK (Negatif)',
            'confidence': query_analysis['confidence'],
            'scores': query_analysis['scores'],
            'votes': sentiment_votes,
            'explanation': explanation,
            'supporting_documents': docs_by_sent[query_sentiment][:5],
            'best_match': {
                'positive': query_analysis['best_positive_example'],
                'negative': query_analysis['best_negative_example'],
                'positive_score': query_analysis['best_positive_score'],
                'negative_score': query_analysis['best_negative_score']
            },
            'all_relevant_documents': [
                {
                    'doc_id': doc_id + 1,
                    'document': doc,
                    'similarity': round(sim, 4),
                    'sentiment': sent['sentiment']
                }
                for doc_id, doc, sim, sent in similar_docs
            ]
        }
        
        print(f"Result: {query_sentiment.upper()}")
        return result
    
    def _generate_explanation(self, query: str, sentiment: str, analysis: Dict,
                             votes: Dict, total_docs: int,
                             top_matching_docs: List[Dict]) -> str:
        label = 'BAIK (Positif)' if sentiment == 'baik' else 'BURUK (Negatif)'
        s = analysis['scores']
        
        exp = f"Query: '{query}'\n\n"
        exp += f"Hasil Klasifikasi: {label}\n"
        exp += f"Confidence: {analysis['confidence']:.1%}\n\n"
        
        exp += "Analisis Kemiripan:\n"
        exp += f"- Rata-rata kemiripan dengan contoh BAIK: {s['positive_avg']:.3f}\n"
        exp += f"- Rata-rata kemiripan dengan contoh BURUK: {s['negative_avg']:.3f}\n"
        exp += f"- Top-3 kemiripan BAIK: {s['positive_top3']:.3f}\n"
        exp += f"- Top-3 kemiripan BURUK: {s['negative_top3']:.3f}\n\n"
        
        exp += f"Contoh paling mirip:\n"
        exp += f"- BAIK: \"{analysis['best_positive_example']}\" (score: {analysis['best_positive_score']:.3f})\n"
        exp += f"- BURUK: \"{analysis['best_negative_example']}\" (score: {analysis['best_negative_score']:.3f})\n\n"
        
        exp += f"Distribusi {total_docs} dokumen relevan:\n"
        exp += f"- {votes['baik']} dokumen bersentimen BAIK\n"
        exp += f"- {votes['buruk']} dokumen bersentimen BURUK\n"
        
        if top_matching_docs:
            exp += "\nDokumen pendukung:\n"
            for i, doc in enumerate(top_matching_docs, 1):
                preview = doc['document'][:150]
                if len(doc['document']) > 150:
                    preview += '...'
                exp += f"{i}. D{doc['doc_id']} (sim: {doc['similarity']:.3f}): {preview}\n"
        
        return exp
    
    def get_statistics(self) -> Dict:
        return {
            'total_documents': len(self.documents_raw),
            'embedding_dimension': self.doc_embeddings.shape[1] if self.doc_embeddings is not None else 0,
            'sentiment_distribution': {
                'baik': sum(1 for s in self.doc_sentiments if s['sentiment'] == 'baik'),
                'buruk': sum(1 for s in self.doc_sentiments if s['sentiment'] == 'buruk'),
                'netral': 0
            }
        }


def main():
    print("="*70)
    print("SENTIMENT CLASSIFICATION AND RETRIEVAL SYSTEM")
    print("SBERT + Example-based Sentiment Analysis")
    print("="*70)
    
    engine = SentimentSearchEngine()
    
    import os
    data_file = os.path.join("versi lama", "Tugas_TKI _1_.xlsx")
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        return
    
    engine.load_data_from_excel(data_file)
    engine.label_documents()
    engine.build_embeddings()
    
    stats = engine.get_statistics()
    print("\nEngine Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    test_queries = [
        "vaksin polio baik ya",
        "anak saya divaksin polio malah sakit",
        "saya senang anak saya sudah divaksin",
        "efek samping vaksin berbahaya"
    ]
    
    for query in test_queries:
        print("\n" + "="*70)
        result = engine.classify_query_sentiment(query, top_k=10)
        print(f"Result: {result['sentiment'].upper()} ({result['confidence']:.2%})")


if __name__ == "__main__":
    main()
