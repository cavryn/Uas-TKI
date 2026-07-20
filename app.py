"""
Flask Web Application for Sentiment Classification and Retrieval System
Using SBERT + Scikit-learn Cosine Similarity

Run: python app.py
Access: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from sentiment_search_engine import SentimentSearchEngine
import os
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sentiment-search-2026'

search_engine = None
engine_ready = False
engine_stats = {}

def initialize_engine():
    global search_engine, engine_ready, engine_stats
    
    try:
        logger.info("Initializing sentiment search engine...")
        
        search_engine = SentimentSearchEngine(
            model_name='paraphrase-multilingual-MiniLM-L12-v2'
        )
        
        data_file = os.path.join("versi lama", "Tugas_TKI _1_.xlsx")
        if not os.path.exists(data_file):
            logger.error(f"Data file not found: {data_file}")
            return
        
        search_engine.load_data_from_excel(data_file)
        search_engine.label_documents()
        search_engine.build_embeddings()
        
        engine_stats = search_engine.get_statistics()
        engine_ready = True
        
        logger.info("Sentiment search engine initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing engine: {e}")
        engine_ready = False

init_thread = threading.Thread(target=initialize_engine, daemon=True)
init_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'ready': engine_ready,
        'stats': engine_stats if engine_ready else {}
    })

@app.route('/api/classify', methods=['POST'])
def classify_sentiment():
    if not engine_ready or search_engine is None:
        return jsonify({
            'error': 'Search engine not ready. Please wait...'
        }), 503
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = int(data.get('top_k', 10))
        
        if not query:
            return jsonify({
                'error': 'Query cannot be empty'
            }), 400
        
        result = search_engine.classify_query_sentiment(query, top_k=top_k)
        
        formatted_result = {
            'success': True,
            'query': result['query'],
            'sentiment': result['sentiment'],
            'sentiment_label': 'BAIK (Positif)' if result['sentiment'] == 'baik' else 'BURUK (Negatif)',
            'confidence': result['confidence'],
            'votes': {
                'baik': result['votes'].get('baik', 0),
                'buruk': result['votes'].get('buruk', 0),
                'netral': result['votes'].get('netral', 0)
            },
            'explanation': result['explanation'],
            'supporting_documents': result['supporting_documents'],
            'all_documents': [
                {
                    'doc_id': doc['doc_id'],
                    'document': doc['document'],
                    'similarity': doc['similarity'],
                    'sentiment': doc['sentiment'],
                    'sentiment_confidence': 0.0
                }
                for doc in result['all_relevant_documents']
            ]
        }
        
        return jsonify(formatted_result)
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/example-queries', methods=['GET'])
def get_example_queries():
    examples = [
        "anak divaksin polio malah sakit",
        "vaksin polio penting untuk mencegah penyakit",
        "alhamdulillah anak sudah imunisasi lengkap",
        "efek samping vaksin berbahaya",
        "polio sudah tereradikasi di indonesia"
    ]
    return jsonify({'examples': examples})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Sentiment Classification & Retrieval System")
    print("SBERT + Scikit-learn Cosine Similarity")
    print("=" * 60)
    print("\nStarting server...")
    print("Access at: http://localhost:5000")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
