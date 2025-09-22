#!/usr/bin/env python3
"""
Embedding Service using BGE-M3 Model
Provides text embedding API for the RAG system
"""

import os
import json
import time
import torch
import numpy as np
from flask import Flask, request, jsonify
from FlagEmbedding import FlagModel
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model instance
embedding_model = None

class EmbeddingService:
    """BGE-M3 based embedding service"""

    def __init__(self):
        self.model = None
        self.device = None
        self.model_name = os.getenv("MODEL_NAME", "BAAI/bge-m3")
        self.batch_size = 32
        self.max_length = 8192  # BGE-M3 supports up to 8192 tokens

    def load_model(self):
        """Load BGE-M3 model"""
        logger.info(f"🔄 Loading embedding model: {self.model_name}")

        # Device selection
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.info("⚠️ No GPU detected, using CPU")

        # Load model
        self.model = FlagModel(
            self.model_name,
            query_instruction_for_retrieval="Represent this query for retrieving relevant documents: ",
            use_fp16=True if self.device == "cuda" else False
        )

        logger.info("✅ Embedding model loaded successfully!")

    def embed_texts(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Generate embeddings for texts

        Args:
            texts: List of text strings to embed
            is_query: Whether these are query texts (vs document texts)

        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        if not texts:
            return []

        # Encode with appropriate method
        if is_query:
            # For queries, use query encoding
            embeddings = self.model.encode_queries(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length
            )
        else:
            # For documents, use standard encoding
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length
            )

        # Convert to list for JSON serialization
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        return embeddings

    def embed_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process batch embedding request

        Args:
            batch_data: Dictionary with 'texts' and optional 'is_query'

        Returns:
            Dictionary with embeddings and metadata
        """
        texts = batch_data.get('texts', [])
        is_query = batch_data.get('is_query', False)

        start_time = time.time()
        embeddings = self.embed_texts(texts, is_query)
        elapsed_time = time.time() - start_time

        return {
            'embeddings': embeddings,
            'count': len(embeddings),
            'dimension': len(embeddings[0]) if embeddings else 0,
            'model': self.model_name,
            'processing_time': elapsed_time,
            'is_query': is_query
        }

# Flask endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global embedding_model

    status = {
        'status': 'healthy' if embedding_model and embedding_model.model else 'loading',
        'model': embedding_model.model_name if embedding_model else None,
        'device': embedding_model.device if embedding_model else None,
        'timestamp': time.time()
    }

    return jsonify(status)

@app.route('/embed', methods=['POST'])
def embed():
    """Single text embedding endpoint"""
    global embedding_model

    if not embedding_model or not embedding_model.model:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        data = request.json
        text = data.get('text', '')
        is_query = data.get('is_query', False)

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Generate embedding
        embeddings = embedding_model.embed_texts([text], is_query)

        return jsonify({
            'embedding': embeddings[0] if embeddings else [],
            'dimension': len(embeddings[0]) if embeddings else 0,
            'model': embedding_model.model_name
        })

    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/embed/batch', methods=['POST'])
def embed_batch():
    """Batch text embedding endpoint"""
    global embedding_model

    if not embedding_model or not embedding_model.model:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        data = request.json

        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400

        result = embedding_model.embed_batch(data)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models"""
    global embedding_model

    return jsonify({
        'models': [
            {
                'id': 'bge-m3',
                'name': 'BAAI/bge-m3',
                'description': 'Multilingual embedding model supporting Korean and English',
                'max_length': 8192,
                'dimension': 1024,
                'loaded': embedding_model and embedding_model.model is not None
            }
        ]
    })

def initialize_service():
    """Initialize the embedding service"""
    global embedding_model

    logger.info("=" * 60)
    logger.info("🚀 Embedding Service Starting")
    logger.info("=" * 60)

    embedding_model = EmbeddingService()
    embedding_model.load_model()

    logger.info(f"📍 Service ready on port 8002")
    logger.info(f"🌐 Endpoints:")
    logger.info("   - GET  /health")
    logger.info("   - POST /embed")
    logger.info("   - POST /embed/batch")
    logger.info("   - GET  /models")

# Initialize on module load if running directly
if __name__ == '__main__':
    initialize_service()
    app.run(host='0.0.0.0', port=8002, debug=False)
else:
    # For gunicorn
    initialize_service()