from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from static_memory_cache import StaticMemoryCache
from telemetries.logger import logger
from telemetries.metrics import metrics


class EmbeddingService:
    """Embedding service using sentence-transformers miniLM-6 model."""
    
    _instance = None
    _model = None
    
    def __init__(self):
        """Initialize embedding service."""
        self.config = StaticMemoryCache.get_embedding_config()
        self.model_name = self.config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.device = self.config.get("device", "cpu")
        self.batch_size = self.config.get("batch_size", 32)
        
        self._load_model()
        logger.info(f"EmbeddingService initialized with model: {self.model_name}")
    
    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_model(self):
        """Load the embedding model."""
        if self._model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.success("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                raise
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text.
        
        Args:
            text: Single text string or list of texts
        
        Returns:
            Numpy array of embeddings
        """
        import time
        start_time = time.time()
        
        try:
            if isinstance(text, str):
                text = [text]
            
            embeddings = self._model.encode(
                text,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            duration = time.time() - start_time
            metrics.track_performance("embedding_generation", duration)
            metrics.track_metric("embedding_count", len(text))
            
            logger.debug(f"Generated embeddings for {len(text)} texts in {duration:.3f}s")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query.
        
        Args:
            query: Query text
        
        Returns:
            Embedding vector
        """
        return self.embed_text(query)
    
    def get_model_dimension(self) -> int:
        """Get the dimension of the embedding model."""
        if self._model is None:
            self._load_model()
        return self._model.get_sentence_embedding_dimension()

