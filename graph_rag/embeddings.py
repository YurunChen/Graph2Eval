"""
Embedding management and vector indexing for GraphRAG
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
import torch
from loguru import logger

from .node_types import Node, NodeType
from config_manager import get_config


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    device: str = "auto"  # auto, cpu, cuda
    
    @classmethod
    def from_config(cls):
        """从配置文件创建嵌入配置"""
        config = get_config()
        embeddings_config = config.graph_rag.get('embeddings', {})
        
        return cls(
            model_name=embeddings_config.get('model_name', "all-MiniLM-L6-v2"),
            batch_size=embeddings_config.get('batch_size', 32),
            max_seq_length=embeddings_config.get('max_seq_length', 512),
            normalize_embeddings=embeddings_config.get('normalize_embeddings', True),
            cache_embeddings=embeddings_config.get('cache_embeddings', True),
            device=embeddings_config.get('device', "auto")
        )


class EmbeddingManager:
    """Manages embedding generation and caching"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.embedding_cache = {}
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            device = self._get_device()
            logger.info(f"Loading embedding model: {self.config.model_name} on {device}")
            
            self.model = SentenceTransformer(self.config.model_name, device=device)
            self.model.max_seq_length = self.config.max_seq_length
            
            logger.info(f"Model loaded. Embedding dimension: {self.get_embedding_dim()}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _get_device(self) -> str:
        """Determine device to use"""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        if self.model is None:
            return 384  # Default for MiniLM
        return self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        if not text or not text.strip():
            return np.zeros(self.get_embedding_dim())
        
        # Check cache first
        if self.config.cache_embeddings and text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True
            )
            
            # Cache result
            if self.config.cache_embeddings:
                self.embedding_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            return np.zeros(self.get_embedding_dim())
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return np.zeros((len(texts), self.get_embedding_dim()))
        
        try:
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=len(valid_texts) > 100
            )
            
            # Cache results
            if self.config.cache_embeddings:
                for text, embedding in zip(valid_texts, embeddings):
                    self.embedding_cache[text] = embedding
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            return np.zeros((len(texts), self.get_embedding_dim()))
    
    def embed_node(self, node: Node) -> np.ndarray:
        """Generate embedding for a node"""
        text = node.get_text_for_embedding()
        embedding = self.embed_text(text)
        
        # Store embedding in node
        node.embedding = embedding
        
        return embedding
    
    def embed_nodes(self, nodes: List[Node]) -> List[np.ndarray]:
        """Generate embeddings for multiple nodes"""
        texts = [node.get_text_for_embedding() for node in nodes]
        embeddings = self.embed_texts(texts)
        
        # Store embeddings in nodes
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
        
        return embeddings.tolist()
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
        
        # Normalize if not already normalized
        if not self.config.normalize_embeddings:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        return float(np.dot(embedding1, embedding2))
    
    def find_similar_nodes(
        self, 
        query_node: Node, 
        candidate_nodes: List[Node], 
        threshold: float = 0.7,
        top_k: int = 10
    ) -> List[Tuple[Node, float]]:
        """Find nodes similar to query node"""
        if query_node.embedding is None:
            self.embed_node(query_node)
        
        similarities = []
        for candidate in candidate_nodes:
            if candidate.embedding is None:
                self.embed_node(candidate)
            
            if candidate.node_id != query_node.node_id:
                similarity = self.compute_similarity(
                    query_node.embedding, 
                    candidate.embedding
                )
                
                if similarity >= threshold:
                    similarities.append((candidate, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save_cache(self, cache_path: str):
        """Save embedding cache to disk"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved embedding cache to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def load_cache(self, cache_path: str):
        """Load embedding cache from disk"""
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded embedding cache from {cache_path}")
            except Exception as e:
                logger.error(f"Failed to load embedding cache: {e}")


class VectorIndex(ABC):
    """Abstract base class for vector indices"""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, ids: List[str]):
        """Add vectors to index"""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save index to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load index from disk"""
        pass


class FAISSIndex(VectorIndex):
    """FAISS-based vector index"""
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.id_map = {}  # Map internal indices to node IDs
        self.reverse_id_map = {}  # Map node IDs to internal indices
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index"""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        logger.info(f"Built FAISS {self.index_type} index with dimension {self.dimension}")
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str]):
        """Add vectors to FAISS index"""
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors must match number of IDs")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Convert to float32 for FAISS
        vectors = vectors.astype(np.float32)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            if len(vectors) >= 100:  # Need enough training data
                self.index.train(vectors)
            else:
                logger.warning("Not enough vectors to train IVF index, using flat index")
                self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add vectors
        start_idx = self.index.ntotal
        self.index.add(vectors)
        
        # Update ID mappings
        for i, node_id in enumerate(ids):
            internal_idx = start_idx + i
            self.id_map[internal_idx] = node_id
            self.reverse_id_map[node_id] = internal_idx
        
        logger.info(f"Added {len(vectors)} vectors to index. Total: {self.index.ntotal}")
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """Search for similar vectors"""
        if self.index.ntotal == 0:
            return [], []
        
        # Ensure query vector is float32 and 2D
        query_vector = query_vector.astype(np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        # Convert internal indices to node IDs
        result_ids = []
        result_scores = []
        
        for idx, score in zip(indices[0], scores[0]):
            if idx in self.id_map and idx != -1:  # -1 indicates no result
                result_ids.append(self.id_map[idx])
                result_scores.append(float(score))
        
        return result_ids, result_scores
    
    def remove_vector(self, node_id: str):
        """Remove vector from index (FAISS doesn't support direct removal)"""
        # FAISS doesn't support efficient removal, so we mark as removed
        if node_id in self.reverse_id_map:
            internal_idx = self.reverse_id_map[node_id]
            del self.id_map[internal_idx]
            del self.reverse_id_map[node_id]
    
    def save(self, path: str):
        """Save FAISS index to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.faiss")
            
            # Save ID mappings
            metadata = {
                "id_map": self.id_map,
                "reverse_id_map": self.reverse_id_map,
                "dimension": self.dimension,
                "index_type": self.index_type
            }
            
            with open(f"{path}.metadata", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved FAISS index to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise
    
    def load(self, path: str):
        """Load FAISS index from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{path}_faiss.faiss")
            
            # Load ID mappings
            with open(f"{path}_faiss.metadata", 'rb') as f:
                metadata = pickle.load(f)
            
            self.id_map = metadata["id_map"]
            self.reverse_id_map = metadata["reverse_id_map"]
            self.dimension = metadata["dimension"]
            self.index_type = metadata["index_type"]
            
            logger.info(f"Loaded FAISS index from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise


class NodeVectorIndex:
    """High-level interface for node-based vector operations"""
    
    def __init__(self, embedding_manager: EmbeddingManager, index_type: str = "flat"):
        self.embedding_manager = embedding_manager
        self.index = FAISSIndex(
            dimension=embedding_manager.get_embedding_dim(),
            index_type=index_type
        )
        self.nodes = {}  # Map node_id to Node object
    
    def add_node(self, node: Node):
        """Add a single node to the index"""
        if node.embedding is None:
            self.embedding_manager.embed_node(node)
        
        self.index.add_vectors(
            node.embedding.reshape(1, -1),
            [node.node_id]
        )
        self.nodes[node.node_id] = node
    
    def add_nodes(self, nodes: List[Node]):
        """Add multiple nodes to the index"""
        # Generate embeddings for nodes that don't have them
        nodes_to_embed = [node for node in nodes if node.embedding is None]
        if nodes_to_embed:
            self.embedding_manager.embed_nodes(nodes_to_embed)
        
        # Collect vectors and IDs
        vectors = []
        ids = []
        
        for node in nodes:
            if node.embedding is not None:
                vectors.append(node.embedding)
                ids.append(node.node_id)
                self.nodes[node.node_id] = node
        
        if vectors:
            vectors_array = np.vstack(vectors)
            self.index.add_vectors(vectors_array, ids)
    
    def search_similar_nodes(
        self, 
        query_node: Node, 
        k: int = 10,
        node_types: Optional[List[NodeType]] = None
    ) -> List[Tuple[Node, float]]:
        """Search for nodes similar to query node"""
        if query_node.embedding is None:
            self.embedding_manager.embed_node(query_node)
        
        # Search index
        node_ids, scores = self.index.search(query_node.embedding, k * 2)  # Get more than needed for filtering
        
        # Filter by node type if specified
        results = []
        for node_id, score in zip(node_ids, scores):
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                # Skip self
                if node.node_id == query_node.node_id:
                    continue
                
                # Filter by node type
                if node_types and node.node_type not in node_types:
                    continue
                
                results.append((node, score))
                
                if len(results) >= k:
                    break
        
        return results
    
    def search_by_text(
        self, 
        query_text: str, 
        k: int = 10,
        node_types: Optional[List[NodeType]] = None
    ) -> List[Tuple[Node, float]]:
        """Search for nodes similar to query text"""
        query_embedding = self.embedding_manager.embed_text(query_text)
        
        # Search index
        node_ids, scores = self.index.search(query_embedding, k * 2)
        
        # Filter and return results
        results = []
        for node_id, score in zip(node_ids, scores):
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                # Filter by node type
                if node_types and node.node_type not in node_types:
                    continue
                
                results.append((node, score))
                
                if len(results) >= k:
                    break
        
        return results
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID"""
        return self.nodes.get(node_id)
    
    def remove_node(self, node_id: str):
        """Remove node from index"""
        if node_id in self.nodes:
            self.index.remove_vector(node_id)
            del self.nodes[node_id]
    
    def save(self, path: str):
        """Save index and nodes to disk"""
        # Save FAISS index
        self.index.save(f"{path}_faiss")
        
        # Save nodes
        with open(f"{path}_nodes.pkl", 'wb') as f:
            pickle.dump(self.nodes, f)
        
        logger.info(f"Saved node vector index to {path}")
    
    def load(self, path: str):
        """Load index and nodes from disk"""
        # Load FAISS index
        self.index.load(f"{path}/vectors")
        
        # Load nodes
        with open(f"{path}/vectors_nodes.pkl", 'rb') as f:
            self.nodes = pickle.load(f)
        
        logger.info(f"Loaded node vector index from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_nodes": len(self.nodes),
            "index_size": self.index.index.ntotal,
            "dimension": self.index.dimension,
            "index_type": self.index.index_type,
            "node_type_counts": {
                node_type.value: sum(1 for node in self.nodes.values() if node.node_type == node_type)
                for node_type in NodeType
            }
        }
