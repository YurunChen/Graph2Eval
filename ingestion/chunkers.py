"""
Document chunkers for creating meaningful text chunks
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from enum import Enum

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger

from .parsers import ParsedElement, DocumentStructure
from config_manager import get_config


class ChunkType(Enum):
    """Types of chunks"""
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    FIXED_SIZE = "fixed_size"


@dataclass
class Chunk:
    """Represents a document chunk"""
    chunk_id: str
    chunk_type: ChunkType
    content: str
    metadata: Dict[str, Any]
    source_elements: List[str]  # IDs of source ParsedElements
    start_char: int = 0
    end_char: int = 0
    page_num: Optional[int] = None
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = None
    
    def __post_init__(self):
        if self.child_chunk_ids is None:
            self.child_chunk_ids = []


class DocumentChunker(ABC):
    """Abstract base class for document chunkers"""
    
    @abstractmethod
    def chunk(self, document: DocumentStructure) -> List[Chunk]:
        """Chunk document into meaningful segments"""
        pass


class SemanticChunker(DocumentChunker):
    """Semantic chunker that groups text by semantic similarity"""
    
    def __init__(
        self, 
        model_name: str = None,
        similarity_threshold: float = None,
        min_chunk_size: int = None,
        max_chunk_size: int = None,
        overlap_size: int = None
    ):
        # 从配置文件获取参数
        config = get_config()
        chunking_config = config.ingestion.get('chunking', {})
        
        # 使用配置或默认值
        self.model = SentenceTransformer(model_name or "all-MiniLM-L6-v2")
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else 0.7
        self.min_chunk_size = min_chunk_size or chunking_config.get('min_chunk_size', 50)
        self.max_chunk_size = max_chunk_size or chunking_config.get('chunk_size', 1000)
        self.overlap_size = overlap_size or chunking_config.get('overlap_size', 100)
        
        # Load spaCy model for sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using simple sentence splitting.")
            self.nlp = None
    
    def chunk(self, document: DocumentStructure) -> List[Chunk]:
        """Create semantic chunks based on sentence similarity"""
        # Type check to ensure document is DocumentStructure
        if not hasattr(document, 'elements'):
            logger.error(f"Expected DocumentStructure, got {type(document).__name__}. Document must have 'elements' attribute.")
            return []
        
        chunks = []
        
        # Group elements by type
        text_elements = [
            elem for elem in document.elements 
            if elem.element_type in ["paragraph", "heading", "list_item"]
        ]
        
        if not text_elements:
            return chunks
        
        # Extract sentences from all text elements
        sentences = []
        element_mapping = []  # Track which element each sentence belongs to
        
        for elem in text_elements:
            elem_sentences = self._extract_sentences(elem.content)
            sentences.extend(elem_sentences)
            element_mapping.extend([elem.element_id] * len(elem_sentences))
        
        if not sentences:
            return chunks
        
        # Compute embeddings
        logger.info(f"Computing embeddings for {len(sentences)} sentences")
        embeddings = self.model.encode(sentences)
        
        # Group sentences into semantic clusters
        clusters = self._cluster_sentences(sentences, embeddings, element_mapping)
        
        # Convert clusters to chunks
        for i, cluster in enumerate(clusters):
            chunk_content = " ".join(cluster["sentences"])
            
            # Ensure chunk size constraints
            if len(chunk_content) < self.min_chunk_size:
                # Merge with previous chunk if too small
                if chunks and len(chunks[-1].content) + len(chunk_content) <= self.max_chunk_size:
                    chunks[-1].content += " " + chunk_content
                    chunks[-1].source_elements.extend(cluster["elements"])
                    chunks[-1].end_char += len(chunk_content) + 1
                    continue
            
            # Split if too large
            if len(chunk_content) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk_content, cluster["elements"], i)
                chunks.extend(sub_chunks)
            else:
                chunk = Chunk(
                    chunk_id=f"semantic_chunk_{i}",
                    chunk_type=ChunkType.SEMANTIC,
                    content=chunk_content,
                    metadata={
                        "sentence_count": len(cluster["sentences"]),
                        "avg_similarity": cluster.get("avg_similarity", 0.0),
                        "source_element_count": len(set(cluster["elements"]))
                    },
                    source_elements=list(set(cluster["elements"])),
                    start_char=0,  # Would need document-level tracking
                    end_char=len(chunk_content)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _cluster_sentences(
        self, 
        sentences: List[str], 
        embeddings: np.ndarray,
        element_mapping: List[str]
    ) -> List[Dict[str, Any]]:
        """Cluster sentences by semantic similarity"""
        if len(sentences) == 0:
            return []
        
        clusters = []
        current_cluster = {
            "sentences": [sentences[0]],
            "embeddings": [embeddings[0]],
            "elements": [element_mapping[0]]
        }
        
        for i in range(1, len(sentences)):
            # Calculate similarity with current cluster
            cluster_embedding = np.mean(current_cluster["embeddings"], axis=0)
            similarity = cosine_similarity([embeddings[i]], [cluster_embedding])[0][0]
            
            if similarity >= self.similarity_threshold:
                # Add to current cluster
                current_cluster["sentences"].append(sentences[i])
                current_cluster["embeddings"].append(embeddings[i])
                current_cluster["elements"].append(element_mapping[i])
            else:
                # Start new cluster
                if current_cluster["sentences"]:
                    current_cluster["avg_similarity"] = self._calculate_avg_similarity(
                        current_cluster["embeddings"]
                    )
                    clusters.append(current_cluster)
                
                current_cluster = {
                    "sentences": [sentences[i]],
                    "embeddings": [embeddings[i]],
                    "elements": [element_mapping[i]]
                }
        
        # Add last cluster
        if current_cluster["sentences"]:
            current_cluster["avg_similarity"] = self._calculate_avg_similarity(
                current_cluster["embeddings"]
            )
            clusters.append(current_cluster)
        
        return clusters
    
    def _calculate_avg_similarity(self, embeddings: List[np.ndarray]) -> float:
        """Calculate average pairwise similarity within cluster"""
        if len(embeddings) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 1.0
    
    def _split_large_chunk(
        self, 
        content: str, 
        elements: List[str], 
        base_id: int
    ) -> List[Chunk]:
        """Split large chunk into smaller pieces"""
        chunks = []
        words = content.split()
        
        # Calculate approximate words per chunk
        words_per_chunk = self.max_chunk_size // 6  # Rough estimate: 6 chars per word
        
        for i in range(0, len(words), words_per_chunk - self.overlap_size // 6):
            chunk_words = words[i:i + words_per_chunk]
            chunk_content = " ".join(chunk_words)
            
            chunk = Chunk(
                chunk_id=f"semantic_chunk_{base_id}_{i // words_per_chunk}",
                chunk_type=ChunkType.SEMANTIC,
                content=chunk_content,
                metadata={
                    "is_split": True,
                    "original_chunk_id": f"semantic_chunk_{base_id}",
                    "split_index": i // words_per_chunk
                },
                source_elements=elements,
                start_char=0,
                end_char=len(chunk_content)
            )
            chunks.append(chunk)
        
        return chunks


class HierarchicalChunker(DocumentChunker):
    """Hierarchical chunker that respects document structure"""
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap_size: int = 50,
        respect_boundaries: bool = True
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.respect_boundaries = respect_boundaries
    
    def chunk(self, document: DocumentStructure) -> List[Chunk]:
        """Create hierarchical chunks respecting document structure"""
        # Type check to ensure document is DocumentStructure
        if not hasattr(document, 'elements'):
            logger.error(f"Expected DocumentStructure, got {type(document).__name__}. Document must have 'elements' attribute.")
            return []
        
        chunks = []
        
        # Build hierarchy
        hierarchy = self._build_hierarchy(document.elements)
        
        # Create chunks at different levels
        chunks.extend(self._create_document_chunks(hierarchy))
        chunks.extend(self._create_section_chunks(hierarchy))
        chunks.extend(self._create_paragraph_chunks(hierarchy))
        
        return chunks
    
    def _build_hierarchy(self, elements: List[ParsedElement]) -> Dict[str, Any]:
        """Build document hierarchy"""
        hierarchy = {
            "type": "document",
            "content": "",
            "children": [],
            "elements": []
        }
        
        current_section = None
        current_subsection = None
        
        for element in elements:
            if element.element_type == "heading":
                level = element.metadata.get("level", 1)
                
                if level == 1:
                    # New main section
                    current_section = {
                        "type": "section",
                        "title": element.content,
                        "content": element.content,
                        "children": [],
                        "elements": [element.element_id],
                        "level": level
                    }
                    hierarchy["children"].append(current_section)
                    current_subsection = None
                    
                elif level == 2 and current_section:
                    # New subsection
                    current_subsection = {
                        "type": "subsection",
                        "title": element.content,
                        "content": element.content,
                        "children": [],
                        "elements": [element.element_id],
                        "level": level
                    }
                    current_section["children"].append(current_subsection)
                    
                elif level > 2:
                    # Sub-subsection - add to current subsection or section
                    target = current_subsection or current_section or hierarchy
                    if target:
                        target["content"] += " " + element.content
                        target["elements"].append(element.element_id)
                        
            else:
                # Add content to appropriate level
                target = current_subsection or current_section or hierarchy
                if target:
                    target["content"] += " " + element.content
                    target["elements"].append(element.element_id)
        
        return hierarchy
    
    def _create_document_chunks(self, hierarchy: Dict[str, Any]) -> List[Chunk]:
        """Create document-level chunks"""
        chunks = []
        
        full_content = self._extract_full_content(hierarchy)
        if len(full_content) > self.max_chunk_size:
            # Split into multiple document chunks
            sub_chunks = self._split_content(
                full_content, 
                hierarchy["elements"],
                "hierarchical_doc"
            )
            chunks.extend(sub_chunks)
        else:
            chunk = Chunk(
                chunk_id="hierarchical_doc_0",
                chunk_type=ChunkType.HIERARCHICAL,
                content=full_content,
                metadata={
                    "hierarchy_level": "document",
                    "section_count": len(hierarchy["children"])
                },
                source_elements=hierarchy["elements"]
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_section_chunks(self, hierarchy: Dict[str, Any]) -> List[Chunk]:
        """Create section-level chunks"""
        chunks = []
        
        for i, section in enumerate(hierarchy["children"]):
            content = self._extract_full_content(section)
            
            if len(content) > self.max_chunk_size:
                sub_chunks = self._split_content(
                    content,
                    section["elements"],
                    f"hierarchical_section_{i}"
                )
                chunks.extend(sub_chunks)
            else:
                chunk = Chunk(
                    chunk_id=f"hierarchical_section_{i}",
                    chunk_type=ChunkType.HIERARCHICAL,
                    content=content,
                    metadata={
                        "hierarchy_level": "section",
                        "title": section.get("title", ""),
                        "subsection_count": len(section["children"])
                    },
                    source_elements=section["elements"]
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_paragraph_chunks(self, hierarchy: Dict[str, Any]) -> List[Chunk]:
        """Create paragraph-level chunks"""
        chunks = []
        chunk_id_counter = 0
        
        def process_node(node: Dict[str, Any], prefix: str = ""):
            nonlocal chunk_id_counter
            
            # Process children first
            for child in node.get("children", []):
                process_node(child, f"{prefix}_sub")
            
            # Create chunks from direct content
            if node.get("elements"):
                content = node["content"]
                if len(content) >= self.min_chunk_size:
                    if len(content) > self.max_chunk_size:
                        sub_chunks = self._split_content(
                            content,
                            node["elements"],
                            f"hierarchical_para_{chunk_id_counter}"
                        )
                        chunks.extend(sub_chunks)
                        chunk_id_counter += len(sub_chunks)
                    else:
                        chunk = Chunk(
                            chunk_id=f"hierarchical_para_{chunk_id_counter}",
                            chunk_type=ChunkType.HIERARCHICAL,
                            content=content,
                            metadata={
                                "hierarchy_level": "paragraph",
                                "node_type": node.get("type", "unknown")
                            },
                            source_elements=node["elements"]
                        )
                        chunks.append(chunk)
                        chunk_id_counter += 1
        
        process_node(hierarchy)
        return chunks
    
    def _extract_full_content(self, node: Dict[str, Any]) -> str:
        """Extract full content from hierarchy node"""
        content_parts = [node.get("content", "")]
        
        for child in node.get("children", []):
            child_content = self._extract_full_content(child)
            if child_content:
                content_parts.append(child_content)
        
        return " ".join(part for part in content_parts if part.strip())
    
    def _split_content(
        self, 
        content: str, 
        elements: List[str], 
        base_id: str
    ) -> List[Chunk]:
        """Split content into smaller chunks"""
        chunks = []
        words = content.split()
        words_per_chunk = self.max_chunk_size // 6  # Rough estimate
        
        for i in range(0, len(words), words_per_chunk - self.overlap_size // 6):
            chunk_words = words[i:i + words_per_chunk]
            chunk_content = " ".join(chunk_words)
            
            chunk = Chunk(
                chunk_id=f"{base_id}_{i // words_per_chunk}",
                chunk_type=ChunkType.HIERARCHICAL,
                content=chunk_content,
                metadata={
                    "is_split": True,
                    "split_index": i // words_per_chunk
                },
                source_elements=elements
            )
            chunks.append(chunk)
        
        return chunks


class FixedSizeChunker(DocumentChunker):
    """Simple fixed-size chunker with overlap"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap_size: int = 100,
        separator: str = " "
    ):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.separator = separator
    
    def chunk(self, document: DocumentStructure) -> List[Chunk]:
        """Create fixed-size chunks with overlap"""
        chunks = []
        
        # Combine all text content
        text_parts = []
        element_ids = []
        
        for element in document.elements:
            if element.element_type in ["paragraph", "heading", "list_item"]:
                text_parts.append(element.content)
                element_ids.append(element.element_id)
        
        full_text = self.separator.join(text_parts)
        
        # Split into fixed-size chunks
        start = 0
        chunk_id = 0
        
        while start < len(full_text):
            end = min(start + self.chunk_size, len(full_text))
            
            # Try to break at word boundary
            if end < len(full_text):
                last_space = full_text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_content = full_text[start:end].strip()
            
            if chunk_content:
                chunk = Chunk(
                    chunk_id=f"fixed_chunk_{chunk_id}",
                    chunk_type=ChunkType.FIXED_SIZE,
                    content=chunk_content,
                    metadata={
                        "start_pos": start,
                        "end_pos": end,
                        "char_count": len(chunk_content)
                    },
                    source_elements=element_ids,
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.overlap_size if end < len(full_text) else end
        
        return chunks
