"""
Graph builder for constructing knowledge graphs from documents
"""

import re
import uuid
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger

from ingestion.parsers import DocumentStructure, ParsedElement
from ingestion.chunkers import Chunk, DocumentChunker, SemanticChunker
from .node_types import Node, NodeType, create_node, ChunkNode, ParagraphNode, TableNode, HeadingNode, FigureNode, EntityNode, WebPageNode, WebElementNode
from .edge_types import Edge, EdgeType, create_edge, SequenceEdge, ContainsEdge, ReferenceEdge, SemanticEdge, EntityRelationEdge, TableContextEdge, FigureContextEdge, WebNavigationEdge, WebInteractionEdge

# Web Agent imports
try:
    from ingestion.web_collector import WebPageData, WebElement
    WEB_AGENT_AVAILABLE = True
except ImportError:
    WEB_AGENT_AVAILABLE = False
from .embeddings import EmbeddingManager, NodeVectorIndex
from .storage import GraphStorage, JSONStorage
from config_manager import get_config


@dataclass
class WebGraphBuildConfig:
    """Configuration for web graph building"""
    
    # Web-specific settings
    create_navigation_edges: bool = True
    create_interaction_edges: bool = True
    create_layout_edges: bool = True
    create_data_flow_edges: bool = True
    
    # Interaction analysis
    analyze_click_triggers: bool = True
    analyze_form_submissions: bool = True
    analyze_data_flow: bool = True
    
    # Spatial analysis
    spatial_threshold: float = 50.0  # pixels
    layout_analysis: bool = True
    
    # Multi-page settings
    cross_page_links: bool = True
    session_tracking: bool = True
    
    # Inherit from GraphBuildConfig
    create_chunk_nodes: bool = True
    create_element_nodes: bool = True
    create_entity_nodes: bool = True
    create_sequence_edges: bool = True
    create_containment_edges: bool = True
    create_reference_edges: bool = True
    create_semantic_edges: bool = True
    create_entity_edges: bool = True
    create_context_edges: bool = True
    semantic_similarity_threshold: float = 0.7
    reference_detection_threshold: float = 0.8
    entity_confidence_threshold: float = 0.7
    use_chunking: bool = True
    chunk_size: int = 500
    chunk_overlap: int = 50
    entity_types: List[str] = None
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "DATE", "MONEY"]


@dataclass
class GraphBuildConfig:
    """Configuration for graph building"""
    # Node creation settings
    create_chunk_nodes: bool = True
    create_element_nodes: bool = True
    create_entity_nodes: bool = True
    
    # Edge creation settings
    create_sequence_edges: bool = True
    create_containment_edges: bool = True
    create_reference_edges: bool = True
    create_semantic_edges: bool = True
    create_entity_edges: bool = True
    create_context_edges: bool = True
    
    # Thresholds
    semantic_similarity_threshold: float = 0.7
    reference_detection_threshold: float = 0.8
    entity_confidence_threshold: float = 0.7
    
    # Chunking settings
    use_chunking: bool = True
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Entity extraction settings
    entity_types: List[str] = None
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "DATE", "MONEY"]
    
    @classmethod
    def from_config(cls):
        """从配置文件创建图构建配置"""
        config = get_config()
        graph_config = config.graph_rag.get('graph_builder', {})
        
        return cls(
            create_chunk_nodes=graph_config.get('create_chunk_nodes', True),
            create_element_nodes=graph_config.get('create_element_nodes', True),
            create_entity_nodes=graph_config.get('create_entity_nodes', True),
            create_sequence_edges=graph_config.get('create_sequence_edges', True),
            create_containment_edges=graph_config.get('create_containment_edges', True),
            create_reference_edges=graph_config.get('create_reference_edges', True),
            create_semantic_edges=graph_config.get('create_semantic_edges', True),
            create_entity_edges=graph_config.get('create_entity_edges', True),
            create_context_edges=graph_config.get('create_context_edges', True),
            semantic_similarity_threshold=graph_config.get('semantic_similarity_threshold', 0.7),
            reference_detection_threshold=graph_config.get('reference_detection_threshold', 0.8),
            entity_confidence_threshold=graph_config.get('entity_confidence_threshold', 0.7),
            use_chunking=graph_config.get('use_chunking', True),
            chunk_size=graph_config.get('chunk_size', 500),
            chunk_overlap=graph_config.get('chunk_overlap', 50)
        )


class DocumentGraph:
    """Represents a document as a knowledge graph"""
    
    def __init__(self, 
                 storage: GraphStorage,
                 embedding_manager: EmbeddingManager,
                 vector_index: Optional[NodeVectorIndex] = None):
        self.storage = storage
        self.embedding_manager = embedding_manager
        self.vector_index = vector_index or NodeVectorIndex(embedding_manager)
        
        # Graph statistics
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "node_types": {},
            "edge_types": {}
        }
    
    def add_node(self, node: Node):
        """Add node to graph"""
        self.storage.add_node(node)
        self.vector_index.add_node(node)
        self._update_stats("node", node.node_type.value)
    
    def add_edge(self, edge: Edge):
        """Add edge to graph"""
        self.storage.add_edge(edge)
        self._update_stats("edge", edge.edge_type.value)
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID"""
        return self.storage.get_node(node_id)
    
    def get_subgraph(self, node_ids: List[str], max_hops: int = 1) -> Tuple[List[Node], List[Edge]]:
        """Get subgraph around nodes"""
        return self.storage.get_subgraph(node_ids, max_hops)
    
    def find_similar_nodes(self, query_text: str, k: int = 10, node_types: Optional[List[NodeType]] = None) -> List[Tuple[Node, float]]:
        """Find nodes similar to query text"""
        return self.vector_index.search_by_text(query_text, k, node_types)
    
    def get_neighbors(self, node_id: str, edge_types: Optional[List[EdgeType]] = None) -> List[Tuple[Node, Edge]]:
        """Get neighbors of a node"""
        return self.storage.get_neighbors(node_id, edge_types)
    
    def _update_stats(self, item_type: str, item_subtype: str):
        """Update graph statistics"""
        if item_type == "node":
            self.stats["total_nodes"] += 1
            self.stats["node_types"][item_subtype] = self.stats["node_types"].get(item_subtype, 0) + 1
        elif item_type == "edge":
            self.stats["total_edges"] += 1
            self.stats["edge_types"][item_subtype] = self.stats["edge_types"].get(item_subtype, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return self.stats.copy()
    
    def save(self, path: str, vector_save_path: str = None):
        """Save graph to storage"""
        # Use the path as directory and save with specific filename
        from pathlib import Path
        graph_dir = Path(path)
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_file = graph_dir / "knowledge_graph.json"
        self.storage.save(str(graph_file))
        
        # Use provided vector path or fallback to same directory as graph
        if vector_save_path:
            from pathlib import Path
            vector_dir = Path(vector_save_path)
            vector_dir.mkdir(parents=True, exist_ok=True)
            vector_file_path = vector_dir / "vectors"
            self.vector_index.save(str(vector_file_path))
        else:
            self.vector_index.save(f"{path}_vectors")
    
    def load(self, graph_path: str, vector_path: str):
        """Load graph from storage"""
        # Handle graph_path as directory - look for knowledge_graph.json inside
        from pathlib import Path
        graph_dir = Path(graph_path)
        if graph_dir.is_dir():
            graph_file = graph_dir / "knowledge_graph.json"
            if graph_file.exists():
                self.storage.load(str(graph_file))
            else:
                raise FileNotFoundError(f"Graph file not found: {graph_file}")
        else:
            # Assume it's a direct file path
            self.storage.load(graph_path)
        
        self.vector_index.load(vector_path)


class GraphBuilder:
    """Builds knowledge graphs from documents"""
    
    def __init__(self, 
                 config: Optional[GraphBuildConfig] = None,
                 embedding_manager: Optional[EmbeddingManager] = None,
                 storage: Optional[GraphStorage] = None):
        self.config = config or GraphBuildConfig()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.storage = storage or JSONStorage()
        
        # Initialize components
        self.chunker = SemanticChunker(
            max_chunk_size=self.config.chunk_size,
            overlap_size=self.config.chunk_overlap
        ) if self.config.use_chunking else None
        
        # Load spaCy model for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Entity extraction will be disabled.")
            self.nlp = None
            self.config.create_entity_nodes = False
    
    def build_graph(self, document: DocumentStructure) -> DocumentGraph:
        """Build complete graph from document"""
        # Type check to ensure document is DocumentStructure
        logger.info(f"Document type: {type(document).__name__}")
        logger.info(f"Document attributes: {dir(document)}")
        
        if not hasattr(document, 'elements'):
            raise TypeError(f"Expected DocumentStructure, got {type(document).__name__}. Document must have 'elements' attribute.")
        
        logger.info(f"Building graph for document: {document.file_path}")
        
        # Create graph
        graph = DocumentGraph(
            storage=self.storage,
            embedding_manager=self.embedding_manager
        )
        
        # Step 1: Create chunk nodes if enabled
        chunks = []
        if self.config.use_chunking and self.chunker:
            chunks = self.chunker.chunk(document)
            if self.config.create_chunk_nodes:
                self._create_chunk_nodes(graph, chunks)
        
        # Step 2: Create element nodes
        if self.config.create_element_nodes:
            self._create_element_nodes(graph, document.elements)
        
        # Step 3: Create entity nodes
        if self.config.create_entity_nodes and self.nlp:
            self._create_entity_nodes(graph, document.elements)
        
        # Step 4: Create edges
        self._create_edges(graph, document, chunks)
        
        logger.info(f"Graph built successfully: {graph.get_stats()}")
        return graph
    
    def _create_chunk_nodes(self, graph: DocumentGraph, chunks: List[Chunk]):
        """Create nodes from document chunks"""
        logger.info(f"Creating {len(chunks)} chunk nodes")
        
        for chunk in chunks:
            node = ChunkNode(
                node_id=chunk.chunk_id,
                node_type=NodeType.CHUNK,
                content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "chunk_type": chunk.chunk_type.value,
                    "source_elements": chunk.source_elements
                },
                page_num=chunk.page_num,
                chunk_size=len(chunk.content),
                chunk_index=getattr(chunk, 'chunk_index', 0)
            )
            
            graph.add_node(node)
    
    def _create_element_nodes(self, graph: DocumentGraph, elements: List[ParsedElement]):
        """Create nodes from document elements"""
        logger.info(f"Creating nodes from {len(elements)} elements")
        
        for i, element in enumerate(elements):
            logger.debug(f"Processing element {i}: type={type(element).__name__}, element_type={getattr(element, 'element_type', 'N/A')}")
            
            # Check if element has the expected attributes
            if not hasattr(element, 'element_type'):
                logger.warning(f"Element {i} does not have element_type attribute: {dir(element)}")
                continue
                
            node = self._element_to_node(element)
            if node:
                graph.add_node(node)
    
    def _element_to_node(self, element: ParsedElement) -> Optional[Node]:
        """Convert ParsedElement to appropriate Node type"""
        node_type_mapping = {
            "paragraph": NodeType.PARAGRAPH,
            "heading": NodeType.HEADING,
            "table": NodeType.TABLE,
            "figure": NodeType.FIGURE,
            "list": NodeType.PARAGRAPH,  # Treat lists as paragraphs
            "list_item": NodeType.PARAGRAPH
        }
        
        node_type = node_type_mapping.get(element.element_type)
        if not node_type:
            return None
        
        # Create appropriate node subclass
        if node_type == NodeType.PARAGRAPH:
            return ParagraphNode(
                node_id=element.element_id or f"para_{uuid.uuid4().hex[:8]}",
                node_type=node_type,
                content=element.content,
                metadata=element.metadata.copy(),
                source_file=None,  # Will be set by document context
                page_num=element.page_num,
                bbox=element.bbox,
                word_count=len(element.content.split()),
                paragraph_index=element.metadata.get("paragraph_index", 0)
            )
        
        elif node_type == NodeType.HEADING:
            return HeadingNode(
                node_id=element.element_id or f"heading_{uuid.uuid4().hex[:8]}",
                node_type=node_type,
                content=element.content,
                metadata=element.metadata.copy(),
                source_file=None,
                page_num=element.page_num,
                bbox=element.bbox,
                level=element.metadata.get("level", 1)
            )
        
        elif node_type == NodeType.TABLE:
            return TableNode(
                node_id=element.element_id or f"table_{uuid.uuid4().hex[:8]}",
                node_type=node_type,
                content=element.content,
                metadata=element.metadata.copy(),
                source_file=None,
                page_num=element.page_num,
                bbox=element.bbox,
                table_data=element.metadata.get("table_data", []),
                rows=element.metadata.get("rows", 0),
                cols=element.metadata.get("cols", 0),
                has_header=element.metadata.get("has_header", False)
            )
        
        elif node_type == NodeType.FIGURE:
            return FigureNode(
                node_id=element.element_id or f"figure_{uuid.uuid4().hex[:8]}",
                node_type=node_type,
                content=element.content,
                metadata=element.metadata.copy(),
                source_file=None,
                page_num=element.page_num,
                bbox=element.bbox,
                figure_type=element.metadata.get("figure_type", "image"),
                caption=element.metadata.get("caption", ""),
                alt_text=element.metadata.get("alt_text", "")
            )
        
        return None
    
    def _create_entity_nodes(self, graph: DocumentGraph, elements: List[ParsedElement]):
        """Extract and create entity nodes"""
        # Type check to ensure elements is a list
        if not isinstance(elements, list):
            logger.error(f"Expected list of ParsedElement, got {type(elements).__name__}")
            return
        
        logger.info("Extracting entities from document")
        logger.info(f"Number of elements: {len(elements)}")
        
        entity_mentions = {}  # canonical_name -> EntityNode
        
        for i, element in enumerate(elements):
            logger.debug(f"Processing element {i}: type={type(element).__name__}, element_type={getattr(element, 'element_type', 'N/A')}")
            
            # Check if element has the expected attributes
            if not hasattr(element, 'element_type'):
                logger.warning(f"Element {i} does not have element_type attribute: {dir(element)}")
                continue
                
            if not hasattr(element, 'content'):
                logger.warning(f"Element {i} does not have content attribute: {dir(element)}")
                continue
                
            if element.element_type in ["paragraph", "heading"]:
                entities = self._extract_entities(element.content)
                
                for ent_text, ent_type, confidence in entities:
                    canonical_name = ent_text.strip()
                    
                    if confidence < self.config.entity_confidence_threshold:
                        continue
                    
                    # Get or create entity node
                    if canonical_name not in entity_mentions:
                        entity_node = EntityNode(
                            node_id=f"entity_{uuid.uuid4().hex[:8]}",
                            node_type=NodeType.ENTITY,
                            content=ent_text,
                            canonical_name=canonical_name,
                            entity_type=ent_type,
                            confidence=confidence
                        )
                        entity_mentions[canonical_name] = entity_node
                        graph.add_node(entity_node)
                    
                    # Add mention to entity
                    entity_node = entity_mentions[canonical_name]
                    entity_node.add_mention(
                        context=element.content[:200] + "..." if len(element.content) > 200 else element.content,
                        source_node_id=element.element_id
                    )
        
        logger.info(f"Created {len(entity_mentions)} entity nodes")
    
    def _extract_entities(self, text: str) -> List[Tuple[str, str, float]]:
        """Extract named entities from text"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in self.config.entity_types:
                entities.append((ent.text, ent.label_, 1.0))  # spaCy doesn't provide confidence scores directly
        
        return entities
    
    def _create_edges(self, graph: DocumentGraph, document: DocumentStructure, chunks: List[Chunk]):
        """Create all types of edges in the graph"""
        logger.info("Creating edges in graph")
        
        # Get all nodes for edge creation
        all_nodes = []
        for node_type in NodeType:
            nodes = graph.storage.find_nodes(node_type=node_type)
            all_nodes.extend(nodes)
        
        # Create different types of edges
        if self.config.create_sequence_edges:
            self._create_sequence_edges(graph, all_nodes)
        
        if self.config.create_containment_edges:
            self._create_containment_edges(graph, all_nodes, document.elements)
        
        if self.config.create_reference_edges:
            self._create_reference_edges(graph, all_nodes)
        
        if self.config.create_semantic_edges:
            self._create_semantic_edges(graph, all_nodes)
        
        if self.config.create_entity_edges:
            self._create_entity_edges(graph, all_nodes)
        
        if self.config.create_context_edges:
            self._create_context_edges(graph, all_nodes)
    
    def _create_sequence_edges(self, graph: DocumentGraph, nodes: List[Node]):
        """Create sequential edges between adjacent elements"""
        # Group nodes by page and sort by position
        page_nodes = {}
        for node in nodes:
            if node.page_num and node.node_type in [NodeType.PARAGRAPH, NodeType.HEADING, NodeType.TABLE, NodeType.FIGURE]:
                page = node.page_num
                if page not in page_nodes:
                    page_nodes[page] = []
                page_nodes[page].append(node)
        
        # Create sequence edges within each page
        edge_count = 0
        for page, page_node_list in page_nodes.items():
            # Sort by bbox position (top to bottom, left to right)
            if all(node.bbox for node in page_node_list):
                page_node_list.sort(key=lambda n: (n.bbox[1], n.bbox[0]))  # Sort by y0, then x0
            
            # Create sequence edges
            for i in range(len(page_node_list) - 1):
                edge = SequenceEdge(
                    edge_id=f"seq_{uuid.uuid4().hex[:8]}",
                    edge_type=EdgeType.SEQUENCE,
                    source_node_id=page_node_list[i].node_id,
                    target_node_id=page_node_list[i + 1].node_id,
                    sequence_type="next",
                    distance=1
                )
                graph.add_edge(edge)
                edge_count += 1
        
        logger.info(f"Created {edge_count} sequence edges")
    
    def _create_containment_edges(self, graph: DocumentGraph, nodes: List[Node], elements: List[ParsedElement]):
        """Create containment edges (heading -> paragraphs, etc.)"""
        # Type check to ensure elements is a list
        if not isinstance(elements, list):
            logger.error(f"Expected list of ParsedElement, got {type(elements).__name__}")
            return
        
        edge_count = 0
        
        # Find hierarchical relationships
        headings = [n for n in nodes if n.node_type == NodeType.HEADING]
        other_nodes = [n for n in nodes if n.node_type != NodeType.HEADING]
        
        for heading in headings:
            # Find elements that belong to this heading
            for node in other_nodes:
                if self._is_contained_by(node, heading):
                    edge = ContainsEdge(
                        edge_id=f"contains_{uuid.uuid4().hex[:8]}",
                        edge_type=EdgeType.CONTAINS,
                        source_node_id=heading.node_id,
                        target_node_id=node.node_id,
                        containment_type="contains"
                    )
                    graph.add_edge(edge)
                    edge_count += 1
        
        logger.info(f"Created {edge_count} containment edges")
    
    def _is_contained_by(self, child_node: Node, parent_node: Node) -> bool:
        """Determine if child node is contained by parent node"""
        # Simple heuristic: same page and child comes after parent
        if (child_node.page_num == parent_node.page_num and 
            child_node.bbox and parent_node.bbox):
            # Child should come after parent vertically
            return child_node.bbox[1] > parent_node.bbox[1]  # y0 of child > y0 of parent
        
        return False
    
    def _create_reference_edges(self, graph: DocumentGraph, nodes: List[Node]):
        """Create reference edges based on explicit mentions"""
        edge_count = 0
        
        # Look for references like "Table 1", "Figure 2", "Section 3"
        reference_patterns = [
            r'\b(?:table|Table|TABLE)\s+(\d+|\w+)',
            r'\b(?:figure|Figure|FIGURE)\s+(\d+|\w+)',
            r'\b(?:section|Section|SECTION)\s+(\d+|\w+)',
            r'\b(?:appendix|Appendix|APPENDIX)\s+(\w+)',
        ]
        
        text_nodes = [n for n in nodes if n.node_type in [NodeType.PARAGRAPH, NodeType.HEADING]]
        target_nodes = [n for n in nodes if n.node_type in [NodeType.TABLE, NodeType.FIGURE, NodeType.HEADING]]
        
        for text_node in text_nodes:
            for pattern in reference_patterns:
                matches = re.finditer(pattern, text_node.content, re.IGNORECASE)
                
                for match in matches:
                    # Find matching target node
                    for target_node in target_nodes:
                        if self._is_reference_match(match.group(), target_node):
                            edge = ReferenceEdge(
                                edge_id=f"ref_{uuid.uuid4().hex[:8]}",
                                edge_type=EdgeType.REFERS_TO,
                                source_node_id=text_node.node_id,
                                target_node_id=target_node.node_id,
                                reference_type="mentions",
                                reference_text=match.group(),
                                confidence=0.9
                            )
                            graph.add_edge(edge)
                            edge_count += 1
                            break
        
        logger.info(f"Created {edge_count} reference edges")
    
    def _is_reference_match(self, reference_text: str, target_node: Node) -> bool:
        """Check if reference text matches target node"""
        # Simple matching logic - can be enhanced
        ref_lower = reference_text.lower()
        
        if target_node.node_type == NodeType.TABLE and "table" in ref_lower:
            return True
        elif target_node.node_type == NodeType.FIGURE and "figure" in ref_lower:
            return True
        elif target_node.node_type == NodeType.HEADING and "section" in ref_lower:
            return True
        
        return False
    
    def _create_semantic_edges(self, graph: DocumentGraph, nodes: List[Node]):
        """Create semantic similarity edges"""
        edge_count = 0
        
        # Get embeddings for all nodes
        text_nodes = [n for n in nodes if n.node_type in [NodeType.PARAGRAPH, NodeType.HEADING]]
        
        if len(text_nodes) < 2:
            return
        
        # Ensure all nodes have embeddings
        nodes_without_embeddings = [n for n in text_nodes if n.embedding is None]
        if nodes_without_embeddings:
            graph.embedding_manager.embed_nodes(nodes_without_embeddings)
        
        # Compute pairwise similarities
        embeddings = np.array([n.embedding for n in text_nodes])
        similarities = cosine_similarity(embeddings)
        
        for i, node1 in enumerate(text_nodes):
            for j, node2 in enumerate(text_nodes[i+1:], i+1):
                similarity = similarities[i][j]
                
                if similarity >= self.config.semantic_similarity_threshold:
                    edge = SemanticEdge(
                        edge_id=f"sem_{uuid.uuid4().hex[:8]}",
                        edge_type=EdgeType.SEMANTIC_SIM,
                        source_node_id=node1.node_id,
                        target_node_id=node2.node_id,
                        similarity_score=float(similarity),
                        similarity_method="cosine",
                        threshold_used=self.config.semantic_similarity_threshold,
                        bidirectional=True
                    )
                    graph.add_edge(edge)
                    edge_count += 1
        
        logger.info(f"Created {edge_count} semantic edges")
    
    def _create_entity_edges(self, graph: DocumentGraph, nodes: List[Node]):
        """Create edges between entities and their mention contexts"""
        edge_count = 0
        
        entity_nodes = [n for n in nodes if n.node_type == NodeType.ENTITY]
        text_nodes = [n for n in nodes if n.node_type in [NodeType.PARAGRAPH, NodeType.HEADING]]
        
        # Create entity relation edges
        for entity in entity_nodes:
            for mention in entity.mentions:
                source_node_id = mention.get("source_node_id")
                if source_node_id:
                    # Find the source node
                    source_node = next((n for n in text_nodes if n.node_id == source_node_id), None)
                    if source_node:
                        edge = EntityRelationEdge(
                            edge_id=f"ent_rel_{uuid.uuid4().hex[:8]}",
                            edge_type=EdgeType.ENTITY_RELATION,
                            source_node_id=source_node.node_id,
                            target_node_id=entity.node_id,
                            relation_type="MENTIONS",
                            relation_confidence=entity.confidence,
                            relation_context=mention.get("context", "")
                        )
                        graph.add_edge(edge)
                        edge_count += 1
        
        logger.info(f"Created {edge_count} entity relation edges")
    
    def _create_context_edges(self, graph: DocumentGraph, nodes: List[Node]):
        """Create context edges for tables and figures"""
        edge_count = 0
        
        table_nodes = [n for n in nodes if n.node_type == NodeType.TABLE]
        figure_nodes = [n for n in nodes if n.node_type == NodeType.FIGURE]
        text_nodes = [n for n in nodes if n.node_type in [NodeType.PARAGRAPH, NodeType.HEADING]]
        
        # Create table-context edges
        for table_node in table_nodes:
            context_nodes = self._find_nearby_nodes(table_node, text_nodes, max_distance=2)
            
            for context_node, distance in context_nodes:
                edge = TableContextEdge(
                    edge_id=f"table_ctx_{uuid.uuid4().hex[:8]}",
                    edge_type=EdgeType.TABLE_CONTEXT,
                    source_node_id=context_node.node_id,
                    target_node_id=table_node.node_id,
                    context_type="explains",
                    context_position="before" if self._is_before(context_node, table_node) else "after",
                    distance_sentences=distance
                )
                graph.add_edge(edge)
                edge_count += 1
        
        # Create figure-context edges
        for figure_node in figure_nodes:
            context_nodes = self._find_nearby_nodes(figure_node, text_nodes, max_distance=2)
            
            for context_node, distance in context_nodes:
                edge = FigureContextEdge(
                    edge_id=f"fig_ctx_{uuid.uuid4().hex[:8]}",
                    edge_type=EdgeType.FIGURE_CONTEXT,
                    source_node_id=context_node.node_id,
                    target_node_id=figure_node.node_id,
                    context_type="describes",
                    context_position="before" if self._is_before(context_node, figure_node) else "after"
                )
                graph.add_edge(edge)
                edge_count += 1
        
        logger.info(f"Created {edge_count} context edges")
    
    def _find_nearby_nodes(self, target_node: Node, candidate_nodes: List[Node], max_distance: int = 2) -> List[Tuple[Node, int]]:
        """Find nodes near the target node"""
        nearby = []
        
        for candidate in candidate_nodes:
            if (candidate.page_num == target_node.page_num and 
                candidate.bbox and target_node.bbox):
                
                # Simple distance metric based on vertical position
                vertical_distance = abs(candidate.bbox[1] - target_node.bbox[1])
                
                if vertical_distance < 100 * max_distance:  # 100 pixels per distance unit
                    distance = int(vertical_distance // 100) + 1
                    nearby.append((candidate, distance))
        
        return nearby
    
    def _is_before(self, node1: Node, node2: Node) -> bool:
        """Check if node1 comes before node2"""
        if node1.bbox and node2.bbox:
            return node1.bbox[1] < node2.bbox[1]  # Compare y-coordinates
        return False
    
    # Web Graph Building Methods
    def build_web_graph(self, web_pages: List[WebPageData]) -> DocumentGraph:
        """Build web interaction graph from collected page data"""
        
        if not WEB_AGENT_AVAILABLE:
            logger.warning("Web Agent not available, cannot build web graph")
            return None
        
        logger.info(f"Building web graph from {len(web_pages)} pages")
        
        # Create graph
        graph = DocumentGraph(
            storage=self.storage,
            embedding_manager=self.embedding_manager
        )
        
        # Step 1: Create page nodes
        page_nodes = self._create_page_nodes(graph, web_pages)
        
        # Step 2: Create element nodes
        element_nodes = self._create_web_element_nodes(graph, web_pages)
        
        # Step 3: Create web-specific edges
        self._create_web_edges(graph, web_pages, page_nodes, element_nodes)
        
        logger.info(f"Web graph built successfully: {graph.get_stats()}")
        return graph
    
    def _create_page_nodes(self, graph: DocumentGraph, web_pages: List[WebPageData]) -> Dict[str, WebPageNode]:
        """Create nodes for web pages"""
        
        page_nodes = {}
        
        for page_data in web_pages:
            # Determine page type
            page_type = self._classify_page_type(page_data)
            
            # Create page node
            page_node = WebPageNode(
                node_id=f"page_{uuid.uuid4().hex[:8]}",
                node_type=NodeType.WEB_PAGE,
                content=f"Page: {page_data.title}",
                metadata={
                    "url": page_data.url,
                    "title": page_data.title,
                    "page_type": page_type,
                    "load_time": page_data.load_time,
                    "page_size": page_data.page_size,
                    "links": page_data.links,
                    "clickable_elements": page_data.clickable_elements,
                    "form_elements": page_data.form_elements,
                    "table_elements": page_data.table_elements
                },
                url=page_data.url,
                title=page_data.title,
                page_type=page_type,
                load_time=page_data.load_time,
                page_size=page_data.page_size
            )
            
            graph.add_node(page_node)
            page_nodes[page_data.url] = page_node
            
            logger.info(f"Created page node: {page_node.node_id} for {page_data.url}")
        
        return page_nodes
    
    def _create_web_element_nodes(self, graph: DocumentGraph, web_pages: List[WebPageData]) -> Dict[str, WebElementNode]:
        """Create nodes for web elements"""
        
        element_nodes = {}
        
        for page_data in web_pages:
            if not page_data.elements:
                continue
                
            # Check if elements are WebElement objects
            if not hasattr(page_data.elements[0], 'element_type'):
                logger.warning(f"Skipping element node creation for page {page_data.url}: elements are not WebElement objects")
                continue
                
            for element in page_data.elements:
                # Create element node based on type
                element_node = self._create_element_node(element, page_data.url)
                
                graph.add_node(element_node)
                element_nodes[element.element_id] = element_node
        
        logger.info(f"Created {len(element_nodes)} element nodes")
        return element_nodes
    
    def _create_element_node(self, element: WebElement, page_url: str) -> WebElementNode:
        """Create appropriate element node based on element type"""
        
        # Map element type to node type
        type_mapping = {
            'button': NodeType.WEB_BUTTON,
            'input_text': NodeType.WEB_INPUT,
            'input_choice': NodeType.WEB_INPUT,
            'link': NodeType.WEB_LINK,
            'table': NodeType.WEB_TABLE,
            'image': NodeType.WEB_IMAGE,
            'form': NodeType.WEB_FORM,
            'select': NodeType.WEB_INPUT,
            'textarea': NodeType.WEB_INPUT
        }
        
        node_type = type_mapping.get(element.element_type, NodeType.WEB_ELEMENT)
        
        # Create base element node
        element_node = WebElementNode(
            node_id=element.element_id,
            node_type=node_type,
            content=element.text_content or element.placeholder or element.value or f"{element.element_type} element",
            metadata={
                "page_url": page_url,
                "tag_name": element.tag_name,
                "css_selector": element.css_selector,
                "attributes": element.attributes
            },
            source_file=page_url,
            bbox=(element.x, element.y, element.x + element.width, element.y + element.height),
            element_type=element.element_type,
            tag_name=element.tag_name,
            text_content=element.text_content,
            placeholder=element.placeholder,
            value=element.value,
            href=element.href,
            src=element.src,
            x=element.x,
            y=element.y,
            width=element.width,
            height=element.height,
            css_classes=element.css_classes,
            css_selector=element.css_selector,
            is_clickable=element.is_clickable,
            is_input=element.is_input,
            is_visible=element.is_visible,
            is_enabled=element.is_enabled,
            input_type=element.input_type,
            required=element.required,
            options=element.options
        )
        
        return element_node
    
    def _create_web_edges(self, graph: DocumentGraph, web_pages: List[WebPageData], 
                         page_nodes: Dict[str, WebPageNode], 
                         element_nodes: Dict[str, WebElementNode]):
        """Create web-specific edges"""
        
        # Create navigation edges between pages
        if self.config.create_navigation_edges:
            self._create_navigation_edges(graph, web_pages, page_nodes, element_nodes)
        
        # Create interaction edges
        if self.config.create_interaction_edges:
            self._create_interaction_edges(graph, web_pages, element_nodes)
    
    def _create_navigation_edges(self, graph: DocumentGraph, web_pages: List[WebPageData],
                                page_nodes: Dict[str, WebPageNode], 
                                element_nodes: Dict[str, WebElementNode]):
        """Create navigation edges between pages"""
        
        for page_data in web_pages:
            source_page = page_nodes.get(page_data.url)
            if not source_page:
                continue
            
            # Find links to other pages
            for element in page_data.elements:
                if not hasattr(element, 'element_type') or not hasattr(element, 'href'):
                    continue
                if element.element_type == 'a' and element.href:
                    target_url = element.href
                    target_page = page_nodes.get(target_url)
                    
                    if target_page and target_page != source_page:
                        # Create navigation edge
                        nav_edge = WebNavigationEdge(
                            edge_id=f"nav_{source_page.node_id}_{target_page.node_id}",
                            edge_type=EdgeType.WEB_NAVIGATION,
                            source_node_id=source_page.node_id,
                            target_node_id=target_page.node_id,
                            weight=1.0,
                            navigation_type="link",
                            target_url=target_url,
                            navigation_method="GET"
                        )
                        
                        graph.add_edge(nav_edge)
    
    def _create_interaction_edges(self, graph: DocumentGraph, web_pages: List[WebPageData],
                                 element_nodes: Dict[str, WebElementNode]):
        """Create interaction edges between elements"""
        
        for page_data in web_pages:
            # Type check to ensure elements are WebElement objects
            if not page_data.elements:
                continue
                
            # Check if elements are WebElement objects
            if not hasattr(page_data.elements[0], 'text_content'):
                logger.warning(f"Skipping interaction edge creation for page {page_data.url}: elements are not WebElement objects")
                continue
            
            # Find form elements and their submit buttons
            form_elements = [e for e in page_data.elements if e.element_type in ['input_text', 'input_choice', 'select', 'textarea']]
            submit_buttons = [e for e in page_data.elements if e.element_type == 'button' and hasattr(e, 'text_content') and 'submit' in e.text_content.lower()]
            
            # Create form submission edges
            for button in submit_buttons:
                for form_elem in form_elements:
                    button_node = element_nodes.get(button.element_id)
                    form_node = element_nodes.get(form_elem.element_id)
                    
                    if button_node and form_node:
                        interaction_edge = WebInteractionEdge(
                            edge_id=f"interaction_{form_node.node_id}_{button_node.node_id}",
                            edge_type=EdgeType.WEB_INTERACTION,
                            source_node_id=form_node.node_id,
                            target_node_id=button_node.node_id,
                            weight=1.0,
                            interaction_type="form_submit",
                            interaction_data={},
                            interaction_result="success"
                        )
                        graph.add_edge(interaction_edge)
    
    def _classify_page_type(self, page_data: WebPageData) -> str:
        """Classify the type of web page"""
        
        title_lower = page_data.title.lower()
        
        # Check for form elements
        if page_data.form_elements:
            return "form"
        
        # Check for search functionality
        if page_data.elements and hasattr(page_data.elements[0], 'text_content'):
            if any('search' in elem.text_content.lower() for elem in page_data.elements):
                return "search"
        
        # Check for product pages
        if any(word in title_lower for word in ['product', 'item', 'buy', 'shop']):
            return "product"
        
        # Check for listing pages
        if any(word in title_lower for word in ['list', 'results', 'catalog']):
            return "listing"
        
        # Default classification
        return "content"
