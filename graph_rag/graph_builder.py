"""
Graph builder for constructing knowledge graphs from documents
"""

import re
import uuid
import gc
import psutil
import os
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger
import json
from tqdm import tqdm

from ingestion.parsers import DocumentStructure, ParsedElement
from ingestion.chunkers import Chunk, DocumentChunker, SemanticChunker
from .node_types import Node, NodeType, create_node, ChunkNode, ParagraphNode, TableNode, HeadingNode, FigureNode, EntityNode, GraphNode, NodeMetadata
from .edge_types import Edge, EdgeType, create_edge, SequenceEdge, ContainsEdge, ReferenceEdge, SemanticEdge, EntityRelationEdge, TableContextEdge, FigureContextEdge, WebNavigationEdge, WebInteractionEdge, GraphEdge
from .storage import GraphStorage, JSONStorage


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
    
    # Memory optimization settings
    enable_memory_optimization: bool = False
    batch_size: int = 100
    max_elements_per_page: int = 2000
    max_semantic_edges_per_page: int = 500
    max_spatial_edges_per_page: int = 500
    max_functional_edges_per_page: int = 500
    max_interaction_edges_per_page: int = 500
    enable_progressive_building: bool = False
    memory_cleanup_interval: int = 50
    vector_batch_size: int = 50
    
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
            chunk_overlap=graph_config.get('chunk_overlap', 50),
            # Memory optimization settings
            enable_memory_optimization=graph_config.get('enable_memory_optimization', False),
            batch_size=graph_config.get('batch_size', 100),
            max_elements_per_page=graph_config.get('max_elements_per_page', 200),
            max_semantic_edges_per_page=graph_config.get('max_semantic_edges_per_page', 50),
            max_spatial_edges_per_page=graph_config.get('max_spatial_edges_per_page', 30),
            max_functional_edges_per_page=graph_config.get('max_functional_edges_per_page', 25),
            max_interaction_edges_per_page=graph_config.get('max_interaction_edges_per_page', 20),
            enable_progressive_building=graph_config.get('enable_progressive_building', False),
            memory_cleanup_interval=graph_config.get('memory_cleanup_interval', 50),
            vector_batch_size=graph_config.get('vector_batch_size', 50)
        )


@dataclass
class WebGraphBuildConfig(GraphBuildConfig):
    """Configuration for web graph building - extends GraphBuildConfig"""
    
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
    spatial_threshold: float = 50.0
    
    # Memory optimization settings
    enable_memory_optimization: bool = True
    batch_size: int = 100  # Process elements in batches
    max_elements_per_page: int = 200  # Limit elements per page
    max_semantic_edges_per_page: int = 50  # Limit semantic edges per page
    enable_progressive_building: bool = True  # Build graph progressively
    memory_cleanup_interval: int = 50  # Cleanup every N operations  # pixels
    layout_analysis: bool = True
    
    # Multi-page settings
    cross_page_links: bool = True
    session_tracking: bool = True


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
        
        # Graph metadata for additional information
        self.metadata = {}
    
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
        from graph_rag.node_types import node_from_dict
        from graph_rag.edge_types import edge_from_dict
        graph_dir = Path(graph_path)
        if graph_dir.is_dir():
            graph_file = graph_dir / "knowledge_graph.json"
            if graph_file.exists():
                # Load graph data from JSON file
                with open(graph_file, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)

                # Clear existing data
                self.storage.clear()

                # Convert nodes from dict format to list format (handle both old and new formats)
                nodes_data = graph_data.get("nodes", {})
                if isinstance(nodes_data, dict):
                    logger.info("Detected old dict format for nodes, converting to list format")
                    nodes_list = list(nodes_data.values())
                else:
                    nodes_list = nodes_data

                # Load nodes
                for node_data in nodes_list:
                    try:
                        node = node_from_dict(node_data)
                        self.storage.add_node(node)
                    except Exception as e:
                        logger.warning(f"Failed to load node {node_data.get('node_id', 'unknown')}: {e}")

                # Convert edges from dict format to list format (handle both old and new formats)
                edges_data = graph_data.get("edges", {})
                if isinstance(edges_data, dict):
                    logger.info("Detected old dict format for edges, converting to list format")
                    edges_list = list(edges_data.values())
                else:
                    edges_list = edges_data

                # Load edges
                for edge_data in edges_list:
                    try:
                        # Convert old field names to new ones
                        if "source_node" in edge_data and "source_node_id" not in edge_data:
                            edge_data["source_node_id"] = edge_data.pop("source_node")
                        if "target_node" in edge_data and "target_node_id" not in edge_data:
                            edge_data["target_node_id"] = edge_data.pop("target_node")
                        edge = edge_from_dict(edge_data)
                        self.storage.add_edge(edge)
                    except Exception as e:
                        logger.warning(f"Failed to load edge {edge_data.get('edge_id', 'unknown')}: {e}")

                logger.info(f"Loaded graph from {graph_file}")
                logger.info(f"Nodes: {len(self.storage.nodes)}, Edges: {len(self.storage.edges)}")
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
        
        # Memory optimization
        self.operation_count = 0
        self.initial_memory = self._get_memory_usage()
        logger.info(f"🚀 GraphBuilder initialized. Initial memory: {self.initial_memory:.2f} MB")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _log_memory_usage(self, operation: str):
        """Log memory usage for an operation"""
        current_memory = self._get_memory_usage()
        memory_increase = current_memory - self.initial_memory
        logger.debug(f"🧠 Memory after {operation}: {current_memory:.2f} MB (+{memory_increase:.2f} MB)")
    
    def _force_memory_cleanup(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        current_memory = self._get_memory_usage()
        logger.debug(f"🧹 Memory cleanup completed. Current: {current_memory:.2f} MB")
    
    def _should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        if not hasattr(self.config, 'memory_cleanup_interval'):
            return False
        return self.operation_count % self.config.memory_cleanup_interval == 0
    
    def _process_in_batches(self, items: List[Any], batch_size: int, operation_name: str):
        """Process items in batches to reduce memory usage"""
        total_batches = (len(items) + batch_size - 1) // batch_size
        logger.info(f"📦 Processing {len(items)} items in {total_batches} batches for {operation_name}")
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.debug(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
            
            yield batch
            
            # Memory cleanup after each batch
            self.operation_count += 1
            if self._should_cleanup():
                self._force_memory_cleanup()
                self._log_memory_usage(f"batch {batch_num} of {operation_name}")
    
    def _limit_elements_per_page(self, web_elements: List[WebElement]) -> List[WebElement]:
        """Limit the number of elements per page to prevent memory explosion"""
        if not hasattr(self.config, 'max_elements_per_page'):
            return web_elements
        
        # Group elements by page
        page_elements = {}
        for elem in web_elements:
            page_url = getattr(elem, 'page_url', 'unknown')
            if page_url not in page_elements:
                page_elements[page_url] = []
            page_elements[page_url].append(elem)
        
        # Limit elements per page
        limited_elements = []
        for page_url, elements in page_elements.items():
            if len(elements) > self.config.max_elements_per_page:
                logger.warning(f"⚠️ Page {page_url} has {len(elements)} elements, limiting to {self.config.max_elements_per_page}")
                # Prioritize interactive elements
                interactive_elements = [e for e in elements if self._is_interactive_element(e)]
                other_elements = [e for e in elements if not self._is_interactive_element(e)]
                
                # Take all interactive elements + some others
                limited = interactive_elements[:self.config.max_elements_per_page]
                remaining_slots = self.config.max_elements_per_page - len(limited)
                limited.extend(other_elements[:remaining_slots])
                limited_elements.extend(limited)
            else:
                limited_elements.extend(elements)
        
        logger.info(f"📊 Limited elements from {len(web_elements)} to {len(limited_elements)}")
        return limited_elements
    
    def _is_interactive_element(self, element: WebElement) -> bool:
        """Check if element is interactive (buttons, links, inputs, etc.)"""
        tag = getattr(element, 'tag', '').lower()
        interactive_tags = {'button', 'a', 'input', 'select', 'textarea', 'form'}
        return tag in interactive_tags
    
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
    
    def build_task_graph_from_web_data(self, web_pages: List[WebPageData], web_elements: List[WebElement]) -> "TaskGraph":
        """直接从web_collector数据构建TaskGraph（内存优化版本）"""
        # Use unified NodeType and EdgeType
        from .node_types import NodeType
        
        logger.info(f"🕸️ Building TaskGraph from {len(web_pages)} pages and {len(web_elements)} elements")
        self._log_memory_usage("task graph build start")
        
        # Apply memory optimization: limit element count
        if hasattr(self.config, 'enable_memory_optimization') and self.config.enable_memory_optimization:
            web_elements = self._limit_elements_per_page(web_elements)
            logger.info(f"📊 Memory optimization: limited to {len(web_elements)} elements")
        
        task_graph = TaskGraph()
        
        # Set website information
        if web_pages:
            first_page = web_pages[0]
            task_graph.website_type = getattr(first_page, 'website_type', 'unknown')
            task_graph.website_description = getattr(first_page, 'website_description', '')
            logger.debug(f"🕸️ Website type: {task_graph.website_type}")
            logger.debug(f"🕸️ Website description: {task_graph.website_description}")
        
        # Build page nodes
        logger.debug(f"🕸️ Creating page nodes...")
        for i, page in enumerate(web_pages):
            page_node = GraphNode(
                node_id=f"page_{i}",
                node_type=NodeType.PAGE,
                url=page.url,
                metadata=NodeMetadata(
                    text_content=page.title,
                    som_mark=f"P{i+1}"
                )
            )
            task_graph.add_node(page_node)
        logger.debug(f"🕸️ Created {len(web_pages)} page nodes")
        
        # Build element nodes - extract elements from each page (batch processing optimization)
        logger.info(f"🕸️ Creating element nodes...")
        element_counter = 0
        
        # Use batch processing to create element nodes
        if hasattr(self.config, 'enable_memory_optimization') and self.config.enable_memory_optimization:
            element_counter = self._create_element_nodes_batch_optimized(task_graph, web_pages)
        else:
            element_counter = self._create_element_nodes_traditional(task_graph, web_pages)
        
        logger.info(f"🕸️ Created {element_counter} element nodes")
        
        # Memory cleanup
        self._force_memory_cleanup()
        self._log_memory_usage("element nodes created")
        
        # Create interaction edges between elements
        logger.info(f"🕸️ Creating interaction edges...")
        # Extract elements from all pages
        all_elements = []
        for page in web_pages:
            if hasattr(page, 'elements') and page.elements:
                all_elements.extend(page.elements)
        initial_edge_count = len(task_graph.edges)
        self._create_interaction_edges(task_graph, all_elements)
        final_edge_count = len(task_graph.edges)
        logger.info(f"🕸️ Interaction edges created: {final_edge_count - initial_edge_count} new edges (total: {final_edge_count})")
        
        # 创建向量索引（内存优化版本）
        logger.info(f"🕸️ Creating vector index...")
        self._log_memory_usage("vector index creation start")
        
        try:
            from graph_rag.embeddings import NodeVectorIndex
            task_graph.vector_index = NodeVectorIndex(self.embedding_manager)
            logger.debug(f"🕸️ NodeVectorIndex created successfully")
            
            # 批处理创建向量索引
            node_count = self._create_vector_index_batch(task_graph)
            
            logger.info(f"🕸️ Vector index created with {node_count} nodes out of {len(task_graph.nodes)} total nodes")
            logger.debug(f"🕸️ Vector index nodes: {len(task_graph.vector_index.nodes)}")
        except Exception as e:
            logger.warning(f"🕸️ Failed to create vector index: {e}")
            import traceback
            logger.debug(f"🕸️ Vector index creation error: {traceback.format_exc()}")
            task_graph.vector_index = None
        
        self._log_memory_usage("vector index creation end")
        
        logger.info(f"🕸️ TaskGraph built successfully: {len(task_graph.nodes)} nodes, {len(task_graph.edges)} edges")
        
        # 最终内存清理
        self._force_memory_cleanup()
        self._log_memory_usage("task graph build end")
        
        return task_graph
    
    def _create_element_nodes_traditional(self, task_graph: "TaskGraph", web_pages: List[WebPageData]) -> int:
        """传统方式创建元素节点"""
        element_counter = 0
        for page_idx, page in enumerate(web_pages):
            for element in page.elements:
                # 映射元素类型（智能判断）
                element_type = self._map_web_element_type_to_node_type(element.element_type, element)
                
                # 创建元素节点
                # 检查是否是业务数据元素
                business_tags = set()
                element_type_for_task = element_type  # 默认使用映射后的类型
                
                if getattr(element, 'som_type', '') == 'business_data':
                    # 导入BusinessTag
                    from .node_types import BusinessTag
                    # 使用NodeType枚举，因为我们在TaskGraph上下文中
                    
                    # 根据内容特征确定通用的业务数据类型
                    text_content = element.text_content
                    text_content_lower = text_content.lower()
                    
                    # 使用更通用的分类逻辑
                    if self._is_user_data(text_content, text_content_lower):
                        element_type_for_task = NodeType.USER_DATA
                    elif self._is_product_data(text_content, text_content_lower):
                        element_type_for_task = NodeType.PRODUCT_DATA
                    elif self._is_order_data(text_content, text_content_lower):
                        element_type_for_task = NodeType.ORDER_DATA
                    elif self._is_content_data(text_content, text_content_lower):
                        element_type_for_task = NodeType.CONTENT_DATA
                    elif self._is_financial_data(text_content, text_content_lower):
                        element_type_for_task = NodeType.FINANCIAL_DATA
                    elif self._is_location_data(text_content, text_content_lower):
                        element_type_for_task = NodeType.LOCATION_DATA
                    elif self._is_time_data(text_content, text_content_lower):
                        element_type_for_task = NodeType.TIME_DATA
                    else:
                        # 默认使用通用业务数据类型
                        element_type_for_task = NodeType.BUSINESS_DATA
                    
                    # 确保业务数据节点被正确标记
                    logger.debug(f"🎯 Business data node detected: {text_content[:50]} -> {element_type_for_task}")
                
                element_node = GraphNode(
                    node_id=f"element_{element_counter}",
                    node_type=element_type_for_task,  # 使用处理后的类型，包括业务数据类型
                    url=page.url,  # 使用页面URL作为元素的来源
                    metadata=NodeMetadata(
                        is_visible=element.is_visible,
                        is_clickable=element.is_clickable,
                        is_input=element.is_input,
                        is_enabled=element.is_enabled,
                        text_content=element.text_content,
                        placeholder=element.placeholder,
                        som_mark=getattr(element, 'som_mark', f"M{element_counter+1}"),
                        xpath=getattr(element, 'xpath', ''),
                        css_selector=element.css_selector,
                        business_tags=business_tags,
                        # 添加HTML属性信息
                        tag_name=getattr(element, 'tag_name', ''),
                        href=getattr(element, 'href', ''),
                        css_classes=getattr(element, 'css_classes', []),
                        role=getattr(element, 'role', '')
                    )
                )
                
                # 为task generator添加element_type字段
                # 确保element_type_for_task是字符串
                if hasattr(element_type_for_task, 'value'):
                    element_node.element_type = element_type_for_task.value
                else:
                    element_node.element_type = str(element_type_for_task)
                task_graph.add_node(element_node)
                
                # 创建包含关系边
                page_node_id = f"page_{page_idx}"
                if page_node_id in task_graph.nodes:
                    edge = GraphEdge(
                        edge_id=f"contains_{page_node_id}_{element_node.node_id}",
                        source_node_id=page_node_id,
                        target_node_id=element_node.node_id,
                        edge_type=EdgeType.CONTAINS
                    )
                    task_graph.add_edge(edge)
                
                element_counter += 1
        
        return element_counter
    
    def _create_element_nodes_batch_optimized(self, task_graph: "TaskGraph", web_pages: List[WebPageData]) -> int:
        """批处理优化方式创建元素节点"""
        element_counter = 0
        batch_size = getattr(self.config, 'batch_size', 100)
        
        # 收集所有元素
        all_elements = []
        for page_idx, page in enumerate(web_pages):
            for element in page.elements:
                all_elements.append((page_idx, element))
        
        # 按批次处理元素
        for batch in self._process_in_batches(all_elements, batch_size, "element node creation"):
            for page_idx, element in batch:
                # 映射元素类型（智能判断）
                element_type = self._map_web_element_type_to_node_type(element.element_type, element)
                
                # 创建元素节点（简化版本，减少内存使用）
                element_node = self._create_single_element_node(element, element_type, element_counter, page_idx)
                task_graph.add_node(element_node)
                
                # 创建包含关系边
                page_node_id = f"page_{page_idx}"
                if page_node_id in task_graph.nodes:
                    edge = GraphEdge(
                        edge_id=f"contains_{page_node_id}_{element_node.node_id}",
                        source_node_id=page_node_id,
                        target_node_id=element_node.node_id,
                        edge_type=EdgeType.CONTAINS
                    )
                    task_graph.add_edge(edge)
                
                element_counter += 1
            
            # 批次后清理内存
            if self._should_cleanup():
                self._force_memory_cleanup()
                self._log_memory_usage(f"element batch {element_counter // batch_size}")
        
        return element_counter
    
    def _create_single_element_node(self, element, element_type, element_counter: int, page_idx: int) -> "GraphNode":
        """创建单个元素节点（简化版本）"""
        from .node_types import NodeType, BusinessTag
        
        # 检查是否是业务数据元素
        business_tags = set()
        element_type_for_task = element_type
        
        if getattr(element, 'som_type', '') == 'business_data':
            text_content = element.text_content
            text_content_lower = text_content.lower()
            
            # 使用更通用的分类逻辑
            if self._is_user_data(text_content, text_content_lower):
                element_type_for_task = NodeType.USER_DATA
            elif self._is_product_data(text_content, text_content_lower):
                element_type_for_task = NodeType.PRODUCT_DATA
            elif self._is_order_data(text_content, text_content_lower):
                element_type_for_task = NodeType.ORDER_DATA
            elif self._is_content_data(text_content, text_content_lower):
                element_type_for_task = NodeType.CONTENT_DATA
            elif self._is_financial_data(text_content, text_content_lower):
                element_type_for_task = NodeType.FINANCIAL_DATA
            elif self._is_location_data(text_content, text_content_lower):
                element_type_for_task = NodeType.LOCATION_DATA
            elif self._is_time_data(text_content, text_content_lower):
                element_type_for_task = NodeType.TIME_DATA
            else:
                element_type_for_task = NodeType.BUSINESS_DATA
        
        element_node = GraphNode(
            node_id=f"element_{element_counter}",
            node_type=element_type_for_task,
            url=getattr(element, 'page_url', ''),
            metadata=NodeMetadata(
                is_visible=element.is_visible,
                is_clickable=element.is_clickable,
                is_input=element.is_input,
                is_enabled=element.is_enabled,
                text_content=element.text_content,
                placeholder=element.placeholder,
                som_mark=getattr(element, 'som_mark', f"M{element_counter+1}"),
                xpath=getattr(element, 'xpath', ''),
                css_selector=element.css_selector,
                business_tags=business_tags,
                tag_name=getattr(element, 'tag_name', ''),
                href=getattr(element, 'href', ''),
                css_classes=getattr(element, 'css_classes', []),
                role=getattr(element, 'role', '')
            )
        )
        
        # 为task generator添加element_type字段
        if hasattr(element_type_for_task, 'value'):
            element_node.element_type = element_type_for_task.value
        else:
            element_node.element_type = str(element_type_for_task)
        
        return element_node
    
    
    def _create_vector_index_batch(self, task_graph: "TaskGraph") -> int:
        """批处理创建向量索引，减少内存使用"""
        node_count = 0
        batch_size = getattr(self.config, 'vector_batch_size', 30)  # 向量索引使用专门的批次大小
        
        # 收集所有需要向量化的节点
        nodes_to_vectorize = []
        for node in task_graph.nodes.values():
            if node is None:
                continue
            text = node.metadata.text_content or node.node_type.value
            if text and len(text.strip()) > 0:  # 只处理有文本内容的节点
                nodes_to_vectorize.append((node, text))
        
        logger.info(f"🕸️ Vectorizing {len(nodes_to_vectorize)} nodes in batches of {batch_size}")
        
        # 按批次处理节点
        for i in range(0, len(nodes_to_vectorize), batch_size):
            batch = nodes_to_vectorize[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(nodes_to_vectorize) + batch_size - 1) // batch_size
            
            logger.debug(f"🕸️ Processing vector batch {batch_num}/{total_batches} ({len(batch)} nodes)")
            
            for node, text in batch:
                try:
                    # 创建临时Node对象用于向量化
                    from graph_rag.node_types import Node, NodeType as GraphNodeType
                    temp_node = Node(
                        node_id=node.node_id,
                        content=text,
                        node_type=GraphNodeType.PARAGRAPH  # 使用PARAGRAPH类型
                    )
                    task_graph.vector_index.add_node(temp_node)
                    node_count += 1
                except Exception as e:
                    logger.warning(f"🕸️ Failed to vectorize node {node.node_id}: {e}")
            
            # 批次后清理内存
            self.operation_count += 1
            if self._should_cleanup():
                self._force_memory_cleanup()
                self._log_memory_usage(f"vector batch {batch_num}")
        
        return node_count
    
    def _map_web_element_type_to_node_type(self, element_type: str, element: WebElement = None) -> "NodeType":
        """将WebElement类型映射到NodeType，基于元素类型和内容智能判断 - 优化版本"""
        # 使用统一的NodeType
        from .node_types import NodeType
        
        # 智能映射：检查元素内容和属性来确定更精确的类型
        if element:
            text_content = getattr(element, 'text_content', '').lower()
            placeholder = getattr(element, 'placeholder', '').lower()
            css_classes = getattr(element, 'css_classes', [])
            css_classes_str = ' '.join(css_classes).lower() if css_classes else ''
            som_type = getattr(element, 'som_type', '').lower()
            href = getattr(element, 'href', '')
            tag_name = getattr(element, 'tag_name', '').lower()
            input_type = getattr(element, 'input_type', '').lower()
            
            # 1. 优先检查业务数据类型（Web Collector已识别的）
            if som_type == 'business_data':
                return self._classify_business_data_by_content(element)
            
            # 2. 优先检查导航相关属性
            if href or som_type in ['navigation', 'link'] or 'nav' in css_classes_str or 'menu' in css_classes_str:
                if som_type == 'navigation':
                    return NodeType.NAVIGATION
                elif som_type == 'link':
                    return NodeType.LINK
                elif 'menu' in css_classes_str:
                    return NodeType.MENU
                elif 'breadcrumb' in css_classes_str:
                    return NodeType.BREADCRUMB
                else:
                    return NodeType.LINK  # 默认链接类型
            
            # 2.5. 检查dropdown元素是否实际上是导航元素
            if som_type == 'dropdown' and ('nav' in css_classes_str or 'menu' in css_classes_str):
                if 'top-nav' in css_classes_str or 'nav-link' in css_classes_str:
                    return NodeType.NAVIGATION
                elif 'menu' in css_classes_str:
                    return NodeType.MENU
                else:
                    return NodeType.DROPDOWN
            
            # 3. 基于SoM类型进行精确映射（Web Collector识别的类型）
            if som_type == 'search_box':
                return NodeType.SEARCH_BOX
            elif som_type == 'form_control':
                return NodeType.INPUT
            elif som_type == 'button':
                return NodeType.BUTTON
            elif som_type == 'submit':
                return NodeType.SUBMIT
            elif som_type == 'search_button':
                return NodeType.BUTTON  # 搜索按钮仍然是按钮类型
            elif som_type == 'filter_button':
                return NodeType.BUTTON  # 过滤按钮仍然是按钮类型
            elif som_type == 'input':
                if 'search' in placeholder or 'search' in text_content or 'search' in css_classes_str:
                    return NodeType.SEARCH_BOX
                else:
                    return NodeType.INPUT
            elif som_type == 'email_input':
                return NodeType.INPUT
            elif som_type == 'password_input':
                return NodeType.INPUT
            elif som_type == 'link':
                return NodeType.LINK
            elif som_type == 'navigation':
                return NodeType.NAVIGATION
            elif som_type == 'menu_link':
                return NodeType.LINK
            elif som_type == 'detail_link':
                return NodeType.DETAIL_LINK
            elif som_type == 'card':
                return NodeType.CARD
            elif som_type == 'list':
                return NodeType.LIST
            elif som_type == 'detail':
                return NodeType.DETAIL
            elif som_type == 'dashboard':
                return NodeType.DASHBOARD
            elif som_type == 'select':
                return NodeType.SELECT
            elif som_type == 'textarea':
                return NodeType.TEXTAREA
            elif som_type == 'paginator':
                return NodeType.PAGINATOR
            elif som_type == 'result':
                return NodeType.RESULT_ITEM
            elif som_type == 'dropdown':
                return NodeType.DROPDOWN
            elif som_type == 'tab':
                return NodeType.TAB
            elif som_type == 'submenu':
                return NodeType.SUBMENU
            elif som_type == 'modal':
                return NodeType.MODAL
            elif som_type == 'toast':
                return NodeType.TOAST
            elif som_type == 'content':
                return NodeType.CONTENT
            elif som_type == 'table':
                # Web tables should use CONTENT type to avoid mixing with document-specific types
                return NodeType.CONTENT  # Use CONTENT type for web tables
            elif som_type == 'image':
                return NodeType.IMAGE
            elif som_type == 'form':
                return NodeType.FORM
            elif som_type == 'filter':
                return NodeType.FILTER
            elif som_type == 'filter_panel':
                return NodeType.FILTER_PANEL
            elif som_type == 'notification_area':
                return NodeType.NOTIFICATION_AREA
            elif som_type == 'tab_container':
                return NodeType.TAB_CONTAINER
            
            # 4. 基于标签名的映射（后备方案）
            if tag_name == 'button':
                if 'submit' in text_content or 'save' in text_content or 'create' in text_content:
                    return NodeType.SUBMIT
                else:
                    return NodeType.BUTTON
            elif tag_name == 'input':
                if input_type == 'search' or 'search' in placeholder or 'search' in text_content:
                    return NodeType.SEARCH_BOX
                else:
                    return NodeType.INPUT
            elif tag_name == 'a':
                if 'nav' in css_classes_str:
                    return NodeType.NAVIGATION
                elif 'menu' in css_classes_str:
                    return NodeType.LINK
                elif 'detail' in css_classes_str:
                    return NodeType.DETAIL_LINK
                else:
                    return NodeType.LINK
            elif tag_name == 'select':
                return NodeType.SELECT
            elif tag_name == 'textarea':
                return NodeType.TEXTAREA
            elif tag_name == 'table':
                # Web tables should use a web-specific type, not document TABLE type
                return NodeType.WEB_TABLE  # Use WEB_TABLE type for web tables
            elif tag_name == 'img':
                return NodeType.IMAGE
            elif tag_name == 'form':
                return NodeType.FORM
            elif tag_name in ['div', 'section']:
                # 智能容器类型识别
                if 'card' in css_classes_str or 'panel' in css_classes_str or 'widget' in css_classes_str:
                    return NodeType.CARD
                elif 'list' in css_classes_str or 'item' in css_classes_str:
                    return NodeType.LIST
                elif 'detail' in css_classes_str or 'info' in css_classes_str:
                    return NodeType.DETAIL
                elif 'dashboard' in css_classes_str or 'overview' in css_classes_str:
                    return NodeType.DASHBOARD
                else:
                    return NodeType.CONTENT
            
            # 5. 基于CSS类的智能推断
            if css_classes:
                if any('card' in cls.lower() for cls in css_classes):
                    return NodeType.CARD
                elif any('list' in cls.lower() for cls in css_classes):
                    return NodeType.LIST
                elif any('detail' in cls.lower() for cls in css_classes):
                    return NodeType.DETAIL
                elif any('dashboard' in cls.lower() for cls in css_classes):
                    return NodeType.DASHBOARD
                elif any('form' in cls.lower() for cls in css_classes):
                    return NodeType.FORM
                elif any('nav' in cls.lower() for cls in css_classes):
                    return NodeType.NAVIGATION
                elif any('menu' in cls.lower() for cls in css_classes):
                    return NodeType.MENU
                elif any('tab' in cls.lower() for cls in css_classes):
                    return NodeType.TAB
                elif any('filter' in cls.lower() for cls in css_classes):
                    return NodeType.FILTER
                elif any('search' in cls.lower() for cls in css_classes):
                    return NodeType.SEARCH_BOX
                elif any('button' in cls.lower() for cls in css_classes):
                    return NodeType.BUTTON
                elif any('input' in cls.lower() for cls in css_classes):
                    return NodeType.INPUT
            
            # 6. 默认类型（基于原始element_type）
            if element_type == 'clickable':
                return NodeType.BUTTON
            elif element_type == 'unknown':
                return NodeType.CONTENT
            else:
                # 尝试直接映射，但排除文档类型
                try:
                    node_type = NodeType(element_type)
                    # 如果是文档类型（如TABLE），则使用CONTENT类型
                    if node_type in [NodeType.TABLE, NodeType.PARAGRAPH, NodeType.HEADING, NodeType.FIGURE, NodeType.ENTITY, NodeType.CHUNK]:
                        return NodeType.CONTENT
                    return node_type
                except ValueError:
                    # 如果无法直接映射，返回通用内容类型
                    return NodeType.CONTENT
        
        # 如果没有element对象，使用element_type进行基本映射
        basic_mapping = {
            'business_data': NodeType.BUSINESS_DATA,
            'search_box': NodeType.SEARCH_BOX,
            'form_control': NodeType.INPUT,
            'button': NodeType.BUTTON,
            'submit': NodeType.SUBMIT,
            'input': NodeType.INPUT,
            'link': NodeType.LINK,
            'navigation': NodeType.NAVIGATION,
            'card': NodeType.CARD,
            'list': NodeType.LIST,
            'detail': NodeType.DETAIL,
            'dashboard': NodeType.DASHBOARD,
            'select': NodeType.SELECT,
            'textarea': NodeType.TEXTAREA,
            'table': NodeType.WEB_TABLE,  # Web tables use WEB_TABLE type
            'image': NodeType.IMAGE,
            'form': NodeType.FORM,
            'filter': NodeType.FILTER,
            'modal': NodeType.MODAL,
            'toast': NodeType.TOAST,
            'content': NodeType.CONTENT,
            'clickable': NodeType.BUTTON,
            'unknown': NodeType.CONTENT
        }
        
        return basic_mapping.get(element_type, NodeType.CONTENT)
    

       
    
    def _create_interaction_edges(self, task_graph: "TaskGraph", web_elements: List[WebElement]):
        """创建元素间的交互边"""
        # 使用统一的GraphEdge和EdgeType
        
        logger.debug(f"🕸️ Creating interaction edges for {len(web_elements)} elements")
        
        # 分析表单关系
        form_elements = [elem for elem in web_elements if elem.is_input]
        # 正确识别submit按钮：区分form_control和submit按钮
        submit_buttons = []
        for elem in web_elements:
            # 1. 明确的submit类型
            if elem.element_type == 'submit':
                submit_buttons.append(elem)
            # 2. input[type="submit"]元素
            elif (elem.tag_name == 'input' and 
                  hasattr(elem, 'input_type') and elem.input_type == 'submit'):
                submit_buttons.append(elem)
            # 3. button标签且文本内容包含提交相关关键词
            elif (elem.tag_name == 'button' and 
                  any(keyword in elem.text_content.lower() 
                      for keyword in ['submit', 'save', 'send', 'confirm', 'log in', 'login', 'sign in'])):
                submit_buttons.append(elem)
            # 4. 有submit相关CSS类的按钮
            elif (elem.tag_name == 'button' and 
                  any('submit' in cls.lower() or 'primary' in cls.lower() 
                      for cls in elem.css_classes)):
                submit_buttons.append(elem)
        
        # 创建表单控制关系 - 改进版本
        logger.debug(f"🕸️ Found {len(submit_buttons)} submit buttons and {len(form_elements)} form elements")
        controls_created = 0
        
        # 调试：检查SoM标记匹配情况
        elements_with_som = sum(1 for elem in web_elements if getattr(elem, 'som_mark', None))
        logger.debug(f"🕸️ Elements with som_mark: {elements_with_som}/{len(web_elements)}")
        logger.debug(f"🕸️ Total nodes in graph: {len(task_graph.nodes)}")
        
        # 按页面分组元素
        page_elements = {}
        elements_grouped = 0
        for elem in web_elements:
            # 使用与节点创建时相同的逻辑获取som_mark
            elem_som = getattr(elem, 'som_mark', None)
            if not elem_som:
                # 如果元素没有som_mark，跳过（这些元素在节点创建时也没有被处理）
                continue
            elem_node = self._find_node_by_som(task_graph, elem_som)
            if not elem_node:
                logger.debug(f"🕸️ Could not find node for element with som_mark: {elem_som}")
                continue
            page = self._find_element_page(task_graph, elem_node)
            if not page:
                logger.debug(f"🕸️ Could not find page for element with som_mark: {elem_som}")
                continue
            if page.node_id not in page_elements:
                page_elements[page.node_id] = {'submit_buttons': [], 'form_elements': []}
            if elem in submit_buttons:
                page_elements[page.node_id]['submit_buttons'].append(elem)
            if elem in form_elements:
                page_elements[page.node_id]['form_elements'].append(elem)
            elements_grouped += 1
        
        logger.debug(f"🕸️ Grouped {elements_grouped} elements into {len(page_elements)} pages")
        
        # 为每个页面创建表单控制关系
        logger.debug(f"🕸️ Processing {len(page_elements)} pages for controls edges")
        for page_id, elements in page_elements.items():
            page_submit_buttons = elements['submit_buttons']
            page_form_elements = elements['form_elements']
            logger.debug(f"🕸️ Page {page_id}: {len(page_submit_buttons)} submit buttons, {len(page_form_elements)} form elements")
            
            # 只为同一页面的提交按钮和表单元素创建关系
            for submit_btn in page_submit_buttons:
                for form_elem in page_form_elements:
                    submit_som = getattr(submit_btn, 'som_mark', None)
                    form_som = getattr(form_elem, 'som_mark', None)
                    
                    if not submit_som or not form_som:
                        logger.debug(f"🕸️ Missing som_mark: submit={submit_som}, form={form_som}")
                        continue
                        
                    submit_node = self._find_node_by_som(task_graph, submit_som)
                    form_node = self._find_node_by_som(task_graph, form_som)
                    
                    if not submit_node:
                        logger.debug(f"🕸️ Could not find submit node for som_mark: {submit_som}")
                        continue
                    if not form_node:
                        logger.debug(f"🕸️ Could not find form node for som_mark: {form_som}")
                        continue
                    
                    edge = GraphEdge(
                        edge_id=f"controls_{submit_node.node_id}_{form_node.node_id}",
                        source_node_id=submit_node.node_id,
                        target_node_id=form_node.node_id,
                        edge_type=EdgeType.CONTROLS
                    )
                    task_graph.add_edge(edge)
                    controls_created += 1
                    logger.debug(f"🕸️ Created controls edge: {submit_node.node_id} -> {form_node.node_id}")
        
        logger.debug(f"🕸️ Created {controls_created} controls edges")
        
        # 使用代码默认Web图构建配置
        # 改进的导航关系分析
        self._create_navigation_edges(task_graph, web_elements)
        
        # 创建点击触发关系
        self._create_click_trigger_edges(task_graph, web_elements)
        
        # 创建数据流关系
        self._create_data_flow_edges(task_graph, web_elements)
        
        # 创建SAME_ENTITY边 - 暂时禁用以避免边爆炸问题
        logger.info(f"🕸️ Skipping same entity edges creation (temporarily disabled)")
        same_entity_edges = 0
        logger.info(f"🕸️ Created {same_entity_edges} same entity edges")
        
        # 动态边类型发现 - 分析元素间的实际关系模式
        logger.info(f"🕸️ Calling dynamic edge type discovery...")
        dynamic_edges = self._discover_dynamic_edge_patterns(task_graph, web_elements)
        logger.info(f"🕸️ Created {dynamic_edges} dynamic relationship edges")
        
        logger.info(f"🕸️ === INTERACTION EDGES CREATION END ===")
    
    def _create_navigation_edges(self, task_graph: "TaskGraph", web_elements: List[WebElement]):
        """创建导航边"""
        
        logger.debug(f"🕸️ Creating navigation edges...")
        
        # 获取所有页面节点
        page_nodes = {node.node_id: node for node in task_graph.nodes.values() 
                     if hasattr(node, 'node_type') and node.node_type.value == 'Page'}
        
        # 精确的导航元素识别
        navigation_elements = []
        for elem in web_elements:
            if self._is_navigation_element(elem):
                navigation_elements.append(elem)
        
        logger.debug(f"🕸️ Found {len(navigation_elements)} navigation elements")
        
        # 调试：检查导航元素的详细信息
        for i, nav_elem in enumerate(navigation_elements[:5]):  # 只显示前5个
            som_mark = getattr(nav_elem, 'som_mark', None)
            href = getattr(nav_elem, 'href', '')
            text = getattr(nav_elem, 'text_content', '')[:50]
            logger.debug(f"🕸️ Nav element {i+1}: som={som_mark}, href={href}, text={text}")
        
        # 为每个导航元素创建边
        nav_edges_created = 0
        for i, nav_elem in enumerate(navigation_elements):
            # 确保SoM标记存在
            som_mark = getattr(nav_elem, 'som_mark', None)
            if not som_mark:
                logger.debug(f"🕸️ Nav element {i+1}: No som_mark")
                continue
                
            nav_node = self._find_node_by_som(task_graph, som_mark)
            if not nav_node:
                logger.debug(f"🕸️ Nav element {i+1}: Could not find node for SoM mark: {som_mark}")
                continue
            
            # 获取导航元素所在的页面
            source_page = self._find_element_page(task_graph, nav_node)
            if not source_page:
                logger.debug(f"🕸️ Nav element {i+1}: Could not find source page for node {nav_node.node_id}")
                continue
            
            # 尝试找到目标页面
            target_page = None
            if hasattr(nav_elem, 'href') and nav_elem.href:
                # 根据href找到目标页面
                target_page = self._find_page_by_url_pattern(task_graph, nav_elem.href)
            
            if target_page and target_page != source_page:
                # 只创建跨页面导航边
                if self._validate_navigation_edge(nav_node, target_page, nav_elem):
                    edge = GraphEdge(
                        edge_id=f"nav_{nav_node.node_id}_{target_page.node_id}",
                        source_node_id=nav_node.node_id,
                        target_node_id=target_page.node_id,
                        edge_type=EdgeType.NAV_TO
                    )
                    task_graph.add_edge(edge)
                    nav_edges_created += 1
                    logger.debug(f"🕸️ Created cross-page navigation edge: {nav_node.node_id} -> {target_page.node_id}")
            else:
                # 对于页面内导航，创建到页面内特定元素的边（如果有的话）
                page_target_elements = self._find_page_navigation_targets(task_graph, source_page, nav_elem)
                for target_element in page_target_elements:
                    if self._validate_navigation_edge(nav_node, target_element, nav_elem):
                        edge = GraphEdge(
                            edge_id=f"nav_{nav_node.node_id}_{target_element.node_id}",
                        source_node_id=nav_node.node_id,
                        target_node_id=target_element.node_id,
                            edge_type=EdgeType.NAV_TO
                        )
                        task_graph.add_edge(edge)
                        nav_edges_created += 1
                        logger.debug(f"🕸️ Created page-internal navigation edge: {nav_node.node_id} -> {target_element.node_id}")
        
        logger.debug(f"🕸️ Created {nav_edges_created} navigation edges")
    
    def _create_click_trigger_edges(self, task_graph: "TaskGraph", web_elements: List[WebElement]):
        """创建点击触发关系边 - 基于实际UI交互模式"""
        
        logger.debug(f"🕸️ Creating click trigger edges...")
        
        # 统计可点击元素
        clickable_count = sum(1 for elem in web_elements if elem.is_clickable)
        logger.debug(f"🕸️ Found {clickable_count} clickable elements out of {len(web_elements)} total elements")
        
        # 按页面分组可点击元素
        page_clickables = {}
        clickables_grouped = 0
        for elem in web_elements:
            if not elem.is_clickable:
                continue
                
            elem_som = getattr(elem, 'som_mark', None)
            if not elem_som:
                continue
                
            elem_node = self._find_node_by_som(task_graph, elem_som)
            if not elem_node:
                continue
                
            page = self._find_element_page(task_graph, elem_node)
            if not page:
                continue
                
            if page.node_id not in page_clickables:
                page_clickables[page.node_id] = []
            page_clickables[page.node_id].append((elem, elem_node))
            clickables_grouped += 1
        
        logger.debug(f"🕸️ Grouped {clickables_grouped} clickable elements into {len(page_clickables)} pages")
        
        edges_created = 0
        
        logger.debug(f"🕸️ Grouped clickable elements into {len(page_clickables)} pages")
        
        # 为每个页面分析交互模式
        for page_id, clickables in page_clickables.items():
            logger.debug(f"🕸️ Processing page {page_id} with {len(clickables)} clickable elements")
            page_edges = self._analyze_page_interaction_patterns(task_graph, page_id, clickables)
            edges_created += page_edges
            logger.debug(f"🕸️ Page {page_id} created {page_edges} edges")
        
        logger.debug(f"🕸️ Created {edges_created} click trigger edges")
    
    def _analyze_page_interaction_patterns(self, task_graph: "TaskGraph", page_id: str, clickables: List[tuple]) -> int:
        """分析页面内的交互模式 - 简化版本"""
        
        edges_created = 0
        
        # 1. 按钮 → 模态框/弹窗模式
        modal_buttons = []
        modal_elements = []
        
        # 2. 按钮 → 表单提交模式  
        submit_buttons = []
        form_elements = []
        
        # 3. 链接 → 页面跳转模式
        navigation_links = []
        
        # 4. 按钮 → 内容显示/隐藏模式
        toggle_buttons = []
        content_elements = []
        
        # 分类可点击元素
        for elem, node in clickables:
            elem_type = getattr(elem, 'element_type', '').lower()
            text_content = getattr(elem, 'text_content', '').lower()
            css_classes = getattr(elem, 'css_classes', [])
            css_str = ' '.join(css_classes).lower()
            
            # 模态框按钮识别
            if (elem_type == 'button' and 
                (any(keyword in text_content for keyword in ['open', 'show', 'modal', 'popup', 'dialog']) or
                 any(keyword in css_str for keyword in ['modal', 'popup', 'dialog']))):
                modal_buttons.append((elem, node))
            
            # 提交按钮识别 - 修复逻辑
            elif (elem_type == 'submit' or 
                  (elem.tag_name == 'button' and any(keyword in text_content for keyword in ['submit', 'save', 'send', 'confirm', 'log in', 'login'])) or
                  (elem.tag_name == 'input' and hasattr(elem, 'input_type') and elem.input_type == 'submit') or
                  any(keyword in css_str for keyword in ['submit', 'primary', 'btn-primary'])):
                submit_buttons.append((elem, node))
            
            # 导航链接识别
            elif (elem_type == 'link' or 
                  hasattr(elem, 'href') and elem.href and elem.href != '#'):
                navigation_links.append((elem, node))
            
            # 切换按钮识别
            elif (elem_type == 'button' and 
                  (any(keyword in text_content for keyword in ['toggle', 'show', 'hide', 'expand', 'collapse']) or
                   any(keyword in css_str for keyword in ['toggle', 'collapse', 'expand']))):
                toggle_buttons.append((elem, node))
        
        # 获取页面内的其他元素
        page_elements = [node for node in task_graph.nodes.values() 
                        if hasattr(node, 'url') and node.url and 
                        self._find_element_page(task_graph, node) and
                        self._find_element_page(task_graph, node).node_id == page_id]
        
        # 分类页面元素
        for node in page_elements:
            node_type = node.node_type.value.lower()
            
            if node_type in ['modal', 'popup', 'dialog']:
                modal_elements.append(node)
            elif node_type in ['form', 'input', 'select', 'textarea', 'form_control']:
                form_elements.append(node)
            elif node_type in ['content', 'text', 'div', 'section']:
                content_elements.append(node)
        
        # 创建基于模式的边
        
        # 1. 按钮 → 模态框
        for button_elem, button_node in modal_buttons:
            for modal_node in modal_elements[:2]:  # 限制数量
                if self._validate_interaction_edge(button_node, modal_node, 'opens_modal'):
                    edge = GraphEdge(
                        edge_id=f"opens_modal_{button_node.node_id}_{modal_node.node_id}",
                        source_node_id=button_node.node_id,
                        target_node_id=modal_node.node_id,
                        edge_type=EdgeType.OPENS
                    )
                    task_graph.add_edge(edge)
                    edges_created += 1
                    logger.debug(f"🕸️ Created modal edge: {button_node.node_id} -> {modal_node.node_id}")
        
        # 2. 提交按钮 → 表单元素
        for submit_elem, submit_node in submit_buttons:
            for form_node in form_elements[:3]:  # 限制数量
                if self._validate_interaction_edge(submit_node, form_node, 'controls_form'):
                    edge = GraphEdge(
                        edge_id=f"controls_form_{submit_node.node_id}_{form_node.node_id}",
                        source_node_id=submit_node.node_id,
                        target_node_id=form_node.node_id,
                        edge_type=EdgeType.CONTROLS
                    )
                    task_graph.add_edge(edge)
                    edges_created += 1
                    logger.debug(f"🕸️ Created form control edge: {submit_node.node_id} -> {form_node.node_id}")
        
        # 3. 切换按钮 → 内容元素
        for toggle_elem, toggle_node in toggle_buttons:
            for content_node in content_elements[:2]:  # 限制数量
                if self._validate_interaction_edge(toggle_node, content_node, 'toggles_content'):
                    edge = GraphEdge(
                        edge_id=f"toggles_{toggle_node.node_id}_{content_node.node_id}",
                        source_node_id=toggle_node.node_id,
                        target_node_id=content_node.node_id,
                        edge_type=EdgeType.CONTROLS
                    )
                    task_graph.add_edge(edge)
                    edges_created += 1
                    logger.debug(f"🕸️ Created toggle edge: {toggle_node.node_id} -> {content_node.node_id}")
        
        # 4. 添加调试信息
        logger.debug(f"🕸️ Page {page_id} interaction analysis:")
        logger.debug(f"🕸️   - Modal buttons: {len(modal_buttons)}, Modal elements: {len(modal_elements)}")
        logger.debug(f"🕸️   - Submit buttons: {len(submit_buttons)}, Form elements: {len(form_elements)}")
        logger.debug(f"🕸️   - Toggle buttons: {len(toggle_buttons)}, Content elements: {len(content_elements)}")
        logger.debug(f"🕸️   - Navigation links: {len(navigation_links)}")
        
        return edges_created
    
    def _validate_interaction_edge(self, source_node: "GraphNode", target_node: "GraphNode", interaction_type: str) -> bool:
        """验证交互边的合理性"""
        # 基本验证
        if source_node.node_id == target_node.node_id:
            return False
        
        # 基于交互类型的验证
        if interaction_type == 'opens_modal':
            # 按钮应该能打开模态框
            return (source_node.node_type.value.lower() in ['button', 'link'] and 
                    target_node.node_type.value.lower() in ['modal', 'popup', 'dialog'])
        
        elif interaction_type == 'controls_form':
            # 提交按钮应该能控制表单元素
            return (source_node.node_type.value.lower() in ['button', 'submit'] and 
                    target_node.node_type.value.lower() in ['form', 'input', 'select', 'textarea'])
        
        elif interaction_type == 'toggles_content':
            # 切换按钮应该能控制内容显示
            return (source_node.node_type.value.lower() in ['button', 'toggle'] and 
                    target_node.node_type.value.lower() in ['content', 'text', 'div', 'section'])
        
        return True
    
    def _is_navigation_element(self, elem: WebElement) -> bool:
        """精确识别导航元素"""
        # 1. 明确的导航标识
        if hasattr(elem, 'som_type') and elem.som_type == 'navigation':
            return True
        
        # 2. 有效的href链接（排除锚点）
        if hasattr(elem, 'href') and elem.href and elem.href not in ['#', 'javascript:void(0)', 'javascript:;']:
            # 进一步验证href是否指向其他页面
            if self._is_valid_navigation_href(elem.href):
                return True
        
        # 3. 导航相关的CSS类
        css_classes = getattr(elem, 'css_classes', [])
        css_str = ' '.join(css_classes).lower()
        nav_css_keywords = ['nav', 'navigation', 'menu', 'breadcrumb', 'navbar', 'nav-item']
        if any(keyword in css_str for keyword in nav_css_keywords):
            return True
        
        # 4. 导航相关的文本内容
        text_content = getattr(elem, 'text_content', '').lower().strip()
        if text_content:
            # 明确的导航文本
            nav_text_keywords = [
                'home', 'back', 'next', 'previous', 'menu', 'navigation',
                'about', 'contact', 'services', 'products', 'login', 'register',
                'dashboard', 'profile', 'settings', 'help', 'support'
            ]
            if any(keyword in text_content for keyword in nav_text_keywords):
                return True
            
            # 面包屑导航
            if '>' in text_content or '→' in text_content or 'breadcrumb' in text_content:
                return True
        
        # 5. 元素类型和属性的组合判断
        elem_type = getattr(elem, 'element_type', '').lower()
        if elem_type == 'link' and hasattr(elem, 'href') and elem.href:
            return True
        
        # 6. 排除非导航元素
        # 排除表单提交按钮
        if elem_type == 'submit' or elem_type == 'button':
            text_content = getattr(elem, 'text_content', '').lower()
            if any(keyword in text_content for keyword in ['submit', 'save', 'send', 'confirm', 'cancel']):
                return False
        
        # 排除模态框触发器
        if elem_type == 'button':
            text_content = getattr(elem, 'text_content', '').lower()
            css_str = ' '.join(getattr(elem, 'css_classes', [])).lower()
            if any(keyword in text_content for keyword in ['open', 'show', 'modal', 'popup']) or \
               any(keyword in css_str for keyword in ['modal', 'popup', 'dialog']):
                return False
        
        return False
    
    def _is_valid_navigation_href(self, href: str) -> bool:
        """验证href是否为有效的导航链接"""
        if not href or href in ['#', 'javascript:void(0)', 'javascript:;']:
            return False
        
        # 检查是否为外部链接
        if href.startswith(('http://', 'https://', '//')):
            return True
        
        # 检查是否为内部页面链接
        if href.startswith('/') and len(href) > 1:
            return True
        
        # 检查是否为相对路径
        if not href.startswith('#') and '.' in href:
            return True
        
        return False
    
    def _validate_navigation_edge(self, source_node: "GraphNode", target_node: "GraphNode", nav_elem: WebElement) -> bool:
        """验证导航边的合理性"""
        # 基本验证
        if source_node.node_id == target_node.node_id:
            return False
        
        # 验证源节点是导航元素
        if not self._is_navigation_element(nav_elem):
            return False
        
        # 验证目标节点类型
        target_type = target_node.node_type.value.lower()
        if target_type not in ['page', 'content', 'section', 'div']:
            return False
        
        # 验证href与目标的一致性
        if hasattr(nav_elem, 'href') and nav_elem.href:
            # 如果href指向特定页面，目标应该是页面节点
            if self._is_valid_navigation_href(nav_elem.href):
                return target_type == 'page'
        
        return True
    
    def _find_page_navigation_targets(self, task_graph: "TaskGraph", page: "GraphNode", nav_elem: WebElement) -> List["GraphNode"]:
        """查找页面内导航的目标元素"""
        targets = []
        
        # 获取页面内的所有元素
        page_elements = [node for node in task_graph.nodes.values() 
                        if hasattr(node, 'url') and node.url == page.url and node != page]
        
        # 基于导航元素的href或文本内容查找目标
        href = getattr(nav_elem, 'href', '')
        text_content = getattr(nav_elem, 'text_content', '').lower()
        
        for element in page_elements:
            element_text = getattr(element.metadata, 'text_content', '').lower()
            element_id = getattr(element.metadata, 'som_mark', '')
            
            # 如果href包含锚点，查找对应的元素
            if href and '#' in href:
                anchor = href.split('#')[-1]
                if anchor in element_id or anchor in element_text:
                    targets.append(element)
            
            # 如果文本内容匹配，可能是页面内导航
            elif text_content and any(keyword in element_text for keyword in ['section', 'content', 'main']):
                if len(targets) < 2:  # 限制数量
                    targets.append(element)
        
        return targets
    
    def _create_data_flow_edges(self, task_graph: "TaskGraph", web_elements: List[WebElement]):
        """创建数据流关系边 - 优化版本"""
        
        logger.debug(f"🕸️ Creating data flow edges...")
        
        # 统计输入和输出元素
        input_count = sum(1 for elem in web_elements if elem.is_input)
        output_count = sum(1 for elem in web_elements if elem.is_clickable and not elem.is_input)
        logger.debug(f"🕸️ Found {input_count} input elements and {output_count} output elements")
        
        # 按页面分组输入和输出元素
        page_data_flow = {}
        data_flow_grouped = 0
        for elem in web_elements:
            elem_som = getattr(elem, 'som_mark', None)
            if not elem_som:
                continue
                
            elem_node = self._find_node_by_som(task_graph, elem_som)
            if not elem_node:
                continue
                
            page = self._find_element_page(task_graph, elem_node)
            if not page:
                continue
                
            if page.node_id not in page_data_flow:
                page_data_flow[page.node_id] = {'inputs': [], 'outputs': []}
            
            if elem.is_input:
                page_data_flow[page.node_id]['inputs'].append((elem, elem_node))
            elif elem.is_clickable and not elem.is_input:
                page_data_flow[page.node_id]['outputs'].append((elem, elem_node))
            data_flow_grouped += 1
        
        logger.debug(f"🕸️ Grouped {data_flow_grouped} data flow elements into {len(page_data_flow)} pages")
        
        edges_created = 0
        
        # 为每个页面创建数据流边
        for page_id, elements in page_data_flow.items():
            inputs_count = len(elements['inputs'])
            outputs_count = len(elements['outputs'])
            logger.debug(f"🕸️ Processing page {page_id}: {inputs_count} inputs, {outputs_count} outputs")
            page_edges = self._create_page_data_flow_edges(task_graph, elements)
            edges_created += page_edges
            logger.debug(f"🕸️ Page {page_id}: Created {page_edges} data flow edges")
        
        logger.debug(f"🕸️ Created {edges_created} data flow edges")
    
    def _create_page_data_flow_edges(self, task_graph: "TaskGraph", elements: dict) -> int:
        """为单个页面创建数据流边"""
        
        edges_created = 0
        inputs = elements['inputs']
        outputs = elements['outputs']
        
        # 限制边的数量，避免边爆炸
        max_edges_per_input = 2
        max_edges_per_output = 2
        
        # 创建输入到输出的数据流
        for i, (input_elem, input_node) in enumerate(inputs):
            input_edges_created = 0
            logger.debug(f"🕸️ Processing input {i+1}/{len(inputs)}: {input_node.node_id}")
            
            for j, (output_elem, output_node) in enumerate(outputs):
                if input_edges_created >= max_edges_per_input:
                    logger.debug(f"🕸️ Input {i+1}: Reached max edges per input ({max_edges_per_input})")
                    break
                    
                # 验证数据流边的合理性
                if self._validate_data_flow_edge(input_node, output_node, input_elem, output_elem):
                    edge_type = self._infer_edge_type(input_node, output_node, 'data_flow')
                    
                    edge = GraphEdge(
                        edge_id=f"flows_{input_node.node_id}_{output_node.node_id}",
                        source_node_id=input_node.node_id,
                        target_node_id=output_node.node_id,
                        edge_type=edge_type
                    )
                    task_graph.add_edge(edge)
                    edges_created += 1
                    input_edges_created += 1
                    logger.debug(f"🕸️ Created data flow edge: {input_node.node_id} -> {output_node.node_id} ({edge_type.value})")
                else:
                    logger.debug(f"🕸️ Input {i+1} -> Output {j+1}: Data flow edge validation failed")
        
        return edges_created
    
    def _validate_data_flow_edge(self, input_node: "GraphNode", output_node: "GraphNode", 
                                input_elem: WebElement, output_elem: WebElement) -> bool:
        """验证数据流边的合理性 - 放宽验证条件"""
        # 基本验证
        if input_node.node_id == output_node.node_id:
            return False
        
        # 放宽输入元素类型验证
        input_type = input_node.node_type.value.lower()
        valid_input_types = ['input', 'select', 'textarea', 'form', 'form_control', 'search_box']
        if input_type not in valid_input_types:
            return False
        
        # 放宽输出元素类型验证
        output_type = output_node.node_type.value.lower()
        valid_output_types = ['button', 'submit', 'form', 'content', 'clickable', 'navigation', 'menu', 'tab']
        if output_type not in valid_output_types:
            return False
        
        # 放宽元素属性验证
        if not input_elem.is_input:
            return False
        
        # 输出元素可以是可点击的或者有特定类型
        if not (output_elem.is_clickable or output_type in ['button', 'submit', 'clickable']):
            return False
        
        # 简化语义合理性验证
        input_text = getattr(input_elem, 'text_content', '').lower()
        output_text = getattr(output_elem, 'text_content', '').lower()
        
        # 避免创建无意义的边
        if input_text and output_text:
            # 如果输入和输出文本完全相同，可能不是数据流关系
            if input_text == output_text:
                return False
        
        return True
    
    def _find_element_page(self, task_graph: "TaskGraph", element_node) -> Optional[Any]:
        """找到元素所在的页面节点"""
        if not hasattr(element_node, 'url') or not element_node.url:
            return None
        
        for node in task_graph.nodes.values():
            if (hasattr(node, 'node_type') and node.node_type.value == 'Page' and 
                hasattr(node, 'url') and node.url == element_node.url):
                return node
        return None
    
    def _find_page_by_url_pattern(self, task_graph: "TaskGraph", href: str) -> Optional[Any]:
        """根据href模式找到目标页面 - 改进的URL匹配逻辑"""
        from urllib.parse import urljoin
        import re
        
        if not href:
            return None
        
        # 获取所有页面节点
        page_nodes = []
        for node in task_graph.nodes.values():
            if (hasattr(node, 'node_type') and node.node_type.value == 'Page' and 
                hasattr(node, 'url') and node.url):
                page_nodes.append(node)
        
        if not page_nodes:
            return None
        
        # 1. 精确匹配（完全相同的URL）
        for node in page_nodes:
            if node.url == href:
                logger.debug(f"🎯 Exact URL match: {href}")
                return node
        
        # 2. 规范化URL匹配
        normalized_href = self._normalize_url(href)
        for node in page_nodes:
            normalized_page_url = self._normalize_url(node.url)
            if normalized_page_url == normalized_href:
                logger.debug(f"🎯 Normalized URL match: {href} -> {node.url}")
                return node
        
        # 3. 相对URL处理
        base_urls = [self._get_base_url(node.url) for node in page_nodes]
        unique_base_urls = list(set(base_urls))
        
        for base_url in unique_base_urls:
            try:
                # 将相对URL转换为绝对URL
                absolute_href = urljoin(base_url, href)
                normalized_absolute_href = self._normalize_url(absolute_href)
                
                for node in page_nodes:
                    normalized_page_url = self._normalize_url(node.url)
                    if normalized_page_url == normalized_absolute_href:
                        logger.debug(f"🎯 Relative URL match: {href} -> {absolute_href} -> {node.url}")
                        return node
            except Exception as e:
                logger.debug(f"Failed to join URLs: {base_url} + {href}: {e}")
                continue
        
        # 4. 路径匹配（忽略域名和协议）
        href_path = self._extract_path(href)
        if href_path and href_path != '/':
            for node in page_nodes:
                page_path = self._extract_path(node.url)
                if page_path == href_path:
                    logger.debug(f"🎯 Path match: {href_path}")
                    return node
        
        # 5. 智能部分匹配（最后的回退策略）
        href_lower = href.lower().strip('/')
        best_match = None
        best_score = 0
        
        for node in page_nodes:
            page_url_lower = node.url.lower().strip('/')
            
            # 计算匹配得分
            score = self._calculate_url_similarity(href_lower, page_url_lower)
            if score > best_score and score > 0.7:  # 阈值为0.7
                best_score = score
                best_match = node
        
        if best_match:
            logger.debug(f"🎯 Smart match: {href} -> {best_match.url} (score: {best_score:.2f})")
            return best_match
        
        logger.debug(f"❌ No URL match found for: {href}")
        return None
    
    def _normalize_url(self, url: str) -> str:
        """规范化URL"""
        if not url:
            return ""
        
        # 移除片段标识符（#）
        url = url.split('#')[0]
        
        # 移除查询参数（?）
        url = url.split('?')[0]
        
        # 统一斜杠
        url = url.replace('\\', '/')
        
        # 移除末尾斜杠（除非是根路径）
        if url.endswith('/') and len(url) > 1:
            url = url.rstrip('/')
        
        # 转换为小写（只有域名部分）
        try:
            parsed = urlparse(url)
            if parsed.netloc:
                # 协议和域名转小写，路径保持原样
                normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}"
                return normalized
        except:
            pass
        
        return url
    
    def _get_base_url(self, url: str) -> str:
        """获取URL的基础部分（协议+域名）"""
        try:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except:
            return url
    
    def _extract_path(self, url: str) -> str:
        """提取URL的路径部分"""
        try:
            parsed = urlparse(url)
            path = parsed.path
            if not path or path == '':
                return '/'
            return path.rstrip('/')
        except:
            return url
    
    def _calculate_url_similarity(self, url1: str, url2: str) -> float:
        """计算两个URL的相似度"""
        if not url1 or not url2:
            return 0.0
        
        # 简单的相似度计算
        if url1 == url2:
            return 1.0
        
        # 检查一个是否包含另一个
        if url1 in url2 or url2 in url1:
            shorter = min(len(url1), len(url2))
            longer = max(len(url1), len(url2))
            return shorter / longer
        
        # 检查公共子字符串
        common_length = 0
        min_length = min(len(url1), len(url2))
        
        for i in range(min_length):
            if url1[i] == url2[i]:
                common_length += 1
            else:
                break
        
        return common_length / max(len(url1), len(url2))
    
    def _find_node_by_som(self, task_graph: "TaskGraph", som_mark: str) -> Optional["GraphNode"]:
        """根据SoM标记查找节点"""
        if not som_mark:
            return None
            
        for node in task_graph.nodes.values():
            if hasattr(node, 'metadata') and hasattr(node.metadata, 'som_mark'):
                if node.metadata.som_mark == som_mark:
                    return node
        return None
    
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
        
        # Use progress bar for element processing
        nodes_created = 0
        with tqdm(total=len(elements), desc="📝 Creating element nodes", unit="elem", ncols=100) as pbar:
            for i, element in enumerate(elements):
                # Check if element has the expected attributes
                if not hasattr(element, 'element_type'):
                    logger.warning(f"Element {i} does not have element_type attribute")
                    pbar.update(1)
                    continue
                    
                node = self._element_to_node(element)
                if node:
                    graph.add_node(node)
                    nodes_created += 1
                
                pbar.update(1)
                pbar.set_postfix({"Nodes": nodes_created})
        
        logger.info(f"Created {nodes_created} element nodes")
    
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
        
        # Use progress bar for element processing
        with tqdm(total=len(elements), desc="🔍 Extracting entities", unit="elem", ncols=100) as pbar:
            for i, element in enumerate(elements):
                # Check if element has the expected attributes
                if not hasattr(element, 'element_type'):
                    logger.warning(f"Element {i} does not have element_type attribute")
                    pbar.update(1)
                    continue
                    
                if not hasattr(element, 'content'):
                    logger.warning(f"Element {i} does not have content attribute")
                    pbar.update(1)
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
                
                pbar.update(1)
                pbar.set_postfix({"Entities": len(entity_mentions)})
        
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
    
    # Generic business data classification methods
    def _is_user_data(self, text_content: str, text_lower: str) -> bool:
        """Check if content represents user data (names, emails, phones, etc.)"""
        import re

        # Email pattern - enhanced
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, text_content):
            return True

        # Phone pattern - enhanced for international formats
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
            r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}\b',  # International
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'  # (123) 456-7890
        ]
        for pattern in phone_patterns:
            if re.search(pattern, text_content):
                return True

        # Name patterns - enhanced with more formats
        name_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # John Smith
            r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b',  # John Michael Smith
            r'\b[A-Z]\. [A-Z][a-z]+\b',  # J. Smith
            r'\b[A-Z][a-z]+ [A-Z]\.\b',  # John S.
        ]
        for pattern in name_patterns:
            if re.search(pattern, text_content):
                return True

        # User-related keywords - expanded
        user_keywords = [
            'user', 'customer', 'client', 'member', 'account', 'profile',
            'contact', 'person', 'individual', 'login', 'username', 'userid'
        ]
        if any(keyword in text_lower for keyword in user_keywords):
            return True

        # Check for common user data fields
        user_fields = ['name', 'first name', 'last name', 'full name', 'email', 'phone', 'mobile', 'address']
        if any(field in text_lower for field in user_fields):
            return True

        return False
    
    def _is_product_data(self, text_content: str, text_lower: str) -> bool:
        """Check if content represents product data"""
        import re

        # Price pattern - enhanced for multiple currencies
        price_patterns = [
            r'\$\d+(?:\.\d{2})?',  # USD
            r'€\d+(?:[,.]\d{2})?',  # EUR
            r'£\d+(?:[,.]\d{2})?',  # GBP
            r'¥\d+',  # JPY
            r'\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP|CNY|JPY)',  # Amount + currency
        ]
        for pattern in price_patterns:
            if re.search(pattern, text_content):
                return True

        # Product-related keywords - expanded
        product_keywords = [
            'product', 'item', 'goods', 'merchandise', 'inventory', 'stock',
            'catalog', 'sku', 'quantity', 'price', 'cost', 'sale', 'discount',
            'brand', 'model', 'category', 'description'
        ]
        if any(keyword in text_lower for keyword in product_keywords):
            return True

        # SKU/Product ID patterns
        sku_patterns = [
            r'\bSKU[-\s]?\w+',  # SKU-12345
            r'\b[A-Z]{2,}\d+',  # Product codes like AB123
            r'\b\d{4,}',  # Product IDs
        ]
        for pattern in sku_patterns:
            if re.search(pattern, text_content):
                return True

        return False
    
    def _is_order_data(self, text_content: str, text_lower: str) -> bool:
        """Check if content represents order data"""
        import re
        
        # Order number pattern
        order_pattern = r'\b[A-Z]{2,}\d{4,}\b'  # ORD12345, INV2023001
        if re.search(order_pattern, text_content):
            return True
        
        # Order-related keywords
        order_keywords = ['order', 'purchase', 'transaction', 'booking', 'reservation']
        if any(keyword in text_lower for keyword in order_keywords):
            return True
        
        return False
    
    def _is_content_data(self, text_content: str, text_lower: str) -> bool:
        """Check if content represents content data (articles, posts, etc.)"""
        content_keywords = ['article', 'post', 'blog', 'news', 'story', 'content', 'text']
        if any(keyword in text_lower for keyword in content_keywords):
            return True
        
        # Check if it's a title or heading
        if len(text_content.split()) <= 10 and text_content.isupper():
            return True
        
        return False
    
    def _is_financial_data(self, text_content: str, text_lower: str) -> bool:
        """Check if content represents financial data"""
        import re
        
        # Currency patterns
        currency_patterns = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # $1,234.56
            r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)',  # 1,234.56 USD
        ]
        for pattern in currency_patterns:
            if re.search(pattern, text_content):
                return True
        
        # Financial keywords
        financial_keywords = ['amount', 'total', 'balance', 'payment', 'cost', 'price', 'revenue']
        if any(keyword in text_lower for keyword in financial_keywords):
            return True
        
        return False
    
    def _is_location_data(self, text_content: str, text_lower: str) -> bool:
        """Check if content represents location data"""
        location_keywords = ['address', 'street', 'city', 'state', 'country', 'zip', 'postal']
        if any(keyword in text_lower for keyword in location_keywords):
            return True
        
        # Common location patterns
        location_patterns = ['USA', 'United States', 'New York', 'London', 'Paris', 'Tokyo']
        if any(pattern in text_content for pattern in location_patterns):
            return True
        
        return False
    
    def _is_time_data(self, text_content: str, text_lower: str) -> bool:
        """Check if content represents time data"""
        import re
        
        # Date patterns
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY
        ]
        for pattern in date_patterns:
            if re.search(pattern, text_content):
                return True
        
        # Time patterns
        time_patterns = [
            r'\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?',  # HH:MM or HH:MM:SS
        ]
        for pattern in time_patterns:
            if re.search(pattern, text_content):
                return True
        
        # Time-related keywords
        time_keywords = ['date', 'time', 'created', 'updated', 'modified', 'published']
        if any(keyword in text_lower for keyword in time_keywords):
            return True
        
        return False
    
    def _classify_business_data_by_content(self, element: WebElement) -> "NodeType":
        """根据内容智能分类业务数据类型"""
        from .node_types import NodeType
        
        text_content = getattr(element, 'text_content', '')
        text_lower = text_content.lower()
        
        # 用户数据检测
        if self._is_user_data(text_content, text_lower):
            return NodeType.USER_DATA
        
        # 产品数据检测
        if self._is_product_data(text_content, text_lower):
            return NodeType.PRODUCT_DATA
        
        # 订单数据检测
        if self._is_order_data(text_content, text_lower):
            return NodeType.ORDER_DATA
        
        # 内容数据检测
        if self._is_content_data(text_content, text_lower):
            return NodeType.CONTENT_DATA
        
        # 财务数据检测
        if self._is_financial_data(text_content, text_lower):
            return NodeType.FINANCIAL_DATA
        
        # 位置数据检测
        if self._is_location_data(text_content, text_lower):
            return NodeType.LOCATION_DATA
        
        # 时间数据检测
        if self._is_time_data(text_content, text_lower):
            return NodeType.TIME_DATA
        
        # 默认使用通用业务数据类型
        return NodeType.BUSINESS_DATA
    
    def _infer_edge_type(self, source_node: "GraphNode", target_node: "GraphNode", relationship: str) -> "EdgeType":
        """从节点关系推断边类型"""
        
        # 基于节点类型和关系推断边类型
        source_type = source_node.node_type
        target_type = target_node.node_type
        
        # 导航关系
        if relationship == 'navigates_to' or relationship == 'nav_to':
            return EdgeType.NAV_TO
        
        # 包含关系
        if relationship == 'contains' or relationship == 'contains_element':
            return EdgeType.CONTAINS
        
        # 控制关系
        if (source_type.value in ['button', 'submit', 'input'] and 
            target_type.value in ['form', 'input', 'select']):
            return EdgeType.CONTROLS
        
        # 填充关系
        if (source_type.value in ['input', 'select'] and 
            target_type.value in ['form', 'table', 'content']):
            return EdgeType.FILLS
        
        # 过滤关系
        if (source_type.value in ['filter', 'input'] and 
            target_type.value in ['table', 'list', 'content']):
            return EdgeType.FILTERS
        
        # 打开关系
        if (source_type.value in ['button', 'link'] and 
            target_type.value in ['modal', 'page', 'content']):
            return EdgeType.OPENS
        
        # 引用关系
        if (source_type.value in ['link', 'detail_link'] and 
            target_type.value in ['content', 'detail', 'page']):
            return EdgeType.REFERS_TO
        
        # 触发关系
        if relationship == 'triggers':
            if source_type.value in ['button', 'link']:
                return EdgeType.OPENS
            elif source_type.value in ['input', 'select']:
                return EdgeType.CONTROLS
            else:
                return EdgeType.CONTAINS
        
        # 数据流关系
        if relationship == 'data_flow':
            if source_type.value in ['input', 'select'] and target_type.value in ['button', 'submit']:
                return EdgeType.FILLS
            elif source_type.value in ['button', 'submit'] and target_type.value in ['form', 'content']:
                return EdgeType.CONTROLS
            else:
                return EdgeType.CONTAINS
        
        # 默认包含关系
        return EdgeType.CONTAINS    
    
    def _create_same_entity_edges(self, task_graph: "TaskGraph", web_elements: List[WebElement]) -> int:
        """创建SAME_ENTITY边 - 识别同一实体的不同视图"""
        
        logger.debug(f"🕸️ Creating same entity edges...")
        
        # 找到业务数据元素
        business_data_elements = [elem for elem in web_elements 
                                if getattr(elem, 'som_type', '') == 'business_data']
        
        logger.debug(f"🕸️ Found {len(business_data_elements)} business data elements for same entity analysis")
        
        edges_created = 0
        max_same_entity_edges = 1000  # 限制SAME_ENTITY边的数量
        
        # 分析同一实体的不同视图
        for i, elem1 in enumerate(business_data_elements):
            if edges_created >= max_same_entity_edges:
                logger.debug(f"🕸️ Reached maximum SAME_ENTITY edges limit: {max_same_entity_edges}")
                break
            elem1_som = getattr(elem1, 'som_mark', None)
            if not elem1_som:
                continue
                
            elem1_node = self._find_node_by_som(task_graph, elem1_som)
            if not elem1_node:
                continue
                
            elem1_text = getattr(elem1, 'text_content', '').lower()
            if not elem1_text:
                continue
            
            # 寻找可能表示同一实体的其他元素
            for j, elem2 in enumerate(business_data_elements[i+1:], i+1):
                elem2_som = getattr(elem2, 'som_mark', None)
                if not elem2_som:
                    continue
                    
                elem2_node = self._find_node_by_som(task_graph, elem2_som)
                if not elem2_node:
                    continue
                    
                elem2_text = getattr(elem2, 'text_content', '').lower()
                if not elem2_text:
                    continue
                
                # 检查是否表示同一实体（简化逻辑）
                if self._is_same_entity(elem1_text, elem2_text):
                    edge = GraphEdge(
                        edge_id=f"same_entity_{elem1_node.node_id}_{elem2_node.node_id}",
                        source_node_id=elem1_node.node_id,
                        target_node_id=elem2_node.node_id,
                        edge_type=EdgeType.SAME_ENTITY
                    )
                    task_graph.add_edge(edge)
                    edges_created += 1
                    logger.debug(f"🕸️ Created SAME_ENTITY edge: {elem1_text[:20]} <-> {elem2_text[:20]}")
        
        logger.debug(f"🕸️ Created {edges_created} same entity edges")
        return edges_created
    
    def _is_same_entity(self, text1: str, text2: str) -> bool:
        """判断两个文本是否表示同一实体 - 严格版本"""
        # 清理文本
        text1 = text1.strip().lower()
        text2 = text2.strip().lower()
        
        # 基本验证
        if not text1 or not text2 or text1 == text2:
            return False
        
        # 检查长度差异（避免过短文本的误判）
        if abs(len(text1) - len(text2)) > max(len(text1), len(text2)) * 0.5:
            return False
        
        # 检查是否包含相同的ID、邮箱、电话号码等标识符
        import re
        
        # 邮箱模式
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails1 = set(re.findall(email_pattern, text1))
        emails2 = set(re.findall(email_pattern, text2))
        if emails1 and emails2 and emails1.intersection(emails2):
            return True
        
        # 电话号码模式
        phone_pattern = r'\b\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'
        phones1 = set(re.findall(phone_pattern, text1))
        phones2 = set(re.findall(phone_pattern, text2))
        if phones1 and phones2 and phones1.intersection(phones2):
            return True
        
        # ID模式（数字ID）
        id_pattern = r'\b\d{4,}\b'
        ids1 = set(re.findall(id_pattern, text1))
        ids2 = set(re.findall(id_pattern, text2))
        if ids1 and ids2 and ids1.intersection(ids2):
            return True
        
        # 检查关键业务实体名称
        business_entities = ['account', 'contact', 'lead', 'opportunity', 'case', 'order', 'invoice']
        for entity in business_entities:
            if entity in text1 and entity in text2:
                # 提取实体名称（实体后的词）
                words1 = text1.split()
                words2 = text2.split()
                
                # 找到实体关键词的位置
                for i, word in enumerate(words1):
                    if entity in word and i + 1 < len(words1):
                        entity_name1 = words1[i + 1]
                        for j, word2 in enumerate(words2):
                            if entity in word2 and j + 1 < len(words2):
                                entity_name2 = words2[j + 1]
                                if entity_name1 == entity_name2:
                                    return True
        
        # 检查是否有足够的具体共同词（提高阈值）
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # 过滤掉常见的停用词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if len(words1) < 2 or len(words2) < 2:
            return False
        
        common_words = words1.intersection(words2)
        # 提高阈值到60%，并且要求至少有2个共同词
        min_common_words = max(2, min(len(words1), len(words2)) * 0.6)
        
        return len(common_words) >= min_common_words
    
    def _discover_dynamic_edge_patterns(self, task_graph: "TaskGraph", web_elements: List[WebElement]) -> int:
        """动态边类型发现 - 分析元素间的实际关系模式"""
        
        logger.info(f"🔍 === DYNAMIC EDGE TYPE DISCOVERY START ===")
        logger.info(f"🔍 Analyzing {len(web_elements)} elements for dynamic relationship patterns")
        
        edges_created = 0
        
        # 1. 基于空间布局的关系发现
        spatial_edges = self._discover_spatial_relationships(task_graph, web_elements)
        edges_created += spatial_edges
        
        # 2. 基于功能组合的关系发现
        functional_edges = self._discover_functional_relationships(task_graph, web_elements)
        edges_created += functional_edges
        
        # 3. 基于语义相似度的关系发现
        semantic_edges = self._discover_semantic_relationships(task_graph, web_elements)
        edges_created += semantic_edges
        
        # 4. 基于交互模式的关系发现
        interaction_edges = self._discover_interaction_patterns(task_graph, web_elements)
        edges_created += interaction_edges
        
        logger.info(f"🔍 === DYNAMIC EDGE TYPE DISCOVERY END ===")
        logger.info(f"🔍 Created {edges_created} dynamic relationship edges")
        
        return edges_created
    
    def _discover_spatial_relationships(self, task_graph: "TaskGraph", web_elements: List[WebElement]) -> int:
        """基于空间布局发现关系（内存优化版本）"""
        
        logger.debug(f"🔍 Discovering spatial relationships...")
        edges_created = 0
        
        # 添加内存管理
        gc.collect()
        self._log_memory_usage("spatial relationships start")
        
        # 限制元素数量进行空间分析
        if hasattr(self.config, 'max_elements_per_page') and len(web_elements) > self.config.max_elements_per_page:
            logger.warning(f"⚠️ Too many elements ({len(web_elements)}) for spatial analysis, sampling...")
            # 优先选择可见和可交互的元素
            interactive_elements = [e for e in web_elements if getattr(e, 'is_visible', True) and getattr(e, 'is_clickable', False)]
            other_elements = [e for e in web_elements if not (getattr(e, 'is_visible', True) and getattr(e, 'is_clickable', False))]
            
            # 取交互元素 + 一些其他元素
            limited_elements = interactive_elements[:self.config.max_elements_per_page]
            remaining_slots = self.config.max_elements_per_page - len(limited_elements)
            limited_elements.extend(other_elements[:remaining_slots])
            web_elements = limited_elements
            logger.info(f"📊 Limited spatial analysis to {len(web_elements)} elements")
        
        # 按页面分组元素
        page_elements = {}
        for elem in web_elements:
            elem_node = self._find_node_by_som(task_graph, elem.som_mark)
            if not elem_node:
                continue
            
            page = self._find_element_page(task_graph, elem_node)
            if not page:
                continue
            
            if page.node_id not in page_elements:
                page_elements[page.node_id] = []
            page_elements[page.node_id].append((elem, elem_node))
        
        # 分析每个页面内的空间关系（批处理优化）
        max_spatial_edges_per_page = getattr(self.config, 'max_semantic_edges_per_page', 30)  # 复用配置
        
        for page_id, elements in page_elements.items():
            # 按位置排序元素（从上到下，从左到右）
            sorted_elements = sorted(elements, key=lambda x: (x[0].y, x[0].x))
            
            # 使用批处理计算空间关系
            page_edges = self._calculate_spatial_relationships_batch(task_graph, sorted_elements, max_spatial_edges_per_page)
            edges_created += page_edges
            
            # 每页后清理内存
            self.operation_count += 1
            if self._should_cleanup():
                self._force_memory_cleanup()
                self._log_memory_usage(f"spatial analysis page {page_id}")
        
        # 清理内存
        del page_elements
        gc.collect()
        self._log_memory_usage("spatial relationships end")
        
        logger.debug(f"🔍 Created {edges_created} spatial relationship edges")
        return edges_created
    
    def _calculate_spatial_relationships_batch(self, task_graph: "TaskGraph", elements: List[Tuple[WebElement, Node]], max_edges: int) -> int:
        """批处理计算空间关系，减少内存使用"""
        edges_created = 0
        batch_size = getattr(self.config, 'batch_size', 50)
        
        # 按批次处理元素对
        for i in range(0, len(elements), batch_size):
            batch_elements = elements[i:i + batch_size]
            
            # 计算批次内的空间关系
            for j, (elem1, node1) in enumerate(batch_elements):
                for k, (elem2, node2) in enumerate(batch_elements[j+1:], j+1):
                    # 限制每页的边数量
                    if edges_created >= max_edges:
                        logger.debug(f"🔍 Reached max spatial edges per page ({max_edges})")
                        break
                    
                    # 计算空间距离
                    distance = self._calculate_spatial_distance(elem1, elem2)
                    
                    # 如果元素足够接近，创建空间关系边
                    if distance < 100:  # 100像素阈值
                        # 推断空间关系类型
                        edge_type = self._infer_spatial_relationship(elem1, elem2, distance)
                        
                        if edge_type:
                            edge = GraphEdge(
                                edge_id=f"spatial_{node1.node_id}_{node2.node_id}",
                                source_node_id=node1.node_id,
                                target_node_id=node2.node_id,
                                edge_type=edge_type
                            )
                            task_graph.add_edge(edge)
                            edges_created += 1
                            logger.debug(f"🔍 Created spatial edge: {node1.node_id} -[{edge_type.value}]-> {node2.node_id}")
                
                if edges_created >= max_edges:
                    break
            
            # 批次后清理内存
            gc.collect()
        
        return edges_created
    
    def _discover_functional_relationships(self, task_graph: "TaskGraph", web_elements: List[WebElement]) -> int:
        """基于功能组合发现关系（内存优化版本）"""
        
        logger.debug(f"🔍 Discovering functional relationships...")
        edges_created = 0
        
        # 添加内存管理
        gc.collect()
        self._log_memory_usage("functional relationships start")
        
        # 限制元素数量进行功能分析
        if hasattr(self.config, 'max_elements_per_page') and len(web_elements) > self.config.max_elements_per_page:
            logger.warning(f"⚠️ Too many elements ({len(web_elements)}) for functional analysis, sampling...")
            # 优先选择表单和交互元素
            form_elements = [e for e in web_elements if getattr(e, 'is_input', False) or getattr(e, 'tag', '').lower() in ['form', 'input', 'select', 'textarea']]
            interactive_elements = [e for e in web_elements if getattr(e, 'is_clickable', False)]
            other_elements = [e for e in web_elements if not (getattr(e, 'is_input', False) or getattr(e, 'is_clickable', False))]
            
            # 取表单元素 + 交互元素 + 一些其他元素
            limited_elements = form_elements[:self.config.max_elements_per_page // 2]
            remaining_slots = self.config.max_elements_per_page - len(limited_elements)
            limited_elements.extend(interactive_elements[:remaining_slots // 2])
            remaining_slots = self.config.max_elements_per_page - len(limited_elements)
            limited_elements.extend(other_elements[:remaining_slots])
            web_elements = limited_elements
            logger.info(f"📊 Limited functional analysis to {len(web_elements)} elements")
        
        # 1. 表单控制关系发现
        form_controls = self._discover_form_control_relationships(task_graph, web_elements)
        edges_created += form_controls
        gc.collect()  # 每个子方法后清理内存
        
        # 2. 搜索过滤关系发现
        search_filters = self._discover_search_filter_relationships(task_graph, web_elements)
        edges_created += search_filters
        gc.collect()
        
        # 3. 数据展示关系发现
        data_display = self._discover_data_display_relationships(task_graph, web_elements)
        edges_created += data_display
        gc.collect()
        
        self._log_memory_usage("functional relationships end")
        logger.debug(f"🔍 Created {edges_created} functional relationship edges")
        return edges_created
    
    def _discover_semantic_relationships(self, task_graph: "TaskGraph", web_elements: List[WebElement]) -> int:
        """基于语义相似度发现关系（内存优化版本）"""
        
        logger.debug(f"🔍 Discovering semantic relationships...")
        edges_created = 0
        
        # 添加内存管理
        gc.collect()  # 强制垃圾回收
        self._log_memory_usage("semantic relationships start")
        
        # 按页面分组元素
        page_elements = {}
        for elem in web_elements:
            elem_node = self._find_node_by_som(task_graph, elem.som_mark)
            if not elem_node:
                continue
            
            page = self._find_element_page(task_graph, elem_node)
            if not page:
                continue
            
            if page.node_id not in page_elements:
                page_elements[page.node_id] = []
            page_elements[page.node_id].append((elem, elem_node))
        
        # 分析每个页面内的语义关系（批处理优化版本）
        for page_id, elements in page_elements.items():
            # 限制每页元素数量
            if hasattr(self.config, 'max_elements_per_page') and len(elements) > self.config.max_elements_per_page:
                logger.warning(f"⚠️ Page {page_id} has {len(elements)} elements, limiting for semantic analysis")
                # 优先选择有文本内容的元素
                elements_with_text = [(e, n) for e, n in elements if getattr(e, 'text', '').strip()]
                elements_without_text = [(e, n) for e, n in elements if not getattr(e, 'text', '').strip()]
                
                # 取有文本的元素 + 一些无文本的元素
                limited_elements = elements_with_text[:self.config.max_elements_per_page]
                remaining_slots = self.config.max_elements_per_page - len(limited_elements)
                limited_elements.extend(elements_without_text[:remaining_slots])
                elements = limited_elements
            
            # 使用批处理计算语义相似度
            page_edges = self._calculate_semantic_similarities_batch(task_graph, elements)
            edges_created += page_edges
            
            # 每页后清理内存
            self.operation_count += 1
            if self._should_cleanup():
                self._force_memory_cleanup()
                self._log_memory_usage(f"semantic analysis page {page_id}")
        
        logger.debug(f"🔍 Created {edges_created} semantic relationship edges")
        
        # 清理内存
        del page_elements
        gc.collect()
        self._log_memory_usage("semantic relationships end")
        
        return edges_created
    
    def _calculate_semantic_similarities_batch(self, task_graph: "TaskGraph", elements: List[Tuple[WebElement, Node]]) -> int:
        """批处理计算语义相似度，减少内存使用"""
        edges_created = 0
        batch_size = getattr(self.config, 'batch_size', 50)
        max_edges_per_page = getattr(self.config, 'max_semantic_edges_per_page', 50)
        
        # 按批次处理元素对
        for i in range(0, len(elements), batch_size):
            batch_elements = elements[i:i + batch_size]
            
            # 计算批次内的相似度
            for j, (elem1, node1) in enumerate(batch_elements):
                for k, (elem2, node2) in enumerate(batch_elements[j+1:], j+1):
                    # 限制每页的边数量
                    if edges_created >= max_edges_per_page:
                        logger.debug(f"🔍 Reached max semantic edges per page ({max_edges_per_page})")
                        break
                    
                    # 快速预筛选：只对有文本内容的元素计算相似度
                    text1 = getattr(elem1, 'text', '').strip()
                    text2 = getattr(elem2, 'text', '').strip()
                    
                    if not text1 or not text2:
                        continue
                    
                    # 计算语义相似度
                    similarity = self._calculate_semantic_similarity(elem1, elem2)
                    
                    # 提高阈值，只创建真正相似的边
                    if similarity > 0.85:
                        edge = GraphEdge(
                            edge_id=f"semantic_{node1.node_id}_{node2.node_id}",
                            source_node_id=node1.node_id,
                            target_node_id=node2.node_id,
                            edge_type=EdgeType.REFERS_TO
                        )
                        task_graph.add_edge(edge)
                        edges_created += 1
                        logger.debug(f"🔍 Created semantic edge: {node1.node_id} -[REFERS_TO]-> {node2.node_id} (similarity: {similarity:.2f})")
                
                if edges_created >= max_edges_per_page:
                    break
            
            # 批次后清理内存
            gc.collect()
        
        return edges_created
    
    def _discover_interaction_patterns(self, task_graph: "TaskGraph", web_elements: List[WebElement]) -> int:
        """基于交互模式发现关系（内存优化版本）"""
        
        logger.debug(f"🔍 Discovering interaction patterns...")
        edges_created = 0
        
        # 添加内存管理
        gc.collect()
        self._log_memory_usage("interaction patterns start")
        
        # 限制元素数量进行交互分析
        if hasattr(self.config, 'max_elements_per_page') and len(web_elements) > self.config.max_elements_per_page:
            logger.warning(f"⚠️ Too many elements ({len(web_elements)}) for interaction analysis, sampling...")
            # 优先选择交互元素
            interactive_elements = [e for e in web_elements if getattr(e, 'is_clickable', False) or getattr(e, 'is_input', False)]
            button_elements = [e for e in web_elements if getattr(e, 'tag', '').lower() == 'button']
            link_elements = [e for e in web_elements if getattr(e, 'tag', '').lower() == 'a']
            other_elements = [e for e in web_elements if not (getattr(e, 'is_clickable', False) or getattr(e, 'is_input', False))]
            
            # 取按钮 + 链接 + 交互元素 + 一些其他元素
            limited_elements = button_elements[:self.config.max_elements_per_page // 4]
            limited_elements.extend(link_elements[:self.config.max_elements_per_page // 4])
            limited_elements.extend(interactive_elements[:self.config.max_elements_per_page // 2])
            remaining_slots = self.config.max_elements_per_page - len(limited_elements)
            limited_elements.extend(other_elements[:remaining_slots])
            web_elements = limited_elements
            logger.info(f"📊 Limited interaction analysis to {len(web_elements)} elements")
        
        # 1. 按钮-内容关系发现
        button_content = self._discover_button_content_relationships(task_graph, web_elements)
        edges_created += button_content
        gc.collect()
        
        # 2. 输入-输出关系发现
        input_output = self._discover_input_output_relationships(task_graph, web_elements)
        edges_created += input_output
        gc.collect()
        
        # 3. 导航-目标关系发现
        navigation_target = self._discover_navigation_target_relationships(task_graph, web_elements)
        edges_created += navigation_target
        gc.collect()
        
        self._log_memory_usage("interaction patterns end")
        logger.debug(f"🔍 Created {edges_created} interaction pattern edges")
        return edges_created
    
    # Helper methods for dynamic edge discovery
    def _calculate_spatial_distance(self, elem1: WebElement, elem2: WebElement) -> float:
        """计算两个元素之间的空间距离"""
        center1_x = elem1.x + elem1.width / 2
        center1_y = elem1.y + elem1.height / 2
        center2_x = elem2.x + elem2.width / 2
        center2_y = elem2.y + elem2.height / 2
        
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        return distance
    
    def _infer_spatial_relationship(self, elem1: WebElement, elem2: WebElement, distance: float) -> Optional[EdgeType]:
        """推断空间关系类型"""
        
        # 如果元素非常接近，可能是包含关系
        if distance < 20:
            return EdgeType.CONTAINS
        
        # 如果元素在垂直方向上接近，可能是序列关系
        vertical_distance = abs((elem1.y + elem1.height/2) - (elem2.y + elem2.height/2))
        if vertical_distance < 30 and distance < 100:
            return EdgeType.SEQUENCE
        
        # 如果元素在水平方向上接近，可能是布局关系
        horizontal_distance = abs((elem1.x + elem1.width/2) - (elem2.x + elem2.width/2))
        if horizontal_distance < 50 and distance < 100:
            return EdgeType.WEB_LAYOUT
        
        return None
    
    def _discover_form_control_relationships(self, task_graph: "TaskGraph", web_elements: List[WebElement]) -> int:
        """发现表单控制关系（内存优化版本）"""
        
        edges_created = 0
        
        # 添加内存管理
        import gc
        gc.collect()
        
        # 找到表单元素和提交按钮
        form_elements = [elem for elem in web_elements if elem.is_input]
        submit_buttons = [elem for elem in web_elements if elem.element_type == 'submit']
        
        for submit_btn in submit_buttons:
            submit_node = self._find_node_by_som(task_graph, submit_btn.som_mark)
            if not submit_node:
                continue
            
            page = self._find_element_page(task_graph, submit_node)
            if not page:
                continue
            
            # 找到同一页面中的表单元素
            for form_elem in form_elements:
                form_node = self._find_node_by_som(task_graph, form_elem.som_mark)
                if not form_node:
                    continue
                
                form_page = self._find_element_page(task_graph, form_node)
                if form_page and form_page.node_id == page.node_id:
                    # 创建控制关系
                    edge = GraphEdge(
                        edge_id=f"controls_{submit_node.node_id}_{form_node.node_id}",
                        source_node_id=submit_node.node_id,
                        target_node_id=form_node.node_id,
                        edge_type=EdgeType.CONTROLS
                    )
                    task_graph.add_edge(edge)
                    edges_created += 1
        
        # 清理内存
        del form_elements, submit_buttons
        gc.collect()
        
        return edges_created
    
    def _discover_search_filter_relationships(self, task_graph: "TaskGraph", web_elements: List[WebElement]) -> int:
        """发现搜索过滤关系（内存优化版本）"""
        
        edges_created = 0
        
        # 添加内存管理
        import gc
        gc.collect()
        
        # 找到搜索输入框和结果表格
        search_inputs = [elem for elem in web_elements if elem.is_input and 
                        (getattr(elem, 'placeholder', '').lower().find('search') != -1 or
                         getattr(elem, 'text_content', '').lower().find('search') != -1)]
        
        tables = [elem for elem in web_elements if elem.element_type == 'table']
        
        # 限制处理数量
        if len(search_inputs) > 10:
            search_inputs = search_inputs[:10]
        if len(tables) > 15:
            tables = tables[:15]
        
        for search_input in search_inputs:
            search_node = self._find_node_by_som(task_graph, search_input.som_mark)
            if not search_node:
                continue
            
            page = self._find_element_page(task_graph, search_node)
            if not page:
                continue
            
            # 找到同一页面中的表格
            for table in tables:
                table_node = self._find_node_by_som(task_graph, table.som_mark)
                if not table_node:
                    continue
                
                table_page = self._find_element_page(task_graph, table_node)
                if table_page and table_page.node_id == page.node_id:
                    # 创建过滤关系
                    edge = GraphEdge(
                        edge_id=f"filters_{search_node.node_id}_{table_node.node_id}",
                        source_node_id=search_node.node_id,
                        target_node_id=table_node.node_id,
                        edge_type=EdgeType.FILTERS
                    )
                    task_graph.add_edge(edge)
                    edges_created += 1
        
        # 清理内存
        del search_inputs, tables
        gc.collect()
        
        return edges_created
    
    def _discover_data_display_relationships(self, task_graph: "TaskGraph", web_elements: List[WebElement]) -> int:
        """发现数据展示关系（内存优化版本）"""
        
        edges_created = 0
        
        # 添加内存管理
        import gc
        gc.collect()
        
        # 找到输入元素和结果展示元素
        input_elements = [elem for elem in web_elements if elem.is_input]
        result_elements = [elem for elem in web_elements if elem.element_type in ['table', 'resultitem', 'card']]
        
        # 限制处理数量
        if len(input_elements) > 20:
            input_elements = input_elements[:20]
        if len(result_elements) > 25:
            result_elements = result_elements[:25]
        
        for input_elem in input_elements:
            input_node = self._find_node_by_som(task_graph, input_elem.som_mark)
            if not input_node:
                continue
            
            page = self._find_element_page(task_graph, input_node)
            if not page:
                continue
            
            # 找到同一页面中的结果展示元素
            for result_elem in result_elements:
                result_node = self._find_node_by_som(task_graph, result_elem.som_mark)
                if not result_node:
                    continue
                
                result_page = self._find_element_page(task_graph, result_node)
                if result_page and result_page.node_id == page.node_id:
                    # 创建填充关系
                    edge = GraphEdge(
                        edge_id=f"fills_{input_node.node_id}_{result_node.node_id}",
                        source_node_id=input_node.node_id,
                        target_node_id=result_node.node_id,
                        edge_type=EdgeType.FILLS
                    )
                    task_graph.add_edge(edge)
                    edges_created += 1
        
        # 清理内存
        del input_elements, result_elements
        gc.collect()
        
        return edges_created
    
    def _calculate_semantic_similarity(self, elem1: WebElement, elem2: WebElement) -> float:
        """计算两个元素的语义相似度"""
        text1 = getattr(elem1, 'text_content', '').lower()
        text2 = getattr(elem2, 'text_content', '').lower()
        
        if not text1 or not text2:
            return 0.0
        
        # 简单的词汇重叠相似度
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _discover_button_content_relationships(self, task_graph: "TaskGraph", web_elements: List[WebElement]) -> int:
        """发现按钮-内容关系（内存优化版本）"""
        
        edges_created = 0
        
        # 添加内存管理
        import gc
        gc.collect()
        
        # 找到按钮和可能的内容元素
        buttons = [elem for elem in web_elements if elem.element_type == 'button']
        content_elements = [elem for elem in web_elements if elem.element_type in ['card', 'detail', 'modal']]
        
        for button in buttons:
            button_node = self._find_node_by_som(task_graph, button.som_mark)
            if not button_node:
                continue
            
            page = self._find_element_page(task_graph, button_node)
            if not page:
                continue
            
            # 找到同一页面中的内容元素
            for content_elem in content_elements:
                content_node = self._find_node_by_som(task_graph, content_elem.som_mark)
                if not content_node:
                    continue
                
                content_page = self._find_element_page(task_graph, content_node)
                if content_page and content_page.node_id == page.node_id:
                    # 创建打开关系
                    edge = GraphEdge(
                        edge_id=f"opens_{button_node.node_id}_{content_node.node_id}",
                        source_node_id=button_node.node_id,
                        target_node_id=content_node.node_id,
                        edge_type=EdgeType.OPENS
                    )
                    task_graph.add_edge(edge)
                    edges_created += 1
        
        # 清理内存
        del buttons, content_elements
        gc.collect()
        
        return edges_created
    
    def _discover_input_output_relationships(self, task_graph: "TaskGraph", web_elements: List[WebElement]) -> int:
        """发现输入-输出关系（内存优化版本）"""
        
        edges_created = 0
        
        # 添加内存管理
        import gc
        gc.collect()
        
        # 找到输入元素和输出元素
        input_elements = [elem for elem in web_elements if elem.is_input]
        output_elements = [elem for elem in web_elements if elem.element_type in ['button', 'link']]
        
        for input_elem in input_elements:
            input_node = self._find_node_by_som(task_graph, input_elem.som_mark)
            if not input_node:
                continue
            
            page = self._find_element_page(task_graph, input_node)
            if not page:
                continue
            
            # 找到同一页面中的输出元素
            for output_elem in output_elements:
                output_node = self._find_node_by_som(task_graph, output_elem.som_mark)
                if not output_node:
                    continue
                
                output_page = self._find_element_page(task_graph, output_node)
                if output_page and output_page.node_id == page.node_id:
                    # 创建数据流关系
                    edge = GraphEdge(
                        edge_id=f"dataflow_{input_node.node_id}_{output_node.node_id}",
                        source_node_id=input_node.node_id,
                        target_node_id=output_node.node_id,
                        edge_type=EdgeType.DATA_FLOW
                    )
                    task_graph.add_edge(edge)
                    edges_created += 1
        
        # 清理内存
        del input_elements, output_elements
        gc.collect()
        
        return edges_created
    
    def _discover_navigation_target_relationships(self, task_graph: "TaskGraph", web_elements: List[WebElement]) -> int:
        """发现导航-目标关系（内存优化版本）"""
        
        edges_created = 0
        
        # 添加内存管理
        import gc
        gc.collect()
        
        # 找到导航元素和目标元素
        navigation_elements = [elem for elem in web_elements if elem.element_type in ['navigation', 'link']]
        target_elements = [elem for elem in web_elements if elem.element_type in ['page', 'card', 'detail']]
        
        for nav_elem in navigation_elements:
            nav_node = self._find_node_by_som(task_graph, nav_elem.som_mark)
            if not nav_node:
                continue
            
            page = self._find_element_page(task_graph, nav_node)
            if not page:
                continue
            
            # 找到同一页面中的目标元素
            for target_elem in target_elements:
                target_node = self._find_node_by_som(task_graph, target_elem.som_mark)
                if not target_node:
                    continue
                
                target_page = self._find_element_page(task_graph, target_node)
                if target_page and target_page.node_id == page.node_id:
                    # 创建导航关系
                    edge = GraphEdge(
                        edge_id=f"navto_{nav_node.node_id}_{target_node.node_id}",
                        source_node_id=nav_node.node_id,
                        target_node_id=target_node.node_id,
                        edge_type=EdgeType.NAV_TO
                    )
                    task_graph.add_edge(edge)
                    edges_created += 1

        # 清理内存
        del navigation_elements, target_elements
        gc.collect()
        
        return edges_created


@dataclass
class TaskGraph:
    """任务图 - 完整的图-任务抽象"""
    storage: GraphStorage = field(default_factory=lambda: JSONStorage())
    website_type: str = ""
    website_description: str = ""
    vector_index: Optional[Any] = None  # 向量索引，延迟初始化
    
    @property
    def nodes(self) -> Dict[str, 'GraphNode']:
        """获取所有节点的字典"""
        return self.storage.nodes
    
    @property
    def edges(self) -> Dict[str, GraphEdge]:
        """获取所有边的字典"""
        return self.storage.edges
    
    def add_node(self, node: 'GraphNode'):
        """添加节点"""
        if node is None:
            logger.warning("Attempted to add None node to TaskGraph")
            return
        self.storage.add_node(node)
    
    def add_edge(self, edge: GraphEdge):
        """添加边"""
        if edge is None:
            logger.warning("Attempted to add None edge to TaskGraph")
            return
        self.storage.add_edge(edge)
    
    def get_node_by_som(self, som_mark: str) -> Optional['GraphNode']:
        """根据SoM标记获取节点"""
        for node in self.nodes.values():
            if node is None:
                continue
            if hasattr(node, 'metadata') and node.metadata and hasattr(node.metadata, 'som_mark') and node.metadata.som_mark == som_mark:
                return node
        return None
    
    def get_neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List['GraphNode']:
        """获取节点的邻居"""
        neighbor_tuples = self.storage.get_neighbors(node_id, [edge_type] if edge_type else None)
        return [neighbor for neighbor, edge in neighbor_tuples]
    
    def get_node(self, node_id: str) -> Optional['GraphNode']:
        """获取节点"""
        return self.storage.get_node(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        """获取边"""
        return self.storage.get_edge(edge_id)
    
    def find_nodes(self, **criteria) -> List['GraphNode']:
        """查找节点"""
        return self.storage.find_nodes(**criteria)
    
    def find_edges(self, **criteria) -> List[GraphEdge]:
        """查找边"""
        return self.storage.find_edges(**criteria)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "website_type": self.website_type,
            "website_description": self.website_description,
            "nodes": [node.to_dict() for node in self.nodes.values() if node is not None],
            "edges": [edge.to_dict() for edge in self.edges.values() if edge is not None]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskGraph':
        """从字典创建TaskGraph"""
        from .node_types import NodeType, NodeMetadata, GraphNode

        # 创建storage实例
        storage = JSONStorage()
        graph = cls(
            storage=storage,
            website_type=data.get("website_type", ""),
            website_description=data.get("website_description", "")
        )

        # 重建节点 - 支持数组格式
        nodes_data = data.get("nodes", [])
        if isinstance(nodes_data, dict):
            # 兼容旧的字典格式
            for node_id, node_data in nodes_data.items():
                try:
                    node_type = NodeType(node_data["node_type"])
                    metadata = NodeMetadata(**node_data["metadata"])
                    node = GraphNode(
                        node_id=node_id,
                        node_type=node_type,
                        url=node_data.get("url", ""),
                        metadata=metadata
                    )
                    # 恢复element_type字段（如果存在）
                    if "element_type" in node_data:
                        node.element_type = node_data["element_type"]
                    graph.add_node(node)
                except Exception as e:
                    logger.warning(f"Failed to create node {node_id} from data: {e}")
                    continue
        else:
            # 新数组格式
            for node_data in nodes_data:
                try:
                    node_id = node_data["node_id"]
                    node_type = NodeType(node_data["node_type"])
                    metadata = NodeMetadata(**node_data["metadata"])
                    node = GraphNode(
                        node_id=node_id,
                        node_type=node_type,
                        url=node_data.get("url", ""),
                        metadata=metadata
                    )
                    # 恢复element_type字段（如果存在）
                    if "element_type" in node_data:
                        node.element_type = node_data["element_type"]
                    graph.add_node(node)
                except Exception as e:
                    logger.warning(f"Failed to create node from data: {e}")
                    continue

        # 重建边 - 支持数组格式
        edges_data = data.get("edges", [])
        if isinstance(edges_data, dict):
            # 兼容旧的字典格式
            for edge_id, edge_data in edges_data.items():
                edge_type = EdgeType(edge_data["edge_type"])
                edge = GraphEdge(
                    edge_id=edge_id,
                    source_node_id=edge_data["source_node"],
                    target_node_id=edge_data["target_node"],
                    edge_type=edge_type,
                    weight=edge_data.get("weight", 1.0),
                    metadata=edge_data.get("metadata", {})
                )
                graph.add_edge(edge)
        else:
            # 新数组格式
            for edge_data in edges_data:
                edge_id = edge_data["edge_id"]
                edge_type = EdgeType(edge_data["edge_type"])
                edge = GraphEdge(
                    edge_id=edge_id,
                    source_node_id=edge_data["source_node"],
                    target_node_id=edge_data["target_node"],
                    edge_type=edge_type,
                    weight=edge_data.get("weight", 1.0),
                    metadata=edge_data.get("metadata", {})
                )
                graph.add_edge(edge)

        return graph
    
    def save(self, graph_path: str, vector_path: str = None):
        """保存TaskGraph到文件，包括向量索引"""
        import json
        from pathlib import Path
        from loguru import logger
        
        # 保存图结构
        graph_dir = Path(graph_path)
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_file = graph_dir / "knowledge_graph.json"
        with open(graph_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.debug(f"🕸️ TaskGraph saved to {graph_file}")
        
        # 保存向量索引
        if self.vector_index and vector_path:
            vector_dir = Path(vector_path)
            vector_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"🕸️ Saving vector index to {vector_dir}")
            self.vector_index.save(str(vector_dir / "vectors"))
            logger.debug(f"🕸️ Vector index saved successfully")
        else:
            logger.warning(f"🕸️ No vector index to save: vector_index={self.vector_index}, vector_path={vector_path}")
    
    @classmethod
    def load(cls, graph_path: str, vector_path: str = None) -> 'TaskGraph':
        """Load TaskGraph from storage"""
        from pathlib import Path
        from graph_rag.node_types import node_from_dict
        from graph_rag.edge_types import edge_from_dict
        
        graph_dir = Path(graph_path)
        if graph_dir.is_dir():
            graph_file = graph_dir / "knowledge_graph.json"
            if graph_file.exists():
                # Load graph data from JSON file
                with open(graph_file, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)

                # Create new TaskGraph instance with storage
                storage = JSONStorage()
                task_graph = cls(
                    storage=storage,
                    website_type=graph_data.get("website_type", ""),
                    website_description=graph_data.get("website_description", "")
                )

                # Convert nodes from dict format to list format (handle both old and new formats)
                nodes_data = graph_data.get("nodes", {})
                if isinstance(nodes_data, dict):
                    logger.info("Detected old dict format for nodes, converting to list format")
                    nodes_list = list(nodes_data.values())
                else:
                    nodes_list = nodes_data

                # Load nodes
                for node_data in nodes_list:
                    try:
                        node = node_from_dict(node_data, is_web_graph=True)  # TaskGraph is for web tasks
                        task_graph.add_node(node)
                    except Exception as e:
                        logger.warning(f"Failed to load node {node_data.get('node_id', 'unknown')}: {e}")

                # Convert edges from dict format to list format (handle both old and new formats)
                edges_data = graph_data.get("edges", {})
                if isinstance(edges_data, dict):
                    logger.info("Detected old dict format for edges, converting to list format")
                    edges_list = list(edges_data.values())
                else:
                    edges_list = edges_data

                # Load edges
                for edge_data in edges_list:
                    try:
                        # Convert old field names to new ones
                        if "source_node" in edge_data and "source_node_id" not in edge_data:
                            edge_data["source_node_id"] = edge_data.pop("source_node")
                        if "target_node" in edge_data and "target_node_id" not in edge_data:
                            edge_data["target_node_id"] = edge_data.pop("target_node")
                        edge = edge_from_dict(edge_data)
                        task_graph.add_edge(edge)
                    except Exception as e:
                        logger.warning(f"Failed to load edge {edge_data.get('edge_id', 'unknown')}: {e}")

                # Load vector index if available
                if vector_path:
                    try:
                        from graph_rag.embeddings import NodeVectorIndex
                        from graph_rag.embeddings import EmbeddingManager
                        embedding_manager = EmbeddingManager()
                        task_graph.vector_index = NodeVectorIndex(embedding_manager)
                        task_graph.vector_index.load(vector_path)
                        logger.info(f"🕸️ Vector index loaded from {vector_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load vector index: {e}")

                logger.info(f"Loaded TaskGraph from {graph_file}")
                logger.info(f"Nodes: {len(task_graph.nodes)}, Edges: {len(task_graph.edges)}")
                return task_graph
            else:
                raise FileNotFoundError(f"Graph file not found: {graph_file}")
        else:
            raise FileNotFoundError(f"Graph directory not found: {graph_dir}")
