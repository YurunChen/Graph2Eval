"""
Retrievers for finding relevant subgraphs and context for tasks
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from loguru import logger

from graph_rag.graph_builder import DocumentGraph
from graph_rag.node_types import Node, NodeType
from graph_rag.edge_types import Edge, EdgeType
from task_craft.task_generator import TaskInstance
from config_manager import get_config


@dataclass
class RetrievalConfig:
    """Configuration for retrieval"""
    max_nodes: int = 10
    max_hops: int = 3
    similarity_threshold: float = 0.7
    include_neighbors: bool = True
    expand_with_context: bool = True
    prefer_gold_nodes: bool = True
    
    @classmethod
    def from_config(cls):
        """从配置文件创建检索配置"""
        config = get_config()
        retrieval_config = config.agent.get('retrieval', {})
        
        return cls(
            max_nodes=retrieval_config.get('max_nodes', 10),
            max_hops=retrieval_config.get('max_hops', 3),
            similarity_threshold=retrieval_config.get('similarity_threshold', 0.7),
            include_neighbors=retrieval_config.get('include_neighbors', True),
            expand_with_context=retrieval_config.get('expand_with_context', True),
            prefer_gold_nodes=retrieval_config.get('prefer_gold_nodes', True)
        )


@dataclass
class RetrievalResult:
    """Result of retrieval operation"""
    nodes: List[Node]
    edges: List[Edge]
    scores: Dict[str, float]  # node_id -> relevance score
    retrieval_method: str
    total_nodes_considered: int
    
    def get_context_text(self, max_length: int = 2000) -> str:
        """Get concatenated context text from retrieved nodes"""
        context_parts = []
        current_length = 0
        
        # Sort nodes by score (highest first)
        sorted_nodes = sorted(
            self.nodes, 
            key=lambda n: self.scores.get(n.node_id, 0.0), 
            reverse=True
        )
        
        for node in sorted_nodes:
            node_text = f"[{node.node_type.value.upper()}] {node.content}"
            
            if current_length + len(node_text) > max_length:
                # Add truncated version if possible
                remaining = max_length - current_length
                if remaining > 100:  # Only add if meaningful amount of space left
                    context_parts.append(node_text[:remaining] + "...")
                break
            
            context_parts.append(node_text)
            current_length += len(node_text) + 2  # +2 for separators
        
        return "\n\n".join(context_parts)


class Retriever(ABC):
    """Abstract base class for retrievers"""
    
    @abstractmethod
    def retrieve(self, task: TaskInstance, graph: DocumentGraph) -> RetrievalResult:
        """Retrieve relevant context for a task"""
        pass


class SubgraphRetriever(Retriever):
    """Retriever that uses subgraph expansion from task nodes"""
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
    
    def retrieve(self, task: TaskInstance, graph: DocumentGraph) -> RetrievalResult:
        """Retrieve subgraph around task nodes"""
        logger.debug(f"Retrieving subgraph for task {task.task_id}")
        
        # Start with task's subgraph nodes if available
        if task.subgraph_nodes:
            start_node_ids = task.subgraph_nodes
        elif task.gold_nodes:
            start_node_ids = task.gold_nodes
        else:
            # Fall back to semantic search
            return self._semantic_fallback_retrieval(task, graph)
        
        # Get starting nodes
        start_nodes = [graph.get_node(node_id) for node_id in start_node_ids]
        start_nodes = [node for node in start_nodes if node is not None]
        
        if not start_nodes:
            return self._semantic_fallback_retrieval(task, graph)
        
        # Expand subgraph
        expanded_nodes, expanded_edges = graph.get_subgraph(
            start_node_ids, max_hops=self.config.max_hops
        )
        
        # Calculate relevance scores
        scores = self._calculate_scores(expanded_nodes, task, graph)
        
        # Filter by relevance and size
        filtered_nodes, filtered_edges = self._filter_by_relevance(
            expanded_nodes, expanded_edges, scores
        )
        
        return RetrievalResult(
            nodes=filtered_nodes,
            edges=filtered_edges,
            scores=scores,
            retrieval_method="subgraph_expansion",
            total_nodes_considered=len(expanded_nodes)
        )
    
    def _semantic_fallback_retrieval(self, task: TaskInstance, graph: DocumentGraph) -> RetrievalResult:
        """Fallback retrieval using semantic search"""
        logger.debug("Using semantic fallback retrieval")
        
        # Use task prompt for semantic search
        similar_nodes = graph.find_similar_nodes(
            task.prompt, 
            k=self.config.max_nodes
        )
        
        if not similar_nodes:
            return RetrievalResult([], [], {}, "semantic_fallback", 0)
        
        nodes = [node for node, score in similar_nodes]
        scores = {node.node_id: score for node, score in similar_nodes}
        
        # Get edges between these nodes
        node_ids = [node.node_id for node in nodes]
        _, edges = graph.get_subgraph(node_ids, max_hops=1)
        
        return RetrievalResult(
            nodes=nodes,
            edges=edges,
            scores=scores,
            retrieval_method="semantic_fallback",
            total_nodes_considered=len(similar_nodes)
        )
    
    def _calculate_scores(self, nodes: List[Node], task: TaskInstance, graph: DocumentGraph) -> Dict[str, float]:
        """Calculate relevance scores for nodes"""
        scores = {}
        
        for node in nodes:
            score = 0.0
            
            # Higher score for gold nodes
            if self.config.prefer_gold_nodes and node.node_id in task.gold_nodes:
                score += 1.0
            
            # Higher score for nodes mentioned in subgraph
            if node.node_id in task.subgraph_nodes:
                score += 0.8
            
            # Score based on node type relevance
            score += self._get_node_type_score(node, task)
            
            # Score based on content relevance (simplified)
            score += self._get_content_relevance_score(node, task)
            
            scores[node.node_id] = score
        
        return scores
    
    def _get_node_type_score(self, node: Node, task: TaskInstance) -> float:
        """Get score based on node type relevance to task"""
        type_scores = {
            "extraction": {"paragraph": 1.0, "heading": 0.7, "table": 0.8, "figure": 0.5},
            "table_qa": {"table": 1.0, "paragraph": 0.5, "heading": 0.3, "figure": 0.2},
            "figure_interpretation": {"figure": 1.0, "paragraph": 0.6, "table": 0.3, "heading": 0.4},
            "reasoning": {"paragraph": 1.0, "heading": 0.6, "table": 0.7, "figure": 0.4},
            "comparison": {"paragraph": 1.0, "table": 0.8, "heading": 0.5, "figure": 0.3},
            "summarization": {"paragraph": 1.0, "heading": 0.8, "table": 0.6, "figure": 0.4}
        }
        
        task_type_scores = type_scores.get(task.task_type.value, {})
        return task_type_scores.get(node.node_type.value, 0.5)
    
    def _get_content_relevance_score(self, node: Node, task: TaskInstance) -> float:
        """Get score based on content relevance (simplified)"""
        # Extract key terms from task prompt
        prompt_words = set(task.prompt.lower().split())
        content_words = set(node.content.lower().split())
        
        # Calculate word overlap
        overlap = len(prompt_words & content_words)
        total = len(prompt_words | content_words)
        
        return overlap / total if total > 0 else 0.0
    
    def _filter_by_relevance(
        self, 
        nodes: List[Node], 
        edges: List[Edge], 
        scores: Dict[str, float]
    ) -> Tuple[List[Node], List[Edge]]:
        """Filter nodes and edges by relevance scores"""
        
        # Sort nodes by score
        sorted_nodes = sorted(
            nodes, 
            key=lambda n: scores.get(n.node_id, 0.0), 
            reverse=True
        )
        
        # Take top nodes up to max_nodes
        filtered_nodes = sorted_nodes[:self.config.max_nodes]
        filtered_node_ids = {node.node_id for node in filtered_nodes}
        
        # Keep edges that connect filtered nodes
        filtered_edges = [
            edge for edge in edges
            if (edge.source_node_id in filtered_node_ids and 
                edge.target_node_id in filtered_node_ids)
        ]
        
        return filtered_nodes, filtered_edges


class HybridRetriever(Retriever):
    """Hybrid retriever combining multiple strategies"""
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.subgraph_retriever = SubgraphRetriever(config)
    
    def retrieve(self, task: TaskInstance, graph: DocumentGraph) -> RetrievalResult:
        """Retrieve using hybrid approach"""
        logger.debug(f"Hybrid retrieval for task {task.task_id}")
        
        # Strategy 1: Subgraph expansion
        subgraph_result = self.subgraph_retriever.retrieve(task, graph)
        
        # Strategy 2: Semantic search
        semantic_result = self._semantic_search(task, graph)
        
        # Strategy 3: Type-specific search
        type_result = self._type_specific_search(task, graph)
        
        # Combine results
        combined_result = self._combine_results([subgraph_result, semantic_result, type_result])
        
        return combined_result
    
    def _semantic_search(self, task: TaskInstance, graph: DocumentGraph) -> RetrievalResult:
        """Semantic search based on task prompt"""
        similar_nodes = graph.find_similar_nodes(
            task.prompt,
            k=self.config.max_nodes // 2  # Use half of max for diversity
        )
        
        if not similar_nodes:
            return RetrievalResult([], [], {}, "semantic_search", 0)
        
        nodes = [node for node, score in similar_nodes]
        scores = {node.node_id: score for node, score in similar_nodes}
        
        # Get connecting edges
        node_ids = [node.node_id for node in nodes]
        _, edges = graph.get_subgraph(node_ids, max_hops=1)
        
        return RetrievalResult(
            nodes=nodes,
            edges=edges,
            scores=scores,
            retrieval_method="semantic_search",
            total_nodes_considered=len(similar_nodes)
        )
    
    def _type_specific_search(self, task: TaskInstance, graph: DocumentGraph) -> RetrievalResult:
        """Search based on task type requirements"""
        
        # Determine preferred node types for task
        preferred_types = self._get_preferred_node_types(task.task_type.value)
        
        type_nodes = []
        for node_type_str in preferred_types:
            try:
                node_type = NodeType(node_type_str)
                nodes = graph.storage.find_nodes(node_type=node_type)
                type_nodes.extend(nodes[:5])  # Limit per type
            except ValueError:
                continue
        
        # Score based on type preference
        scores = {}
        for i, node in enumerate(type_nodes):
            # Higher score for preferred types, decay with position
            type_priority = preferred_types.index(node.node_type.value) if node.node_type.value in preferred_types else len(preferred_types)
            position_decay = 1.0 / (i + 1)
            scores[node.node_id] = (1.0 - type_priority / len(preferred_types)) * position_decay
        
        # Get connecting edges
        node_ids = [node.node_id for node in type_nodes]
        _, edges = graph.get_subgraph(node_ids, max_hops=1)
        
        return RetrievalResult(
            nodes=type_nodes,
            edges=edges,
            scores=scores,
            retrieval_method="type_specific",
            total_nodes_considered=len(type_nodes)
        )
    
    def _get_preferred_node_types(self, task_type: str) -> List[str]:
        """Get preferred node types for task type"""
        preferences = {
            "extraction": ["paragraph", "heading", "table"],
            "table_qa": ["table", "paragraph"],
            "figure_interpretation": ["figure", "paragraph"],
            "reasoning": ["paragraph", "heading", "table"],
            "comparison": ["paragraph", "table", "heading"],
            "summarization": ["paragraph", "heading"],
            "cross_reference": ["paragraph", "heading"],
            "aggregation": ["paragraph", "table", "heading"],
            "safety_check": ["paragraph", "heading"]
        }
        
        return preferences.get(task_type, ["paragraph", "heading", "table", "figure"])
    
    def _combine_results(self, results: List[RetrievalResult]) -> RetrievalResult:
        """Combine multiple retrieval results"""
        
        all_nodes = []
        all_edges = []
        combined_scores = {}
        all_methods = []
        total_considered = 0
        
        # Collect all nodes and scores
        seen_nodes = set()
        seen_edges = set()
        
        for result in results:
            all_methods.append(result.retrieval_method)
            total_considered += result.total_nodes_considered
            
            for node in result.nodes:
                if node.node_id not in seen_nodes:
                    all_nodes.append(node)
                    seen_nodes.add(node.node_id)
                
                # Combine scores (take maximum)
                existing_score = combined_scores.get(node.node_id, 0.0)
                new_score = result.scores.get(node.node_id, 0.0)
                combined_scores[node.node_id] = max(existing_score, new_score)
            
            for edge in result.edges:
                edge_key = (edge.source_node_id, edge.target_node_id, edge.edge_type.value)
                if edge_key not in seen_edges:
                    all_edges.append(edge)
                    seen_edges.add(edge_key)
        
        # Sort by combined scores and limit
        sorted_nodes = sorted(
            all_nodes,
            key=lambda n: combined_scores.get(n.node_id, 0.0),
            reverse=True
        )
        
        final_nodes = sorted_nodes[:self.config.max_nodes]
        final_node_ids = {node.node_id for node in final_nodes}
        
        # Keep edges between final nodes
        final_edges = [
            edge for edge in all_edges
            if (edge.source_node_id in final_node_ids and 
                edge.target_node_id in final_node_ids)
        ]
        
        return RetrievalResult(
            nodes=final_nodes,
            edges=final_edges,
            scores=combined_scores,
            retrieval_method="hybrid_" + "_".join(all_methods),
            total_nodes_considered=total_considered
        )


class ContextualRetriever(Retriever):
    """Retriever that considers task context and history"""
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.task_history = []  # Track previous tasks for context
    
    def retrieve(self, task: TaskInstance, graph: DocumentGraph) -> RetrievalResult:
        """Retrieve with contextual awareness"""
        
        # Use hybrid retrieval as base
        hybrid_retriever = HybridRetriever(self.config)
        base_result = hybrid_retriever.retrieve(task, graph)
        
        # Enhance with contextual information
        enhanced_result = self._enhance_with_context(base_result, task, graph)
        
        # Update task history
        self.task_history.append(task)
        if len(self.task_history) > 10:  # Keep limited history
            self.task_history.pop(0)
        
        return enhanced_result
    
    def _enhance_with_context(
        self, 
        base_result: RetrievalResult, 
        task: TaskInstance, 
        graph: DocumentGraph
    ) -> RetrievalResult:
        """Enhance retrieval with contextual information"""
        
        enhanced_nodes = base_result.nodes.copy()
        enhanced_edges = base_result.edges.copy()
        enhanced_scores = base_result.scores.copy()
        
        # Add context from similar previous tasks
        similar_tasks = self._find_similar_previous_tasks(task)
        
        for similar_task in similar_tasks:
            # Add nodes that were successful in similar tasks
            for node_id in similar_task.gold_nodes:
                if node_id not in enhanced_scores:
                    node = graph.get_node(node_id)
                    if node:
                        enhanced_nodes.append(node)
                        enhanced_scores[node_id] = 0.6  # Moderate score for historical relevance
        
        # Add domain-specific context nodes
        domain_nodes = self._find_domain_context_nodes(task, graph)
        for node in domain_nodes:
            if node.node_id not in enhanced_scores:
                enhanced_nodes.append(node)
                enhanced_scores[node.node_id] = 0.5
        
        # Sort and limit final results
        final_nodes = sorted(
            enhanced_nodes,
            key=lambda n: enhanced_scores.get(n.node_id, 0.0),
            reverse=True
        )[:self.config.max_nodes]
        
        final_node_ids = {node.node_id for node in final_nodes}
        final_edges = [
            edge for edge in enhanced_edges
            if (edge.source_node_id in final_node_ids and 
                edge.target_node_id in final_node_ids)
        ]
        
        return RetrievalResult(
            nodes=final_nodes,
            edges=final_edges,
            scores=enhanced_scores,
            retrieval_method="contextual_" + base_result.retrieval_method,
            total_nodes_considered=base_result.total_nodes_considered + len(domain_nodes)
        )
    
    def _find_similar_previous_tasks(self, current_task: TaskInstance) -> List[TaskInstance]:
        """Find similar tasks from history"""
        similar_tasks = []
        
        for prev_task in self.task_history:
            # Simple similarity based on task type and template
            if (prev_task.task_type == current_task.task_type or 
                prev_task.template_id == current_task.template_id):
                similar_tasks.append(prev_task)
        
        return similar_tasks
    
    def _find_domain_context_nodes(self, task: TaskInstance, graph: DocumentGraph) -> List[Node]:
        """Find nodes that provide domain context"""
        
        # Look for heading nodes that might provide section context
        heading_nodes = graph.storage.find_nodes(node_type=NodeType.HEADING)
        
        # Simple relevance scoring based on content overlap
        relevant_headings = []
        task_words = set(task.prompt.lower().split())
        
        for heading in heading_nodes[:5]:  # Limit search
            heading_words = set(heading.content.lower().split())
            overlap = len(task_words & heading_words)
            
            if overlap > 0:
                relevant_headings.append(heading)
        
        return relevant_headings
