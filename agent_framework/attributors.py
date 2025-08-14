"""
Failure attribution system for mapping failures to graph nodes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import numpy as np
from collections import defaultdict, Counter
import re

from loguru import logger

from graph_rag.graph_builder import DocumentGraph
from graph_rag.node_types import Node, NodeType
from graph_rag.edge_types import Edge, EdgeType
from task_craft.task_generator import TaskInstance
from .executors import ExecutionResult
from .retrievers import RetrievalResult
from .evaluators import EvaluationResult


@dataclass
class AttributionResult:
    """Result of failure attribution analysis"""
    
    task_id: str
    failure_type: str
    attribution_method: str
    
    # Attributed nodes with confidence scores
    attributed_nodes: List[Tuple[str, float]] = field(default_factory=list)  # (node_id, confidence)
    attributed_edges: List[Tuple[str, float]] = field(default_factory=list)  # (edge_id, confidence)
    
    # Attribution analysis
    primary_cause: Optional[str] = None  # Most likely cause node
    contributing_factors: List[str] = field(default_factory=list)
    
    # Confidence and metadata
    overall_confidence: float = 0.0
    attribution_time: float = 0.0
    
    # Detailed analysis
    analysis_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "failure_type": self.failure_type,
            "attribution_method": self.attribution_method,
            "attributed_nodes": self.attributed_nodes,
            "attributed_edges": self.attributed_edges,
            "primary_cause": self.primary_cause,
            "contributing_factors": self.contributing_factors,
            "overall_confidence": self.overall_confidence,
            "attribution_time": self.attribution_time,
            "analysis_details": self.analysis_details
        }


class FailureAttributor(ABC):
    """Abstract base class for failure attribution"""
    
    @abstractmethod
    def attribute_failure(
        self, 
        task: TaskInstance,
        result: ExecutionResult,
        retrieval_result: RetrievalResult,
        evaluation_result: EvaluationResult,
        graph: DocumentGraph
    ) -> AttributionResult:
        """Attribute failure to specific graph nodes"""
        pass


class GraphCentralityAttributor(FailureAttributor):
    """Attribution based on graph centrality and importance"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_attributed_nodes = self.config.get('max_attributed_nodes', 5)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)
    
    def attribute_failure(
        self, 
        task: TaskInstance,
        result: ExecutionResult,
        retrieval_result: RetrievalResult,
        evaluation_result: EvaluationResult,
        graph: DocumentGraph
    ) -> AttributionResult:
        """Attribute failure using graph centrality analysis"""
        
        attribution_result = AttributionResult(
            task_id=task.task_id,
            failure_type=self._classify_failure_type(result, evaluation_result),
            attribution_method="graph_centrality"
        )
        
        # Get nodes involved in the task
        involved_nodes = self._get_involved_nodes(task, retrieval_result)
        
        if not involved_nodes:
            return attribution_result
        
        # Calculate centrality scores
        centrality_scores = self._calculate_centrality_scores(involved_nodes, graph)
        
        # Calculate failure attribution scores
        attribution_scores = self._calculate_attribution_scores(
            involved_nodes, centrality_scores, task, result, evaluation_result
        )
        
        # Select top attributed nodes
        sorted_attributions = sorted(
            attribution_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for node_id, score in sorted_attributions[:self.max_attributed_nodes]:
            if score >= self.confidence_threshold:
                attribution_result.attributed_nodes.append((node_id, score))
        
        # Determine primary cause
        if attribution_result.attributed_nodes:
            attribution_result.primary_cause = attribution_result.attributed_nodes[0][0]
            attribution_result.contributing_factors = [
                node_id for node_id, _ in attribution_result.attributed_nodes[1:]
            ]
        
        # Calculate overall confidence
        if attribution_result.attributed_nodes:
            attribution_result.overall_confidence = np.mean([
                score for _, score in attribution_result.attributed_nodes
            ])
        
        # Add analysis details
        attribution_result.analysis_details = {
            "involved_nodes_count": len(involved_nodes),
            "centrality_scores": centrality_scores,
            "attribution_scores": dict(sorted_attributions[:10]),  # Top 10
            "failure_indicators": self._get_failure_indicators(result, evaluation_result)
        }
        
        return attribution_result
    
    def _classify_failure_type(self, result: ExecutionResult, eval_result: EvaluationResult) -> str:
        """Classify the type of failure"""
        
        if not result.success:
            if result.error_type:
                return f"execution_error_{result.error_type.lower()}"
            else:
                return "execution_failure"
        
        # Classification based on evaluation scores
        if eval_result.citation_f1 < 0.3:
            return "citation_failure"
        elif eval_result.f1_score < 0.3:
            return "answer_quality_failure"
        elif eval_result.reasoning_coherence < 0.5:
            return "reasoning_failure"
        elif eval_result.safety_compliance < 0.8:
            return "safety_failure"
        elif eval_result.retrieval_accuracy < 0.5:
            return "retrieval_failure"
        else:
            return "general_performance_failure"
    
    def _get_involved_nodes(self, task: TaskInstance, retrieval_result: RetrievalResult) -> List[Node]:
        """Get nodes that were involved in task execution"""
        
        involved_nodes = []
        
        # Add retrieved nodes
        if retrieval_result and retrieval_result.nodes:
            involved_nodes.extend(retrieval_result.nodes)
        
        # Add subgraph nodes if available
        if task.subgraph_nodes:
            # Would need to fetch these from graph
            pass
        
        return involved_nodes
    
    def _calculate_centrality_scores(self, nodes: List[Node], graph: DocumentGraph) -> Dict[str, float]:
        """Calculate centrality scores for nodes"""
        
        centrality_scores = {}
        
        for node in nodes:
            # Degree centrality (number of connections)
            neighbors = graph.get_neighbors(node.node_id)
            degree_centrality = len(neighbors)
            
            # Content importance (length, type)
            content_importance = self._calculate_content_importance(node)
            
            # Combined centrality score
            centrality_scores[node.node_id] = degree_centrality * 0.7 + content_importance * 0.3
        
        # Normalize scores
        if centrality_scores:
            max_score = max(centrality_scores.values())
            if max_score > 0:
                centrality_scores = {
                    nid: score / max_score 
                    for nid, score in centrality_scores.items()
                }
        
        return centrality_scores
    
    def _calculate_content_importance(self, node: Node) -> float:
        """Calculate importance of node content"""
        
        importance = 0.0
        
        # Node type importance
        type_weights = {
            NodeType.TABLE: 1.0,
            NodeType.HEADING: 0.8,
            NodeType.ENTITY: 0.9,
            NodeType.PARAGRAPH: 0.6,
            NodeType.FIGURE: 0.7,
            NodeType.CHUNK: 0.5
        }
        importance += type_weights.get(node.node_type, 0.5)
        
        # Content length importance
        content_length = len(node.content) if node.content else 0
        length_score = min(1.0, content_length / 500)  # Normalize to 500 chars
        importance += length_score * 0.3
        
        # Keyword importance (simplified)
        important_keywords = ['result', 'conclusion', 'finding', 'data', 'analysis', 'table', 'figure']
        keyword_score = sum(
            1 for keyword in important_keywords 
            if keyword in node.content.lower()
        ) / len(important_keywords)
        importance += keyword_score * 0.2
        
        return importance
    
    def _calculate_attribution_scores(
        self, 
        nodes: List[Node], 
        centrality_scores: Dict[str, float],
        task: TaskInstance,
        result: ExecutionResult,
        eval_result: EvaluationResult
    ) -> Dict[str, float]:
        """Calculate attribution scores for each node"""
        
        attribution_scores = {}
        
        for node in nodes:
            score = 0.0
            
            # Base score from centrality
            score += centrality_scores.get(node.node_id, 0.0) * 0.3
            
            # Citation mismatch penalty
            if task.requires_citations:
                if node.node_id in task.gold_nodes and node.node_id not in result.citations:
                    score += 0.4  # Missing required citation
                elif node.node_id not in task.gold_nodes and node.node_id in result.citations:
                    score += 0.2  # Unnecessary citation
            
            # Content relevance to failure
            relevance_score = self._calculate_content_relevance_to_failure(
                node, task, result, eval_result
            )
            score += relevance_score * 0.3
            
            attribution_scores[node.node_id] = score
        
        return attribution_scores
    
    def _calculate_content_relevance_to_failure(
        self, 
        node: Node, 
        task: TaskInstance, 
        result: ExecutionResult,
        eval_result: EvaluationResult
    ) -> float:
        """Calculate how relevant node content is to the failure"""
        
        relevance = 0.0
        
        if not node.content or not result.answer:
            return 0.0
        
        # Ensure both are strings
        node_content = str(node.content) if node.content is not None else ""
        answer_content = str(result.answer) if result.answer is not None else ""
        
        # Check if node content was likely used in answer
        node_words = set(node_content.lower().split())
        answer_words = set(answer_content.lower().split())
        
        word_overlap = len(node_words & answer_words)
        total_words = len(node_words | answer_words)
        
        if total_words > 0:
            content_usage = word_overlap / total_words
            relevance += content_usage
        
        # Check for problematic content patterns
        if eval_result.safety_compliance < 0.8:
            # Look for potentially unsafe content in node
            unsafe_patterns = ['violence', 'harm', 'illegal', 'dangerous']
            for pattern in unsafe_patterns:
                if pattern in node.content.lower():
                    relevance += 0.3
                    break
        
        return min(1.0, relevance)
    
    def _get_failure_indicators(self, result: ExecutionResult, eval_result: EvaluationResult) -> List[str]:
        """Get indicators of what went wrong"""
        
        indicators = []
        
        if not result.success:
            indicators.append(f"execution_error: {result.error_type}")
        
        if eval_result.citation_f1 < 0.5:
            indicators.append("poor_citation_quality")
        
        if eval_result.f1_score < 0.5:
            indicators.append("poor_answer_quality")
        
        if eval_result.reasoning_coherence < 0.5:
            indicators.append("poor_reasoning")
        
        if eval_result.safety_compliance < 0.8:
            indicators.append("safety_violation")
        
        if result.execution_time > 30:
            indicators.append("slow_execution")
        
        return indicators


class ErrorPropagationAttributor(FailureAttributor):
    """Attribution based on error propagation through the graph"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.propagation_depth = self.config.get('propagation_depth', 2)
        self.decay_factor = self.config.get('decay_factor', 0.7)
    
    def attribute_failure(
        self, 
        task: TaskInstance,
        result: ExecutionResult,
        retrieval_result: RetrievalResult,
        evaluation_result: EvaluationResult,
        graph: DocumentGraph
    ) -> AttributionResult:
        """Attribute failure using error propagation analysis"""
        
        attribution_result = AttributionResult(
            task_id=task.task_id,
            failure_type=self._classify_failure_type(result, evaluation_result),
            attribution_method="error_propagation"
        )
        
        # Start with nodes that have direct evidence of errors
        error_sources = self._identify_initial_error_sources(task, result, evaluation_result, retrieval_result)
        
        if not error_sources:
            return attribution_result
        
        # Propagate error attribution through graph
        attribution_scores = self._propagate_attribution_scores(error_sources, graph)
        
        # Select top attributed nodes
        sorted_attributions = sorted(
            attribution_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for node_id, score in sorted_attributions[:self.config.get('max_attributed_nodes', 5)]:
            if score >= self.config.get('confidence_threshold', 0.2):
                attribution_result.attributed_nodes.append((node_id, score))
        
        # Set primary cause and contributing factors
        if attribution_result.attributed_nodes:
            attribution_result.primary_cause = attribution_result.attributed_nodes[0][0]
            attribution_result.contributing_factors = [
                node_id for node_id, _ in attribution_result.attributed_nodes[1:]
            ]
            attribution_result.overall_confidence = np.mean([
                score for _, score in attribution_result.attributed_nodes
            ])
        
        attribution_result.analysis_details = {
            "initial_error_sources": error_sources,
            "propagation_depth": self.propagation_depth,
            "final_attribution_scores": dict(sorted_attributions[:10])
        }
        
        return attribution_result
    
    def _classify_failure_type(self, result: ExecutionResult, eval_result: EvaluationResult) -> str:
        """Classify failure type"""
        # Same implementation as GraphCentralityAttributor
        if not result.success:
            return f"execution_error_{result.error_type.lower() if result.error_type else 'unknown'}"
        
        if eval_result.citation_f1 < 0.3:
            return "citation_failure"
        elif eval_result.f1_score < 0.3:
            return "answer_quality_failure"
        elif eval_result.reasoning_coherence < 0.5:
            return "reasoning_failure"
        elif eval_result.safety_compliance < 0.8:
            return "safety_failure"
        else:
            return "general_performance_failure"
    
    def _identify_initial_error_sources(
        self, 
        task: TaskInstance,
        result: ExecutionResult,
        eval_result: EvaluationResult,
        retrieval_result: RetrievalResult
    ) -> Dict[str, float]:
        """Identify initial sources of error with confidence scores"""
        
        error_sources = {}
        
        # Citation errors
        if task.requires_citations and eval_result.citation_f1 < 0.5:
            # Missing citations
            for gold_node in task.gold_nodes:
                if gold_node not in result.citations:
                    error_sources[gold_node] = 0.8
            
            # Wrong citations
            for cited_node in result.citations:
                if cited_node not in task.gold_nodes:
                    error_sources[cited_node] = 0.6
        
        # Content quality errors
        if eval_result.f1_score < 0.5:
            # Nodes that were retrieved but may have poor content
            if retrieval_result:
                for node in retrieval_result.nodes:
                    if node.node_id in task.subgraph_nodes:
                        # This node was expected to help but didn't
                        error_sources[node.node_id] = 0.5
        
        # Safety errors
        if eval_result.safety_compliance < 0.8:
            # Any retrieved nodes might contain problematic content
            if retrieval_result:
                for node in retrieval_result.nodes:
                    if self._contains_problematic_content(node.content):
                        error_sources[node.node_id] = 0.7
        
        return error_sources
    
    def _propagate_attribution_scores(self, initial_sources: Dict[str, float], graph: DocumentGraph) -> Dict[str, float]:
        """Propagate attribution scores through the graph"""
        
        current_scores = initial_sources.copy()
        all_scores = current_scores.copy()
        
        for depth in range(self.propagation_depth):
            next_scores = {}
            
            for node_id, score in current_scores.items():
                # Get neighbors of this node
                neighbors = graph.get_neighbors(node_id)
                
                for neighbor_node, edge in neighbors:
                    # Calculate propagated score
                    edge_weight = self._get_edge_propagation_weight(edge)
                    propagated_score = score * self.decay_factor * edge_weight
                    
                    # Accumulate score for neighbor
                    if neighbor_node.node_id not in next_scores:
                        next_scores[neighbor_node.node_id] = 0.0
                    next_scores[neighbor_node.node_id] += propagated_score
            
            # Update scores
            for node_id, score in next_scores.items():
                if node_id not in all_scores:
                    all_scores[node_id] = score
                else:
                    all_scores[node_id] = max(all_scores[node_id], score)
            
            current_scores = next_scores
            
            # Stop if no more propagation
            if not current_scores:
                break
        
        return all_scores
    
    def _get_edge_propagation_weight(self, edge: Edge) -> float:
        """Get propagation weight for different edge types"""
        
        edge_weights = {
            EdgeType.REFERS_TO: 1.0,      # Strong propagation for references
            EdgeType.CONTAINS: 0.8,       # Strong for containment
            EdgeType.SEMANTIC_SIM: 0.7,   # Medium for semantic similarity
            EdgeType.SEQUENCE: 0.6,       # Medium for sequence
            EdgeType.ENTITY_RELATION: 0.9, # Strong for entity relations
            EdgeType.TABLE_CONTEXT: 0.8,  # Strong for table context
            EdgeType.FIGURE_CONTEXT: 0.8  # Strong for figure context
        }
        
        return edge_weights.get(edge.edge_type, 0.5)
    
    def _contains_problematic_content(self, content: str) -> bool:
        """Check if content contains problematic patterns"""
        
        if not content:
            return False
        
        problematic_patterns = [
            r'\b(?:violence|harm|kill|murder)\b',
            r'\b(?:illegal|criminal|fraud)\b',
            r'\b(?:discriminat|racist|sexist|bias)\b'
        ]
        
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in problematic_patterns)


class AttentionBasedAttributor(FailureAttributor):
    """Attribution based on simulated attention mechanisms"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def attribute_failure(
        self, 
        task: TaskInstance,
        result: ExecutionResult,
        retrieval_result: RetrievalResult,
        evaluation_result: EvaluationResult,
        graph: DocumentGraph
    ) -> AttributionResult:
        """Attribute failure using attention-like analysis"""
        
        attribution_result = AttributionResult(
            task_id=task.task_id,
            failure_type=self._classify_failure_type(result, evaluation_result),
            attribution_method="attention_based"
        )
        
        if not retrieval_result or not retrieval_result.nodes:
            return attribution_result
        
        # Calculate attention scores for each node
        attention_scores = self._calculate_attention_scores(
            task, result, retrieval_result.nodes
        )
        
        # Calculate attribution based on attention and failure
        attribution_scores = self._calculate_attention_attribution(
            attention_scores, task, result, evaluation_result
        )
        
        # Select attributed nodes
        sorted_attributions = sorted(
            attribution_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for node_id, score in sorted_attributions[:self.config.get('max_attributed_nodes', 5)]:
            if score >= self.config.get('confidence_threshold', 0.3):
                attribution_result.attributed_nodes.append((node_id, score))
        
        if attribution_result.attributed_nodes:
            attribution_result.primary_cause = attribution_result.attributed_nodes[0][0]
            attribution_result.overall_confidence = np.mean([
                score for _, score in attribution_result.attributed_nodes
            ])
        
        attribution_result.analysis_details = {
            "attention_scores": attention_scores,
            "attribution_method": "attention_based"
        }
        
        return attribution_result
    
    def _classify_failure_type(self, result: ExecutionResult, eval_result: EvaluationResult) -> str:
        """Classify failure type"""
        # Same as other attributors
        if not result.success:
            return f"execution_error_{result.error_type.lower() if result.error_type else 'unknown'}"
        
        if eval_result.citation_f1 < 0.3:
            return "citation_failure"
        elif eval_result.f1_score < 0.3:
            return "answer_quality_failure"
        else:
            return "general_failure"
    
    def _calculate_attention_scores(self, task: TaskInstance, result: ExecutionResult, nodes: List[Node]) -> Dict[str, float]:
        """Calculate attention scores based on content overlap"""
        
        attention_scores = {}
        
        # Get query representation (task prompt)
        query_words = set(task.prompt.lower().split())
        
        # Get answer representation
        answer_content = str(result.answer) if result.answer is not None else ""
        answer_words = set(answer_content.lower().split())
        
        for node in nodes:
            node_words = set(node.content.lower().split())
            
            # Query-node attention
            query_attention = len(query_words & node_words) / len(query_words) if query_words else 0.0
            
            # Answer-node attention (how much the node influenced the answer)
            answer_attention = len(answer_words & node_words) / len(node_words) if node_words else 0.0
            
            # Combined attention score
            attention_scores[node.node_id] = (query_attention + answer_attention) / 2
        
        return attention_scores
    
    def _calculate_attention_attribution(
        self, 
        attention_scores: Dict[str, float],
        task: TaskInstance,
        result: ExecutionResult,
        eval_result: EvaluationResult
    ) -> Dict[str, float]:
        """Calculate attribution scores based on attention and failure"""
        
        attribution_scores = {}
        
        for node_id, attention_score in attention_scores.items():
            # High attention nodes that led to failure get high attribution
            attribution_score = attention_score
            
            # Boost for citation errors
            if task.requires_citations:
                if node_id in task.gold_nodes and node_id not in result.citations:
                    attribution_score += 0.3  # Should have been cited but wasn't
                elif node_id not in task.gold_nodes and node_id in result.citations:
                    attribution_score += 0.2  # Incorrectly cited
            
            # Boost for content quality issues
            if eval_result.f1_score < 0.5:
                attribution_score += 0.1  # General content issue
            
            attribution_scores[node_id] = attribution_score
        
        return attribution_scores


class HybridAttributor(FailureAttributor):
    """Hybrid attributor combining multiple attribution methods"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize sub-attributors
        self.attributors = [
            GraphCentralityAttributor(config),
            ErrorPropagationAttributor(config),
            AttentionBasedAttributor(config)
        ]
        
        # Weights for combining attributors
        self.attributor_weights = self.config.get('attributor_weights', [0.4, 0.4, 0.2])
    
    def attribute_failure(
        self, 
        task: TaskInstance,
        result: ExecutionResult,
        retrieval_result: RetrievalResult,
        evaluation_result: EvaluationResult,
        graph: DocumentGraph
    ) -> AttributionResult:
        """Combine multiple attribution methods"""
        
        # Get attributions from each method
        sub_attributions = []
        for attributor in self.attributors:
            try:
                attribution = attributor.attribute_failure(
                    task, result, retrieval_result, evaluation_result, graph
                )
                sub_attributions.append(attribution)
            except Exception as e:
                logger.warning(f"Attribution method {attributor.__class__.__name__} failed: {e}")
                sub_attributions.append(None)
        
        # Combine results
        combined_result = AttributionResult(
            task_id=task.task_id,
            failure_type=sub_attributions[0].failure_type if sub_attributions[0] else "unknown",
            attribution_method="hybrid"
        )
        
        # Combine attribution scores
        combined_scores = defaultdict(float)
        combined_counts = defaultdict(int)
        
        for i, attribution in enumerate(sub_attributions):
            if attribution and attribution.attributed_nodes:
                weight = self.attributor_weights[i] if i < len(self.attributor_weights) else 0.1
                
                for node_id, score in attribution.attributed_nodes:
                    combined_scores[node_id] += score * weight
                    combined_counts[node_id] += 1
        
        # Average scores and select top nodes
        final_scores = {}
        for node_id, total_score in combined_scores.items():
            final_scores[node_id] = total_score / combined_counts[node_id]
        
        # Sort and select
        sorted_attributions = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        for node_id, score in sorted_attributions[:self.config.get('max_attributed_nodes', 5)]:
            if score >= self.config.get('confidence_threshold', 0.2):
                combined_result.attributed_nodes.append((node_id, score))
        
        if combined_result.attributed_nodes:
            combined_result.primary_cause = combined_result.attributed_nodes[0][0]
            combined_result.overall_confidence = np.mean([
                score for _, score in combined_result.attributed_nodes
            ])
        
        # Combine analysis details
        combined_result.analysis_details = {
            "sub_attributions": [attr.to_dict() if attr else None for attr in sub_attributions],
            "combined_scores": final_scores,
            "attributor_weights": self.attributor_weights
        }
        
        return combined_result
