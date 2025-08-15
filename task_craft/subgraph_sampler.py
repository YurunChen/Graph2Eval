"""
Subgraph sampling for task generation
"""

import random
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from loguru import logger

from graph_rag.graph_builder import DocumentGraph
from graph_rag.node_types import Node, NodeType
from graph_rag.edge_types import Edge, EdgeType, COMMON_MOTIFS, GraphMotif
from config_manager import get_config
from .task_templates import TaskTemplate


@dataclass
class SamplingConfig:
    """Configuration for subgraph sampling"""
    max_samples_per_template: int = 20
    min_subgraph_size: int = 1
    max_subgraph_size: int = 8
    sampling_strategy: str = "random"  # random, motif_based, centrality_based, semantic_based, context_chain_based, contradiction_based
    avoid_disconnected: bool = True
    ensure_diversity: bool = True
    random_seed: Optional[int] = None
    
    @classmethod
    def from_config(cls):
        """Create sampling configuration from configuration file"""
        config = get_config()
        sampling_config = config.task_craft.get('subgraph_sampling', {})
        global_config = config.global_config
        
        return cls(
            max_samples_per_template=sampling_config.get('max_samples_per_template', 20),
            min_subgraph_size=sampling_config.get('min_subgraph_size', 1),
            max_subgraph_size=sampling_config.get('max_subgraph_size', 8),
            sampling_strategy=sampling_config.get('sampling_strategy', 'hybrid'),
            avoid_disconnected=sampling_config.get('avoid_disconnected', True),
            ensure_diversity=sampling_config.get('ensure_diversity', True),
            random_seed=global_config.get('random_seed', 42)
        )


class SubgraphSampler:
    """Samples subgraphs from document graphs for task generation"""
    
    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()
        
        if self.config.random_seed:
            random.seed(self.config.random_seed)
    
    def sample_subgraphs_for_template(
        self, 
        graph: DocumentGraph, 
        template: TaskTemplate
    ) -> List[Tuple[List[Node], List[Edge]]]:
        """Sample subgraphs that match template requirements"""
        
        logger.debug(f"Sampling subgraphs for template: {template.template_id}")
        
        # Get sampling strategy
        if self.config.sampling_strategy == "motif_based":
            return self._sample_motif_based(graph, template)
        elif self.config.sampling_strategy == "centrality_based":
            return self._sample_centrality_based(graph, template)
        elif self.config.sampling_strategy == "semantic_based":
            return self._sample_semantic_based(graph, template)
        elif self.config.sampling_strategy == "context_chain_based":
            return self._sample_context_chain_based(graph, template)
        elif self.config.sampling_strategy == "contradiction_based":
            return self._sample_contradiction_based(graph, template)
        else:
            return self._sample_random(graph, template)
    
    def _sample_random(
        self, 
        graph: DocumentGraph, 
        template: TaskTemplate
    ) -> List[Tuple[List[Node], List[Edge]]]:
        """Random subgraph sampling"""
        
        subgraphs = []
        attempts = 0
        max_attempts = self.config.max_samples_per_template * 10
        
        # Get candidate nodes that match template requirements
        candidate_nodes = self._get_candidate_nodes(graph, template)
        
        if not candidate_nodes:
            return subgraphs
        
        while len(subgraphs) < self.config.max_samples_per_template and attempts < max_attempts:
            attempts += 1
            
            # Sample initial node
            start_node = random.choice(candidate_nodes)
            
            # Expand to create subgraph
            subgraph_nodes, subgraph_edges = self._expand_subgraph(
                graph, start_node, template
            )
            
            # Check if subgraph meets requirements
            if self._is_valid_subgraph(subgraph_nodes, subgraph_edges, template):
                if not self._is_duplicate_subgraph(subgraph_nodes, subgraphs):
                    subgraphs.append((subgraph_nodes, subgraph_edges))
        
        logger.debug(f"Random sampling generated {len(subgraphs)} subgraphs")
        return subgraphs
    
    def _sample_motif_based(
        self, 
        graph: DocumentGraph, 
        template: TaskTemplate
    ) -> List[Tuple[List[Node], List[Edge]]]:
        """Sample subgraphs based on graph motifs"""
        
        subgraphs = []
        
        # Find applicable motifs for this template
        applicable_motifs = self._find_applicable_motifs(template)
        
        for motif in applicable_motifs:
            motif_subgraphs = self._sample_motif_instances(graph, motif, template)
            subgraphs.extend(motif_subgraphs)
            
            if len(subgraphs) >= self.config.max_samples_per_template:
                break
        
        # Fill remaining slots with random sampling if needed
        if len(subgraphs) < self.config.max_samples_per_template:
            remaining = self.config.max_samples_per_template - len(subgraphs)
            random_subgraphs = self._sample_random(graph, template)
            subgraphs.extend(random_subgraphs[:remaining])
        
        logger.debug(f"Motif-based sampling generated {len(subgraphs)} subgraphs")
        return subgraphs
    
    def _sample_centrality_based(
        self, 
        graph: DocumentGraph, 
        template: TaskTemplate
    ) -> List[Tuple[List[Node], List[Edge]]]:
        """Sample subgraphs starting from high-centrality nodes"""
        
        subgraphs = []
        
        # Get candidate nodes
        candidate_nodes = self._get_candidate_nodes(graph, template)
        
        if not candidate_nodes:
            return subgraphs
        
        # Calculate centrality scores
        centrality_scores = self._calculate_centrality_scores(graph, candidate_nodes)
        
        # Sort nodes by centrality (highest first)
        sorted_nodes = sorted(candidate_nodes, 
                            key=lambda n: centrality_scores.get(n.node_id, 0), 
                            reverse=True)
        
        # Sample subgraphs starting from high-centrality nodes
        for start_node in sorted_nodes[:self.config.max_samples_per_template * 2]:
            subgraph_nodes, subgraph_edges = self._expand_subgraph(
                graph, start_node, template
            )
            
            if self._is_valid_subgraph(subgraph_nodes, subgraph_edges, template):
                if not self._is_duplicate_subgraph(subgraph_nodes, subgraphs):
                    subgraphs.append((subgraph_nodes, subgraph_edges))
                    
                    if len(subgraphs) >= self.config.max_samples_per_template:
                        break
        
        logger.debug(f"Centrality-based sampling generated {len(subgraphs)} subgraphs")
        return subgraphs
    
    def _sample_semantic_based(
        self, 
        graph: DocumentGraph, 
        template: TaskTemplate
    ) -> List[Tuple[List[Node], List[Edge]]]:
        """Sample subgraphs based on semantic similarity"""
        
        subgraphs = []
        
        # Get candidate nodes
        candidate_nodes = self._get_candidate_nodes(graph, template)
        
        if not candidate_nodes:
            return subgraphs
        
        # Find semantically similar clusters
        semantic_clusters = self._find_semantic_clusters(graph, candidate_nodes)
        
        # Sample from each cluster
        for cluster in semantic_clusters:
            if len(subgraphs) >= self.config.max_samples_per_template:
                break
            
            # Use cluster as subgraph if it meets size requirements
            if (self.config.min_subgraph_size <= len(cluster) <= self.config.max_subgraph_size):
                cluster_edges = self._get_edges_between_nodes(graph, cluster)
                
                if self._is_valid_subgraph(cluster, cluster_edges, template):
                    subgraphs.append((cluster, cluster_edges))
            
            # Also expand from cluster centers
            for node in cluster[:2]:  # Take up to 2 nodes from each cluster
                subgraph_nodes, subgraph_edges = self._expand_subgraph(
                    graph, node, template
                )
                
                if self._is_valid_subgraph(subgraph_nodes, subgraph_edges, template):
                    if not self._is_duplicate_subgraph(subgraph_nodes, subgraphs):
                        subgraphs.append((subgraph_nodes, subgraph_edges))
                        
                        if len(subgraphs) >= self.config.max_samples_per_template:
                            break
        
        logger.debug(f"Semantic-based sampling generated {len(subgraphs)} subgraphs")
        return subgraphs
    
    def _get_candidate_nodes(self, graph: DocumentGraph, template: TaskTemplate) -> List[Node]:
        """Get nodes that could serve as starting points for subgraphs"""
        
        candidate_nodes = []
        
        # If template specifies required node types, filter by those
        if template.required_node_types:
            for node_type_str in template.required_node_types:
                try:
                    node_type = NodeType(node_type_str)
                    nodes = graph.storage.find_nodes(node_type=node_type)
                    candidate_nodes.extend(nodes)
                except ValueError:
                    logger.warning(f"Unknown node type: {node_type_str}")
        else:
            # Get all nodes
            for node_type in NodeType:
                nodes = graph.storage.find_nodes(node_type=node_type)
                candidate_nodes.extend(nodes)
        
        return candidate_nodes
    
    def _expand_subgraph(
        self, 
        graph: DocumentGraph, 
        start_node: Node, 
        template: TaskTemplate
    ) -> Tuple[List[Node], List[Edge]]:
        """Expand subgraph from a starting node"""
        
        subgraph_nodes = [start_node]
        subgraph_edges = []
        visited = {start_node.node_id}
        
        # Expand up to max_hops
        current_nodes = [start_node]
        
        for hop in range(template.max_hops):
            if len(subgraph_nodes) >= self.config.max_subgraph_size:
                break
            
            next_nodes = []
            
            for node in current_nodes:
                # Get neighbors
                neighbors = graph.get_neighbors(node.node_id)
                
                for neighbor_node, edge in neighbors:
                    # Check if edge type is allowed
                    if (template.required_edge_types and 
                        edge.edge_type.value not in template.required_edge_types):
                        continue
                    
                    # Add neighbor if not visited and within size limits
                    if (neighbor_node.node_id not in visited and 
                        len(subgraph_nodes) < self.config.max_subgraph_size):
                        
                        subgraph_nodes.append(neighbor_node)
                        subgraph_edges.append(edge)
                        visited.add(neighbor_node.node_id)
                        next_nodes.append(neighbor_node)
            
            current_nodes = next_nodes
            
            if not current_nodes:  # No more nodes to expand
                break
        
        return subgraph_nodes, subgraph_edges
    
    def _is_valid_subgraph(
        self, 
        nodes: List[Node], 
        edges: List[Edge], 
        template: TaskTemplate
    ) -> bool:
        """Check if subgraph meets template requirements"""
        
        # Check size constraints
        if not (template.min_nodes <= len(nodes) <= template.max_nodes):
            return False
        
        if not (self.config.min_subgraph_size <= len(nodes) <= self.config.max_subgraph_size):
            return False
        
        # Check node type requirements
        if template.required_node_types:
            node_types = {node.node_type.value for node in nodes}
            required_types = set(template.required_node_types)
            if not required_types.issubset(node_types):
                return False
        
        # Check edge type requirements
        if template.required_edge_types:
            edge_types = {edge.edge_type.value for edge in edges}
            required_edge_types = set(template.required_edge_types)
            if not required_edge_types.issubset(edge_types):
                return False
        
        # Check connectivity if required
        if self.config.avoid_disconnected and len(nodes) > 1:
            if not self._is_connected_subgraph(nodes, edges):
                return False
        
        return True
    
    def _is_connected_subgraph(self, nodes: List[Node], edges: List[Edge]) -> bool:
        """Check if subgraph is connected"""
        
        if len(nodes) <= 1:
            return True
        
        # Build adjacency list
        adj = {node.node_id: set() for node in nodes}
        node_ids = {node.node_id for node in nodes}
        
        for edge in edges:
            if edge.source_node_id in node_ids and edge.target_node_id in node_ids:
                adj[edge.source_node_id].add(edge.target_node_id)
                adj[edge.target_node_id].add(edge.source_node_id)  # Treat as undirected
        
        # BFS to check connectivity
        visited = set()
        queue = [nodes[0].node_id]
        visited.add(nodes[0].node_id)
        
        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(nodes)
    
    def _is_duplicate_subgraph(
        self, 
        nodes: List[Node], 
        existing_subgraphs: List[Tuple[List[Node], List[Edge]]]
    ) -> bool:
        """Check if subgraph is a duplicate of existing ones"""
        
        if not self.config.ensure_diversity:
            return False
        
        current_node_ids = {node.node_id for node in nodes}
        
        for existing_nodes, _ in existing_subgraphs:
            existing_node_ids = {node.node_id for node in existing_nodes}
            
            # Check for significant overlap (>70%)
            overlap = len(current_node_ids & existing_node_ids)
            total = len(current_node_ids | existing_node_ids)
            
            if total > 0 and overlap / total > 0.7:
                return True
        
        return False
    
    def _find_applicable_motifs(self, template: TaskTemplate) -> List[GraphMotif]:
        """Find motifs that are applicable to the template"""
        
        applicable = []
        
        for motif in COMMON_MOTIFS:
            # Check if motif requirements match template
            if template.required_node_types:
                motif_node_types = set(motif.node_pattern)
                template_node_types = set(template.required_node_types)
                
                if not motif_node_types.issubset(template_node_types):
                    continue
            
            if template.required_edge_types:
                motif_edge_types = set(motif.edge_pattern)
                template_edge_types = set(template.required_edge_types)
                
                if not motif_edge_types.issubset(template_edge_types):
                    continue
            
            # Check size constraints
            if not (template.min_nodes <= motif.max_nodes and 
                    motif.min_nodes <= template.max_nodes):
                continue
            
            applicable.append(motif)
        
        return applicable
    
    def _sample_motif_instances(
        self, 
        graph: DocumentGraph, 
        motif: GraphMotif, 
        template: TaskTemplate
    ) -> List[Tuple[List[Node], List[Edge]]]:
        """Sample instances of a specific motif"""
        
        instances = []
        attempts = 0
        max_attempts = 50
        
        # Get nodes of required types
        candidate_nodes = []
        for node_type_str in motif.node_pattern:
            try:
                node_type = NodeType(node_type_str)
                nodes = graph.storage.find_nodes(node_type=node_type)
                candidate_nodes.extend(nodes)
            except ValueError:
                continue
        
        while len(instances) < 5 and attempts < max_attempts:  # Limit motif instances
            attempts += 1
            
            if not candidate_nodes:
                break
            
            # Sample starting node
            start_node = random.choice(candidate_nodes)
            
            # Try to build motif pattern
            motif_nodes, motif_edges = self._build_motif_instance(
                graph, start_node, motif
            )
            
            if motif_nodes and self._is_valid_subgraph(motif_nodes, motif_edges, template):
                instances.append((motif_nodes, motif_edges))
        
        return instances
    
    def _build_motif_instance(
        self, 
        graph: DocumentGraph, 
        start_node: Node, 
        motif: GraphMotif
    ) -> Tuple[List[Node], List[Edge]]:
        """Build a specific motif instance starting from a node"""
        
        # Simple motif building - could be enhanced for complex patterns
        nodes = [start_node]
        edges = []
        
        # Get neighbors and try to match pattern
        neighbors = graph.get_neighbors(start_node.node_id)
        
        for neighbor_node, edge in neighbors:
            if len(nodes) >= motif.max_nodes:
                break
            
            # Check if neighbor type matches pattern
            if neighbor_node.node_type.value in motif.node_pattern:
                # Check if edge type matches pattern  
                if edge.edge_type.value in motif.edge_pattern:
                    nodes.append(neighbor_node)
                    edges.append(edge)
        
        return nodes, edges
    
    def _calculate_centrality_scores(
        self, 
        graph: DocumentGraph, 
        nodes: List[Node]
    ) -> Dict[str, float]:
        """Calculate centrality scores for nodes"""
        
        centrality_scores = {}
        
        # Simple degree centrality based on number of connections
        for node in nodes:
            neighbors = graph.get_neighbors(node.node_id)
            centrality_scores[node.node_id] = len(neighbors)
        
        # Normalize scores
        if centrality_scores:
            max_score = max(centrality_scores.values())
            if max_score > 0:
                for node_id in centrality_scores:
                    centrality_scores[node_id] /= max_score
        
        return centrality_scores
    
    def _find_semantic_clusters(
        self, 
        graph: DocumentGraph, 
        nodes: List[Node]
    ) -> List[List[Node]]:
        """Find clusters of semantically similar nodes"""
        
        clusters = []
        
        # Find nodes connected by semantic similarity edges
        processed = set()
        
        for node in nodes:
            if node.node_id in processed:
                continue
            
            cluster = [node]
            processed.add(node.node_id)
            
            # Find semantically similar neighbors
            neighbors = graph.get_neighbors(node.node_id, [EdgeType.SEMANTIC_SIM])
            
            for neighbor_node, edge in neighbors:
                if neighbor_node.node_id not in processed and neighbor_node in nodes:
                    cluster.append(neighbor_node)
                    processed.add(neighbor_node.node_id)
            
            if len(cluster) > 1:  # Only keep clusters with multiple nodes
                clusters.append(cluster)
        
        return clusters
    
    def _get_edges_between_nodes(
        self, 
        graph: DocumentGraph, 
        nodes: List[Node]
    ) -> List[Edge]:
        """Get all edges between a set of nodes"""
        
        node_ids = {node.node_id for node in nodes}
        edges = []
        
        # Check all edges in the graph
        for edge_type in EdgeType:
            type_edges = graph.storage.find_edges(edge_type=edge_type)
            
            for edge in type_edges:
                if (edge.source_node_id in node_ids and 
                    edge.target_node_id in node_ids):
                    edges.append(edge)
        
        return edges


class MotifSampler:
    """Specialized sampler for graph motifs"""
    
    def __init__(self):
        self.motif_patterns = COMMON_MOTIFS
    
    def sample_motifs(
        self, 
        graph: DocumentGraph, 
        motif_types: Optional[List[str]] = None,
        max_samples: int = 20
    ) -> List[Tuple[List[Node], List[Edge], GraphMotif]]:
        """Sample motif instances from graph"""
        
        sampled_motifs = []
        
        # Filter motifs if specific types requested
        target_motifs = self.motif_patterns
        if motif_types:
            target_motifs = [m for m in self.motif_patterns if m.motif_id in motif_types]
        
        for motif in target_motifs:
            if len(sampled_motifs) >= max_samples:
                break
            
            instances = self._sample_motif_instances(graph, motif)
            
            for nodes, edges in instances:
                sampled_motifs.append((nodes, edges, motif))
                
                if len(sampled_motifs) >= max_samples:
                    break
        
        return sampled_motifs
    
    def _sample_motif_instances(
        self, 
        graph: DocumentGraph, 
        motif: GraphMotif
    ) -> List[Tuple[List[Node], List[Edge]]]:
        """Sample instances of a specific motif pattern"""
        
        instances = []
        
        # Get candidate starting nodes
        candidate_nodes = []
        if motif.node_pattern:
            first_node_type = motif.node_pattern[0]
            try:
                node_type = NodeType(first_node_type)
                candidate_nodes = graph.storage.find_nodes(node_type=node_type)
            except ValueError:
                return instances
        
        # Try to build motif from each candidate
        for start_node in candidate_nodes[:10]:  # Limit attempts
            instance_nodes, instance_edges = self._build_motif_from_node(
                graph, start_node, motif
            )
            
            if self._matches_motif(instance_nodes, instance_edges, motif):
                instances.append((instance_nodes, instance_edges))
        
        return instances
    
    def _build_motif_from_node(
        self, 
        graph: DocumentGraph, 
        start_node: Node, 
        motif: GraphMotif
    ) -> Tuple[List[Node], List[Edge]]:
        """Build motif instance starting from a specific node"""
        
        # Simple implementation - could be enhanced for complex patterns
        nodes = [start_node]
        edges = []
        
        # Try to extend according to pattern
        current_nodes = [start_node]
        
        for pattern_step in range(1, len(motif.node_pattern)):
            next_nodes = []
            target_node_type = motif.node_pattern[pattern_step]
            target_edge_type = motif.edge_pattern[pattern_step - 1] if pattern_step - 1 < len(motif.edge_pattern) else None
            
            for current_node in current_nodes:
                neighbors = graph.get_neighbors(current_node.node_id)
                
                for neighbor_node, edge in neighbors:
                    # Check type matching
                    if (neighbor_node.node_type.value == target_node_type and
                        (not target_edge_type or edge.edge_type.value == target_edge_type) and
                        neighbor_node not in nodes):
                        
                        nodes.append(neighbor_node)
                        edges.append(edge)
                        next_nodes.append(neighbor_node)
                        break  # Take first match
            
            current_nodes = next_nodes
            if not current_nodes:
                break
        
        return nodes, edges
    
    def _matches_motif(
        self, 
        nodes: List[Node], 
        edges: List[Edge], 
        motif: GraphMotif
    ) -> bool:
        """Check if node/edge combination matches motif pattern"""
        
        # Check basic size constraints
        if not (motif.min_nodes <= len(nodes) <= motif.max_nodes):
            return False
        
        # Check node types
        node_types = [node.node_type.value for node in nodes]
        if len(node_types) != len(motif.node_pattern):
            return False
        
        # Simple pattern matching
        return motif.matches_subgraph(node_types, [edge.edge_type.value for edge in edges])


class ContextChainSampler:
    """Specialized sampler for context chain reasoning tasks"""
    
    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()
    
    def sample_context_chains(
        self, 
        graph: DocumentGraph, 
        template: TaskTemplate,
        chain_length: int = 3
    ) -> List[Tuple[List[Node], List[Edge]]]:
        """Sample subgraphs that form reasoning chains"""
        
        subgraphs = []
        attempts = 0
        max_attempts = self.config.max_samples_per_template * 5
        
        while len(subgraphs) < self.config.max_samples_per_template and attempts < max_attempts:
            attempts += 1
            
            # Find a starting node
            start_node = self._find_chain_start(graph, template)
            if not start_node:
                continue
            
            # Build chain by following edges
            chain_nodes, chain_edges = self._build_reasoning_chain(
                graph, start_node, chain_length, template
            )
            
            if (self._is_valid_subgraph(chain_nodes, chain_edges, template) and 
                not self._is_duplicate_subgraph(chain_nodes, subgraphs)):
                subgraphs.append((chain_nodes, chain_edges))
        
        return subgraphs
    
    def _find_chain_start(self, graph: DocumentGraph, template: TaskTemplate) -> Optional[Node]:
        """Find a good starting node for a reasoning chain"""
        candidate_nodes = self._get_candidate_nodes(graph, template)
        
        # Prefer nodes with outgoing edges (can start a chain)
        chain_starters = []
        for node in candidate_nodes:
            outgoing_edges = graph.get_outgoing_edges(node.node_id)
            if len(outgoing_edges) >= 2:  # Need at least 2 edges to form a chain
                chain_starters.append(node)
        
        return random.choice(chain_starters) if chain_starters else random.choice(candidate_nodes)
    
    def _build_reasoning_chain(
        self, 
        graph: DocumentGraph, 
        start_node: Node, 
        chain_length: int,
        template: TaskTemplate
    ) -> Tuple[List[Node], List[Edge]]:
        """Build a reasoning chain from a starting node"""
        
        chain_nodes = [start_node]
        chain_edges = []
        current_node = start_node
        
        for _ in range(chain_length - 1):
            # Get outgoing edges
            outgoing_edges = graph.get_outgoing_edges(current_node.node_id)
            
            # Filter edges by template requirements
            valid_edges = [
                edge for edge in outgoing_edges
                if edge.edge_type.value in template.required_edge_types
            ]
            
            if not valid_edges:
                break
            
            # Choose next edge and node
            next_edge = random.choice(valid_edges)
            next_node = graph.get_node(next_edge.target_node_id)
            
            if next_node and next_node not in chain_nodes:
                chain_nodes.append(next_node)
                chain_edges.append(next_edge)
                current_node = next_node
            else:
                break
        
        # Get edges between chain nodes
        all_edges = self._get_edges_between_nodes(graph, chain_nodes)
        
        return chain_nodes, all_edges
    
    def _get_candidate_nodes(self, graph: DocumentGraph, template: TaskTemplate) -> List[Node]:
        """Get candidate nodes for chain sampling"""
        candidate_nodes = []
        
        for node_type_str in template.required_node_types:
            try:
                node_type = NodeType(node_type_str)
                nodes = graph.storage.find_nodes(node_type=node_type)
                candidate_nodes.extend(nodes)
            except ValueError:
                continue
        
        return candidate_nodes
    
    def _is_valid_subgraph(
        self, 
        nodes: List[Node], 
        edges: List[Edge], 
        template: TaskTemplate
    ) -> bool:
        """Check if subgraph is valid for template"""
        if not (template.min_nodes <= len(nodes) <= template.max_nodes):
            return False
        
        # Check node types
        node_types = {node.node_type.value for node in nodes}
        required_node_types = set(template.required_node_types)
        if not required_node_types.issubset(node_types):
            return False
        
        return True
    
    def _is_duplicate_subgraph(
        self, 
        nodes: List[Node], 
        existing_subgraphs: List[Tuple[List[Node], List[Edge]]]
    ) -> bool:
        """Check if subgraph is duplicate"""
        node_ids = {node.node_id for node in nodes}
        
        for existing_nodes, _ in existing_subgraphs:
            existing_ids = {node.node_id for node in existing_nodes}
            if node_ids == existing_ids:
                return True
        
        return False
    
    def _get_edges_between_nodes(
        self, 
        graph: DocumentGraph, 
        nodes: List[Node]
    ) -> List[Edge]:
        """Get edges between a set of nodes"""
        edges = []
        node_ids = {node.node_id for node in nodes}
        
        for node in nodes:
            # Get all edges for this node
            all_edges = graph.get_incoming_edges(node.node_id) + graph.get_outgoing_edges(node.node_id)
            
            for edge in all_edges:
                # Check if both endpoints are in our node set
                if (edge.source_node_id in node_ids and 
                    edge.target_node_id in node_ids and
                    edge not in edges):
                    edges.append(edge)
        
        return edges


class ContradictionSampler:
    """Specialized sampler for contradiction detection tasks"""
    
    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()
    
    def sample_contradiction_pairs(
        self, 
        graph: DocumentGraph, 
        template: TaskTemplate
    ) -> List[Tuple[List[Node], List[Edge]]]:
        """Sample subgraphs that contain potential contradictions"""
        
        subgraphs = []
        attempts = 0
        max_attempts = self.config.max_samples_per_template * 10
        
        while len(subgraphs) < self.config.max_samples_per_template and attempts < max_attempts:
            attempts += 1
            
            # Find semantically similar nodes that might contain contradictions
            contradiction_nodes = self._find_potential_contradictions(graph, template)
            
            if len(contradiction_nodes) >= 2:
                # Add some context nodes
                context_nodes = self._add_context_nodes(graph, contradiction_nodes, template)
                all_nodes = contradiction_nodes + context_nodes
                
                # Get edges between nodes
                edges = self._get_edges_between_nodes(graph, all_nodes)
                
                if (self._is_valid_subgraph(all_nodes, edges, template) and 
                    not self._is_duplicate_subgraph(all_nodes, subgraphs)):
                    subgraphs.append((all_nodes, edges))
        
        return subgraphs
    
    def _find_potential_contradictions(
        self, 
        graph: DocumentGraph, 
        template: TaskTemplate
    ) -> List[Node]:
        """Find nodes that might contain contradictions"""
        
        # Get all paragraph nodes
        paragraph_nodes = graph.storage.find_nodes(node_type=NodeType.PARAGRAPH)
        
        # Find semantically similar nodes (potential contradictions)
        contradiction_candidates = []
        
        for i, node1 in enumerate(paragraph_nodes):
            for j, node2 in enumerate(paragraph_nodes[i+1:], i+1):
                # Check semantic similarity
                similarity = self._calculate_semantic_similarity(node1, node2)
                
                # High similarity might indicate potential contradictions
                if 0.7 <= similarity <= 0.9:
                    contradiction_candidates.extend([node1, node2])
                    break  # Found a pair, move to next node
        
        return contradiction_candidates[:4]  # Limit to 4 nodes
    
    def _calculate_semantic_similarity(self, node1: Node, node2: Node) -> float:
        """Calculate semantic similarity between two nodes"""
        # Simple word overlap similarity
        words1 = set(node1.content.lower().split())
        words2 = set(node2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _add_context_nodes(
        self, 
        graph: DocumentGraph, 
        contradiction_nodes: List[Node], 
        template: TaskTemplate
    ) -> List[Node]:
        """Add context nodes around contradiction nodes"""
        
        context_nodes = []
        
        for node in contradiction_nodes:
            # Get neighboring nodes
            neighbors = self._get_neighbor_nodes(graph, node, max_neighbors=2)
            context_nodes.extend(neighbors)
        
        # Remove duplicates and limit
        unique_context = []
        seen_ids = {node.node_id for node in contradiction_nodes}
        
        for node in context_nodes:
            if node.node_id not in seen_ids:
                unique_context.append(node)
                seen_ids.add(node.node_id)
        
        return unique_context[:3]  # Limit context nodes
    
    def _get_neighbor_nodes(
        self, 
        graph: DocumentGraph, 
        node: Node, 
        max_neighbors: int = 2
    ) -> List[Node]:
        """Get neighboring nodes of a given node"""
        
        neighbors = []
        
        # Get incoming and outgoing edges
        incoming_edges = graph.get_incoming_edges(node.node_id)
        outgoing_edges = graph.get_outgoing_edges(node.node_id)
        
        all_edges = incoming_edges + outgoing_edges
        
        for edge in all_edges[:max_neighbors]:
            neighbor_id = edge.target_node_id if edge.source_node_id == node.node_id else edge.source_node_id
            neighbor = graph.get_node(neighbor_id)
            if neighbor:
                neighbors.append(neighbor)
        
        return neighbors
    
    def _is_valid_subgraph(
        self, 
        nodes: List[Node], 
        edges: List[Edge], 
        template: TaskTemplate
    ) -> bool:
        """Check if subgraph is valid for template"""
        if not (template.min_nodes <= len(nodes) <= template.max_nodes):
            return False
        
        # Check node types
        node_types = {node.node_type.value for node in nodes}
        required_node_types = set(template.required_node_types)
        if not required_node_types.issubset(node_types):
            return False
        
        return True
    
    def _is_duplicate_subgraph(
        self, 
        nodes: List[Node], 
        existing_subgraphs: List[Tuple[List[Node], List[Edge]]]
    ) -> bool:
        """Check if subgraph is duplicate"""
        node_ids = {node.node_id for node in nodes}
        
        for existing_nodes, _ in existing_subgraphs:
            existing_ids = {node.node_id for node in existing_nodes}
            if node_ids == existing_ids:
                return True
        
        return False
    
    def _get_edges_between_nodes(
        self, 
        graph: DocumentGraph, 
        nodes: List[Node]
    ) -> List[Edge]:
        """Get edges between a set of nodes"""
        edges = []
        node_ids = {node.node_id for node in nodes}
        
        for node in nodes:
            # Get all edges for this node
            all_edges = graph.get_incoming_edges(node.node_id) + graph.get_outgoing_edges(node.node_id)
            
            for edge in all_edges:
                # Check if both endpoints are in our node set
                if (edge.source_node_id in node_ids and 
                    edge.target_node_id in node_ids and
                    edge not in edges):
                    edges.append(edge)
        
        return edges


# Add the new sampling methods to SubgraphSampler class
def _sample_context_chain_based(
    self, 
    graph: DocumentGraph, 
    template: TaskTemplate
) -> List[Tuple[List[Node], List[Edge]]]:
    """Sample subgraphs using context chain strategy"""
    
    chain_sampler = ContextChainSampler(self.config)
    return chain_sampler.sample_context_chains(graph, template)


def _sample_contradiction_based(
    self, 
    graph: DocumentGraph, 
    template: TaskTemplate
) -> List[Tuple[List[Node], List[Edge]]]:
    """Sample subgraphs using contradiction detection strategy"""
    
    contradiction_sampler = ContradictionSampler(self.config)
    return contradiction_sampler.sample_contradiction_pairs(graph, template)
