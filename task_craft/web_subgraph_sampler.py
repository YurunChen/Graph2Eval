"""
Web Subgraph Sampler - 子图选择策略 - 修复版本
FIXED ISSUES:
1. 简化并发设计 - 只保留种子级并发，避免嵌套并发问题
2. 使用线程池 - 避免序列化问题，提高性能
3. 优化参数 - 调整worker数量和chunk大小
4. 删除复杂嵌套并发 - 移除导致性能问题的代码

按策略取r-hop子图，控长、控噪、控复杂度，确保任务与实际UI结构强绑定
支持简化的并发处理以提高采样效率
"""

import logging
import random
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path

# New: concurrent processing related imports
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from threading import Lock

# Try to import psutil, provide default value if not available
try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("🎯 psutil not available, using default memory values")

from loguru import logger
from collections import defaultdict, deque
from graph_rag.graph_builder import TaskGraph
from graph_rag.edge_types import GraphEdge, EdgeType
from graph_rag.node_types import GraphNode, NodeType
from .task_seeds import TaskSeedPattern, TaskSeedType
from config_manager import get_config

class SamplingStrategy(Enum):
    """子图采样策略"""
    ELEMENT_CENTRIC = "element_centric"  # Element-centric
    RELATION_CENTRIC = "relation_centric"  # Relation-centric
    GOAL_CONDITIONED = "goal_conditioned"  # Goal-conditioned
    CURRICULUM_RADIUS = "curriculum_radius"  # Curriculum radius

    STRUCTURE_FIRST = "structure_first"  # Structure-first
    FUNCTIONAL_MODULE = "functional_module"  # Functional module-oriented

@dataclass
class SubgraphConfig:
    """子图配置"""
    strategy: SamplingStrategy = SamplingStrategy.ELEMENT_CENTRIC
    max_radius: int = 2  # Maximum hops
    min_nodes: int = 3   # Minimum number of nodes
    max_nodes: int = 15   # Maximum number of nodes
    max_subgraphs_per_seed: int = 15  # Maximum subgraphs per seed (ensure quality)
    max_total_subgraphs: int = 200  # Maximum total subgraphs (ensure diversity)
    target_task_type: Optional[str] = None  # Target task type
    curriculum_step: int = 1  # Current step in curriculum learning
    
    # Fixed concurrency configuration
    enable_concurrency: bool = True  # Whether to enable concurrency
    max_workers: int = 4  # Maximum worker threads (reduce worker count)
    chunk_size: int = 8   # Task chunk size (increase chunk size, reduce task fragmentation)
    use_thread_pool: bool = True  # Use thread pool to avoid serialization issues
    
    @classmethod
    def from_config(cls):
        """从配置文件创建配置"""
        config = get_config()
        web_sampling_config = config.task_craft.get('web_subgraph_sampling', {})
        
        # Strategy mapping
        strategy_mapping = {
            "element_centric": SamplingStrategy.ELEMENT_CENTRIC,
            "relation_centric": SamplingStrategy.RELATION_CENTRIC,
            "goal_conditioned": SamplingStrategy.GOAL_CONDITIONED,
            "curriculum_radius": SamplingStrategy.CURRICULUM_RADIUS,
    
        }
        
        strategy_str = web_sampling_config.get('sampling_strategy', 'element_centric')
        strategy = strategy_mapping.get(strategy_str, SamplingStrategy.ELEMENT_CENTRIC)
        
        # Concurrency configuration
        concurrency_config = web_sampling_config.get('concurrency', {})
        
        return cls(
            strategy=strategy,
            max_radius=web_sampling_config.get('max_radius', 2),
            min_nodes=web_sampling_config.get('min_nodes', 3),
            max_nodes=web_sampling_config.get('max_nodes', 15),
            max_subgraphs_per_seed=web_sampling_config.get('max_subgraphs_per_seed', 15),
            max_total_subgraphs=web_sampling_config.get('max_total_subgraphs', 200),
            target_task_type=web_sampling_config.get('target_task_type'),
            curriculum_step=web_sampling_config.get('curriculum_step', 1),
            enable_concurrency=concurrency_config.get('enable', True),
            max_workers=concurrency_config.get('max_workers', min(4, os.cpu_count() or 4)),
            chunk_size=concurrency_config.get('chunk_size', 8),  # Increase default chunk size
            use_thread_pool=concurrency_config.get('use_thread_pool', True)  # Default to use thread pool
        )

@dataclass
class SubgraphSample:
    """子图样本"""
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: Dict[str, GraphEdge] = field(default_factory=dict)
    center_node: Optional[str] = None
    radius: int = 0
    strategy: SamplingStrategy = SamplingStrategy.ELEMENT_CENTRIC
    task_seed: Optional[TaskSeedPattern] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
            "center_node": self.center_node,
            "radius": self.radius,
            "strategy": self.strategy.value,
            "task_seed": self.task_seed.name if self.task_seed else None
        }

class WebSubgraphSampler:
    """Web子图采样器 - 支持并发处理"""
    
    def __init__(self, config: SubgraphConfig):
        self.config = config
        self._lock = threading.Lock()  # Thread safety lock
        self.executor = None
        
        # Initialize executor
        self._init_executor()
    
    def _get_node_safe(self, task_graph: TaskGraph, node_id: str) -> Optional[Any]:
        """安全地获取节点，避免None错误"""
        node = task_graph.nodes.get(node_id)
        if node is None:
            logger.debug(f"🎯 Node {node_id} is None or not found")
        return node
    
    def _init_executor(self):
        """初始化执行器"""
        if self.config.enable_concurrency:
            if self.config.use_thread_pool:
                # Use thread pool to avoid serialization issues
                self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
                logger.info(f"🎯 Initialized ThreadPoolExecutor with {self.config.max_workers} workers")
            else:
                # If configured not to use thread pool, disable concurrency
                self.executor = None
                logger.info("🎯 Thread pool disabled, using sequential processing")
        else:
            self.executor = None
            logger.info("🎯 Concurrency disabled, using sequential processing")
    
    def __del__(self):
        """清理资源"""
        if self.executor:
            self.executor.shutdown(wait=True)
    
# 删除重复的方法定义，保留下面的正确版本
    
    
    
    
    def _curriculum_radius_sampling(self, task_graph: TaskGraph, task_seeds: List[TaskSeedPattern]) -> List[SubgraphSample]:
        """课程式半径采样 - 逐步扩大半径"""
        subgraphs = []
        
        # 根据课程步骤确定当前半径
        current_radius = min(self.config.curriculum_step, self.config.max_radius)
        
        # 从简单任务开始
        simple_seeds = [seed for seed in task_seeds if seed.difficulty == "EASY"]
        medium_seeds = [seed for seed in task_seeds if seed.difficulty == "MEDIUM"]
        hard_seeds = [seed for seed in task_seeds if seed.difficulty == "HARD"]
        
        seed_priority = simple_seeds + medium_seeds + hard_seeds
        
        for seed in seed_priority:
            # 根据当前半径采样
            subgraph = self._radius_based_sampling(task_graph, seed, current_radius)
            if subgraph and self._validate_subgraph(subgraph, seed):
                subgraphs.append(subgraph)
        
        return subgraphs
    
    
    def _deduplicate_subgraphs(self, subgraphs: List[SubgraphSample]) -> List[SubgraphSample]:
        """去重子图，保留最优质的"""
        if not subgraphs:
            return []
        
        # 按节点集合去重
        unique_subgraphs = []
        seen_node_sets = set()
        
        for subgraph in subgraphs:
            node_set = frozenset(subgraph.nodes.keys())
            if node_set not in seen_node_sets:
                seen_node_sets.add(node_set)
                unique_subgraphs.append(subgraph)
        
        logger.info(f"🎯 Deduplication: {len(subgraphs)} -> {len(unique_subgraphs)} subgraphs")
        return unique_subgraphs
    
    def _find_related_business_data(self, task_graph: TaskGraph, subgraph_node_ids: List[str]) -> List[str]:
        """查找与子图相关的业务数据节点 - 修复版本"""
        related_business_data = []
        
        # 获取子图中所有节点的页面URL
        subgraph_urls = set()
        for node_id in subgraph_node_ids:
            node = task_graph.nodes.get(node_id)
            if node is None:
                continue
            if hasattr(node, 'url') and node.url:
                subgraph_urls.add(node.url)
        
        # 查找与子图在同一页面的业务数据节点
        from graph_rag.node_types import NodeType
        business_data_types = {
            NodeType.BUSINESS_DATA,
            NodeType.USER_DATA,
            NodeType.PRODUCT_DATA,
            NodeType.ORDER_DATA,
            NodeType.CONTENT_DATA,
            NodeType.FINANCIAL_DATA,
            NodeType.LOCATION_DATA,
            NodeType.TIME_DATA
        }
        
        # 按优先级排序业务数据节点
        business_data_candidates = []
        
        for node_id, node in task_graph.nodes.items():
            if node_id in subgraph_node_ids:
                continue
            
            # 安全检查：确保节点不为None
            if node is None:
                logger.warning(f"🎯 Warning: node {node_id} is None, skipping")
                continue
                
            if node.node_type in business_data_types:
                # 计算业务数据节点的优先级
                priority = self._calculate_business_data_priority(node, subgraph_urls)
                business_data_candidates.append((node_id, priority))
        
        # 按优先级排序，优先选择高优先级的业务数据节点
        business_data_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前5个最高优先级的业务数据节点
        related_business_data = [node_id for node_id, _ in business_data_candidates[:5]]
        
        logger.debug(f"🎯 Found {len(related_business_data)} related business data nodes")
        return related_business_data
    
    def _calculate_business_data_priority(self, node: Any, subgraph_urls: Set[str]) -> float:
        """计算业务数据节点的优先级"""
        if node is None:
            return 0.0

        priority = 0.0
        
        # 1. 页面匹配优先级（最高）
        if hasattr(node, 'url') and node.url in subgraph_urls:
            priority += 10.0
        
        # 2. 节点类型优先级
        node_type_priority = {
            NodeType.BUSINESS_DATA: 8.0,
            NodeType.USER_DATA: 7.0,
            NodeType.PRODUCT_DATA: 7.0,
            NodeType.ORDER_DATA: 6.0,
            NodeType.CONTENT_DATA: 5.0,
            NodeType.FINANCIAL_DATA: 6.0,
            NodeType.LOCATION_DATA: 4.0,
            NodeType.TIME_DATA: 3.0
        }
        priority += node_type_priority.get(node.node_type, 1.0)
        
        # 3. 内容丰富度优先级
        if hasattr(node, 'content') and node.content and len(str(node.content)) > 10:
            priority += 2.0
        
        if hasattr(node, 'text') and node.text and len(str(node.text)) > 5:
            priority += 1.5
        
        # 4. 属性丰富度优先级
        if hasattr(node, 'properties') and node.properties:
            priority += len(node.properties) * 0.5
        
        return priority
    
    
    
    def _find_nodes_for_seed(self, task_graph: TaskGraph, seed: TaskSeedPattern) -> List[str]:
        """找到满足种子要求的节点"""
        target_nodes = []
        
        logger.debug(f"🎯 Finding nodes for seed: {seed.name}")
        logger.debug(f"🎯 Core slots: {seed.core_slots}")
        logger.debug(f"🎯 Optional slots: {seed.optional_slots}")
        
        for node_id, node in task_graph.nodes.items():
            # 安全检查：确保节点不为None
            if node is None:
                logger.warning(f"🎯 Warning: node {node_id} is None, skipping")
                continue
                
            # 检查核心槽位
            for slot_name, required_type in seed.core_slots.items():
                # 确保类型比较正确
                if hasattr(node.node_type, 'value'):
                    node_type_value = node.node_type.value
                else:
                    node_type_value = str(node.node_type)
                
                if isinstance(required_type, list):
                    # 如果是列表，检查是否匹配其中任何一个
                    for item_type in required_type:
                        if hasattr(item_type, 'value'):
                            item_type_value = item_type.value
                        else:
                            item_type_value = str(item_type)
                        
                        if node_type_value == item_type_value:
                            target_nodes.append(node_id)
                            logger.debug(f"🎯 Found matching node {node_id} for core slot {slot_name}: {item_type_value}")
                            break
                    else:
                        continue
                    break
                else:
                    # 如果是单个类型
                    if hasattr(required_type, 'value'):
                        required_type_value = required_type.value
                    else:
                        required_type_value = str(required_type)
                    
                    if node_type_value == required_type_value:
                        target_nodes.append(node_id)
                        logger.debug(f"🎯 Found matching node {node_id} for core slot {slot_name}: {node_type_value}")
                        break
            
            # 检查可选槽位
            for slot_name, required_type in seed.optional_slots.items():
                # 确保类型比较正确
                if hasattr(node.node_type, 'value'):
                    node_type_value = node.node_type.value
                else:
                    node_type_value = str(node.node_type)
                
                if isinstance(required_type, list):
                    # 如果是列表，检查是否匹配其中任何一个
                    for item_type in required_type:
                        if hasattr(item_type, 'value'):
                            item_type_value = item_type.value
                        else:
                            item_type_value = str(item_type)
                        
                        if node_type_value == item_type_value:
                            target_nodes.append(node_id)
                            logger.debug(f"🎯 Found matching node {node_id} for optional slot {slot_name}: {item_type_value}")
                            break
                    else:
                        continue
                    break
                else:
                    # 如果是单个类型
                    if hasattr(required_type, 'value'):
                        required_type_value = required_type.value
                    else:
                        required_type_value = str(required_type)
                    
                    if node_type_value == required_type_value:
                        target_nodes.append(node_id)
                        logger.debug(f"🎯 Found matching node {node_id} for optional slot {slot_name}: {node_type_value}")
                        break
        
        logger.debug(f"🎯 Found {len(target_nodes)} target nodes for seed {seed.name}")
        return target_nodes
    
    def _bfs_sampling(self, task_graph: TaskGraph, center_node_id: str, seed: TaskSeedPattern) -> Optional[SubgraphSample]:
        """BFS采样"""
        visited = set()
        queue = deque([(center_node_id, 0)])  # (node_id, distance)
        sampled_nodes = {}
        sampled_edges = {}
        business_data_nodes = []
        
        # 调试：检查图中的边类型
        edge_types_in_graph = {edge.edge_type for edge in task_graph.edges.values()}
        logger.debug(f"🎯 BFS sampling for seed {seed.name}")
        logger.debug(f"🎯 Edge types in graph: {edge_types_in_graph}")
        logger.debug(f"🎯 Seed required edge types: {seed.required_edge_types}")
        logger.debug(f"🎯 Starting from center node: {center_node_id}")
        
        while queue and len(sampled_nodes) < self.config.max_nodes:
            current_node_id, distance = queue.popleft()
            
            if current_node_id in visited or distance > self.config.max_radius:
                continue
            
            visited.add(current_node_id)
            current_node = task_graph.nodes.get(current_node_id)
            if current_node is None:
                logger.warning(f"🎯 Current node {current_node_id} is None, skipping")
                continue
            sampled_nodes[current_node_id] = current_node
            
            # 检查是否为业务数据节点（使用NodeType枚举）
            from graph_rag.node_types import NodeType
            business_data_types = {
                NodeType.BUSINESS_DATA,
                NodeType.USER_DATA,
                NodeType.PRODUCT_DATA,
                NodeType.ORDER_DATA,
                NodeType.CONTENT_DATA,
                NodeType.FINANCIAL_DATA,
                NodeType.LOCATION_DATA,
                NodeType.TIME_DATA
            }
            
            if current_node.node_type in business_data_types:
                business_data_nodes.append(current_node_id)
            
                # 添加相关的边
            edges_added = 0
            for edge in task_graph.edges.values():
                if edge.source_node_id == current_node_id and edge.target_node_id in task_graph.nodes:
                    # 精准检查边类型匹配
                    edge_matches_requirement = False
                    
                    # 1. 优先添加导航边（如果种子需要导航）
                    if edge.edge_type == EdgeType.NAV_TO:
                        needs_nav = self._does_seed_need_navigation(seed)
                        if needs_nav:
                            edge_matches_requirement = True
                            logger.debug(f"🎯 Edge {edge.edge_id} included: NAV_TO edge for navigation-requiring seed")
                        else:
                            logger.debug(f"🎯 Edge {edge.edge_id} excluded: NAV_TO edge but seed doesn't need navigation")
                    
                    # 2. 检查是否匹配种子要求的边类型
                    elif seed.required_edge_types and edge.edge_type in seed.required_edge_types:
                        edge_matches_requirement = True
                        logger.debug(f"🎯 Edge {edge.edge_id} included: matches required edge type {edge.edge_type}")
                    
                    # 3. 检查是否是内容包含关系（CONTAINS边）
                    elif edge.edge_type == EdgeType.CONTAINS:
                        # 允许通过CONTAINS边扩展子图
                        if edge.source_node_id in sampled_nodes:
                            # 如果源节点已采样，添加目标节点
                            edge_matches_requirement = True
                            logger.debug(f"🎯 Edge {edge.edge_id} included: CONTAINS edge from sampled node to {edge.target_node_id}")
                        elif edge.target_node_id in sampled_nodes:
                            # 如果目标节点已采样，添加源节点
                            edge_matches_requirement = True
                            logger.debug(f"🎯 Edge {edge.edge_id} included: CONTAINS edge from {edge.source_node_id} to sampled node")
                        else:
                            logger.debug(f"🎯 Edge {edge.edge_id} excluded: CONTAINS edge not connected to sampled nodes")
                    
                    # 4. 检查是否是功能性边且与种子相关
                    elif edge.edge_type in {EdgeType.CONTROLS, EdgeType.FILLS, EdgeType.FILTERS, EdgeType.OPENS}:
                        if self._is_functional_edge_relevant(edge, seed, sampled_nodes):
                            edge_matches_requirement = True
                            logger.debug(f"🎯 Edge {edge.edge_id} included: functional edge {edge.edge_type} is relevant")
                        else:
                            logger.debug(f"🎯 Edge {edge.edge_id} excluded: functional edge {edge.edge_type} not relevant")
                    
                    # 5. 如果没有明确理由，不包含该边
                    else:
                        logger.debug(f"🎯 Edge {edge.edge_id} excluded: no clear reason to include {edge.edge_type}")
                    
                    if edge_matches_requirement:
                        sampled_edges[edge.edge_id] = edge
                        queue.append((edge.target_node_id, distance + 1))
                        edges_added += 1
                        logger.debug(f"🎯 Added edge {edge.edge_id} ({edge.edge_type}) to target {edge.target_node_id}")
                
                # 反向边检查（目标节点是当前节点）
                if edge.target_node_id == current_node_id and edge.source_node_id in task_graph.nodes:
                    # 精准检查边类型匹配
                    edge_matches_requirement = False

                    # 1. 优先添加导航边（如果种子需要导航）
                    if edge.edge_type == EdgeType.NAV_TO:
                        needs_nav = self._does_seed_need_navigation(seed)
                        if needs_nav:
                            edge_matches_requirement = True
                            logger.debug(f"🎯 Edge {edge.edge_id} included: NAV_TO edge for navigation-requiring seed")
                        else:
                            logger.debug(f"🎯 Edge {edge.edge_id} excluded: NAV_TO edge but seed doesn't need navigation")

                    # 2. 检查是否匹配种子要求的边类型
                    elif seed.required_edge_types and edge.edge_type in seed.required_edge_types:
                        edge_matches_requirement = True
                        logger.debug(f"🎯 Edge {edge.edge_id} included: matches required edge type {edge.edge_type}")

                    # 3. 检查是否是内容包含关系（CONTAINS边）
                    elif edge.edge_type == EdgeType.CONTAINS:
                        # 允许通过CONTAINS边扩展子图
                        if edge.source_node_id in sampled_nodes:
                            # 如果源节点已采样，添加目标节点
                            edge_matches_requirement = True
                            logger.debug(f"🎯 Edge {edge.edge_id} included: CONTAINS edge from sampled node to {edge.source_node_id}")
                        elif edge.target_node_id in sampled_nodes:
                            # 如果目标节点已采样，添加源节点
                            edge_matches_requirement = True
                            logger.debug(f"🎯 Edge {edge.edge_id} included: CONTAINS edge from {edge.target_node_id} to sampled node")
                        else:
                            logger.debug(f"🎯 Edge {edge.edge_id} excluded: CONTAINS edge not connected to sampled nodes")

                    # 4. 检查是否是功能性边且与种子相关
                    elif edge.edge_type in {EdgeType.CONTROLS, EdgeType.FILLS, EdgeType.FILTERS, EdgeType.OPENS}:
                        if self._is_functional_edge_relevant(edge, seed, sampled_nodes):
                            edge_matches_requirement = True
                            logger.debug(f"🎯 Edge {edge.edge_id} included: functional edge {edge.edge_type} is relevant")
                        else:
                            logger.debug(f"🎯 Edge {edge.edge_id} excluded: functional edge {edge.edge_type} not relevant")

                    # 5. 如果没有明确理由，不包含该边
                    else:
                        logger.debug(f"🎯 Edge {edge.edge_id} excluded: no clear reason to include {edge.edge_type}")

                    if edge_matches_requirement:
                        sampled_edges[edge.edge_id] = edge
                        queue.append((edge.source_node_id, distance + 1))
                        edges_added += 1
                        logger.debug(f"🎯 Added edge {edge.edge_id} ({edge.edge_type}) to target {edge.source_node_id}")
            
            logger.debug(f"🎯 Added {edges_added} edges from node {current_node_id}")
        
        # 如果子图中没有业务数据节点，尝试添加一些
        if not business_data_nodes and len(sampled_nodes) < self.config.max_nodes:
            # 查找与当前子图相关的业务数据节点
            related_business_data = self._find_related_business_data(task_graph, list(sampled_nodes.keys()))
            for node_id in related_business_data:
                if len(sampled_nodes) < self.config.max_nodes:
                    node = task_graph.nodes.get(node_id)
                    if node is not None:
                        sampled_nodes[node_id] = node
                    logger.debug(f"🎯 Added business data node {node_id} to subgraph")
        
        if len(sampled_nodes) >= self.config.min_nodes:
            return SubgraphSample(
                nodes=sampled_nodes,
                edges=sampled_edges,
                center_node=center_node_id,
                radius=self.config.max_radius,
                strategy=self.config.strategy,
                task_seed=seed
            )
        
        return None
    
    
    
    def _radius_based_sampling(self, task_graph: TaskGraph, seed: TaskSeedPattern, radius: int) -> Optional[SubgraphSample]:
        """基于半径的采样"""
        # 找到种子要求的节点类型
        target_nodes = self._find_nodes_for_seed(task_graph, seed)
        
        if not target_nodes:
            return None
        
        # 随机选择一个目标节点作为中心
        center_node_id = random.choice(target_nodes)
        
        # 使用BFS采样，但限制半径
        return self._bfs_sampling_with_radius(task_graph, center_node_id, seed, radius)
    
    def _bfs_sampling_with_radius(self, task_graph: TaskGraph, center_node_id: str, seed: TaskSeedPattern, radius: int) -> Optional[SubgraphSample]:
        """带半径限制的BFS采样"""
        visited = set()
        queue = deque([(center_node_id, 0)])
        sampled_nodes = {}
        sampled_edges = {}
        
        while queue and len(sampled_nodes) < self.config.max_nodes:
            current_node_id, distance = queue.popleft()
            
            if current_node_id in visited or distance > radius:
                continue
            
            visited.add(current_node_id)
            current_node = task_graph.nodes.get(current_node_id)
            if current_node is None:
                logger.warning(f"🎯 Current node {current_node_id} is None, skipping")
                continue
            sampled_nodes[current_node_id] = current_node
            
            # 添加相关的边
            for edge in task_graph.edges.values():
                if edge.source_node_id == current_node_id and edge.target_node_id in task_graph.nodes:
                    # 检查边类型匹配
                    edge_matches_requirement = False
                    
                    # 优先添加导航边
                    if edge.edge_type == EdgeType.NAV_TO:
                        edge_matches_requirement = True
                    # 检查是否匹配种子要求的边类型
                    elif seed.required_edge_types and edge.edge_type in seed.required_edge_types:
                        edge_matches_requirement = True
                    # 如果没有特定要求，添加所有边
                    elif not seed.required_edge_types:
                        edge_matches_requirement = True
                    
                    if edge_matches_requirement:
                        sampled_edges[edge.edge_id] = edge
                        queue.append((edge.target_node_id, distance + 1))
                        logger.debug(f"🎯 BFS with radius: added edge {edge.edge_id} ({edge.edge_type}) to target {edge.target_node_id}")
        
        if len(sampled_nodes) >= self.config.min_nodes:
            return SubgraphSample(
                nodes=sampled_nodes,
                edges=sampled_edges,
                center_node=center_node_id,
                radius=radius,
                strategy=self.config.strategy,
                task_seed=seed
            )
        
        return None
    
    def _validate_subgraph(self, subgraph: SubgraphSample, seed: TaskSeedPattern) -> bool:
        """验证子图是否满足种子要求 - 增强质量控制"""
        logger.debug(f"🎯 Validating subgraph for seed {seed.name} with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges")

        # 检查节点数量 - 严格控制质量
        if len(subgraph.nodes) < self.config.min_nodes:  # 使用配置的最小节点数
            logger.debug(f"🎯 Subgraph validation failed: too few nodes ({len(subgraph.nodes)} < {self.config.min_nodes})")
            return False

        if len(subgraph.nodes) > self.config.max_nodes:
            logger.debug(f"🎯 Subgraph validation failed: too many nodes ({len(subgraph.nodes)} > {self.config.max_nodes})")
            return False

        # 检查子图连通性 - 新增质量检查
        if not self._validate_subgraph_connectivity(subgraph.nodes, subgraph.edges):
            logger.debug(f"🎯 Subgraph validation failed: disconnected components")
            return False

        # 检查可执行性分数 - 确保质量阈值
        executability_score = self._calculate_executability_score(subgraph)
        min_quality_score = 5.0  # 设置最小质量分数
        if executability_score < min_quality_score:
            logger.debug(f"🎯 Subgraph validation failed: low executability score ({executability_score} < {min_quality_score})")
            return False
        
        # 检查是否包含必需的节点类型 - 放宽检查
        subgraph_node_types = {node.node_type for node in subgraph.nodes.values() if node is not None}
        logger.debug(f"🎯 Subgraph node types: {subgraph_node_types}")
        
        # 检查核心槽位 - 放宽检查，只要有任何匹配即可
        core_slot_types = set()
        for slot_value in seed.core_slots.values():
            if isinstance(slot_value, list):
                core_slot_types.update(slot_value)
            else:
                core_slot_types.add(slot_value)
        
        if core_slot_types:
            intersection = subgraph_node_types.intersection(core_slot_types)
            if not intersection:
                logger.debug(f"🎯 Subgraph validation failed: no core slot types found. Required: {core_slot_types}, Available: {subgraph_node_types}")
                return False
            else:
                logger.debug(f"🎯 Found matching core slot types: {intersection}")
        
        # 检查是否包含必需的边类型 - 放宽检查，边类型不是必需的
        subgraph_edge_types = {edge.edge_type for edge in subgraph.edges.values()}
        logger.debug(f"🎯 Subgraph edge types: {subgraph_edge_types}")
        
        if seed.required_edge_types:
            intersection = subgraph_edge_types.intersection(seed.required_edge_types)
            if not intersection:
                logger.debug(f"🎯 Warning: no required edge types found, but continuing. Required: {seed.required_edge_types}, Available: {subgraph_edge_types}")
            else:
                logger.debug(f"🎯 Found matching edge types: {intersection}")
        
        # 特别检查导航相关的种子 - 放宽检查
        if seed.seed_type in [TaskSeedType.MULTI_HOP_NAVIGATION, TaskSeedType.BASIC_NAVIGATION, TaskSeedType.BREADCRUMB_NAVIGATION]:
            # 确保有导航节点
            navigation_types = {NodeType.NAVIGATION, NodeType.LINK, NodeType.BREADCRUMB, NodeType.BUTTON}
            if not subgraph_node_types.intersection(navigation_types):
                logger.debug(f"🎯 Navigation subgraph validation failed: no navigation nodes found")
                return False
            
            # 如果有导航边更好，但不是必需的
            if EdgeType.NAV_TO not in subgraph_edge_types:
                logger.debug(f"🎯 Navigation subgraph has no NAV_TO edges, but continuing with navigation nodes")
        
        logger.debug(f"🎯 Subgraph validation passed for seed {seed.name}")
        return True
    
    
    
    def _should_include_edge_for_seed(self, edge: Any, seed: TaskSeedPattern, sampled_nodes: Dict[str, Any]) -> bool:
        """精准判断边是否应该被包含在子图中"""
        # 1. 检查边类型是否匹配种子要求
        if seed.required_edge_types and edge.edge_type in seed.required_edge_types:
            # 如果源节点或目标节点在已采样节点中，就包含这条边（用于扩展子图）
            if edge.source_node_id in sampled_nodes or edge.target_node_id in sampled_nodes:
                logger.debug(f"🎯 Edge {edge.edge_id} included: matches required edge type {edge.edge_type} and connected to sampled nodes")
                return True
            else:
                logger.debug(f"🎯 Edge {edge.edge_id} excluded: matches required edge type {edge.edge_type} but not connected to sampled nodes")
                return False
        
        # 2. 检查是否是导航边且种子需要导航
        if edge.edge_type == EdgeType.NAV_TO:
            needs_nav = self._does_seed_need_navigation(seed)
            if needs_nav:
                # 如果源节点或目标节点在已采样节点中，就包含这条边（用于扩展子图）
                if edge.source_node_id in sampled_nodes or edge.target_node_id in sampled_nodes:
                    logger.debug(f"🎯 Edge {edge.edge_id} included: NAV_TO edge connected to sampled nodes")
                    return True
                else:
                    logger.debug(f"🎯 Edge {edge.edge_id} excluded: NAV_TO edge not connected to sampled nodes")
                    return False
            else:
                logger.debug(f"🎯 Edge {edge.edge_id} excluded: NAV_TO edge but seed doesn't need navigation")
                return False
        
        # 3. 检查是否是内容包含关系
        if edge.edge_type == EdgeType.CONTAINS:
            # 如果源节点或目标节点在已采样节点中，就包含这条边（用于扩展子图）
            if edge.source_node_id in sampled_nodes or edge.target_node_id in sampled_nodes:
                logger.debug(f"🎯 Edge {edge.edge_id} included: CONTAINS edge connected to sampled nodes")
                return True
            else:
                logger.debug(f"🎯 Edge {edge.edge_id} excluded: CONTAINS edge not connected to sampled nodes")
                return False
        
        # 4. 检查是否是功能性边（如控制、填充等）
        functional_edge_types = {EdgeType.CONTROLS, EdgeType.FILLS, EdgeType.FILTERS, EdgeType.OPENS}
        if edge.edge_type in functional_edge_types:
            # 检查是否与已采样的节点形成有意义的连接
            if self._is_functional_edge_relevant(edge, seed, sampled_nodes):
                logger.debug(f"🎯 Edge {edge.edge_id} included: functional edge {edge.edge_type} is relevant")
                return True
            else:
                logger.debug(f"🎯 Edge {edge.edge_id} excluded: functional edge {edge.edge_type} not relevant")
                return False
        
        # 5. 默认不包含，除非有明确理由
        logger.debug(f"🎯 Edge {edge.edge_id} excluded: no clear reason to include {edge.edge_type}")
        return False
    
    def _is_functional_edge_relevant(self, edge: Any, seed: TaskSeedPattern, sampled_nodes: Dict[str, Any]) -> bool:
        """判断功能性边是否与种子相关"""
        # 如果源节点或目标节点在已采样节点中，就包含这条边（用于扩展子图）
        if edge.source_node_id not in sampled_nodes and edge.target_node_id not in sampled_nodes:
            return False
        
        # 检查边的端点是否与种子要求的节点类型匹配
        source_node = sampled_nodes.get(edge.source_node_id)
        target_node = sampled_nodes.get(edge.target_node_id)

        # 检查节点是否存在
        if source_node is None or target_node is None:
            return False

        # 检查是否形成种子要求的模式
        if seed.core_slots:
            source_type = source_node.node_type
            target_type = target_node.node_type
            
            # 检查是否匹配核心槽位要求
            for slot_name, required_type in seed.core_slots.items():
                if isinstance(required_type, list):
                    for item_type in required_type:
                        if source_type == item_type or target_type == item_type:
                            logger.debug(f"🎯 Functional edge relevant: matches core slot {slot_name} ({item_type})")
                            return True
                elif source_type == required_type or target_type == required_type:
                    logger.debug(f"🎯 Functional edge relevant: matches core slot {slot_name} ({required_type})")
                    return True
        
        # 检查是否与种子名称描述的功能匹配
        seed_name_lower = seed.name.lower()
        if edge.edge_type == EdgeType.CONTROLS and 'control' in seed_name_lower:
            return True
        if edge.edge_type == EdgeType.FILLS and 'fill' in seed_name_lower:
            return True
        if edge.edge_type == EdgeType.FILTERS and 'filter' in seed_name_lower:
            return True
        
        return False
    
    
    def _calculate_node_distance(self, task_graph: TaskGraph, node_a: str, node_b: str) -> int:
        """计算两个节点之间的距离（BFS）"""
        if node_a == node_b:
            return 0
        
        queue = deque([(node_a, 0)])
        visited = {node_a}
        
        while queue:
            current_node, distance = queue.popleft()
            
            for edge in task_graph.edges.values():
                if edge.source_node_id == current_node and edge.target_node_id not in visited:
                    if edge.target_node_id == node_b:
                        return distance + 1
                    visited.add(edge.target_node_id)
                    queue.append((edge.target_node_id, distance + 1))
                
                if edge.target_node_id == current_node and edge.source_node_id not in visited:
                    if edge.source_node_id == node_b:
                        return distance + 1
                    visited.add(edge.source_node_id)
                    queue.append((edge.source_node_id, distance + 1))
        
        return float('inf')  # 不可达
    
    
    # ==================== 重构后的强绑定采样方法 ====================
    
    def _analyze_graph_structure(self, task_graph: TaskGraph) -> None:
        """分析图结构，记录节点和边类型分布"""
        # 节点类型分布
        node_type_counts = {}
        for node in task_graph.nodes.values():
            if node is None:
                continue
            node_type = node.node_type.value
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        
        logger.info(f"🎯 Node type distribution: {node_type_counts}")
        
        # 边类型分布
        edge_type_counts = {}
        for edge in task_graph.edges.values():
            edge_type = edge.edge_type.value
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        logger.info(f"🎯 Edge type distribution: {edge_type_counts}")
    
    def _get_seed_anchor_nodes(self, seed: TaskSeedPattern) -> Set[Union[NodeType, List[NodeType]]]:
        """获取种子必需的节点类型（锚点）"""
        anchor_nodes = set()
        
        # 1. 核心槽位节点类型
        if seed.core_slots:
            for slot_name, slot_value in seed.core_slots.items():
                if isinstance(slot_value, list):
                    # 如果是列表，将整个列表作为一个要求
                    anchor_nodes.add(tuple(slot_value))
                else:
                    # 如果是单个类型，直接添加
                    anchor_nodes.add(slot_value)
        
        # 2. 根据种子类型添加特定锚点
        if seed.seed_type == TaskSeedType.BUSINESS_SEARCH_FILTER:
            anchor_nodes.update([NodeType.SEARCH_BOX, NodeType.BUTTON, NodeType.RESULT_ITEM])
        elif seed.seed_type == TaskSeedType.MULTI_HOP_NAVIGATION:
            anchor_nodes.update([NodeType.NAVIGATION, NodeType.LINK, NodeType.BUTTON])
        elif seed.seed_type == TaskSeedType.USER_NAVIGATION:
            anchor_nodes.update([NodeType.USER_DATA, NodeType.NAVIGATION, NodeType.PAGE])
        elif seed.seed_type == TaskSeedType.PRODUCT_NAVIGATION:
            anchor_nodes.update([NodeType.PRODUCT_DATA, NodeType.NAVIGATION, NodeType.PAGE])
        elif seed.seed_type == TaskSeedType.ORDER_NAVIGATION:
            anchor_nodes.update([NodeType.ORDER_DATA, NodeType.NAVIGATION, NodeType.PAGE])
        elif seed.seed_type == TaskSeedType.MIXED_DATA_NAVIGATION:
            anchor_nodes.update([NodeType.NAVIGATION, NodeType.PAGE])
            # 混合数据类型需要至少两种不同的数据类型
            anchor_nodes.update([NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA, NodeType.ORDER_DATA])
        
        # 3. 根据种子名称推断锚点
        seed_name_lower = seed.name.lower()
        if 'form' in seed_name_lower:
            anchor_nodes.add(NodeType.FORM)
        if 'search' in seed_name_lower:
            anchor_nodes.add(NodeType.SEARCH_BOX)
        if 'navigation' in seed_name_lower:
            anchor_nodes.add(NodeType.NAVIGATION)
        if 'table' in seed_name_lower:
            anchor_nodes.add(NodeType.TABLE)
        # 新增：根据数据类型推断锚点
        if 'user' in seed_name_lower:
            anchor_nodes.add(NodeType.USER_DATA)
        if 'product' in seed_name_lower:
            anchor_nodes.add(NodeType.PRODUCT_DATA)
        if 'order' in seed_name_lower:
            anchor_nodes.add(NodeType.ORDER_DATA)
        if 'business' in seed_name_lower:
            anchor_nodes.add(NodeType.BUSINESS_DATA)
        if 'mixed' in seed_name_lower:
            # 混合数据类型需要多种数据类型支持
            anchor_nodes.update([NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA, NodeType.ORDER_DATA])
        
        logger.debug(f"🎯 Anchor nodes for {seed.name}: {anchor_nodes}")
        return anchor_nodes
    
    def _get_recommended_strategies_for_seed(self, seed: TaskSeedPattern) -> List[SamplingStrategy]:
        """智能策略选择 - 根据种子类型和特征推荐最适合的采样策略 - 修复版本"""
        recommended_strategies = []
        
        # 获取种子分类
        seed_category = getattr(seed, 'seed_category', 'interaction')
        
        # 1. 基于种子分类的策略选择
        if seed_category == "business":
            # 业务种子优先使用目标导向和关系中心策略
            recommended_strategies.extend([
                SamplingStrategy.GOAL_CONDITIONED,  # 业务任务通常是目标导向的
                SamplingStrategy.RELATION_CENTRIC,  # 业务数据间的关系很重要
                SamplingStrategy.ELEMENT_CENTRIC,   # 业务元素是核心
            ])
        else:
            # 交互种子使用元素中心和功能模块策略
            recommended_strategies.extend([
                SamplingStrategy.ELEMENT_CENTRIC,   # 交互元素是核心
                SamplingStrategy.FUNCTIONAL_MODULE, # 功能模块导向
                SamplingStrategy.GOAL_CONDITIONED,  # 目标导向作为备选
            ])
        
        # 2. 基于种子名称的智能匹配
        seed_name_lower = seed.name.lower()
        
        if 'search' in seed_name_lower or 'filter' in seed_name_lower:
            recommended_strategies.extend([
                SamplingStrategy.ELEMENT_CENTRIC,  # 搜索框和过滤器通常是关键元素
                SamplingStrategy.GOAL_CONDITIONED,  # 目标导向的搜索任务
            ])
        
        elif 'form' in seed_name_lower or 'filling' in seed_name_lower:
            recommended_strategies.extend([
                SamplingStrategy.ELEMENT_CENTRIC,  # 表单元素是核心
                SamplingStrategy.RELATION_CENTRIC,  # 表单字段间的关系很重要
            ])
        
        elif 'navigation' in seed_name_lower or 'nav' in seed_name_lower:
            recommended_strategies.extend([
                SamplingStrategy.RELATION_CENTRIC,  # 导航关系是核心
                SamplingStrategy.GOAL_CONDITIONED,  # 目标导向的导航
            ])
        
        elif 'data' in seed_name_lower or 'extraction' in seed_name_lower:
            recommended_strategies.extend([
                SamplingStrategy.RELATION_CENTRIC,  # 数据关系很重要
                SamplingStrategy.ELEMENT_CENTRIC,  # 数据元素是核心
            ])
        
        elif 'comparison' in seed_name_lower or 'compare' in seed_name_lower:
            recommended_strategies.extend([
                SamplingStrategy.RELATION_CENTRIC,  # 比较关系是核心
                SamplingStrategy.GOAL_CONDITIONED,  # 目标导向的比较
            ])
        
        # 新增：通用交互元素的策略匹配
        elif any(keyword in seed_name_lower for keyword in ['button', 'menu', 'tab', 'modal', 'toast', 'breadcrumb', 'pagination', 'expand', 'scroll']):
            recommended_strategies.extend([
                SamplingStrategy.ELEMENT_CENTRIC,  # 交互元素是核心
                SamplingStrategy.FUNCTIONAL_MODULE,  # 功能模块导向
                SamplingStrategy.RELATION_CENTRIC,  # 交互关系也很重要
            ])
        
        # 3. 基于种子类型的精确匹配
        if seed.seed_type in [TaskSeedType.MULTI_HOP_NAVIGATION, TaskSeedType.BASIC_NAVIGATION]:
            recommended_strategies.extend([
                SamplingStrategy.RELATION_CENTRIC,
                SamplingStrategy.GOAL_CONDITIONED,
            ])

        elif seed.seed_type in [TaskSeedType.BUSINESS_SEARCH_FILTER]:
            recommended_strategies.extend([
                SamplingStrategy.ELEMENT_CENTRIC,
                SamplingStrategy.RELATION_CENTRIC,
            ])

        # 4. 基于种子复杂度的策略调整
        complexity = self._assess_seed_complexity(seed)
        if complexity == "high":
            # 复杂种子优先使用高效策略
            recommended_strategies = [SamplingStrategy.ELEMENT_CENTRIC] + recommended_strategies
        elif complexity == "medium":
            # 中等复杂度使用平衡策略
            if SamplingStrategy.ELEMENT_CENTRIC not in recommended_strategies:
                recommended_strategies.insert(0, SamplingStrategy.ELEMENT_CENTRIC)
        
        # 5. 去重并保持顺序
        seen = set()
        unique_strategies = []
        for strategy in recommended_strategies:
            if strategy not in seen:
                seen.add(strategy)
                unique_strategies.append(strategy)
        
        # 6. 如果没有推荐策略，使用智能默认策略
        if not unique_strategies:
            unique_strategies = self._get_default_strategies_for_seed(seed)
        
        # 7. 限制策略数量，提高效率
        max_strategies = 3
        if len(unique_strategies) > max_strategies:
            unique_strategies = unique_strategies[:max_strategies]
        
        logger.debug(f"🎯 Smart strategy selection for {seed.name} (category: {seed_category}): {[s.value for s in unique_strategies]} (complexity: {complexity})")
        return unique_strategies
    
    def _assess_seed_complexity(self, seed: TaskSeedPattern) -> str:
        """评估种子复杂度"""
        complexity_score = 0
        
        # 1. 基于核心槽位数量
        complexity_score += len(seed.core_slots) * 2
        
        # 2. 基于可选槽位数量
        complexity_score += len(seed.optional_slots)
        
        # 3. 基于必需边类型数量
        if seed.required_edge_types:
            complexity_score += len(seed.required_edge_types)
        
        # 4. 基于种子类型
        if seed.seed_type in [TaskSeedType.MULTI_HOP_NAVIGATION]:
            complexity_score += 3
        elif seed.seed_type in [TaskSeedType.BUSINESS_SEARCH_FILTER]:
            complexity_score += 2
        
        # 5. 基于种子名称关键词
        seed_name_lower = seed.name.lower()
        if any(keyword in seed_name_lower for keyword in ['multi', 'complex', 'advanced']):
            complexity_score += 2
        elif any(keyword in seed_name_lower for keyword in ['simple', 'basic']):
            complexity_score -= 1
        
        # 6. 分类复杂度
        if complexity_score >= 8:
            return "high"
        elif complexity_score >= 4:
            return "medium"
        else:
            return "low"
    
    def _get_default_strategies_for_seed(self, seed: TaskSeedPattern) -> List[SamplingStrategy]:
        """为种子获取默认策略"""
        # 根据种子复杂度选择默认策略
        complexity = self._assess_seed_complexity(seed)
        
        if complexity == "high":
            return [SamplingStrategy.ELEMENT_CENTRIC, SamplingStrategy.RELATION_CENTRIC]
        elif complexity == "medium":
            return [SamplingStrategy.ELEMENT_CENTRIC, SamplingStrategy.GOAL_CONDITIONED]
        else:
            return [SamplingStrategy.ELEMENT_CENTRIC]
    
    def _execute_seed_bound_sampling(self, task_graph: TaskGraph, seed: TaskSeedPattern, 
                                    required_node_types: Set[NodeType], 
                                    recommended_strategies: List[SamplingStrategy]) -> List[SubgraphSample]:
        """简化的种子绑定采样 - 直接基于策略采样"""
        subgraphs = []
        
        logger.info(f"🎯 Executing simple seed-bound sampling for {seed.name}")
        
        # 直接基于推荐策略进行采样，不进行复杂的锚点补充
        for strategy in recommended_strategies:
            try:
                if strategy == SamplingStrategy.ELEMENT_CENTRIC:
                    strategy_subgraphs = self._simple_element_centric_sampling(task_graph, seed)
                elif strategy == SamplingStrategy.GOAL_CONDITIONED:
                    strategy_subgraphs = self._simple_goal_conditioned_sampling(task_graph, seed)
                elif strategy == SamplingStrategy.RELATION_CENTRIC:
                    strategy_subgraphs = self._simple_relation_centric_sampling(task_graph, seed)
                elif strategy == SamplingStrategy.CURRICULUM_RADIUS:
                    strategy_subgraphs = self._curriculum_radius_sampling(task_graph, [seed])
                elif strategy == SamplingStrategy.FUNCTIONAL_MODULE:
                    strategy_subgraphs = self._functional_module_sampling(task_graph, [seed])
                else:
                    # 默认使用元素中心采样
                    strategy_subgraphs = self._simple_element_centric_sampling(task_graph, seed)
                
                subgraphs.extend(strategy_subgraphs)
                logger.info(f"🎯 Strategy {strategy.value} generated {len(strategy_subgraphs)} subgraphs")
                
                # 限制每个策略生成的子图数量
                if len(subgraphs) >= 10:
                    break
                    
            except Exception as e:
                logger.error(f"🎯 Strategy {strategy.value} failed: {e}")
                continue
        
        logger.info(f"🎯 Total subgraphs generated for {seed.name}: {len(subgraphs)}")
        return subgraphs
    
    def _simple_element_centric_sampling(self, task_graph: TaskGraph, seed: TaskSeedPattern) -> List[SubgraphSample]:
        """简化的元素中心采样"""
        subgraphs = []
        
        # 找到满足种子要求的节点
        target_nodes = self._find_nodes_for_seed(task_graph, seed)
        logger.debug(f"🎯 Found {len(target_nodes)} target nodes for simple element-centric sampling")
        
        # 限制处理数量
        max_attempts = min(len(target_nodes), 5)
        
        for i, center_node_id in enumerate(target_nodes[:max_attempts]):
            center_node = task_graph.nodes.get(center_node_id)
            if center_node is None:
                continue
            
            # 使用简单的BFS采样
            subgraph = self._bfs_sampling(task_graph, center_node_id, seed)
            
            if subgraph and self._validate_subgraph(subgraph, seed):
                subgraphs.append(subgraph)
                logger.debug(f"🎯 Added simple element-centric subgraph with {len(subgraph.nodes)} nodes")
        
        return subgraphs
    
    def _simple_goal_conditioned_sampling(self, task_graph: TaskGraph, seed: TaskSeedPattern) -> List[SubgraphSample]:
        """简化的目标条件采样"""
        subgraphs = []
        
        # 找到满足种子要求的节点
        target_nodes = self._find_nodes_for_seed(task_graph, seed)
        logger.debug(f"🎯 Found {len(target_nodes)} target nodes for simple goal-conditioned sampling")
        
        # 限制处理数量
        max_attempts = min(len(target_nodes), 5)
        
        for i, target_node_id in enumerate(target_nodes[:max_attempts]):
            # 使用简单的反向采样
            subgraph = self._simple_backward_sampling(task_graph, target_node_id, seed)
            
            if subgraph and self._validate_subgraph(subgraph, seed):
                subgraphs.append(subgraph)
                logger.debug(f"🎯 Added simple goal-conditioned subgraph with {len(subgraph.nodes)} nodes")
        
        return subgraphs
    
    def _simple_relation_centric_sampling(self, task_graph: TaskGraph, seed: TaskSeedPattern) -> List[SubgraphSample]:
        """简化的关系中心采样"""
        subgraphs = []
        
        # 找到满足种子要求的节点
        target_nodes = self._find_nodes_for_seed(task_graph, seed)
        logger.debug(f"🎯 Found {len(target_nodes)} target nodes for simple relation-centric sampling")
        
        # 限制处理数量
        max_attempts = min(len(target_nodes), 5)
        
        for i, target_node_id in enumerate(target_nodes[:max_attempts]):
            # 使用简单的反向采样
            subgraph = self._simple_backward_sampling(task_graph, target_node_id, seed)
            
            if subgraph and self._validate_subgraph(subgraph, seed):
                subgraphs.append(subgraph)
                logger.debug(f"🎯 Added simple relation-centric subgraph with {len(subgraph.nodes)} nodes")
        
        return subgraphs
    
    def _simple_backward_sampling(self, task_graph: TaskGraph, target_node_id: str, seed: TaskSeedPattern) -> Optional[SubgraphSample]:
        """简化的反向采样"""
        target_node = task_graph.nodes.get(target_node_id)
        if target_node is None:
            return None
        
        sampled_nodes = {target_node_id: target_node}
        sampled_edges = {}
        
        # 简单的反向BFS
        queue = deque([target_node_id])
        visited = {target_node_id}
        
        while queue and len(sampled_nodes) < self.config.max_nodes:
            current_node_id = queue.popleft()
            
            # 限制深度
            if len(visited) > 10:  # 限制访问的节点数量
                break
            
            for edge in task_graph.edges.values():
                if edge.target_node_id == current_node_id and edge.source_node_id not in visited:
                    # 简单的边包含判断
                    if self._should_include_edge_for_seed(edge, seed, sampled_nodes):
                        sampled_edges[edge.edge_id] = edge
                        source_node = task_graph.nodes.get(edge.source_node_id)
                        if source_node is not None:
                            sampled_nodes[edge.source_node_id] = source_node
                        visited.add(edge.source_node_id)
                        queue.append(edge.source_node_id)
        
        if len(sampled_nodes) >= self.config.min_nodes:
            return SubgraphSample(
                nodes=sampled_nodes,
                edges=sampled_edges,
                center_node=target_node_id,
                radius=3,
                strategy=SamplingStrategy.GOAL_CONDITIONED,
                task_seed=seed
            )
        
        return None
    
    def _goal_conditioned_sampling_with_anchors(self, task_graph: TaskGraph, seed: TaskSeedPattern, 
                                              required_node_types: Set[NodeType]) -> List[SubgraphSample]:
        """带锚点约束的目标条件采样"""
        subgraphs = []
        
        # 找到满足种子要求的节点
        target_nodes = self._find_nodes_for_seed(task_graph, seed)
        logger.debug(f"🎯 Found {len(target_nodes)} target nodes for seed {seed.name}")
        
        if not target_nodes:
            return subgraphs
        
        for target_node_id in target_nodes:
            # 检查目标节点是否包含必需的锚点类型
            target_node = task_graph.nodes.get(target_node_id)
            if target_node is None:
                logger.warning(f"🎯 Target node {target_node_id} is None, skipping")
                continue
            if target_node.node_type in required_node_types:
                logger.debug(f"🎯 Target node {target_node_id} contains required anchor type {target_node.node_type}")
                
                # 执行反向采样，确保包含锚点
                subgraph = self._backward_sampling_with_anchors(
                    task_graph, target_node_id, seed, required_node_types
                )
                
                if subgraph and self._validate_subgraph_with_anchors(subgraph, seed, required_node_types):
                    subgraphs.append(subgraph)
                    logger.debug(f"🎯 Added anchor-validated subgraph with {len(subgraph.nodes)} nodes")
        
        return subgraphs
    
    def _backward_sampling_with_anchors(self, task_graph: TaskGraph, target_node_id: str, 
                                       seed: TaskSeedPattern, required_node_types: Set[NodeType]) -> Optional[SubgraphSample]:
        """带锚点约束的反向采样"""
        target_node = task_graph.nodes.get(target_node_id)
        if target_node is None:
            logger.warning(f"🎯 Target node {target_node_id} is None in backward sampling")
            return None
        sampled_nodes = {target_node_id: target_node}
        sampled_edges = {}
        
        # 反向BFS，但限制跳数和类型
        queue = deque([(target_node_id, 0)])
        visited = {target_node_id}
        
        while queue and len(sampled_nodes) < self.config.max_nodes:
            current_node_id, distance = queue.popleft()
            
            # 跳数限制：最多3跳
            if distance >= 3:
                continue
            
            for edge in task_graph.edges.values():
                if edge.target_node_id == current_node_id and edge.source_node_id not in visited:
                    # 精准判断边是否应该被包含
                    should_include_edge = self._should_include_edge_for_seed(edge, seed, sampled_nodes)
                    
                    if should_include_edge:
                        sampled_edges[edge.edge_id] = edge
                        source_node = task_graph.nodes.get(edge.source_node_id)
                        if source_node is not None:
                            sampled_nodes[edge.source_node_id] = source_node
                        visited.add(edge.source_node_id)
                        queue.append((edge.source_node_id, distance + 1))
                        
                        logger.debug(f"🎯 Backward sampling: included edge {edge.edge_id} ({edge.edge_type}) from {edge.source_node_id}")
        
        # 只有在锚点验证失败时才补充锚点
        if not self._validate_subgraph_with_anchors(
            SubgraphSample(
                nodes=sampled_nodes,
                edges=sampled_edges,
                center_node=target_node_id,
                radius=3,
                strategy=SamplingStrategy.GOAL_CONDITIONED,
                task_seed=seed
            ), seed, required_node_types
        ):
            # 锚点验证失败，尝试补充缺失的锚点
            self._ensure_anchor_nodes_present(task_graph, sampled_nodes, sampled_edges, required_node_types)
            
            # 再次验证
            if not self._validate_subgraph_with_anchors(
                SubgraphSample(
                    nodes=sampled_nodes,
                    edges=sampled_edges,
                    center_node=target_node_id,
                    radius=3,
                    strategy=SamplingStrategy.GOAL_CONDITIONED,
                    task_seed=seed
                ), seed, required_node_types
            ):
                logger.debug(f"🎯 Subgraph still invalid after anchor supplementation")
                return None
        
        if len(sampled_nodes) >= self.config.min_nodes:
            return SubgraphSample(
                nodes=sampled_nodes,
                edges=sampled_edges,
                center_node=target_node_id,
                radius=3,
                strategy=SamplingStrategy.GOAL_CONDITIONED,
                task_seed=seed
            )
        
        return None
    
    def _does_seed_need_navigation(self, seed: TaskSeedPattern) -> bool:
        """判断种子是否需要导航功能"""
        # 根据种子名称和类型判断是否需要导航
        seed_name_lower = seed.name.lower()
        seed_type = seed.seed_type
        
        # 导航相关的关键词
        nav_keywords = ['navigation', 'navigate', 'nav', 'menu', 'link', 'page', 'site']
        
        # 检查种子名称是否包含导航关键词
        if any(keyword in seed_name_lower for keyword in nav_keywords):
            return True
        
        # 检查种子类型
        if seed_type in [TaskSeedType.BUSINESS_NAVIGATION, TaskSeedType.USER_NAVIGATION, 
                        TaskSeedType.PRODUCT_NAVIGATION, TaskSeedType.ORDER_NAVIGATION, 
                        TaskSeedType.MIXED_DATA_NAVIGATION]:
            return True
        
        return False
    
    def _ensure_anchor_nodes_present(self, task_graph: TaskGraph, sampled_nodes: Dict[str, Any], 
                                    sampled_edges: Dict[str, Any], required_node_types: Set[NodeType], max_attempts: int = 3) -> None:
        """确保子图包含所有必需的锚点节点，带循环检测"""
        # 循环检测：记录尝试次数
        attempt_count = 0
        
        while attempt_count < max_attempts:
            attempt_count += 1
            current_node_types = {node.node_type for node in sampled_nodes.values() if node is not None}
        
        # 处理列表类型的锚点要求
        processed_required_types = set()
        for anchor_type in required_node_types:
            if isinstance(anchor_type, list):
                # 如果是列表，只要满足其中一种类型即可
                processed_required_types.update(anchor_type)
            else:
                processed_required_types.add(anchor_type)
        
            # 检查是否满足锚点要求 - 支持类型映射
        missing_anchor_types = set()
        for required_type in processed_required_types:
            # 检查是否已经满足该类型要求
            if required_type not in current_node_types:
                    # 检查是否有兼容的类型可以满足要求
                    compatible_types = self._get_compatible_node_types(required_type)
                    if not any(compatible_type in current_node_types for compatible_type in compatible_types):
                        missing_anchor_types.add(required_type)
        
            if not missing_anchor_types:
                # 所有锚点要求都满足，退出循环
                break
            
            logger.debug(f"🎯 Missing anchor types (attempt {attempt_count}): {missing_anchor_types}")
            
            # 尝试添加缺失的锚点节点
            added_any = False
            
            for anchor_type in missing_anchor_types:
                # 处理列表类型的锚点要求
                if isinstance(anchor_type, list):
                    # 如果是列表类型，只要找到其中一种类型即可
                    for item_type in anchor_type:
                        anchor_nodes = [node_id for node_id, node in task_graph.nodes.items() 
                                      if node is not None and node.node_type == item_type and node_id not in sampled_nodes]
                        
                        if anchor_nodes:
                            # 选择最接近的锚点节点
                            best_anchor_node = self._find_closest_anchor_node(
                                task_graph, sampled_nodes, anchor_nodes
                            )
                            
                            if best_anchor_node:
                                anchor_node = task_graph.nodes.get(best_anchor_node)
                                if anchor_node is not None:
                                    sampled_nodes[best_anchor_node] = anchor_node
                                logger.debug(f"🎯 Added missing anchor node {best_anchor_node} (type: {item_type}) from list {anchor_type}")
                                
                                # 添加相关的边
                                self._add_anchor_related_edges(task_graph, best_anchor_node, sampled_nodes, sampled_edges)
                                
                                # 验证边是否真正添加成功
                                logger.debug(f"🎯 Added anchor node {best_anchor_node} to subgraph")
                                added_any = True
                                
                                # 找到一个就够了，跳出内层循环
                                break
                else:
                    # 单个类型的处理 - 支持兼容类型
                    compatible_types = self._get_compatible_node_types(anchor_type)
                    anchor_nodes = []
                    
                    # 优先寻找精确匹配的类型
                    for node_id, node in task_graph.nodes.items():
                        if node is not None and node.node_type == anchor_type and node_id not in sampled_nodes:
                            anchor_nodes.append(node_id)
                    
                    # 如果没有精确匹配，寻找兼容类型
                    if not anchor_nodes:
                        for compatible_type in compatible_types:
                            if compatible_type != anchor_type:  # 避免重复
                                for node_id, node in task_graph.nodes.items():
                                    if node is not None and node.node_type == compatible_type and node_id not in sampled_nodes:
                                        anchor_nodes.append(node_id)
                    
                    if anchor_nodes:
                        # 选择最接近的锚点节点
                        best_anchor_node = self._find_closest_anchor_node(
                            task_graph, sampled_nodes, anchor_nodes
                        )
                        
                        if best_anchor_node:
                            anchor_node = task_graph.nodes.get(best_anchor_node)
                            if anchor_node is not None:
                                sampled_nodes[best_anchor_node] = anchor_node
                            logger.debug(f"🎯 Added missing anchor node {best_anchor_node} (type: {anchor_node.node_type if anchor_node else 'unknown'}, compatible with {anchor_type})")
                            
                            # 添加相关的边
                            self._add_anchor_related_edges(task_graph, best_anchor_node, sampled_nodes, sampled_edges)
                            
                            # 验证边是否真正添加成功
                            logger.debug(f"🎯 Added anchor node {best_anchor_node} to subgraph")
                            added_any = True
            
            # 移除不可达的节点
            if sampled_nodes:
                reachable_nodes = self._get_reachable_nodes(sampled_nodes, sampled_edges)
                unreachable_nodes = set(sampled_nodes.keys()) - reachable_nodes
                for node_id in unreachable_nodes:
                    del sampled_nodes[node_id]
                    logger.debug(f"🎯 Removed unreachable node: {node_id}")
                
                edges_to_remove = []
                for edge_id, edge in sampled_edges.items():
                    if edge.source_node_id not in sampled_nodes or edge.target_node_id not in sampled_nodes:
                        edges_to_remove.append(edge_id)
                
                for edge_id in edges_to_remove:
                    del sampled_edges[edge_id]
                    logger.debug(f"🎯 Removed orphaned edge: {edge_id}")
            
            # 如果没有添加任何节点，退出循环
            if not added_any:
                logger.warning(f"🎯 No anchor nodes could be added after attempt {attempt_count}")
                break
        
        # 循环结束检查
        if attempt_count >= max_attempts:
            logger.warning(f"🎯 Anchor supplementation reached max attempts ({max_attempts})")
    
    def _find_closest_anchor_node(self, task_graph: TaskGraph, sampled_nodes: Dict[str, Any], 
                                 anchor_nodes: List[str]) -> Optional[str]:
        """找到最接近已采样节点的锚点节点"""
        if not anchor_nodes:
            return None
        
        min_distance = float('inf')
        best_node = None
        
        for anchor_node in anchor_nodes:
            total_distance = 0
            for sampled_node_id in sampled_nodes:
                distance = self._calculate_node_distance(task_graph, anchor_node, sampled_node_id)
                total_distance += distance
            
            if total_distance < min_distance:
                min_distance = total_distance
                best_node = anchor_node
        
        return best_node
    
    def _validate_subgraph_connectivity(self, sampled_nodes: Dict[str, Any], sampled_edges: Dict[str, Any]) -> bool:
        """验证子图的连通性 - 放宽条件，允许锚点桥接"""
        if not sampled_nodes:
            return False
        
        if len(sampled_nodes) == 1:
            return True
        
        # 构建邻接表
        adjacency = {}
        for node_id in sampled_nodes:
            adjacency[node_id] = []
        
        for edge in sampled_edges.values():
            if edge.source_node_id in adjacency and edge.target_node_id in adjacency:
                adjacency[edge.source_node_id].append(edge.target_node_id)
                adjacency[edge.target_node_id].append(edge.source_node_id)
        
        # 检查连通性，但允许一些节点通过锚点桥接
        start_node = next(iter(sampled_nodes))
        visited = set()
        queue = deque([start_node])
        visited.add(start_node)
        
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # 严格要求完全连通性，确保所有节点都可达
        reachable_ratio = len(visited) / len(sampled_nodes)
        is_fully_connected = reachable_ratio >= 1.0  # 要求100%连通

        logger.debug(f"🎯 Connectivity check: {len(visited)}/{len(sampled_nodes)} nodes reachable (ratio: {reachable_ratio:.2f})")

        if not is_fully_connected:
            unreachable_nodes = set(sampled_nodes.keys()) - visited
            logger.warning(f"🎯 Subgraph not fully connected: {len(unreachable_nodes)} unreachable nodes: {list(unreachable_nodes)[:3]}...")
            return False

        return True

    def _ensure_subgraph_reachability(self, sampled_nodes: Dict[str, Any], sampled_edges: Dict[str, Any]) -> bool:
        """确保子图中所有节点都可达，如果有不可达节点则移除它们"""
        if not sampled_nodes or len(sampled_nodes) <= 1:
            return True

        # 构建邻接表
        adjacency = {}
        for node_id in sampled_nodes:
            adjacency[node_id] = []

        for edge in sampled_edges.values():
            if edge.source_node_id in adjacency and edge.target_node_id in adjacency:
                adjacency[edge.source_node_id].append(edge.target_node_id)
                adjacency[edge.target_node_id].append(edge.source_node_id)

        # 找到所有连通组件
        visited = set()
        components = []

        for node_id in sampled_nodes:
            if node_id not in visited:
                # 开始新的连通组件
                component = set()
                queue = deque([node_id])
                component.add(node_id)
                visited.add(node_id)

                while queue:
                    current = queue.popleft()
                    for neighbor in adjacency[current]:
                        if neighbor not in component:
                            component.add(neighbor)
                            visited.add(neighbor)
                            queue.append(neighbor)

                components.append(component)

        # 如果只有一个连通组件，所有节点都可达
        if len(components) == 1:
            return True

        # 如果有多个连通组件，选择最大的组件，移除其他组件
        largest_component = max(components, key=len)
        logger.debug(f"🎯 Found {len(components)} disconnected components, keeping largest with {len(largest_component)} nodes")

        # 移除不在最大组件中的节点
        nodes_to_remove = set(sampled_nodes.keys()) - largest_component
        for node_id in nodes_to_remove:
            del sampled_nodes[node_id]
            logger.debug(f"🎯 Removed unreachable node: {node_id}")

        # 移除与已删除节点相关的边
        edges_to_remove = []
        for edge_id, edge in sampled_edges.items():
            if edge.source_node_id not in sampled_nodes or edge.target_node_id not in sampled_nodes:
                edges_to_remove.append(edge_id)

        for edge_id in edges_to_remove:
            del sampled_edges[edge_id]
            logger.debug(f"🎯 Removed orphaned edge: {edge_id}")

        # 验证清理后的连通性
        return self._validate_subgraph_connectivity(sampled_nodes, sampled_edges)

    def _remove_unreachable_nodes(self, sampled_nodes: Dict[str, Any], sampled_edges: Dict[str, Any]) -> None:
        """移除不可达的节点和相关的边"""
        if not sampled_nodes:
            return
        
        # 构建邻接表
        adjacency = {}
        for node_id in sampled_nodes:
            adjacency[node_id] = []
        
        for edge in sampled_edges.values():
            if edge.source_node_id in adjacency and edge.target_node_id in adjacency:
                adjacency[edge.source_node_id].append(edge.target_node_id)
                adjacency[edge.target_node_id].append(edge.source_node_id)
        
        # 找到连通分量
        start_node = next(iter(sampled_nodes))
        visited = set()
        queue = deque([start_node])
        visited.add(start_node)
        
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # 移除不可达的节点
        unreachable_nodes = set(sampled_nodes.keys()) - visited
        for node_id in unreachable_nodes:
            del sampled_nodes[node_id]
            logger.debug(f"🎯 Removed unreachable node: {node_id}")
        
        # 移除相关的边
        edges_to_remove = []
        for edge_id, edge in sampled_edges.items():
            if edge.source_node_id not in sampled_nodes or edge.target_node_id not in sampled_nodes:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            del sampled_edges[edge_id]
            logger.debug(f"🎯 Removed orphaned edge: {edge_id}")
    
    def _add_anchor_related_edges(self, task_graph: TaskGraph, node_id: str, 
                                 sampled_nodes: Dict[str, Any], sampled_edges: Dict[str, Any]) -> None:
        """添加与锚点节点相关的边"""
        for edge in task_graph.edges.values():
            if (edge.source_node_id == node_id and edge.target_node_id in sampled_nodes) or \
               (edge.target_node_id == node_id and edge.source_node_id in sampled_nodes):
                if edge.edge_id not in sampled_edges:
                    sampled_edges[edge.edge_id] = edge
                    logger.debug(f"🎯 Added anchor-related edge {edge.edge_id} ({edge.edge_type})")
    
    def _validate_subgraph_with_anchors(self, subgraph: SubgraphSample, seed: TaskSeedPattern, 
                                       required_node_types: Set[NodeType]) -> bool:
        """简化的子图验证 - 只进行基础验证"""
        # 只进行基础验证，不进行复杂的锚点检查
        return self._validate_subgraph(subgraph, seed)
    
    def _get_compatible_node_types(self, required_type: NodeType) -> Set[NodeType]:
        """获取与指定类型兼容的节点类型"""
        compatible_types = {required_type}  # 包含自身
        
        # 定义类型兼容性映射
        compatibility_map = {
            NodeType.INPUT: {NodeType.SEARCH_BOX, NodeType.FILTER},
            NodeType.SEARCH_BOX: {NodeType.INPUT, NodeType.FILTER},
            NodeType.FILTER: {NodeType.INPUT, NodeType.SEARCH_BOX},
            NodeType.FORM: {NodeType.INPUT, NodeType.BUTTON, NodeType.SEARCH_BOX},
            NodeType.BUTTON: {NodeType.FORM, NodeType.INPUT},
            NodeType.CONTENT: {NodeType.TABLE, NodeType.LIST, NodeType.CARD, NodeType.DETAIL, NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA, NodeType.ORDER_DATA},
            NodeType.TABLE: {NodeType.CONTENT, NodeType.LIST, NodeType.BUSINESS_DATA},
            NodeType.LIST: {NodeType.CONTENT, NodeType.TABLE, NodeType.BUSINESS_DATA},
            NodeType.CARD: {NodeType.CONTENT, NodeType.LIST, NodeType.BUSINESS_DATA},
            NodeType.DETAIL: {NodeType.CONTENT, NodeType.BUSINESS_DATA},
            NodeType.NAVIGATION: {NodeType.LINK, NodeType.BUTTON},
            NodeType.LINK: {NodeType.NAVIGATION, NodeType.BUTTON},
            # 业务数据类型之间的兼容性
            NodeType.BUSINESS_DATA: {NodeType.CONTENT, NodeType.TABLE, NodeType.LIST, NodeType.CARD, NodeType.DETAIL},
            NodeType.USER_DATA: {NodeType.BUSINESS_DATA, NodeType.CONTENT},
            NodeType.PRODUCT_DATA: {NodeType.BUSINESS_DATA, NodeType.CONTENT},
            NodeType.ORDER_DATA: {NodeType.BUSINESS_DATA, NodeType.CONTENT},
        }
        
        if required_type in compatibility_map:
            compatible_types.update(compatibility_map[required_type])
        
        return compatible_types
    
    def _get_reachable_nodes(self, nodes: Dict[str, Any], edges: Dict[str, Any]) -> Set[str]:
        """获取从锚点可达的所有节点"""
        if not nodes or not edges:
            return set(nodes.keys()) if nodes else set()
        
        # 找到所有锚点（通常是INPUT, BUTTON, FORM等关键节点）
        anchor_nodes = set()
        for node_id, node in nodes.items():
            if hasattr(node, 'node_type') and node.node_type in [
                NodeType.INPUT, NodeType.BUTTON, NodeType.FORM, 
                NodeType.NAVIGATION, NodeType.SEARCH_BOX
            ]:
                anchor_nodes.add(node_id)
        
        # 如果没有找到锚点，返回所有节点
        if not anchor_nodes:
            return set(nodes.keys())
        
        # 从锚点开始BFS，找到所有可达节点
        reachable = set()
        queue = list(anchor_nodes)
        
        while queue:
            current_node = queue.pop(0)
            if current_node in reachable:
                continue
                
            reachable.add(current_node)
            
            # 找到与当前节点相连的所有边
            for edge_id, edge in edges.items():
                if edge.source_node_id == current_node and edge.target_node_id in nodes:
                    if edge.target_node_id not in reachable:
                        queue.append(edge.target_node_id)
                elif edge.target_node_id == current_node and edge.source_node_id in nodes:
                    if edge.source_node_id not in reachable:
                        queue.append(edge.source_node_id)
        
        return reachable
    
    def _enhanced_validation_and_deduplication(self, subgraphs: List[SubgraphSample]) -> List[SubgraphSample]:
        """增强验证和去重 - 移除过度严格的元路径去重"""
        if not subgraphs:
            return []
        
        # 1. 基础验证
        valid_subgraphs = []
        for subgraph in subgraphs:
            if subgraph.task_seed and self._validate_subgraph(subgraph, subgraph.task_seed):
                valid_subgraphs.append(subgraph)
        
        logger.info(f"🎯 Enhanced validation: {len(subgraphs)} -> {len(valid_subgraphs)} valid subgraphs")
        
        # 2. 直接使用宽松去重，避免过度去重问题
        unique_subgraphs = self._relaxed_deduplication(valid_subgraphs, self.config.max_total_subgraphs)
        logger.info(f"🎯 Relaxed deduplication: {len(valid_subgraphs)} -> {len(unique_subgraphs)} unique subgraphs")
        
        return unique_subgraphs
    
    
    def _relaxed_deduplication(self, subgraphs: List[SubgraphSample], max_subgraphs: int = 200) -> List[SubgraphSample]:
        """宽松去重 - 当严格去重结果太少时使用，支持内容多样性"""
        if not subgraphs:
            return []
        
        unique_subgraphs = []
        seen_basic_patterns = set()
        seen_content_patterns = set()
        
        for subgraph in subgraphs:
            # 1. 基本节点类型模式
            basic_pattern = "->".join(sorted([node.node_type.value for node in subgraph.nodes.values() if node is not None]))
            
            # 2. 内容模式（基于节点内容）
            content_pattern = self._generate_content_pattern(subgraph)
            
            # 3. 决定是否保留子图
            should_keep = False
            
            # 如果基本模式没见过，保留
            if basic_pattern not in seen_basic_patterns:
                should_keep = True
                seen_basic_patterns.add(basic_pattern)
                logger.debug(f"🎯 Relaxed deduplication: new basic pattern {basic_pattern}")
            
            # 如果内容模式没见过，也保留（即使基本模式相同）
            elif content_pattern not in seen_content_patterns:
                should_keep = True
                seen_content_patterns.add(content_pattern)
                logger.debug(f"🎯 Relaxed deduplication: new content pattern for {basic_pattern}")
            
            # 如果子图数量还很少，也保留
            elif len(unique_subgraphs) < max_subgraphs:  # 使用配置中的数量限制
                should_keep = True
                logger.debug(f"🎯 Relaxed deduplication: keeping for diversity (count: {len(unique_subgraphs)})")
            
            if should_keep:
                unique_subgraphs.append(subgraph)
        
        logger.info(f"🎯 Relaxed deduplication: {len(subgraphs)} -> {len(unique_subgraphs)} subgraphs")
        return unique_subgraphs
    
    def _generate_content_pattern(self, subgraph: SubgraphSample) -> str:
        """生成基于节点内容的模式，用于内容多样性检测"""
        if not subgraph.nodes:
            return ""
        
        content_signatures = []
        for node_id, node in subgraph.nodes.items():
            if node is None:
                continue
            
            # 提取节点的内容签名
            content_parts = []
            
            # 文本内容
            if hasattr(node, 'text_content') and node.text_content:
                content_parts.append(f"text:{node.text_content[:50]}")
            elif hasattr(node, 'text') and node.text:
                content_parts.append(f"text:{node.text[:50]}")
            
            # 值内容
            if hasattr(node, 'value') and node.value:
                content_parts.append(f"value:{str(node.value)[:50]}")
            
            # 属性内容
            if hasattr(node, 'properties') and node.properties:
                for key, value in node.properties.items():
                    if value and len(str(value)) > 0:
                        content_parts.append(f"{key}:{str(value)[:30]}")
            
            # 如果有内容，添加到签名中
            if content_parts:
                content_signatures.append(f"{node_id}:{'|'.join(content_parts)}")
        
        return "||".join(content_signatures)
    
    def _sort_by_executability_score(self, subgraphs: List[SubgraphSample]) -> List[SubgraphSample]:
        """按可执行性分数排序"""
        if not subgraphs:
            return []
        
        # 计算每个子图的可执行性分数
        scored_subgraphs = []
        for subgraph in subgraphs:
            score = self._calculate_executability_score(subgraph)
            scored_subgraphs.append((subgraph, score))
        
        # 按分数降序排序
        scored_subgraphs.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序后的子图
        sorted_subgraphs = [subgraph for subgraph, score in scored_subgraphs]
        
        logger.info(f"🎯 Sorted {len(sorted_subgraphs)} subgraphs by executability score")
        return sorted_subgraphs
    
    def _calculate_executability_score(self, subgraph: SubgraphSample) -> float:
        """计算子图的可执行性分数 - 优先选择结构完整、功能丰富的子图"""
        score = 0.0
        
        # 1. 业务数据覆盖度分数（权重最高）
        business_data_nodes = [node for node in subgraph.nodes.values() 
                             if node is not None and node.node_type in {NodeType.BUSINESS_DATA, NodeType.USER_DATA, 
                                                  NodeType.PRODUCT_DATA, NodeType.ORDER_DATA}]
        score += len(business_data_nodes) * 3.0  # 提高权重
        
        # 2. 功能模块多样性分数（新增）
        node_types = {node.node_type for node in subgraph.nodes.values() if node is not None}
        functional_modules = {
            NodeType.SEARCH_BOX, NodeType.INPUT, NodeType.FORM,  # 输入模块
            NodeType.BUTTON, NodeType.LINK, NodeType.NAVIGATION,  # 交互模块
            NodeType.TABLE, NodeType.RESULT_ITEM, NodeType.CONTENT_DATA,  # 展示模块
            NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA, NodeType.ORDER_DATA  # 数据模块
        }
        module_coverage = len(node_types.intersection(functional_modules))
        score += module_coverage * 2.0  # 功能模块越多分数越高
        
        # 3. 节点可用性分数
        for node in subgraph.nodes.values():
            if node is None:
                continue
            # 检查是否有可用的label/value
            if hasattr(node, 'text') and node.text and node.text.strip():
                score += 1.0
            if hasattr(node, 'value') and node.value:
                score += 1.0
        
        # 4. 结构完整性分数（提高权重）
        if len(subgraph.nodes) >= 4:  # 提高节点数量要求
            score += 5.0  # 基础结构分
        elif len(subgraph.nodes) >= 3:
            score += 3.0
        if len(subgraph.edges) >= 3:  # 提高边数量要求
            score += 4.0  # 连接性分
        elif len(subgraph.edges) >= 2:
            score += 2.0
        
        # 5. 交互元素丰富度分数（新增）
        interactive_nodes = [node for node in subgraph.nodes.values() 
                           if node is not None and node.node_type in {NodeType.BUTTON, NodeType.LINK, NodeType.NAVIGATION, NodeType.SEARCH_BOX, NodeType.INPUT}]
        score += len(interactive_nodes) * 2.0  # 交互元素越多分数越高
        
        # 6. 边类型多样性分数（新增）
        edge_types = {edge.edge_type for edge in subgraph.edges.values()}
        score += len(edge_types) * 1.5  # 边类型越多分数越高
        
        # 7. 业务逻辑完整性奖励（新增）
        # 如果同时包含输入、交互、数据、展示等模块，给予额外奖励
        if (NodeType.SEARCH_BOX in node_types or NodeType.INPUT in node_types) and \
           (NodeType.BUTTON in node_types or NodeType.LINK in node_types) and \
           (NodeType.BUSINESS_DATA in node_types or NodeType.USER_DATA in node_types or NodeType.PRODUCT_DATA in node_types):
            score += 5.0  # 完整的业务逻辑奖励
        
        logger.debug(f"🎯 Enhanced executability score for subgraph: {score}")
        return score
    
    # ==================== 补充缺失的锚点采样方法 ====================
    
    def _element_centric_sampling_with_anchors(self, task_graph: TaskGraph, seed: TaskSeedPattern, 
                                              required_node_types: Set[NodeType]) -> List[SubgraphSample]:
        """优化的元素中心采样 - 智能节点选择和扩展"""
        subgraphs = []
        
        # 1. 优先寻找能直接满足所有锚点要求的节点组合
        complete_anchor_sets = self._find_anchor_complete_node_sets(task_graph, seed, required_node_types)
        
        if complete_anchor_sets:
            logger.debug(f"🎯 Found {len(complete_anchor_sets)} complete anchor sets for {seed.name}")
            
            # 使用完整的锚点组合进行采样
            for anchor_set in complete_anchor_sets[:3]:  # 最多尝试3个组合
                subgraph = self._sample_from_anchor_set(task_graph, anchor_set, seed, required_node_types)
                if subgraph and self._validate_subgraph_with_anchors(subgraph, seed, required_node_types):
                    subgraphs.append(subgraph)
                    logger.debug(f"🎯 Added subgraph from complete anchor set: {len(subgraph.nodes)} nodes")
                    
                    if len(subgraphs) >= 3:  # 限制数量
                        break
        
        # 2. 如果完整锚点组合不够，使用传统的单节点采样
        if len(subgraphs) < 2:
            logger.debug(f"🎯 Falling back to traditional single-node sampling for {seed.name}")
            candidate_nodes = self._find_smart_candidate_nodes(task_graph, seed, required_node_types)
            logger.debug(f"🎯 Found {len(candidate_nodes)} smart candidate nodes")
            
            max_attempts = min(len(candidate_nodes), 10)  # 减少尝试次数
            
            for i, center_node_id in enumerate(candidate_nodes[:max_attempts]):
                center_node = task_graph.nodes.get(center_node_id)
                if center_node is None:
                    continue
            
                # 快速预检查：节点是否满足基本要求
                if not self._quick_node_check(center_node, seed, required_node_types):
                    continue
                
                subgraph = self._optimized_bfs_sampling(
                    task_graph, center_node_id, seed, required_node_types
                )
                
                if subgraph and self._validate_subgraph_with_anchors(subgraph, seed, required_node_types):
                    subgraphs.append(subgraph)
                    logger.debug(f"🎯 Added optimized subgraph with {len(subgraph.nodes)} nodes from {center_node_id}")
            
                    # 5. 限制子图数量，避免过度采样
                    if len(subgraphs) >= 10:  # 每个种子最多10个子图
                        break
        
        logger.info(f"🎯 Element-centric sampling completed: {len(subgraphs)} subgraphs for {seed.name}")
        return subgraphs
    
    def _find_smart_candidate_nodes(self, task_graph: TaskGraph, seed: TaskSeedPattern, 
                                   required_node_types: Set[NodeType]) -> List[str]:
        """智能选择候选节点 - 优先选择能直接满足锚点要求的节点组合"""
        candidates = []
        
        # 1. 优先选择包含必需锚点类型的节点
        for node_id, node in task_graph.nodes.items():
            if node is None:
                continue
                
            # 检查是否包含必需的锚点类型
            if node.node_type in required_node_types:
                candidates.append(node_id)
                continue
            
            # 2. 根据种子类型选择相关节点
            if self._is_node_relevant_for_seed(node, seed):
                candidates.append(node_id)
        
        # 3. 按相关性排序
        candidates.sort(key=lambda node_id: self._calculate_node_relevance(
            task_graph.nodes[node_id], seed, required_node_types
        ), reverse=True)
        
        return candidates
    
    def _find_anchor_complete_node_sets(self, task_graph: TaskGraph, seed: TaskSeedPattern, 
                                       required_node_types: Set[NodeType]) -> List[List[str]]:
        """寻找能直接满足所有锚点要求的节点组合，避免后续补充"""
        # 按类型分组所有节点
        nodes_by_type = {}
        for node_id, node in task_graph.nodes.items():
            if node is None:
                continue
            node_type = node.node_type
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node_id)
        
        # 寻找能覆盖所有必需类型的节点组合
        complete_sets = []
        
        # 对于每个必需的类型，找到对应的节点
        for required_type in required_node_types:
            compatible_types = self._get_compatible_node_types(required_type)
            available_nodes = []
            
            for compatible_type in compatible_types:
                if compatible_type in nodes_by_type:
                    available_nodes.extend(nodes_by_type[compatible_type])
            
            if not available_nodes:
                logger.debug(f"🎯 No nodes found for required type {required_type}")
                return []  # 如果某个必需类型没有对应节点，返回空列表
        
        # 生成节点组合（简化版本：选择每个类型的第一个节点）
        node_combination = []
        for required_type in required_node_types:
            compatible_types = self._get_compatible_node_types(required_type)
            for compatible_type in compatible_types:
                if compatible_type in nodes_by_type and nodes_by_type[compatible_type]:
                    node_combination.append(nodes_by_type[compatible_type][0])
                    break
        
        if len(node_combination) >= len(required_node_types):
            complete_sets.append(node_combination)
            logger.debug(f"🎯 Found complete anchor set: {node_combination}")
        
        return complete_sets
    
    def _sample_from_anchor_set(self, task_graph: TaskGraph, anchor_set: List[str], 
                               seed: TaskSeedPattern, required_node_types: Set[NodeType]) -> Optional[SubgraphSample]:
        """从完整的锚点集合开始采样子图"""
        if not anchor_set:
            return None
        
        # 初始化子图
        sampled_nodes = {}
        sampled_edges = {}
        
        # 添加所有锚点节点
        for node_id in anchor_set:
            node = task_graph.nodes.get(node_id)
            if node is not None:
                sampled_nodes[node_id] = node
        
        # 添加锚点之间的边
        for edge in task_graph.edges.values():
            if edge.source_node_id in sampled_nodes and edge.target_node_id in sampled_nodes:
                sampled_edges[edge.edge_id] = edge
        
        # 从锚点集合扩展，寻找相关节点
        queue = list(anchor_set)
        visited = set(anchor_set)
        
        while queue and len(sampled_nodes) < self.config.max_nodes:
            current_node_id = queue.pop(0)
            
            # 寻找相邻节点
            for edge in task_graph.edges.values():
                if edge.source_node_id == current_node_id and edge.target_node_id not in visited:
                    neighbor_id = edge.target_node_id
                    neighbor_node = task_graph.nodes.get(neighbor_id)
                    
                    if neighbor_node is not None:
                        # 检查是否应该包含这个邻居节点
                        if self._should_include_neighbor(neighbor_node, seed, required_node_types):
                            sampled_nodes[neighbor_id] = neighbor_node
                            sampled_edges[edge.edge_id] = edge
                            queue.append(neighbor_id)
                            visited.add(neighbor_id)
                
                elif edge.target_node_id == current_node_id and edge.source_node_id not in visited:
                    neighbor_id = edge.source_node_id
                    neighbor_node = task_graph.nodes.get(neighbor_id)
                    
                    if neighbor_node is not None:
                        # 检查是否应该包含这个邻居节点
                        if self._should_include_neighbor(neighbor_node, seed, required_node_types):
                            sampled_nodes[neighbor_id] = neighbor_node
                            sampled_edges[edge.edge_id] = edge
                            queue.append(neighbor_id)
                            visited.add(neighbor_id)
        
        # 验证连通性
        if not self._validate_subgraph_connectivity(sampled_nodes, sampled_edges):
            logger.debug(f"🎯 Subgraph from anchor set failed connectivity validation")
            return None
        
        if len(sampled_nodes) >= self.config.min_nodes:
            return SubgraphSample(
                nodes=sampled_nodes,
                edges=sampled_edges,
                center_node=anchor_set[0],  # 使用第一个锚点作为中心节点
                strategy=SamplingStrategy.ELEMENT_CENTRIC
            )
        
        return None
    
    def _should_include_neighbor(self, neighbor_node: Any, seed: TaskSeedPattern, 
                               required_node_types: Set[NodeType]) -> bool:
        """判断是否应该包含邻居节点"""
        if neighbor_node is None:
            return False
        
        # 优先包含必需类型的节点
        if neighbor_node.node_type in required_node_types:
            return True
        
        # 包含与种子相关的节点
        if self._is_node_relevant_for_seed(neighbor_node, seed):
            return True
        
        # 增强业务数据节点处理（对于业务种子）
        seed_category = getattr(seed, 'seed_category', 'interaction')
        if seed_category == "business":
            # 直接匹配业务数据类型
            business_types = {
                NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA,
                NodeType.ORDER_DATA, NodeType.CONTENT_DATA, NodeType.FINANCIAL_DATA,
                NodeType.LOCATION_DATA, NodeType.TIME_DATA
            }
            if neighbor_node.node_type in business_types:
                return True

            # 检查内容是否包含业务关键词（针对SuiteCRM等系统）
            text_content = getattr(neighbor_node.metadata, 'text_content', '').lower()
            business_keywords = [
                'account', 'contact', 'lead', 'opportunity', 'quote', 'invoice',
                'product', 'campaign', 'report', 'dashboard', 'calendar', 'meeting',
                'task', 'call', 'email', 'note', 'document', 'project'
            ]
            if any(keyword in text_content for keyword in business_keywords):
                return True

        return False
    
    def _is_node_relevant_for_seed(self, node: Any, seed: TaskSeedPattern) -> bool:
        """判断节点是否与种子相关"""
        if node is None:
            return False
        seed_name_lower = seed.name.lower()
        node_type = node.node_type.value.lower()
        
        # 根据种子名称匹配相关节点类型
        if 'search' in seed_name_lower and node_type in ['search_box', 'input', 'button']:
            return True
        elif 'form' in seed_name_lower and node_type in ['form', 'input', 'button', 'submit']:
            return True
        elif 'navigation' in seed_name_lower and node_type in ['navigation', 'link', 'button', 'menu']:
            return True
        elif 'data' in seed_name_lower and node_type in ['business_data', 'content', 'table', 'user_data', 'product_data', 'order_data']:
            return True
        elif 'comparison' in seed_name_lower and node_type in ['content', 'table', 'card', 'business_data']:
            return True
        elif 'business' in seed_name_lower and node_type in ['business_data', 'user_data', 'product_data', 'order_data', 'navigation', 'menu']:
            return True
        
        return False
    
    def _calculate_node_relevance(self, node: Any, seed: TaskSeedPattern, 
                                 required_node_types: Set[NodeType]) -> float:
        """计算节点与种子的相关性分数"""
        if node is None:
            return 0.0
        score = 0.0
        
        # 1. 锚点类型匹配
        if node.node_type in required_node_types:
            score += 10.0
        
        # 2. 种子类型匹配
        if self._is_node_relevant_for_seed(node, seed):
            score += 5.0
        
        # 3. 节点类型重要性
        high_importance_types = {'business_data', 'form', 'input', 'button', 'navigation'}
        if node.node_type.value.lower() in high_importance_types:
            score += 3.0

        # 4. 业务数据节点额外加分
        business_types = {'business_data', 'user_data', 'product_data', 'order_data', 'content_data'}
        if node.node_type.value.lower() in business_types:
            score += 4.0  # 业务数据节点更高优先级

        # 5. 内容相关性（针对SuiteCRM等系统）
        text_content = getattr(node.metadata if hasattr(node, 'metadata') else node, 'text_content', '').lower()
        seed_category = getattr(seed, 'seed_category', 'interaction')

        if seed_category == 'business':
            business_keywords = [
                'account', 'contact', 'lead', 'opportunity', 'quote', 'invoice',
                'product', 'campaign', 'report', 'dashboard', 'calendar', 'meeting'
            ]
            if any(keyword in text_content for keyword in business_keywords):
                score += 2.0
        
        return score
    
    def _quick_node_check(self, node: Any, seed: TaskSeedPattern, 
                         required_node_types: Set[NodeType]) -> bool:
        """快速节点检查 - 避免复杂的验证逻辑"""
        if node is None:
            return False
        # 基本检查：节点类型是否匹配
        if node.node_type in required_node_types:
            return True
        
        # 快速相关性检查
        if self._is_node_relevant_for_seed(node, seed):
            return True
        
        return False
    
    def _optimized_bfs_sampling(self, task_graph: TaskGraph, center_node_id: str, 
                               seed: TaskSeedPattern, required_node_types: Set[NodeType]) -> Optional[SubgraphSample]:
        """优化的BFS采样 - 更智能的扩展策略"""
        sampled_nodes = {center_node_id: task_graph.nodes[center_node_id]}
        sampled_edges = {}
        queue = deque([center_node_id])
        visited = {center_node_id}
        
        # 限制BFS深度和宽度，提高效率
        max_depth = 3
        max_nodes = self.config.max_nodes
        current_depth = 0
        
        while queue and len(sampled_nodes) < max_nodes and current_depth < max_depth:
            current_level_size = len(queue)
            current_depth += 1
            
            for _ in range(current_level_size):
                if not queue:
                    break
                    
                current_node_id = queue.popleft()
                
                # 智能扩展：优先选择相关边
                relevant_edges = self._find_relevant_edges(
                    task_graph, current_node_id, seed, required_node_types
                )
                
                # 动态调整扩展数量：优先选择高质量边，确保可达性
                max_expansions = 2 if len(sampled_nodes) < max_nodes // 2 else 1  # 前半部分允许更多扩展

                for edge in relevant_edges[:max_expansions]:
                    neighbor_id = edge.target_node_id if edge.source_node_id == current_node_id else edge.source_node_id

                    if neighbor_id not in visited and len(sampled_nodes) < max_nodes:
                        visited.add(neighbor_id)
                        sampled_nodes[neighbor_id] = task_graph.nodes[neighbor_id]
                        sampled_edges[edge.edge_id] = edge
                        queue.append(neighbor_id)

                        # 确保至少有一些功能性连接
                        if len([e for e in sampled_edges.values() if e.edge_type.value.lower() in ['controls', 'fills', 'opens']]) == 0:
                            # 如果还没有功能性边，继续寻找
                            break
        
        # 确保包含必需的锚点
        self._ensure_anchor_nodes_present(task_graph, sampled_nodes, sampled_edges, required_node_types)

        # 确保子图的可达性，移除不可达的节点
        if not self._ensure_subgraph_reachability(sampled_nodes, sampled_edges):
            logger.debug(f"🎯 Failed to ensure subgraph reachability for center node {center_node_id}")
            return None

        if len(sampled_nodes) >= 2:  # 至少需要2个节点
            return SubgraphSample(
                nodes=sampled_nodes,
                edges=sampled_edges,
                center_node=center_node_id,
                strategy=SamplingStrategy.ELEMENT_CENTRIC
            )
        
        return None
    
    def _find_relevant_edges(self, task_graph: TaskGraph, node_id: str,
                            seed: TaskSeedPattern, required_node_types: Set[NodeType]) -> List[Any]:
        """找到与节点相关的边 - 增强可达性保证"""
        relevant_edges = []

        for edge in task_graph.edges.values():
            if edge.source_node_id == node_id or edge.target_node_id == node_id:
                neighbor_id = edge.target_node_id if edge.source_node_id == node_id else edge.source_node_id

                # 检查邻居节点是否在图中存在
                if neighbor_id not in task_graph.nodes:
                    continue

                # 优先级排序：功能性边 > 结构边 > 其他边
                edge_type = edge.edge_type.value.lower()
                if edge_type in ['controls', 'fills', 'opens', 'nav_to']:
                    # 功能性边：最高优先级，确保交互可达性
                    relevant_edges.append((edge, 4, 'functional'))
                elif edge_type == 'contains':
                    # 包含边：中等优先级，确保结构完整性
                    relevant_edges.append((edge, 3, 'structural'))
                elif edge_type in ['refers_to', 'same_entity', 'supports_fact']:
                    # 语义边：中等优先级，确保语义连通性
                    relevant_edges.append((edge, 2, 'semantic'))
                else:
                    # 其他边：最低优先级
                    relevant_edges.append((edge, 1, 'other'))

        # 按优先级排序，确保高质量边优先选择
        relevant_edges.sort(key=lambda x: (x[1], x[2] == 'functional'), reverse=True)

        # 返回边对象列表
        return [edge for edge, _, _ in relevant_edges]
    
    def _relation_centric_sampling_with_anchors(self, task_graph: TaskGraph, seed: TaskSeedPattern, 
                                               required_node_types: Set[NodeType]) -> List[SubgraphSample]:
        """优化的关系中心采样 - 并行化处理"""
        subgraphs = []
        
        # 1. 智能选择目标节点
        target_nodes = self._find_smart_target_nodes(task_graph, seed, required_node_types)
        logger.debug(f"🎯 Found {len(target_nodes)} smart target nodes for relation-centric sampling")
        
        # 2. 限制处理数量，提高效率
        max_targets = min(len(target_nodes), 15)  # 最多处理15个目标节点
        
        # 3. 智能并发处理：根据目标节点数量决定是否使用并发
        if len(target_nodes) > 3 and self.config.enable_concurrency:
            # 使用受控的并发处理，限制线程数
            subgraphs = self._controlled_parallel_relation_sampling(
                task_graph, target_nodes[:max_targets], seed, required_node_types
            )
        else:
            # 串行处理（目标节点少时）
            for target_node_id in target_nodes[:max_targets]:
                subgraph = self._optimized_backward_sampling(
                    task_graph, target_node_id, seed, required_node_types
                )
                
                if subgraph and self._validate_subgraph_with_anchors(subgraph, seed, required_node_types):
                    subgraphs.append(subgraph)
        
                    # 限制子图数量
                    if len(subgraphs) >= 15:
                        break
        
        logger.info(f"🎯 Relation-centric sampling completed: {len(subgraphs)} subgraphs for {seed.name}")
        return subgraphs
    
    def _find_smart_target_nodes(self, task_graph: TaskGraph, seed: TaskSeedPattern, 
                                required_node_types: Set[NodeType]) -> List[str]:
        """智能选择目标节点 - 优先选择高价值节点"""
        target_nodes = []
        
        # 1. 优先选择包含必需锚点类型的节点
        for node_id, node in task_graph.nodes.items():
            if node is None:
                continue
                
            if node.node_type in required_node_types:
                target_nodes.append(node_id)
        
        # 2. 如果没有找到锚点节点，选择相关节点
        if not target_nodes:
            for node_id, node in task_graph.nodes.items():
                    
                if self._is_node_relevant_for_seed(node, seed):
                    target_nodes.append(node_id)
        
        # 3. 按重要性排序
        target_nodes.sort(key=lambda node_id: self._calculate_node_importance(
            task_graph.nodes[node_id], seed, required_node_types
        ), reverse=True)
        
        return target_nodes
    
    def _calculate_node_importance(self, node: Any, seed: TaskSeedPattern, 
                                  required_node_types: Set[NodeType]) -> float:
        """计算节点重要性分数"""
        if node is None:
            return 0.0
        score = 0.0
        
        # 1. 锚点类型匹配
        if node.node_type in required_node_types:
            score += 20.0
        
        # 2. 种子类型匹配
        if self._is_node_relevant_for_seed(node, seed):
            score += 10.0
        
        # 3. 节点类型重要性
        importance_map = {
            'business_data': 15.0,
            'form': 12.0,
            'input': 10.0,
            'button': 8.0,
            'navigation': 8.0,
            'content': 6.0,
            'table': 6.0
        }
        
        node_type_lower = node.node_type.value.lower()
        score += importance_map.get(node_type_lower, 3.0)
        
        return score
    
    def _controlled_parallel_relation_sampling(self, task_graph: TaskGraph, target_nodes: List[str], 
                                             seed: TaskSeedPattern, required_node_types: Set[NodeType]) -> List[SubgraphSample]:
        """受控的并行关系采样 - 限制线程数，避免资源竞争"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        subgraphs = []
        
        # 智能计算线程数：确保总线程数不超过系统限制
        import os
        cpu_count = os.cpu_count() or 4
        max_system_threads = cpu_count * 2  # 保守估计
        
        # 计算当前已使用的线程数（估算）
        estimated_current_threads = self.config.max_workers  # 第一层线程数
        
        # 为第三层分配剩余线程数
        available_threads = max(1, max_system_threads - estimated_current_threads)
        max_workers = min(len(target_nodes), available_threads, 3)  # 最多3个线程
        
        logger.debug(f"🎯 Controlled parallel relation sampling: {len(target_nodes)} targets, {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ControlledRelation") as executor:
            # 提交任务
            future_to_node = {}
            for target_node_id in target_nodes:
                future = executor.submit(
                    self._optimized_backward_sampling,
                    task_graph, target_node_id, seed, required_node_types
                )
                future_to_node[future] = target_node_id
            
            # 收集结果
            for future in as_completed(future_to_node):
                target_node_id = future_to_node[future]
                try:
                    subgraph = future.result()
                    if subgraph and self._validate_subgraph_with_anchors(subgraph, seed, required_node_types):
                        subgraphs.append(subgraph)
                        logger.debug(f"🎯 Controlled parallel sampling: added subgraph from {target_node_id}")
                        
                        # 限制子图数量
                        if len(subgraphs) >= 15:
                            break
                except Exception as e:
                    logger.warning(f"🎯 Controlled parallel sampling failed for {target_node_id}: {e}")
        
        return subgraphs
    
    def _controlled_strategy_execution(self, task_graph: TaskGraph, seed: TaskSeedPattern, 
                                     required_node_types: Set[NodeType], 
                                     recommended_strategies: List[SamplingStrategy]) -> List[SubgraphSample]:
        """受控的策略级并发执行 - 限制线程数，避免资源竞争"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        strategy_subgraphs = []
        
        # 智能计算线程数：确保总线程数不超过系统限制
        import os
        cpu_count = os.cpu_count() or 4
        max_system_threads = cpu_count * 2  # 保守估计
        
        # 计算当前已使用的线程数（估算）
        estimated_current_threads = self.config.max_workers  # 第一层线程数
        
        # 为第二层分配剩余线程数
        available_threads = max(1, max_system_threads - estimated_current_threads)
        max_workers = min(len(recommended_strategies), available_threads, 2)  # 最多2个策略并发
        
        logger.debug(f"🎯 Controlled strategy execution: {len(recommended_strategies)} strategies, {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ControlledStrategy") as executor:
            # 提交策略任务
            future_to_strategy = {}
            for strategy in recommended_strategies:
                future = executor.submit(
                    self._execute_single_strategy,
                    task_graph, seed, required_node_types, strategy
                )
                future_to_strategy[future] = strategy
            
            # 收集策略结果
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    strategy_result = future.result()
                    strategy_subgraphs.extend(strategy_result)
                    logger.debug(f"🎯 Controlled strategy {strategy.value} completed: {len(strategy_result)} subgraphs")
                except Exception as e:
                    logger.error(f"🎯 Controlled strategy {strategy.value} failed: {e}")
        
        return strategy_subgraphs
    
    def _optimized_backward_sampling(self, task_graph: TaskGraph, target_node_id: str, 
                                   seed: TaskSeedPattern, required_node_types: Set[NodeType]) -> Optional[SubgraphSample]:
        """优化的反向采样 - 更智能的扩展策略"""
        sampled_nodes = {target_node_id: task_graph.nodes[target_node_id]}
        sampled_edges = {}
        queue = deque([target_node_id])
        visited = {target_node_id}
        
        # 限制采样参数
        max_depth = 4
        max_nodes = self.config.max_nodes
        current_depth = 0
        
        while queue and len(sampled_nodes) < max_nodes and current_depth < max_depth:
            current_level_size = len(queue)
            current_depth += 1
            
            for _ in range(current_level_size):
                if not queue:
                    break
                    
                current_node_id = queue.popleft()
                
                # 智能反向扩展
                relevant_edges = self._find_backward_edges(
                    task_graph, current_node_id, seed, required_node_types
                )
                
                for edge in relevant_edges[:4]:  # 限制每个节点的扩展数量
                    neighbor_id = edge.source_node_id if edge.target_node_id == current_node_id else edge.target_node_id
                    
                    if neighbor_id not in visited and len(sampled_nodes) < max_nodes:
                        visited.add(neighbor_id)
                        sampled_nodes[neighbor_id] = task_graph.nodes[neighbor_id]
                        sampled_edges[edge.edge_id] = edge
                        queue.append(neighbor_id)
        
        # 确保包含必需的锚点
        self._ensure_anchor_nodes_present(task_graph, sampled_nodes, sampled_edges, required_node_types)

        # 确保子图的可达性，移除不可达的节点
        if not self._ensure_subgraph_reachability(sampled_nodes, sampled_edges):
            logger.debug(f"🎯 Failed to ensure subgraph reachability for target node {target_node_id}")
            return None

        if len(sampled_nodes) >= 2:
            return SubgraphSample(
                nodes=sampled_nodes,
                edges=sampled_edges,
                center_node=target_node_id,
                strategy=SamplingStrategy.RELATION_CENTRIC
            )
        
        return None
    
    def _find_backward_edges(self, task_graph: TaskGraph, node_id: str, 
                            seed: TaskSeedPattern, required_node_types: Set[NodeType]) -> List[Any]:
        """找到反向相关的边"""
        relevant_edges = []
        
        for edge in task_graph.edges.values():
            if edge.target_node_id == node_id:  # 反向边：指向当前节点
                # 优先选择功能性边
                if edge.edge_type.value.lower() in ['controls', 'fills', 'opens', 'nav_to']:
                    relevant_edges.append(edge)
                elif edge.edge_type.value.lower() == 'contains':
                    relevant_edges.append(edge)
        
        # 按边类型重要性排序
        edge_priority = {'controls': 4, 'fills': 4, 'opens': 3, 'nav_to': 3, 'contains': 2}
        relevant_edges.sort(key=lambda e: edge_priority.get(e.edge_type.value.lower(), 0), reverse=True)
        
        return relevant_edges
    
    
    
    def _functional_module_sampling(self, task_graph: TaskGraph, task_seeds: List[TaskSeedPattern]) -> List[SubgraphSample]:
        """功能模块导向采样 - 生成通用的、可复用的子图"""
        subgraphs = []
        
        logger.info(f"🎯 Starting functional module sampling")
        
        # 定义功能模块组合 - 这些是通用的功能模式
        functional_combinations = [
            # 搜索功能组合
            {NodeType.SEARCH_BOX, NodeType.BUTTON, NodeType.RESULT_ITEM},
            # 表单功能组合
            {NodeType.FORM, NodeType.INPUT, NodeType.BUTTON},
            # 导航功能组合
            {NodeType.NAVIGATION, NodeType.LINK, NodeType.PAGE},
            # 数据展示组合
            {NodeType.TABLE, NodeType.BUSINESS_DATA, NodeType.BUTTON},
            # 用户交互组合
            {NodeType.USER_DATA, NodeType.BUTTON, NodeType.LINK},
            # 产品浏览组合
            {NodeType.PRODUCT_DATA, NodeType.LINK, NodeType.BUTTON},
            # 订单管理组合
            {NodeType.ORDER_DATA, NodeType.BUTTON, NodeType.TABLE}
        ]
        
        for combination in functional_combinations:
            # 找到包含这些功能模块的节点
            matching_nodes = []
            for node_id, node in task_graph.nodes.items():
                if node is None:
                    continue
                if node.node_type in combination:
                    matching_nodes.append(node_id)
            
            if len(matching_nodes) >= 2:  # 至少需要2个功能模块
                # 从这些节点开始采样，但不绑定特定种子
                for start_node_id in matching_nodes[:3]:  # 最多尝试3个起始节点
                    # 使用通用的BFS采样，不绑定种子
                    subgraph = self._generic_bfs_sampling(task_graph, start_node_id)
                    if subgraph and self._validate_generic_subgraph(subgraph, combination):
                        subgraphs.append(subgraph)
                        logger.debug(f"🎯 Added generic functional module subgraph: {combination} -> {len(subgraph.nodes)} nodes")
                        break  # 找到一个就够了，继续下一个组合
        
        logger.info(f"🎯 Functional module sampling generated {len(subgraphs)} generic subgraphs")
        return subgraphs
    
    def _generic_bfs_sampling(self, task_graph: TaskGraph, center_node_id: str) -> Optional[SubgraphSample]:
        """通用的BFS采样 - 不绑定特定任务种子，生成可复用的子图"""
        sampled_nodes = {center_node_id: task_graph.nodes[center_node_id]}
        sampled_edges = {}
        
        # BFS扩展，但限制跳数和类型
        queue = deque([(center_node_id, 0)])
        visited = {center_node_id}
        
        while queue and len(sampled_nodes) < self.config.max_nodes:
            current_node_id, distance = queue.popleft()
            
            # 跳数限制：最多3跳
            if distance >= 3:
                continue
            
            for edge in task_graph.edges.values():
                if edge.source_node_id == current_node_id and edge.target_node_id not in visited:
                    # 通用的边包含逻辑，不依赖特定种子
                    should_include_edge = self._should_include_edge_generic(edge, sampled_nodes)
                    
                    if should_include_edge:
                        sampled_edges[edge.edge_id] = edge
                        sampled_nodes[edge.target_node_id] = task_graph.nodes[edge.target_node_id]
                        visited.add(edge.target_node_id)
                        queue.append((edge.target_node_id, distance + 1))
                        
                        logger.debug(f"🎯 Generic BFS sampling: included edge {edge.edge_id} ({edge.edge_type}) to {edge.target_node_id}")
        
        if len(sampled_nodes) >= self.config.min_nodes:
            return SubgraphSample(
                nodes=sampled_nodes,
                edges=sampled_edges,
                center_node=center_node_id,
                radius=3,
                strategy=SamplingStrategy.FUNCTIONAL_MODULE,
                task_seed=None  # 不绑定特定种子
            )
        
        return None
    
    def _should_include_edge_generic(self, edge, sampled_nodes: Dict[str, GraphNode]) -> bool:
        """通用的边包含判断逻辑 - 不依赖特定种子"""
        # 1. 基本的连接性检查
        if edge.source_node_id not in sampled_nodes:
            return False
        
        # 2. 优先包含功能性的边类型
        functional_edge_types = {
            EdgeType.CONTROLS, EdgeType.NAV_TO, EdgeType.OPENS, 
            EdgeType.FILTERS, EdgeType.CONTAINS
        }
        
        if edge.edge_type in functional_edge_types:
            return True
        
        # 3. 检查目标节点是否是有价值的节点类型
        target_node = edge.target_node_id
        if target_node in sampled_nodes:
            node = sampled_nodes[target_node]
            valuable_node_types = {
                NodeType.BUTTON, NodeType.LINK, NodeType.INPUT, 
                NodeType.SEARCH_BOX, NodeType.FORM, NodeType.TABLE,
                NodeType.BUSINESS_DATA, NodeType.USER_DATA, 
                NodeType.PRODUCT_DATA, NodeType.ORDER_DATA
            }
            if node.node_type in valuable_node_types:
                return True
        
        return False
    
    def _validate_generic_subgraph(self, subgraph: SubgraphSample, required_modules: Set[NodeType]) -> bool:
        """验证通用子图是否满足功能模块要求"""
        if not subgraph or len(subgraph.nodes) < self.config.min_nodes:
            return False
        
        # 检查是否包含足够的功能模块
        subgraph_node_types = {node.node_type for node in subgraph.nodes.values() if node is not None}
        module_coverage = len(subgraph_node_types.intersection(required_modules))
        
        # 至少需要2个功能模块
        if module_coverage < 2:
            return False
        
        # 检查连接性
        if not self._validate_subgraph_connectivity(subgraph.nodes, subgraph.edges):
            return False
        
        logger.debug(f"🎯 Generic subgraph validation passed: {module_coverage} modules covered")
        return True
    
    # ==================== 删除：并发BFS采样（不再需要）====================
    # 并发BFS采样已被删除，因为：
    # 1. 会导致嵌套并发问题
    # 2. 增加了线程/进程开销
    # 3. BFS遍历本身是顺序的，强制并发没有意义
    
    # ==================== 串行处理（向后兼容） ====================
    
    def _sequential_seed_processing(self, task_graph: TaskGraph, task_seeds: List[TaskSeedPattern]) -> List[SubgraphSample]:
        """串行处理种子（保持原有逻辑）"""
        subgraphs = []
        
        for seed in task_seeds:
            logger.info(f"🎯 Processing seed: {seed.name} (type: {seed.seed_type.value})")
            
            # 确定种子必需的节点类型（锚点）
            required_node_types = self._get_seed_anchor_nodes(seed)
            logger.debug(f"🎯 Seed {seed.name} requires anchor nodes: {required_node_types}")
            
            # 选择最适合的采样策略
            recommended_strategies = self._get_recommended_strategies_for_seed(seed)
            logger.debug(f"🎯 Recommended strategies for {seed.name}: {[s.value for s in recommended_strategies]}")
            
            # 执行强绑定采样
            seed_subgraphs = self._execute_seed_bound_sampling(
                task_graph, seed, required_node_types, recommended_strategies
            )
            
            subgraphs.extend(seed_subgraphs)
            logger.info(f"🎯 Generated {len(seed_subgraphs)} subgraphs for seed {seed.name}")
        
        return subgraphs
    
    # ==================== 并发性能监控和调优 ====================
    
    def monitor_concurrency_performance(self, start_time: float, task_seeds_count: int, 
                                      subgraphs_count: int) -> Dict[str, Any]:
        """监控并发性能指标"""
        elapsed_time = time.time() - start_time
        seeds_per_second = task_seeds_count / elapsed_time if elapsed_time > 0 else 0
        subgraphs_per_second = subgraphs_count / elapsed_time if elapsed_time > 0 else 0
        
        performance_metrics = {
            "total_time": elapsed_time,
            "seeds_processed": task_seeds_count,
            "subgraphs_generated": subgraphs_count,
            "seeds_per_second": seeds_per_second,
            "subgraphs_per_second": subgraphs_per_second,
            "concurrency_enabled": self.config.enable_concurrency,
            "max_workers": self.config.max_workers,
            "chunk_size": self.config.chunk_size
        }
        
        # 记录性能指标
        logger.info(f"🎯 Performance Metrics:")
        logger.info(f"  - Total time: {elapsed_time:.2f}s")
        logger.info(f"  - Seeds processed: {task_seeds_count}")
        logger.info(f"  - Subgraphs generated: {subgraphs_count}")
        logger.info(f"  - Throughput: {seeds_per_second:.2f} seeds/s, {subgraphs_per_second:.2f} subgraphs/s")
        
        return performance_metrics
    
    
    def _execute_single_strategy(self, task_graph: TaskGraph, seed: TaskSeedPattern, 
                                required_node_types: Set[NodeType], strategy: SamplingStrategy) -> List[SubgraphSample]:
        """执行单个采样策略 - 简化版本"""
        try:
            if strategy == SamplingStrategy.ELEMENT_CENTRIC:
                return self._simple_element_centric_sampling(task_graph, seed)
            elif strategy == SamplingStrategy.GOAL_CONDITIONED:
                return self._simple_goal_conditioned_sampling(task_graph, seed)
            elif strategy == SamplingStrategy.RELATION_CENTRIC:
                return self._simple_relation_centric_sampling(task_graph, seed)
            elif strategy == SamplingStrategy.FUNCTIONAL_MODULE:
                return self._functional_module_sampling(task_graph, [seed])
            else:
                # 默认使用元素中心采样
                return self._simple_element_centric_sampling(task_graph, seed)
        except Exception as e:
            logger.error(f"🎯 Strategy {strategy.value} execution failed: {e}")
            return []
    
    
    def update_concurrency_config(self, **kwargs):
        """动态更新并发配置"""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f"🎯 Updated concurrency config: {key} = {value}")
            
            # 重新初始化执行器
            if self.executor:
                self.executor.shutdown(wait=True)
            self._init_executor()
    
    def get_concurrency_stats(self) -> Dict[str, Any]:
        """获取并发统计信息"""
        stats = {
            "enabled": self.config.enable_concurrency,
            "max_workers": self.config.max_workers,
            "chunk_size": self.config.chunk_size,
            "use_thread_pool": self.config.use_thread_pool,
            "executor_type": type(self.executor).__name__ if self.executor else "None"
        }
        
        if self.executor:
            if hasattr(self.executor, '_max_workers'):
                stats["active_workers"] = self.executor._max_workers
            if hasattr(self.executor, '_threads'):
                stats["thread_count"] = len(self.executor._threads)
        
        return stats
    
    def optimize_chunk_size(self, task_seeds_count: int) -> int:
        """智能优化任务分块大小 - 修复后的逻辑"""
        if task_seeds_count <= 4:
            return 1
        elif task_seeds_count <= 16:
            return 4  # 增加chunk大小，减少任务分片
        elif task_seeds_count <= 64:
            return 8  # 增加chunk大小
        else:
            return min(12, task_seeds_count // 8)  # 限制最大chunk大小
    
    
    def sample_subgraphs(self, task_graph: TaskGraph, task_seeds: List[TaskSeedPattern]) -> Tuple[List[SubgraphSample], List[Dict[str, Any]]]:
        """基于任务种子的强绑定子图采样 - 修复后只保留种子级并发"""
        start_time = time.time()
        
        # 过滤掉不需要的业务数据任务种子，只保留导航和搜索过滤任务
        filtered_seeds = self._filter_allowed_seeds(task_seeds)
        
        logger.info(f"🎯 Starting simplified concurrent subgraph sampling with {len(filtered_seeds)} task seeds (filtered from {len(task_seeds)})")
        logger.info(f"🎯 Task graph has {len(task_graph.nodes)} nodes and {len(task_graph.edges)} edges")
        logger.info(f"🎯 Concurrency enabled: {self.config.enable_concurrency}, Workers: {self.config.max_workers}")
        logger.info(f"🎯 Config details: enable_concurrency={self.config.enable_concurrency}, use_thread_pool={self.config.use_thread_pool}, max_workers={self.config.max_workers}")
        logger.info(f"🎯 Executor type: {type(self.executor).__name__ if self.executor else 'None'}")
        
        # 分析图结构
        self._analyze_graph_structure(task_graph)
        
        if self.config.enable_concurrency:
            logger.info(f"🎯 Using CONCURRENT processing with {self.config.max_workers} workers")
            # 修复后：只使用种子级并发，避免嵌套并发问题
            subgraphs = self._concurrent_seed_processing_simple(task_graph, filtered_seeds)
        else:
            logger.info(f"🎯 Using SEQUENTIAL processing (concurrency disabled)")
            # 串行处理种子（保持原有逻辑）
            subgraphs = self._sequential_seed_processing(task_graph, filtered_seeds)
        
        # 增强验证和去重
        validated_subgraphs = self._enhanced_validation_and_deduplication(subgraphs)
        
        # 按可执行性分数排序
        sorted_subgraphs = self._sort_by_executability_score(validated_subgraphs)
        
        # 限制总子图数量
        if len(sorted_subgraphs) > self.config.max_total_subgraphs:
            logger.info(f"🎯 Limiting total subgraphs from {len(sorted_subgraphs)} to {self.config.max_total_subgraphs}")
            sorted_subgraphs = sorted_subgraphs[:self.config.max_total_subgraphs]
        
        # 性能监控
        performance_metrics = self.monitor_concurrency_performance(
            start_time, len(task_seeds), len(sorted_subgraphs)
        )
        
        # 转换为详细的子图信息
        detailed_subgraphs = self._create_detailed_subgraphs(sorted_subgraphs)
        
        return sorted_subgraphs, detailed_subgraphs

    def _get_node_content(self, node) -> str:
        """获取节点的内容，支持不同类型的节点"""
        if not node:
            return ''

        # 首先尝试获取节点的content属性（用于传统Node类型）
        if hasattr(node, 'content') and node.content:
            content = str(node.content)
            return content[:100] + "..." if len(content) > 100 else content

        # 对于GraphNode，尝试从metadata中获取text_content
        if hasattr(node, 'metadata') and node.metadata:
            text_content = getattr(node.metadata, 'text_content', '')
            if text_content:
                content = str(text_content)
                return content[:100] + "..." if len(content) > 100 else content

            # 如果没有text_content，尝试其他metadata字段
            placeholder = getattr(node.metadata, 'placeholder', '')
            if placeholder:
                content = str(placeholder)
                return content[:100] + "..." if len(content) > 100 else content

        # 如果都没有内容，返回空字符串
        return ''

    def _create_detailed_subgraphs(self, sorted_subgraphs: List[SubgraphSample]) -> List[Dict[str, Any]]:
        """转换为详细的子图信息，供 benchmark_runner 保存"""
        detailed_subgraphs = []
        for subgraph in sorted_subgraphs:
            detailed_info = {
                "subgraph_id": getattr(subgraph, 'subgraph_id', f"subgraph_{len(detailed_subgraphs)}"),
                "nodes": {node_id: {
                    "node_id": node_id,
                    "node_type": node.node_type.value if node and hasattr(node.node_type, 'value') else str(node.node_type) if node else "unknown",
                    "som_annotation": getattr(node, 'som_annotation', '') if node else '',
                    "content": self._get_node_content(node) if node else '',
                    "is_clickable": getattr(node, 'is_clickable', False) if node else False,
                    "is_input": getattr(node, 'is_input', False) if node else False,
                    "properties": getattr(node, 'properties', {}) if node else {}
                } for node_id, node in subgraph.nodes.items() if node is not None},
                "edges": {edge_id: {
                    "edge_id": edge_id,
                    "edge_type": edge.edge_type.value if edge and hasattr(edge.edge_type, 'value') else str(edge.edge_type) if edge else "unknown",
                    "source": edge.source_node_id if edge else "unknown",
                    "target": edge.target_node_id if edge else "unknown",
                    "properties": getattr(edge, 'properties', {}) if edge else {}
                } for edge_id, edge in subgraph.edges.items() if edge is not None},
                "sampling_strategy": getattr(subgraph, 'strategy', 'unknown').value if hasattr(getattr(subgraph, 'strategy', 'unknown'), 'value') else str(getattr(subgraph, 'strategy', 'unknown')),
                "seed_pattern": subgraph.task_seed.name if subgraph.task_seed else "unknown",
                "executability_score": getattr(subgraph, 'executability_score', 0.0),
                "center_node": getattr(subgraph, 'center_node', 'unknown'),
                "radius": getattr(subgraph, 'radius', 0),
                "node_count": len(subgraph.nodes),
                "edge_count": len(subgraph.edges)
            }
            detailed_subgraphs.append(detailed_info)
        
        return detailed_subgraphs
    
    def _concurrent_seed_processing_simple(self, task_graph: TaskGraph, task_seeds: List[TaskSeedPattern]) -> List[SubgraphSample]:
        """修复后的简化种子级并发处理 - 避免嵌套并发和内存共享问题"""
        all_subgraphs = []
        
        # 智能调整chunk大小，确保有足够的并发
        if len(task_seeds) <= self.config.max_workers:
            # 如果种子数量少于worker数量，每个种子一个chunk
            chunk_size = 1
        else:
            # 否则使用配置的chunk大小，但确保至少有2个chunk
            chunk_size = max(1, min(self.config.chunk_size, len(task_seeds) // 2))
        
        # 将种子分块
        seed_chunks = [task_seeds[i:i + chunk_size] 
                      for i in range(0, len(task_seeds), chunk_size)]
        
        logger.info(f"🎯 Processing {len(task_seeds)} seeds in {len(seed_chunks)} chunks (chunk_size: {chunk_size})")
        logger.info(f"🎯 Using {self.config.max_workers} workers for concurrent processing")
        
        # 提交种子块任务 - 每个chunk独立处理，避免嵌套并发
        future_to_chunk = {}
        logger.info(f"🎯 Submitting {len(seed_chunks)} chunks to thread pool executor...")
        for chunk_idx, seed_chunk in enumerate(seed_chunks):
            logger.debug(f"🎯 Submitting chunk {chunk_idx} with {len(seed_chunk)} seeds")
            future = self.executor.submit(
                self._process_seed_chunk_simple, task_graph, seed_chunk, chunk_idx
            )
            future_to_chunk[future] = chunk_idx
            logger.debug(f"🎯 Chunk {chunk_idx} submitted to executor")
        
        # 收集结果
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                chunk_subgraphs = future.result()
                all_subgraphs.extend(chunk_subgraphs)
                logger.info(f"🎯 Chunk {chunk_idx} completed: {len(chunk_subgraphs)} subgraphs")
            except Exception as e:
                logger.error(f"🎯 Chunk {chunk_idx} failed: {e}")
        
        logger.info(f"🎯 Simplified concurrent seed processing completed: {len(all_subgraphs)} total subgraphs")
        return all_subgraphs
    
    def _process_seed_chunk_simple(self, task_graph: TaskGraph, seed_chunk: List[TaskSeedPattern], chunk_idx: int) -> List[SubgraphSample]:
        """处理种子块 - 修复后串行处理，避免嵌套并发"""
        chunk_subgraphs = []
        
        # 添加线程信息，验证并发执行
        import threading
        thread_name = threading.current_thread().name
        logger.info(f"🎯 Chunk {chunk_idx} started processing in thread: {thread_name}")
        
        for seed in seed_chunk:
            logger.debug(f"🎯 Processing seed {seed.name} in chunk {chunk_idx} (thread: {thread_name})")
            
            # 记录处理这个种子之前的子图数量
            seed_start_count = len(chunk_subgraphs)
            
            # 确定种子必需的节点类型（锚点）
            required_node_types = self._get_seed_anchor_nodes(seed)
            
            # 选择最适合的采样策略
            recommended_strategies = self._get_recommended_strategies_for_seed(seed)
            
            # 智能策略级并发：根据策略复杂度和数量决定是否使用并发
            if len(recommended_strategies) > 2 and self.config.enable_concurrency:
                # 使用受控的策略级并发
                seed_subgraphs = self._controlled_strategy_execution(
                    task_graph, seed, required_node_types, recommended_strategies
                )
                # 限制每个种子生成的子图数量
                if len(seed_subgraphs) > self.config.max_subgraphs_per_seed:
                    logger.debug(f"🎯 Limiting seed {seed.name} subgraphs from {len(seed_subgraphs)} to {self.config.max_subgraphs_per_seed}")
                    seed_subgraphs = seed_subgraphs[:self.config.max_subgraphs_per_seed]
                chunk_subgraphs.extend(seed_subgraphs)
            else:
                # 串行执行策略（策略少或并发被禁用时）
                logger.debug(f"🎯 Using sequential strategy execution for seed {seed.name}")
                seed_subgraphs = []
                for strategy in recommended_strategies:
                    try:
                        strategy_result = self._execute_single_strategy(
                            task_graph, seed, required_node_types, strategy
                        )
                        seed_subgraphs.extend(strategy_result)
                        logger.debug(f"🎯 Strategy {strategy.value} completed: {len(strategy_result)} subgraphs")

                        # 检查是否已经达到种子限制
                        if len(seed_subgraphs) >= self.config.max_subgraphs_per_seed:
                            logger.debug(f"🎯 Reached max subgraphs per seed limit ({self.config.max_subgraphs_per_seed}), stopping")
                            break
                    except Exception as e:
                        logger.error(f"🎯 Strategy {strategy.value} failed: {e}")
                        continue

                # 最终限制每个种子的子图数量
                if len(seed_subgraphs) > self.config.max_subgraphs_per_seed:
                    seed_subgraphs = seed_subgraphs[:self.config.max_subgraphs_per_seed]

                chunk_subgraphs.extend(seed_subgraphs)
            
            # 计算这个种子实际生成的子图数量
            seed_generated_count = len(chunk_subgraphs) - seed_start_count
            logger.debug(f"🎯 Generated {seed_generated_count} subgraphs for seed {seed.name}")
        
        logger.info(f"🎯 Chunk {chunk_idx} completed in thread: {thread_name}, generated {len(chunk_subgraphs)} subgraphs")
        return chunk_subgraphs
    
    def _filter_allowed_seeds(self, task_seeds: List[TaskSeedPattern]) -> List[TaskSeedPattern]:
        """过滤掉不需要的业务数据任务种子，只保留导航和搜索过滤任务"""
        # 定义允许的种子类型（只保留导航和搜索过滤任务）
        allowed_seed_types = {
            TaskSeedType.BUSINESS_SEARCH_FILTER,
            TaskSeedType.BUSINESS_NAVIGATION,
            TaskSeedType.USER_NAVIGATION,
            TaskSeedType.PRODUCT_NAVIGATION,
            TaskSeedType.ORDER_NAVIGATION,
            TaskSeedType.MIXED_DATA_NAVIGATION,
            TaskSeedType.MULTI_HOP_NAVIGATION,
            # 保留所有交互种子
            TaskSeedType.CONTENT_BROWSING,
            TaskSeedType.BASIC_NAVIGATION,
            TaskSeedType.BUTTON_INTERACTION,
            TaskSeedType.MENU_EXPLORATION,
            TaskSeedType.TAB_SWITCHING,
            TaskSeedType.MODAL_INTERACTION,
            TaskSeedType.TOAST_NOTIFICATION,
            TaskSeedType.BREADCRUMB_NAVIGATION,
            TaskSeedType.PAGINATION_BROWSING,
            TaskSeedType.EXPAND_COLLAPSE,
            TaskSeedType.SCROLL_READING
        }
        
        # 过滤种子
        filtered_seeds = []
        filtered_out_count = 0
        
        for seed in task_seeds:
            if seed.seed_type in allowed_seed_types:
                filtered_seeds.append(seed)
            else:
                filtered_out_count += 1
                logger.debug(f"🎯 Filtered out seed: {seed.name} (type: {seed.seed_type.value})")
        
        logger.info(f"🎯 Seed filtering: {len(filtered_seeds)} allowed, {filtered_out_count} filtered out")
        return filtered_seeds
