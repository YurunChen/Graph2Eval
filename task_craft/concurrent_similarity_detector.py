"""
Concurrent Similarity Detector - Optimizes task similarity detection performance
Supports multiple similarity algorithms and concurrent processing
"""

import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import time
from loguru import logger
import numpy as np
import hashlib
import json
import re

from .task_seeds import TaskSeedPattern
from graph_rag.graph_builder import TaskGraph
from graph_rag.node_types import GraphNode
from graph_rag.edge_types import GraphEdge

class SimilarityAlgorithm(Enum):
    """Similarity algorithm types"""
    LLM_SEMANTIC = "llm_semantic"  # LLM-based semantic similarity
    HYBRID = "hybrid"  # Hybrid approach (LLM + fallback)

@dataclass
class SimilarityResult:
    """Similarity detection result"""
    task1_id: str
    task2_id: str
    similarity_score: float
    algorithm: SimilarityAlgorithm
    details: Dict[str, Any] = None
    processing_time: float = 0.0

@dataclass
class BatchSimilarityConfig:
    """Batch similarity detection configuration - uses default settings, no external config needed"""
    max_workers: int = 4
    batch_size: int = 50
    cache_results: bool = True
    algorithms: List[SimilarityAlgorithm] = None
    similarity_threshold: float = 0.7
    enable_early_stop: bool = True
    llm_executor: Any = None  # LLM executor for similarity assessment
    llm_model_name: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 500
    
    def __post_init__(self):
        if self.algorithms is None:
            # Default to LLM semantic similarity
            self.algorithms = [SimilarityAlgorithm.LLM_SEMANTIC]

class ConcurrentSimilarityDetector:
    """Concurrent similarity detector with LLM-based assessment"""
    
    def __init__(self, config: BatchSimilarityConfig):
        self.config = config
        self.similarity_cache = {}
        self.llm_executor = config.llm_executor
        
    def clear_cache(self):
        """Clear cache"""
        self.similarity_cache.clear()
        logger.info("Similarity cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.similarity_cache),
            "cache_hits": getattr(self, '_cache_hits', 0),
            "cache_misses": getattr(self, '_cache_misses', 0)
        }
    
    def _get_cache_key(self, task1_id: str, task2_id: str, algorithm: SimilarityAlgorithm) -> str:
        """Generate cache key"""
        # Ensure task1_id < task2_id for consistent cache keys
        if task1_id > task2_id:
            task1_id, task2_id = task2_id, task1_id
        return f"{task1_id}_{task2_id}_{algorithm.value}"
    
    def _check_cache(self, task1_id: str, task2_id: str, algorithm: SimilarityAlgorithm) -> Optional[float]:
        """Check cache"""
        cache_key = self._get_cache_key(task1_id, task2_id, algorithm)
        if cache_key in self.similarity_cache:
            self._cache_hits = getattr(self, '_cache_hits', 0) + 1
            return self.similarity_cache[cache_key]
        self._cache_misses = getattr(self, '_cache_misses', 0) + 1
        return None
    
    def _store_cache(self, task1_id: str, task2_id: str, algorithm: SimilarityAlgorithm, score: float):
        """Store to cache"""
        if self.config.cache_results:
            cache_key = self._get_cache_key(task1_id, task2_id, algorithm)
            self.similarity_cache[cache_key] = score
    
    def detect_duplicates_batch(self, tasks: List[Any], task_graph: Optional[TaskGraph] = None) -> List[SimilarityResult]:
        """Batch detection of duplicate tasks"""
        logger.info(f"🎯 Starting batch similarity detection for {len(tasks)} tasks")
        start_time = time.time()
        
        # Generate all task pairs
        task_pairs = []
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                task_pairs.append((tasks[i], tasks[j]))
        
        logger.info(f"🎯 Generated {len(task_pairs)} task pairs for similarity analysis")
        
        # Use thread pool for concurrent processing
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_pair = {}
            for task1, task2 in task_pairs:
                for algorithm in self.config.algorithms:
                    future = executor.submit(
                        self._calculate_similarity_single,
                        task1, task2, algorithm, task_graph
                    )
                    future_to_pair[future] = (task1, task2, algorithm)
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_pair):
                task1, task2, algorithm = future_to_pair[future]
                try:
                    result = future.result()
                    if result and result.similarity_score >= self.config.similarity_threshold:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error calculating similarity for {task1.task_id} vs {task2.task_id}: {e}")
        
        processing_time = time.time() - start_time
        logger.info(f"🎯 Batch similarity detection completed in {processing_time:.2f}s")
        logger.info(f"🎯 Found {len(results)} high-similarity pairs")
        
        return results
    
    def _calculate_similarity_single(self, task1: Any, task2: Any, algorithm: SimilarityAlgorithm, task_graph: Optional[TaskGraph] = None) -> Optional[SimilarityResult]:
        """Calculate similarity for a single task pair"""
        start_time = time.time()
        
        # Check cache
        cached_score = self._check_cache(task1.task_id, task2.task_id, algorithm)
        if cached_score is not None:
            return SimilarityResult(
                task1_id=task1.task_id,
                task2_id=task2.task_id,
                similarity_score=cached_score,
                algorithm=algorithm,
                processing_time=time.time() - start_time
            )
        
        # Calculate similarity using LLM
        try:
            if algorithm == SimilarityAlgorithm.LLM_SEMANTIC:
                score = self._llm_semantic_similarity(task1, task2)
            elif algorithm == SimilarityAlgorithm.HYBRID:
                score = self._hybrid_similarity(task1, task2, task_graph)
            else:
                score = 0.0
            
            # Store to cache
            self._store_cache(task1.task_id, task2.task_id, algorithm, score)
            
            processing_time = time.time() - start_time
            return SimilarityResult(
                task1_id=task1.task_id,
                task2_id=task2.task_id,
                similarity_score=score,
                algorithm=algorithm,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error calculating {algorithm.value} similarity: {e}")
            return None
    
    def _llm_semantic_similarity(self, task1: Any, task2: Any) -> float:
        """LLM-based semantic similarity assessment"""
        try:
            if not self.llm_executor:
                logger.warning("LLM executor not available, using fallback similarity")
                return self._fallback_similarity(task1, task2)
            
            # Create LLM prompt for similarity assessment
            prompt = self._create_similarity_assessment_prompt(task1, task2)
            
            # Get LLM response
            response = self.llm_executor.execute_simple(prompt)
            
            # Parse similarity score from response
            similarity_score = self._parse_similarity_response(response)
            
            logger.debug(f"LLM similarity assessment: {task1.task_id} vs {task2.task_id} = {similarity_score:.3f}")
            return similarity_score
            
        except Exception as e:
            logger.error(f"Error in LLM similarity assessment: {e}")
            return self._fallback_similarity(task1, task2)
    
    def _create_similarity_assessment_prompt(self, task1: Any, task2: Any) -> str:
        """Create LLM prompt for similarity assessment"""
        prompt = f"""You are an expert task similarity assessor. Your job is to evaluate how similar two web tasks are based on their content, purpose, and execution steps.

TASK 1:
- Task ID: {task1.task_id}
- Task Type: {getattr(task1, 'web_task_type', getattr(task1, 'task_type', 'unknown'))}
- Prompt: {getattr(task1, 'prompt', 'N/A')}
- Steps: {self._format_task_steps(task1)}
- Elements Used: {getattr(task1, 'som_elements_used', [])}

TASK 2:
- Task ID: {task2.task_id}
- Task Type: {getattr(task2, 'web_task_type', getattr(task2, 'task_type', 'unknown'))}
- Prompt: {getattr(task2, 'prompt', 'N/A')}
- Steps: {self._format_task_steps(task2)}
- Elements Used: {getattr(task2, 'som_elements_used', [])}

ASSESSMENT CRITERIA:
1. **Content Similarity**: How similar are the tasks in terms of what they accomplish?
2. **Step Similarity**: How similar are the execution steps and sequences?
3. **Element Similarity**: How similar are the UI elements and interactions used?
4. **Purpose Similarity**: How similar are the overall goals and objectives?

SIMILARITY SCALE:
- 0.0-0.2: Completely different tasks (different goals, different elements, different steps)
- 0.2-0.4: Mostly different tasks (some overlap but fundamentally different)
- 0.4-0.6: Moderately similar tasks (similar goals but different approaches)
- 0.6-0.8: Very similar tasks (similar goals and approaches, minor differences)
- 0.8-1.0: Nearly identical tasks (same goals, same approach, same elements)

Provide your assessment as a JSON response with the following format:
{{
    "similarity_score": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of your assessment>",
    "content_similarity": <float>,
    "step_similarity": <float>,
    "element_similarity": <float>,
    "purpose_similarity": <float>
}}

Focus on the semantic meaning and practical execution rather than just text similarity."""
        
        return prompt
    
    def _format_task_steps(self, task: Any) -> str:
        """Format task steps for LLM prompt"""
        if hasattr(task, 'task_steps') and task.task_steps:
            steps = []
            for i, step in enumerate(task.task_steps, 1):
                if hasattr(step, 'step_type'):
                    step_desc = f"Step {i}: {step.step_type}"
                    if hasattr(step, 'action_description') and step.action_description:
                        step_desc += f" - {step.action_description}"
                    if hasattr(step, 'target_som_mark') and step.target_som_mark:
                        step_desc += f" (target: {step.target_som_mark})"
                    steps.append(step_desc)
            return "; ".join(steps)
        return "No steps available"
    
    def _parse_similarity_response(self, response: Any) -> float:
        """Parse similarity score from LLM response"""
        try:
            # 处理ExecutionResult对象
            if hasattr(response, 'result'):
                response = response.result
            elif hasattr(response, 'content'):
                response = response.content
            elif not isinstance(response, str):
                response = str(response)
            
            # 尝试多种JSON提取方式
            json_patterns = [
                r'\{[^{}]*"similarity_score"[^{}]*\}',  # 包含similarity score的JSON
                r'\{.*?\}',  # 任何JSON对象
                r'```json\s*(.*?)\s*```',  # 代码块中的JSON
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, response, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group()
                        # 清理JSON字符串
                        json_str = re.sub(r'[^\x20-\x7E]', '', json_str)  # 移除非ASCII字符
                        data = json.loads(json_str)
                        if 'similarity_score' in data:
                            return float(data['similarity_score'])
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue
            
            # 备用：直接提取数字
            number_patterns = [
                r'similarity_score["\s]*:["\s]*([0-9]*\.?[0-9]+)',
                r'similarity["\s]*:["\s]*([0-9]*\.?[0-9]+)',
                r'score["\s]*:["\s]*([0-9]*\.?[0-9]+)',
                r'\b(0\.\d+)\b',  # 任何0-1之间的数字
            ]
            
            for pattern in number_patterns:
                number_match = re.search(pattern, response)
                if number_match:
                    try:
                        score = float(number_match.group(1))
                        if 0.0 <= score <= 1.0:
                            return score
                    except (ValueError, IndexError):
                        continue
            
            # 基于关键词的启发式评分
            response_lower = response.lower()
            if any(word in response_lower for word in ['identical', 'same', 'duplicate', 'exact']):
                return 0.9
            elif any(word in response_lower for word in ['very similar', 'highly similar', 'almost identical']):
                return 0.8
            elif any(word in response_lower for word in ['similar', 'alike', 'comparable']):
                return 0.6
            elif any(word in response_lower for word in ['somewhat similar', 'moderately similar']):
                return 0.4
            elif any(word in response_lower for word in ['different', 'dissimilar', 'unrelated']):
                return 0.2
            else:
                return 0.5  # 默认中等相似度
            
        except Exception as e:
            logger.error(f"Error parsing similarity response: {e}")
            logger.debug(f"Raw response: {response}")
            return 0.5
    
    def _fallback_similarity(self, task1: Any, task2: Any) -> float:
        """Fallback similarity calculation when LLM is not available"""
        try:
            # Simple text-based similarity
            text1 = self._extract_text_features(task1)
            text2 = self._extract_text_features(task2)
            
            if not text1 or not text2:
                return 0.0
            
            # Simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error in fallback similarity: {e}")
            return 0.0
    

    
    def _hybrid_similarity(self, task1: Any, task2: Any, task_graph: Optional[TaskGraph] = None) -> float:
        """Hybrid similarity - LLM + fallback"""
        try:
            # Primary: LLM semantic similarity
            llm_sim = self._llm_semantic_similarity(task1, task2)
            
            # Fallback: simple text similarity
            fallback_sim = self._fallback_similarity(task1, task2)
            
            # Weighted average: 80% LLM, 20% fallback
            weighted_sum = 0.8 * llm_sim + 0.2 * fallback_sim
            return weighted_sum
        except Exception as e:
            logger.error(f"Error in hybrid similarity: {e}")
            return 0.0
    
    def _extract_text_features(self, task: Any) -> str:
        """Extract text features"""
        features = []
        
        # Extract prompt
        if hasattr(task, 'prompt') and task.prompt:
            features.append(task.prompt)
        
        # Extract task description
        if hasattr(task, 'description') and task.description:
            features.append(task.description)
        
        # Extract step descriptions
        if hasattr(task, 'task_steps') and task.task_steps:
            for step in task.task_steps:
                if hasattr(step, 'action_description') and step.action_description:
                    features.append(step.action_description)
        
        return " ".join(features)
    
    def _extract_feature_set(self, task: Any) -> Set[str]:
        """Extract feature set"""
        features = set()
        
        # Extract text features
        text = self._extract_text_features(task)
        words = text.lower().split()
        features.update(words)
        
        # Extract node types
        if hasattr(task, 'subgraph_nodes') and task.subgraph_nodes:
            for node in task.subgraph_nodes.values():
                if hasattr(node, 'node_type'):
                    features.add(f"node_type_{node.node_type.value}")
        
        return features
    
    def _extract_key_terms(self, task: Any) -> Set[str]:
        """Extract key terms"""
        terms = set()
        
        # Extract keywords from prompt
        if hasattr(task, 'prompt') and task.prompt:
            words = [word.lower() for word in task.prompt.split() if len(word) > 3]
            terms.update(words)
        
        # Extract variables
        if hasattr(task, 'variables') and task.variables:
            for key, value in task.variables.items():
                if isinstance(value, str):
                    words = [word.lower() for word in value.split() if len(word) > 3]
                    terms.update(words)
        
        return terms
    
    def _extract_subgraph_features(self, task: Any, task_graph: TaskGraph) -> Optional[Dict[str, Any]]:
        """Extract subgraph features"""
        try:
            if not hasattr(task, 'subgraph_nodes') or not task.subgraph_nodes:
                return None
            
            nodes = {}
            edges = {}
            
            # Extract node features
            for node_id, node in task.subgraph_nodes.items():
                nodes[node_id] = {
                    'type': node.node_type.value,
                    'metadata': getattr(node, 'metadata', {})
                }
            
            # Extract edge features
            if hasattr(task, 'subgraph_edges') and task.subgraph_edges:
                for edge_id, edge in task.subgraph_edges.items():
                    edges[edge_id] = {
                        'type': edge.edge_type.value,
                        'source': edge.source_node_id,
                        'target': edge.target_node_id
                    }
            
            return {'nodes': nodes, 'edges': edges}
        except Exception as e:
            logger.error(f"Error extracting subgraph features: {e}")
            return None
    
    def _calculate_node_similarity(self, nodes1: Dict, nodes2: Dict) -> float:
        """Calculate node similarity"""
        try:
            if not nodes1 or not nodes2:
                return 0.0
            
            # Calculate node type overlap
            types1 = {node['type'] for node in nodes1.values()}
            types2 = {node['type'] for node in nodes2.values()}
            
            intersection = len(types1.intersection(types2))
            union = len(types1.union(types2))
            
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating node similarity: {e}")
            return 0.0
    
    def _calculate_edge_similarity(self, edges1: Dict, edges2: Dict) -> float:
        """Calculate edge similarity"""
        try:
            if not edges1 or not edges2:
                return 0.0
            
            # Calculate edge type overlap
            types1 = {edge['type'] for edge in edges1.values()}
            types2 = {edge['type'] for edge in edges2.values()}
            
            intersection = len(types1.intersection(types2))
            union = len(types1.union(types2))
            
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating edge similarity: {e}")
            return 0.0
    
    def find_duplicate_groups(self, tasks: List[Any], task_graph: Optional[TaskGraph] = None) -> List[List[str]]:
        """Find duplicate task groups"""
        logger.info(f"🎯 Finding duplicate groups for {len(tasks)} tasks")
        
        # Batch similarity detection
        similarity_results = self.detect_duplicates_batch(tasks, task_graph)
        
        # Build similarity graph
        similarity_graph = {}
        for result in similarity_results:
            if result.similarity_score >= self.config.similarity_threshold:
                if result.task1_id not in similarity_graph:
                    similarity_graph[result.task1_id] = set()
                if result.task2_id not in similarity_graph:
                    similarity_graph[result.task2_id] = set()
                
                similarity_graph[result.task1_id].add(result.task2_id)
                similarity_graph[result.task2_id].add(result.task1_id)
        
        # Use connected component algorithm to find duplicate groups
        duplicate_groups = []
        visited = set()
        
        for task_id in similarity_graph:
            if task_id not in visited:
                group = self._find_connected_component(similarity_graph, task_id, visited)
                if len(group) > 1:  # Only return groups with duplicates
                    duplicate_groups.append(list(group))
        
        logger.info(f"🎯 Found {len(duplicate_groups)} duplicate groups")
        return duplicate_groups
    
    def _find_connected_component(self, graph: Dict[str, Set[str]], start: str, visited: Set[str]) -> Set[str]:
        """Find connected component"""
        component = set()
        stack = [start]
        
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.add(current)
                
                if current in graph:
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)
        
        return component
    
    def optimize_task_selection(self, tasks: List[Any], target_count: int, task_graph: Optional[TaskGraph] = None) -> List[Any]:
        """优化任务选择，去除重复并保持多样性"""
        logger.info(f"🎯 Optimizing task selection: {len(tasks)} -> {target_count}")
        
        if len(tasks) <= target_count:
            return tasks
        
        # 找到重复组
        duplicate_groups = self.find_duplicate_groups(tasks, task_graph)
        
        # 从每个重复组中选择最佳任务
        selected_tasks = []
        removed_tasks = set()
        
        for group in duplicate_groups:
            # 选择质量最高的任务
            best_task = max(group, key=lambda task_id: self._get_task_quality(tasks, task_id))
            selected_tasks.append(best_task)
            removed_tasks.update(group - {best_task})
        
        # 添加非重复任务
        for task in tasks:
            if task.task_id not in removed_tasks:
                selected_tasks.append(task.task_id)
        
        # 如果仍然超过目标数量，使用多样性选择
        if len(selected_tasks) > target_count:
            selected_tasks = self._diversity_based_selection(tasks, selected_tasks, target_count, task_graph)
        
        logger.info(f"🎯 Task selection optimized: {len(tasks)} -> {len(selected_tasks)}")
        return [task for task in tasks if task.task_id in selected_tasks]
    
    def _get_task_quality(self, tasks: List[Any], task_id: str) -> float:
        """Get task quality score"""
        for task in tasks:
            if task.task_id == task_id:
                return getattr(task, 'quality_score', 0.5)
        return 0.5
    
    def _diversity_based_selection(self, all_tasks: List[Any], candidate_ids: List[str], target_count: int, task_graph: Optional[TaskGraph] = None) -> List[str]:
        """Diversity-based task selection"""
        if len(candidate_ids) <= target_count:
            return candidate_ids
        
        # Use MMR algorithm for diversity selection
        selected = []
        remaining = candidate_ids.copy()
        
        # Select first task (highest quality)
        first_task_id = max(remaining, key=lambda tid: self._get_task_quality(all_tasks, tid))
        selected.append(first_task_id)
        remaining.remove(first_task_id)
        
        # Iteratively select remaining tasks
        while len(selected) < target_count and remaining:
            best_task_id = None
            best_score = -1
            
            for task_id in remaining:
                # Calculate diversity score
                diversity_score = self._calculate_diversity_score(all_tasks, task_id, selected, task_graph)
                quality_score = self._get_task_quality(all_tasks, task_id)
                
                # Combined score
                combined_score = 0.7 * diversity_score + 0.3 * quality_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_task_id = task_id
            
            if best_task_id:
                selected.append(best_task_id)
                remaining.remove(best_task_id)
            else:
                break
        
        return selected
    
    def _calculate_diversity_score(self, all_tasks: List[Any], candidate_id: str, selected_ids: List[str], task_graph: Optional[TaskGraph] = None) -> float:
        """Calculate diversity score"""
        if not selected_ids:
            return 1.0
        
        # Calculate minimum similarity with selected tasks
        min_similarity = 1.0
        candidate_task = next((task for task in all_tasks if task.task_id == candidate_id), None)
        
        if not candidate_task:
            return 0.0
        
        for selected_id in selected_ids:
            selected_task = next((task for task in all_tasks if task.task_id == selected_id), None)
            if selected_task:
                similarity = self._hybrid_similarity(candidate_task, selected_task, task_graph)
                min_similarity = min(min_similarity, similarity)
        
        # Diversity score = 1 - maximum similarity
        return 1.0 - min_similarity
    
    def get_similarity_matrix(self, tasks: List[Any], task_graph: Optional[TaskGraph] = None) -> Dict[str, Dict[str, float]]:
        """Get similarity matrix"""
        logger.info(f"🎯 Computing similarity matrix for {len(tasks)} tasks")
        
        matrix = {}
        task_ids = [task.task_id for task in tasks]
        
        # Initialize matrix
        for task_id in task_ids:
            matrix[task_id] = {tid: 0.0 for tid in task_ids}
        
        # Use concurrent computation for similarity
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for i, task1 in enumerate(tasks):
                for j in range(i + 1, len(tasks)):
                    task2 = tasks[j]
                    future = executor.submit(
                        self._calculate_similarity_single,
                        task1, task2, SimilarityAlgorithm.HYBRID, task_graph
                    )
                    futures.append((future, task1.task_id, task2.task_id))
            
            # Collect results
            for future, task1_id, task2_id in futures:
                try:
                    result = future.result()
                    if result:
                        matrix[task1_id][task2_id] = result.similarity_score
                        matrix[task2_id][task1_id] = result.similarity_score
                except Exception as e:
                    logger.error(f"Error computing similarity for {task1_id} vs {task2_id}: {e}")
        
        return matrix
    
    def analyze_similarity_distribution(self, tasks: List[Any], task_graph: Optional[TaskGraph] = None) -> Dict[str, Any]:
        """Analyze similarity distribution"""
        logger.info(f"🎯 Analyzing similarity distribution for {len(tasks)} tasks")
        
        # Get similarity matrix
        matrix = self.get_similarity_matrix(tasks, task_graph)
        
        # Collect all similarity values
        similarities = []
        for task1_id in matrix:
            for task2_id in matrix[task1_id]:
                if task1_id != task2_id:
                    similarities.append(matrix[task1_id][task2_id])
        
        if not similarities:
            return {
                "total_pairs": 0,
                "mean_similarity": 0.0,
                "std_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "similarity_distribution": {}
            }
        
        # Calculate statistics
        similarities = np.array(similarities)
        stats = {
            "total_pairs": len(similarities),
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "similarity_distribution": {
                "very_low": int(np.sum(similarities < 0.2)),
                "low": int(np.sum((similarities >= 0.2) & (similarities < 0.4))),
                "medium": int(np.sum((similarities >= 0.4) & (similarities < 0.6))),
                "high": int(np.sum((similarities >= 0.6) & (similarities < 0.8))),
                "very_high": int(np.sum(similarities >= 0.8))
            }
        }
        
        logger.info(f"🎯 Similarity analysis: mean={stats['mean_similarity']:.3f}, std={stats['std_similarity']:.3f}")
        return stats
    
    def export_similarity_report(self, tasks: List[Any], output_path: str, task_graph: Optional[TaskGraph] = None):
        """Export similarity report"""
        logger.info(f"🎯 Exporting similarity report to {output_path}")
        
        # Get similarity matrix and analysis results
        matrix = self.get_similarity_matrix(tasks, task_graph)
        analysis = self.analyze_similarity_distribution(tasks, task_graph)
        duplicate_groups = self.find_duplicate_groups(tasks, task_graph)
        
        # Build report
        report = {
            "metadata": {
                "total_tasks": len(tasks),
                "analysis_timestamp": time.time(),
                "config": {
                    "max_workers": self.config.max_workers,
                    "similarity_threshold": self.config.similarity_threshold,
                    "algorithms": [alg.value for alg in self.config.algorithms]
                }
            },
            "similarity_matrix": matrix,
            "analysis": analysis,
            "duplicate_groups": duplicate_groups,
            "cache_stats": self.get_cache_stats()
        }
        
        # Save report
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"🎯 Similarity report exported to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting similarity report: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_stats = self.get_cache_stats()
        
        return {
            "cache_stats": cache_stats,
            "cache_hit_rate": cache_stats["cache_hits"] / (cache_stats["cache_hits"] + cache_stats["cache_misses"]) if (cache_stats["cache_hits"] + cache_stats["cache_misses"]) > 0 else 0.0,
            "total_operations": cache_stats["cache_hits"] + cache_stats["cache_misses"]
        }

class TaskSimilarityOptimizer:
    """Task similarity optimizer - integration interface with task generator"""
    
    def __init__(self, llm_executor: Any = None):
        # Use default configuration with LLM executor
        config = BatchSimilarityConfig(llm_executor=llm_executor)
        self.detector = ConcurrentSimilarityDetector(config)
        self.config = config
    
    def optimize_generated_tasks(self, tasks: List[Any], target_count: int, task_graph: Optional[TaskGraph] = None) -> List[Any]:
        """Optimize generated tasks, remove duplicates and maintain diversity"""
        logger.info(f"🎯 Optimizing {len(tasks)} generated tasks to {target_count}")
        
        if len(tasks) <= target_count:
            return tasks
        
        # Use concurrent similarity detector to optimize task selection
        optimized_tasks = self.detector.optimize_task_selection(tasks, target_count, task_graph)
        
        # Record optimization results
        removed_count = len(tasks) - len(optimized_tasks)
        logger.info(f"🎯 Task optimization completed: removed {removed_count} duplicate tasks")
        
        return optimized_tasks
    
    def remove_duplicates(self, tasks: List[Any], task_graph: Optional[TaskGraph] = None) -> List[Any]:
        """Remove duplicate tasks"""
        logger.info(f"🎯 Removing duplicates from {len(tasks)} tasks")
        
        # Find duplicate groups
        duplicate_groups = self.detector.find_duplicate_groups(tasks, task_graph)
        
        # Remove all tasks except the best one from each duplicate group
        tasks_to_remove = set()
        for group in duplicate_groups:
            if len(group) > 1:
                # Select highest quality task to keep
                best_task_id = max(group, key=lambda tid: self._get_task_quality(tasks, tid))
                tasks_to_remove.update(group - {best_task_id})
        
        # Remove duplicate tasks
        filtered_tasks = [task for task in tasks if task.task_id not in tasks_to_remove]
        
        logger.info(f"🎯 Duplicate removal completed: removed {len(tasks_to_remove)} tasks")
        return filtered_tasks
    
    def _get_task_quality(self, tasks: List[Any], task_id: str) -> float:
        """Get task quality score"""
        for task in tasks:
            if task.task_id == task_id:
                return getattr(task, 'quality_score', 0.5)
        return 0.5
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            "detector_stats": self.detector.get_performance_stats(),
            "config": {
                "max_workers": self.config.max_workers,
                "similarity_threshold": self.config.similarity_threshold,
                "algorithms": [alg.value for alg in self.config.algorithms]
            }
        }
    
    def export_optimization_report(self, tasks: List[Any], output_path: str, task_graph: Optional[TaskGraph] = None):
        """Export optimization report"""
        self.detector.export_similarity_report(tasks, output_path, task_graph)