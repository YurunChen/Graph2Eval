"""
Task Coverage Optimizer - Task selection is "coverage optimization"
Performs multi-objective redundancy removal on candidate task sets to build small but comprehensive datasets
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
import json
import time
from collections import Counter
from loguru import logger
from graph_rag.node_types import GraphNode, NodeType
from graph_rag.edge_types import GraphEdge, EdgeType
from .concurrent_similarity_detector import ConcurrentSimilarityDetector, BatchSimilarityConfig, SimilarityAlgorithm

class CoverageMetric(Enum):
    """Coverage metrics"""
    NODE_TYPE = "node_type"
    EDGE_TYPE = "edge_type"
    PATTERN = "pattern"
    PAGE_LEVEL = "page_level"
    WEBSITE_TYPE = "website_type"

class NoveltyMetric(Enum):
    """Novelty metrics"""
    MMR = "mmr"
    SUBGRAPH_JACCARD = "subgraph_jaccard"
    STEP_SIMILARITY = "step_similarity"
    LLM_SEMANTIC = "llm_semantic"  # New: LLM semantic similarity

@dataclass
class CoverageOptimizationConfig:
    """Coverage optimization configuration"""
    # Coverage weights
    coverage_weights: Dict[CoverageMetric, float] = field(default_factory=lambda: {
        CoverageMetric.NODE_TYPE: 0.30,
        CoverageMetric.EDGE_TYPE: 0.20,
        CoverageMetric.PATTERN: 0.15,
        CoverageMetric.PAGE_LEVEL: 0.15,
        CoverageMetric.WEBSITE_TYPE: 0.10
    })
    
    # Novelty weights
    novelty_weights: Dict[NoveltyMetric, float] = field(default_factory=lambda: {
        NoveltyMetric.MMR: 0.1,
        NoveltyMetric.SUBGRAPH_JACCARD: 0.1,
        NoveltyMetric.STEP_SIMILARITY: 0.1,
        NoveltyMetric.LLM_SEMANTIC: 0.7  # LLM semantic similarity has highest weight
    })
    
    # Solvability check
    enable_solvability_check: bool = True
    solvability_threshold: float = 0.5
    
    # Diversity parameters
    mmr_lambda: float = 0.7  # MMR diversity parameter
    min_jaccard_distance: float = 0.3  # Minimum Jaccard distance
    
    # LLM similarity analysis
    enable_llm_similarity: bool = True
    llm_similarity_threshold: float = 0.85  # Higher similarity threshold, tasks with >85% similarity considered duplicate

@dataclass
class TaskCandidate:
    """Task candidate"""
    task_id: str
    task_type: str
    difficulty: str
    quality_score: float = 0.0
    coverage_score: float = 0.0
    novelty_score: float = 0.0
    solvability_score: float = 1.0
    combined_score: float = 0.0
    diversity_features: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    subgraph: Any = None  # SubgraphSample
    metapath_instance: Any = None  # MetapathInstance

class EnhancedTaskSimilarityAnalyzer:
    """Enhanced task similarity analyzer using LLM-based similarity detection"""
    
    def __init__(self, config: BatchSimilarityConfig = None, llm_executor: Any = None):
        if config is None:
            config = BatchSimilarityConfig(llm_executor=llm_executor)
        elif llm_executor and not config.llm_executor:
            config.llm_executor = llm_executor
        
        self.config = config
        self.similarity_detector = ConcurrentSimilarityDetector(self.config)
        self.similarity_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def clear_cache(self):
        """Clear cache"""
        self.similarity_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.similarity_detector.clear_cache()
        logger.info("Enhanced similarity cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        detector_stats = self.similarity_detector.get_cache_stats()
        return {
            "cache_size": len(self.similarity_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0,
            "detector_stats": detector_stats
        }
    
    def analyze_task_similarity(self, task1: TaskCandidate, task2: TaskCandidate) -> float:
        """Analyze semantic similarity between two tasks using enhanced detection"""
        # Generate cache key
        cache_key = tuple(sorted([task1.task_id, task2.task_id]))
        if cache_key in self.similarity_cache:
            self.cache_hits += 1
            logger.debug(f"Using cached similarity for {task1.task_id} vs {task2.task_id}")
            return self.similarity_cache[cache_key]
        
        self.cache_misses += 1
        
        try:
            # Use concurrent similarity detector
            similarity_results = self.similarity_detector.detect_duplicates_batch([task1, task2])
            
            if similarity_results:
                # Get the similarity score from the result
                similarity_score = similarity_results[0].similarity_score
                self.similarity_cache[cache_key] = similarity_score
                logger.debug(f"Enhanced similarity analysis: {task1.task_id} vs {task2.task_id} = {similarity_score:.3f}")
                return similarity_score
            else:
                # No similarity detected, return low similarity
                similarity_score = 0.1
                self.similarity_cache[cache_key] = similarity_score
                return similarity_score
                
        except Exception as e:
            logger.error(f"Error in enhanced similarity analysis: {e}")
            return self._fallback_similarity_analysis(task1, task2)
    
    def _fallback_similarity_analysis(self, task1: TaskCandidate, task2: TaskCandidate) -> float:
        """Fallback similarity analysis"""
        # Calculate similarity based on SoM element overlap
        som1 = set(task1.diversity_features.get('som_elements', []))
        som2 = set(task2.diversity_features.get('som_elements', []))
        
        if not som1 or not som2:
            return 0.0
        
        intersection = len(som1 & som2)
        union = len(som1 | som2)
        
        return intersection / union if union > 0 else 0.0
    
    def batch_analyze_similarity(self, tasks: List[TaskCandidate]) -> Dict[Tuple[str, str], float]:
        """Batch analyze similarity for multiple tasks"""
        logger.info(f"🎯 Starting batch similarity analysis for {len(tasks)} tasks")
        
        # Use concurrent similarity detector for batch analysis
        similarity_results = self.similarity_detector.detect_duplicates_batch(tasks)
        
        # Convert results to dictionary format
        similarity_dict = {}
        for result in similarity_results:
            key = tuple(sorted([result.task1_id, result.task2_id]))
            similarity_dict[key] = result.similarity_score
        
        logger.info(f"🎯 Batch similarity analysis completed: {len(similarity_dict)} pairs analyzed")
        return similarity_dict

class TaskCoverageOptimizer:
    """Task coverage optimizer"""
    
    def __init__(self, config: CoverageOptimizationConfig, similarity_config: BatchSimilarityConfig = None, llm_executor: Any = None):
        self.config = config
        self.selected_tasks: List[TaskCandidate] = []
        self.coverage_tracker = CoverageTracker()
        self.similarity_analyzer = EnhancedTaskSimilarityAnalyzer(similarity_config, llm_executor) if config.enable_llm_similarity else None
    
    def optimize_task_selection(self, candidates: List[TaskCandidate], target_count: int) -> List[TaskCandidate]:
        """Optimize task selection"""
        logger.info(f"Optimizing selection from {len(candidates)} candidates to {target_count} tasks")
        
        # Initialize coverage tracker
        self.coverage_tracker.reset()
        
        # Calculate coverage scores for candidate tasks
        for candidate in candidates:
            candidate.coverage_score = self._calculate_coverage_score(candidate)
            candidate.solvability_score = self._calculate_solvability_score(candidate)
        
        # Use MMR algorithm to select tasks
        selected_tasks = self._mmr_selection(candidates, target_count)
        
        # Final validation and adjustment
        selected_tasks = self._final_validation(selected_tasks)
        
        logger.info(f"Selected {len(selected_tasks)} tasks with coverage: {self.coverage_tracker.get_coverage_summary()}")
        
        return selected_tasks
    
    def _calculate_coverage_score(self, candidate: TaskCandidate) -> float:
        """Calculate coverage score"""
        if candidate.subgraph is None:
            logger.warning(f"Task {candidate.task_id} has no subgraph - using default coverage score")
            return 0.5  # 返回默认分数
        
        score = 0.0
        
        # Node type coverage
        node_types = self._extract_node_types(candidate)
        node_coverage = self.coverage_tracker.calculate_node_type_coverage(node_types)
        score += self.config.coverage_weights[CoverageMetric.NODE_TYPE] * node_coverage
        
        # Edge type coverage
        edge_types = self._extract_edge_types(candidate)
        edge_coverage = self.coverage_tracker.calculate_edge_type_coverage(edge_types)
        score += self.config.coverage_weights[CoverageMetric.EDGE_TYPE] * edge_coverage
        
        # Pattern coverage
        pattern_coverage = self.coverage_tracker.calculate_pattern_coverage(candidate.task_type)
        score += self.config.coverage_weights[CoverageMetric.PATTERN] * pattern_coverage
        
        # Page level coverage
        page_level_coverage = self.coverage_tracker.calculate_page_level_coverage(candidate)
        score += self.config.coverage_weights[CoverageMetric.PAGE_LEVEL] * page_level_coverage
        
        # Website type coverage
        website_coverage = self.coverage_tracker.calculate_website_type_coverage(candidate)
        score += self.config.coverage_weights[CoverageMetric.WEBSITE_TYPE] * website_coverage
        
        # Difficulty coverage (10% weight)
        difficulty_coverage = self._calculate_difficulty_coverage(candidate.difficulty)
        score += 0.1 * difficulty_coverage
        
        return score
    
    def _calculate_novelty_score(self, candidate: TaskCandidate, selected_tasks: List[TaskCandidate]) -> float:
        """Calculate novelty score"""
        if not selected_tasks:
            return 1.0  # First task has highest novelty
        
        # Check enhanced semantic similarity first - if too similar, return 0 immediately
        if self.similarity_analyzer:
            enhanced_novelty = self._calculate_enhanced_semantic_novelty(candidate, selected_tasks)
            logger.debug(f"🎯 Task {candidate.task_id} enhanced novelty: {enhanced_novelty}")
            if enhanced_novelty == 0.0:  # Enhanced detector detected duplicate
                logger.info(f"🎯 Task {candidate.task_id} filtered out due to enhanced similarity (novelty=0.0)")
                return 0.0
        
        score = 0.0
        
        # MMR novelty
        mmr_novelty = self._calculate_mmr_novelty(candidate, selected_tasks)
        score += self.config.novelty_weights[NoveltyMetric.MMR] * mmr_novelty
        
        # Subgraph Jaccard distance
        jaccard_novelty = self._calculate_jaccard_novelty(candidate, selected_tasks)
        score += self.config.novelty_weights[NoveltyMetric.SUBGRAPH_JACCARD] * jaccard_novelty
        
        # Step similarity
        step_novelty = self._calculate_step_novelty(candidate, selected_tasks)
        score += self.config.novelty_weights[NoveltyMetric.STEP_SIMILARITY] * step_novelty
        
        # Enhanced semantic similarity (if not already checked above)
        if self.similarity_analyzer:
            enhanced_novelty = self._calculate_enhanced_semantic_novelty(candidate, selected_tasks)
            score += self.config.novelty_weights[NoveltyMetric.LLM_SEMANTIC] * enhanced_novelty
        
        return score
    
    def _calculate_enhanced_semantic_novelty(self, candidate: TaskCandidate, selected_tasks: List[TaskCandidate]) -> float:
        """Calculate enhanced semantic novelty using concurrent similarity detection"""
        if not self.similarity_analyzer or not selected_tasks:
            return 1.0
        
        # First check for exact prompt duplicates (strict check)
        candidate_prompt = candidate.prompt.lower().strip() if hasattr(candidate, 'prompt') else ""
        for selected_task in selected_tasks:
            selected_prompt = selected_task.prompt.lower().strip() if hasattr(selected_task, 'prompt') else ""
            if candidate_prompt == selected_prompt and candidate_prompt:
                logger.info(f"Task {candidate.task_id} has exact duplicate prompt with {selected_task.task_id}")
                return 0.0
        
        # Calculate maximum similarity with selected tasks
        max_similarity = 0.0
        most_similar_task = None
        
        for selected_task in selected_tasks:
            similarity = self.similarity_analyzer.analyze_task_similarity(candidate, selected_task)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_task = selected_task.task_id
        
        # Enhanced novelty calculation logic with content-based analysis
        if max_similarity >= self.config.llm_similarity_threshold:
            # Exceeds threshold, consider task duplicate, novelty is 0
            logger.info(f"Task {candidate.task_id} is too similar to {most_similar_task} (similarity: {max_similarity:.3f})")
            return 0.0
        
        # 调整相似性阈值，使其更宽松
        if max_similarity >= 0.8: # 提高阈值，更严格地认为是重复
            logger.info(f"Task {candidate.task_id} is too similar to {most_similar_task} (similarity: {max_similarity:.3f}), novelty: 0.0")
            return 0.0
        elif max_similarity >= 0.5: # 降低中等相似度阈值
            novelty = 0.3 * (1.0 - max_similarity) # 给予较低的新颖度
            logger.debug(f"Task {candidate.task_id} moderate novelty: {novelty:.3f} (similarity: {max_similarity:.3f})")
            return novelty
        else: # 相似度较低，新颖度较高
            novelty = 0.6 + 0.4 * (1.0 - max_similarity)
            logger.debug(f"Task {candidate.task_id} high novelty: {novelty:.3f} (similarity: {max_similarity:.3f})")
            return novelty
    
    def optimize_task_selection_batch(self, candidates: List[TaskCandidate], target_count: int) -> List[TaskCandidate]:
        """Optimize task selection using batch similarity analysis"""
        logger.info(f"🎯 Starting batch optimization for {len(candidates)} candidates to {target_count} tasks")
        
        # Initialize coverage tracker
        self.coverage_tracker.reset()
        
        # Calculate coverage scores for candidate tasks
        for candidate in candidates:
            candidate.coverage_score = self._calculate_coverage_score(candidate)
            candidate.solvability_score = self._calculate_solvability_score(candidate)
        
        # Use batch similarity analysis if available
        if self.similarity_analyzer:
            logger.info("🎯 Using batch similarity analysis for optimization")
            selected_tasks = self._batch_mmr_selection(candidates, target_count)
        else:
            logger.info("🎯 Using standard MMR selection")
            selected_tasks = self._mmr_selection(candidates, target_count)
        
        # Final validation and adjustment
        selected_tasks = self._final_validation(selected_tasks)
        
        logger.info(f"🎯 Batch optimization completed: selected {len(selected_tasks)} tasks")
        logger.info(f"🎯 Coverage summary: {self.coverage_tracker.get_coverage_summary()}")
        
        return selected_tasks
    
    def _batch_mmr_selection(self, candidates: List[TaskCandidate], target_count: int) -> List[TaskCandidate]:
        """MMR selection with batch similarity analysis"""
        if not candidates:
            return []
        
        # Sort candidates by quality score
        candidates.sort(key=lambda x: x.quality_score, reverse=True)
        
        selected_tasks = []
        remaining_candidates = candidates.copy()
        
        # Select first task (highest quality)
        if remaining_candidates:
            selected_tasks.append(remaining_candidates.pop(0))
        
        # Batch analyze similarity for remaining candidates
        if self.similarity_analyzer and remaining_candidates:
            logger.info(f"🎯 Performing batch similarity analysis for {len(remaining_candidates)} candidates")
            similarity_dict = self.similarity_analyzer.batch_analyze_similarity(remaining_candidates)
        
        # Iteratively select remaining tasks
        while len(selected_tasks) < target_count and remaining_candidates:
            best_candidate = None
            best_score = -1
            
            for candidate in remaining_candidates:
                # Calculate novelty score using batch similarity results
                novelty_score = self._calculate_batch_novelty_score(candidate, selected_tasks, similarity_dict)
                
                # Calculate combined score
                combined_score = (
                    0.3 * candidate.quality_score +
                    0.3 * candidate.coverage_score +
                    0.4 * novelty_score
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                selected_tasks.append(best_candidate)
                remaining_candidates.remove(best_candidate)
            else:
                break
        
        return selected_tasks
    
    def _calculate_batch_novelty_score(self, candidate: TaskCandidate, selected_tasks: List[TaskCandidate], similarity_dict: Dict[Tuple[str, str], float]) -> float:
        """Calculate novelty score using batch similarity results"""
        if not selected_tasks:
            return 1.0
        
        # Calculate maximum similarity with selected tasks
        max_similarity = 0.0
        
        for selected_task in selected_tasks:
            key = tuple(sorted([candidate.task_id, selected_task.task_id]))
            if key in similarity_dict:
                similarity = similarity_dict[key]
                max_similarity = max(max_similarity, similarity)
        
        # Calculate novelty based on similarity
        if max_similarity >= 0.8:
            return 0.0  # Too similar
        elif max_similarity >= 0.5:
            return 0.3 * (1.0 - max_similarity)
        else:
            return 0.6 + 0.4 * (1.0 - max_similarity)
    
    def _calculate_solvability_score(self, candidate: TaskCandidate) -> float:
        """Calculate solvability score"""
        if not self.config.enable_solvability_check:
            return 1.0
        
        
        # Check path reachability
        path_solvability = self._check_path_solvability(candidate)
        
        # Check element visibility
        element_solvability = self._check_element_solvability(candidate)
        
        # Check interaction feasibility
        interaction_solvability = self._check_interaction_solvability(candidate)
        
        # Combined solvability score
        solvability_score = (path_solvability + element_solvability + interaction_solvability) / 3.0
        
        return max(0.0, min(1.0, solvability_score))
    
    def _mmr_selection(self, candidates: List[TaskCandidate], target_count: int) -> List[TaskCandidate]:
        """Use MMR algorithm to select tasks with concurrent similarity detection"""
        selected = []
        remaining = candidates.copy()
        
        # If we have a similarity analyzer, use batch processing for better performance
        if self.similarity_analyzer and len(remaining) > 10:
            logger.info("🎯 Using concurrent similarity detection for MMR selection")
            return self._concurrent_mmr_selection(candidates, target_count)
        
        # Standard MMR selection for smaller datasets
        while len(selected) < target_count and remaining:
            # Calculate novelty scores for each candidate task
            for candidate in remaining:
                candidate.novelty_score = self._calculate_novelty_score(candidate, selected)
                candidate.combined_score = (
                    0.4 * candidate.quality_score +
                    0.4 * candidate.coverage_score +
                    0.2 * candidate.novelty_score
                ) * candidate.solvability_score
            
            # Filter out tasks with zero novelty (duplicates) first
            valid_candidates = [c for c in remaining if c.novelty_score > 0.0]
            
            if not valid_candidates:
                # No valid candidates left, break
                logger.warning("No more novel candidates available for selection")
                break
            
            # Select best candidate from valid candidates
            best_candidate = max(valid_candidates, key=lambda c: c.combined_score)
            
            # Check solvability threshold
            if best_candidate.solvability_score >= self.config.solvability_threshold:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                
                # Update coverage tracker
                self.coverage_tracker.update_coverage(best_candidate)
            else:
                # Remove unsolvable tasks
                remaining.remove(best_candidate)
        
        return selected
    
    def _concurrent_mmr_selection(self, candidates: List[TaskCandidate], target_count: int) -> List[TaskCandidate]:
        """Concurrent MMR selection using batch similarity analysis"""
        logger.info(f"🎯 Starting concurrent MMR selection for {len(candidates)} candidates")
        
        selected = []
        remaining = candidates.copy()
        
        # Sort candidates by quality score for initial selection
        remaining.sort(key=lambda x: x.quality_score, reverse=True)
        
        # Select first task (highest quality)
        if remaining:
            selected.append(remaining.pop(0))
            self.coverage_tracker.update_coverage(selected[0])
        
        # Use batch similarity analysis for remaining candidates
        while len(selected) < target_count and remaining:
            # Perform batch similarity analysis
            if self.similarity_analyzer:
                logger.info(f"🎯 Analyzing similarity for {len(remaining)} remaining candidates")
                similarity_dict = self.similarity_analyzer.batch_analyze_similarity(remaining)
            else:
                similarity_dict = {}
            
            # Calculate novelty scores using batch results
            best_candidate = None
            best_score = -1
            
            for candidate in remaining:
                # Calculate novelty score using batch similarity results
                novelty_score = self._calculate_batch_novelty_score(candidate, selected, similarity_dict)
                
                # Calculate combined score
                combined_score = (
                    0.4 * candidate.quality_score +
                    0.4 * candidate.coverage_score +
                    0.2 * novelty_score
                ) * candidate.solvability_score
                
                if combined_score > best_score and candidate.solvability_score >= self.config.solvability_threshold:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                self.coverage_tracker.update_coverage(best_candidate)
                logger.info(f"🎯 Selected task {best_candidate.task_id} with score {best_score:.3f}")
            else:
                logger.warning("No more valid candidates available for selection")
                break
        
        logger.info(f"🎯 Concurrent MMR selection completed: selected {len(selected)} tasks")
        return selected
    
    def _final_validation(self, selected_tasks: List[TaskCandidate]) -> List[TaskCandidate]:
        """Final validation and adjustment"""
        # Check coverage balance
        coverage_summary = self.coverage_tracker.get_coverage_summary()
        
        # If some metrics have insufficient coverage, can adjust
        if coverage_summary['node_type_coverage'] < 0.7:
            logger.warning("Node type coverage is low, consider adding more diverse tasks")
        
        if coverage_summary['edge_type_coverage'] < 0.6:
            logger.warning("Edge type coverage is low, consider adding more interaction tasks")
        
        return selected_tasks
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics including concurrent similarity detection stats"""
        stats = {
            "selected_tasks_count": len(self.selected_tasks),
            "coverage_summary": self.coverage_tracker.get_coverage_summary(),
            "config": {
                "mmr_lambda": self.config.mmr_lambda,
                "solvability_threshold": self.config.solvability_threshold,
                "enable_llm_similarity": self.config.enable_llm_similarity,
                "llm_similarity_threshold": self.config.llm_similarity_threshold
            }
        }
        
        # Add similarity analyzer stats if available
        if self.similarity_analyzer:
            similarity_stats = self.similarity_analyzer.get_cache_stats()
            stats["similarity_analyzer"] = similarity_stats
            stats["concurrent_detection_enabled"] = True
        else:
            stats["concurrent_detection_enabled"] = False
        
        return stats
    
    def export_optimization_report(self, output_path: str):
        """Export optimization report including similarity analysis results"""
        logger.info(f"🎯 Exporting optimization report to {output_path}")
        
        # Get optimization stats
        stats = self.get_optimization_stats()
        
        # Add selected task details
        selected_task_details = []
        for task in self.selected_tasks:
            task_detail = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "difficulty": task.difficulty,
                "quality_score": task.quality_score,
                "coverage_score": task.coverage_score,
                "novelty_score": task.novelty_score,
                "solvability_score": task.solvability_score,
                "combined_score": task.combined_score
            }
            selected_task_details.append(task_detail)
        
        report = {
            "optimization_stats": stats,
            "selected_tasks": selected_task_details,
            "timestamp": time.time()
        }
        
        # Save report
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"🎯 Optimization report exported to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting optimization report: {e}")

    def _extract_node_types(self, candidate: TaskCandidate) -> Set[NodeType]:
        """Extract node types involved in the task"""
        return self.coverage_tracker._extract_node_types(candidate)
    
    def _extract_edge_types(self, candidate: TaskCandidate) -> Set[EdgeType]:
        """Extract edge types involved in the task"""
        return self.coverage_tracker._extract_edge_types(candidate)
    
    def _calculate_mmr_novelty(self, candidate: TaskCandidate, selected_tasks: List[TaskCandidate]) -> float:
        """Calculate MMR novelty"""
        if not selected_tasks:
            return 1.0
        
        # Calculate maximum similarity with selected tasks
        max_similarity = 0.0
        for selected_task in selected_tasks:
            similarity = self._calculate_task_similarity(candidate, selected_task)
            max_similarity = max(max_similarity, similarity)
        
        return 1.0 - max_similarity
    
    def _calculate_jaccard_novelty(self, candidate: TaskCandidate, selected_tasks: List[TaskCandidate]) -> float:
        """Calculate Jaccard novelty"""
        
        candidate_nodes = set(candidate.subgraph.nodes.keys())
        min_jaccard = 1.0
        
        for selected_task in selected_tasks:
            if selected_task.subgraph is None:
                continue
            selected_nodes = set(selected_task.subgraph.nodes.keys())
            
            intersection = len(candidate_nodes & selected_nodes)
            union = len(candidate_nodes | selected_nodes)
            
            if union > 0:
                jaccard = intersection / union
                min_jaccard = min(min_jaccard, jaccard)
        
        return 1.0 - min_jaccard
    
    def _calculate_step_novelty(self, candidate: TaskCandidate, selected_tasks: List[TaskCandidate]) -> float:
        """Calculate step novelty"""
        if not selected_tasks:
            return 1.0
        
        candidate_steps = [step.get('step_type', '') for step in candidate.steps]
        max_similarity = 0.0
        
        for selected_task in selected_tasks:
            selected_steps = [step.get('step_type', '') for step in selected_task.steps]
            
            # Calculate step sequence similarity
            similarity = self._calculate_step_sequence_similarity(candidate_steps, selected_steps)
            max_similarity = max(max_similarity, similarity)
        
        return 1.0 - max_similarity
    
    def _calculate_task_similarity(self, task1: TaskCandidate, task2: TaskCandidate) -> float:
        """Calculate similarity between two tasks"""
        # Task type similarity
        type_similarity = 1.0 if task1.task_type == task2.task_type else 0.0
        
        # Difficulty similarity
        difficulty_similarity = 1.0 if task1.difficulty == task2.difficulty else 0.5
        
        # Step count similarity
        step_count_similarity = 1.0 - abs(len(task1.steps) - len(task2.steps)) / max(len(task1.steps), len(task2.steps))
        
        # Combined similarity
        return (type_similarity + difficulty_similarity + step_count_similarity) / 3.0
    
    def _calculate_step_sequence_similarity(self, steps1: List[str], steps2: List[str]) -> float:
        """Calculate step sequence similarity"""
        if not steps1 or not steps2:
            return 0.0
        
        # Use edit distance to calculate similarity
        distance = self._levenshtein_distance(steps1, steps2)
        max_len = max(len(steps1), len(steps2))
        
        return 1.0 - (distance / max_len)
    
    def _levenshtein_distance(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate edit distance"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        return dp[m][n]
    
    def _check_path_solvability(self, candidate: TaskCandidate) -> float:
        """Check path reachability"""
        nodes = set(candidate.subgraph.nodes.keys())
        edges = candidate.subgraph.edges.values()
        
        # Check if all nodes are reachable
        reachable = set()
        if nodes:
            start_node = list(nodes)[0]
            reachable.add(start_node)
            
            # Simple reachability check
            for edge in edges:
                if edge.source_node_id in reachable:
                    reachable.add(edge.target_node_id)
        
        return len(reachable) / len(nodes) if nodes else 1.0
    
    def _check_element_solvability(self, candidate: TaskCandidate) -> float:
        """Check element visibility"""
        if candidate.subgraph is None:
            return 1.0
        
        visible_count = 0
        total_count = 0
        
        for node in candidate.subgraph.nodes.values():
            if node.metadata.is_visible:
                visible_count += 1
            total_count += 1
        
        return visible_count / total_count if total_count > 0 else 1.0
    
    def _check_interaction_solvability(self, candidate: TaskCandidate) -> float:
        
        clickable_count = 0
        input_count = 0
        total_interactive = 0
        
        for node in candidate.subgraph.nodes.values():
            if node.metadata.is_clickable or node.metadata.is_input:
                total_interactive += 1
                if node.metadata.is_clickable:
                    clickable_count += 1
                if node.metadata.is_input:
                    input_count += 1
        
        if total_interactive == 0:
            return 1.0
        
        # Check if there are enough interactive elements
        interaction_ratio = total_interactive / len(candidate.subgraph.nodes)
        return min(1.0, interaction_ratio * 2)  # Adjust weight
    
    def _calculate_difficulty_coverage(self, difficulty: str) -> float:
        """Calculate difficulty coverage for web tasks"""
        if not hasattr(self, 'difficulty_counts'):
            self.difficulty_counts = {}
        
        if difficulty in self.difficulty_counts:
            return 0.0  # Existing difficulty
        else:
            return 1.0  # New difficulty

class CoverageTracker:
    """Coverage tracker"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset coverage tracker"""
        self.node_type_counts = Counter()
        self.edge_type_counts = Counter()
        self.pattern_counts = Counter()
        self.page_level_counts = Counter()
        self.website_type_counts = Counter()
        self.difficulty_counts = Counter()
    
    def update_coverage(self, candidate: TaskCandidate):
        """Update coverage"""
        # Update node type coverage
        node_types = self._extract_node_types(candidate)
        for node_type in node_types:
            self.node_type_counts[node_type] += 1
        
        # Update edge type coverage
        edge_types = self._extract_edge_types(candidate)
        for edge_type in edge_types:
            self.edge_type_counts[edge_type] += 1
        
        # Update pattern coverage
        self.pattern_counts[candidate.task_type] += 1
        
        # Update page level coverage
        page_level = self._calculate_page_level(candidate)
        self.page_level_counts[page_level] += 1
        
        # Update website type coverage
        website_type = getattr(candidate.subgraph, 'website_type', 'unknown')
        self.website_type_counts[website_type] += 1
        
        # Update difficulty coverage
        self.difficulty_counts[candidate.difficulty] += 1
    
    def _extract_node_types(self, candidate: TaskCandidate) -> Set[NodeType]:
        """Extract node types involved in the task"""
        node_types = set()
        for step in candidate.steps:
            if 'target_som_mark' in step and step['target_som_mark']:
                # Find corresponding node from subgraph
                for node in candidate.subgraph.nodes.values():
                    if node.metadata.som_mark == step['target_som_mark']:
                        node_types.add(node.node_type)
                        break
        return node_types
    
    def _extract_edge_types(self, candidate: TaskCandidate) -> Set[EdgeType]:
        """Extract edge types involved in the task"""
        edge_types = set()
        if candidate.subgraph is None:
            return edge_types
        for edge in candidate.subgraph.edges.values():
            edge_types.add(edge.edge_type)
        return edge_types
    
    def _calculate_page_level(self, candidate: TaskCandidate) -> str:
        """Calculate page level"""
        if candidate.subgraph is None:
            return "single_page"
        page_nodes = [node for node in candidate.subgraph.nodes.values() if node.node_type == NodeType.PAGE]
        if len(page_nodes) == 1:
            return "single_page"
        elif len(page_nodes) <= 3:
            return "multi_page"
        else:
            return "deep_navigation"
    
    def calculate_node_type_coverage(self, node_types: Set[NodeType]) -> float:
        """Calculate node type coverage"""
        if not node_types:
            return 0.0
        
        # Calculate new node types
        new_types = node_types - set(self.node_type_counts.keys())
        return len(new_types) / len(node_types)
    
    def calculate_edge_type_coverage(self, edge_types: Set[EdgeType]) -> float:
        """Calculate edge type coverage"""
        if not edge_types:
            return 0.0
        
        # Calculate new edge types
        new_types = edge_types - set(self.edge_type_counts.keys())
        return len(new_types) / len(edge_types)
    
    def calculate_pattern_coverage(self, task_type: str) -> float:
        """Calculate pattern coverage"""
        if task_type in self.pattern_counts:
            return 0.0  # Existing pattern
        else:
            return 1.0  # New pattern
    
    def calculate_page_level_coverage(self, candidate: TaskCandidate) -> float:
        """Calculate page level coverage"""
        page_level = self._calculate_page_level(candidate)
        if page_level in self.page_level_counts:
            return 0.0  # Existing level
        else:
            return 1.0  # New level
    
    def calculate_website_type_coverage(self, candidate: TaskCandidate) -> float:
        """Calculate website type coverage"""
        website_type = getattr(candidate.subgraph, 'website_type', 'unknown')
        if website_type in self.website_type_counts:
            return 0.0  # Existing website type
        else:
            return 1.0  # New website type
    
    def get_coverage_summary(self) -> Dict[str, Any]:
        """Get coverage summary"""
        return {
            'node_type_coverage': len(self.node_type_counts) / len(NodeType),
            'edge_type_coverage': len(self.edge_type_counts) / len(EdgeType),
            'pattern_coverage': len(self.pattern_counts),
            'page_level_coverage': len(self.page_level_counts) / 3,  # 3 levels
            'website_type_coverage': len(self.website_type_counts),
            'total_tasks': sum(self.pattern_counts.values())
        }

# ============================================================================
# Text Task Coverage Optimizer Extension
# ============================================================================

@dataclass
class TextTaskCandidate:
    """Text task candidate for coverage optimization"""
    task_id: str
    task_type: str
    difficulty: str
    prompt: str
    gold_answer: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    coverage_score: float = 0.0
    novelty_score: float = 0.0
    combined_score: float = 0.0
    diversity_features: Dict[str, Any] = field(default_factory=dict)
    template_id: str = ""
    subgraph_nodes: Optional[Dict[str, Any]] = None

class TextTaskCoverageOptimizer:
    """Text task coverage optimizer - extension for text generation tasks"""
    
    def __init__(self, config: CoverageOptimizationConfig = None, similarity_config: BatchSimilarityConfig = None, llm_executor: Any = None):
        self.config = config or CoverageOptimizationConfig()
        self.similarity_config = similarity_config or BatchSimilarityConfig(
            similarity_threshold=0.6,  # Default threshold for text tasks
            algorithms=[SimilarityAlgorithm.LLM_SEMANTIC],  # Use LLM semantic similarity
            llm_executor=llm_executor
        )
        self.similarity_analyzer = EnhancedTaskSimilarityAnalyzer(self.similarity_config, llm_executor)
        self.coverage_tracker = CoverageTracker()
        
        # Text-specific coverage tracking
        self.task_type_counts = Counter()
        self.difficulty_counts = Counter()
        self.template_counts = Counter()
        self.variable_key_counts = Counter()
        self.content_length_ranges = Counter()
        
        # Performance tracking
        self.optimization_stats = {
            'total_candidates': 0,
            'selected_tasks': 0,
            'removed_duplicates': 0,
            'processing_time': 0.0,
            'similarity_checks': 0
        }
    
    def optimize_text_tasks(self, tasks: List[Any], target_count: int = None) -> List[Any]:
        """Optimize text task selection using coverage optimization"""
        logger.info(f"🎯 Starting text task coverage optimization for {len(tasks)} tasks")
        start_time = time.time()
        
        # Convert tasks to TextTaskCandidate format
        candidates = self._convert_to_candidates(tasks)
        self.optimization_stats['total_candidates'] = len(candidates)
        
        
        # Calculate coverage and novelty scores
        self._calculate_coverage_scores(candidates)
        self._calculate_novelty_scores(candidates)
        
        # Select optimal tasks
        selected_candidates = self._select_optimal_tasks(candidates, target_count)
        
        # Convert back to original format
        optimized_tasks = self._convert_from_candidates(selected_candidates, tasks)
        
        # Update statistics
        self.optimization_stats['selected_tasks'] = len(optimized_tasks)
        self.optimization_stats['removed_duplicates'] = len(tasks) - len(optimized_tasks)
        self.optimization_stats['processing_time'] = time.time() - start_time
        
        logger.info(f"🎯 Text task optimization completed: {len(tasks)} -> {len(optimized_tasks)} tasks")
        return optimized_tasks
    
    def _convert_to_candidates(self, tasks: List[Any]) -> List[TextTaskCandidate]:
        """Convert tasks to TextTaskCandidate format"""
        candidates = []
        for task in tasks:
            candidate = TextTaskCandidate(
                task_id=task.task_id,
                task_type=task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type),
                difficulty=task.difficulty.value if hasattr(task.difficulty, 'value') else str(task.difficulty),
                prompt=task.prompt,
                gold_answer=task.gold_answer,
                variables=task.variables or {},
                quality_score=getattr(task, 'quality_score', 0.5),
                template_id=task.template_id,
                subgraph_nodes=getattr(task, 'subgraph_nodes', None)
            )
            
            # Extract diversity features
            candidate.diversity_features = self._extract_text_diversity_features(task)
            candidates.append(candidate)
        
        return candidates
    
    def _convert_from_candidates(self, candidates: List[TextTaskCandidate], original_tasks: List[Any]) -> List[Any]:
        """Convert candidates back to original task format"""
        task_map = {task.task_id: task for task in original_tasks}
        return [task_map[candidate.task_id] for candidate in candidates if candidate.task_id in task_map]
    
    def _extract_text_diversity_features(self, task: Any) -> Dict[str, Any]:
        """Extract diversity features for text tasks"""
        features = {}
        
        # Extract key terms from prompt and answer
        prompt_terms = set(word.lower() for word in task.prompt.split() if len(word) > 3)
        features['prompt_terms'] = list(prompt_terms)
        
        if task.gold_answer:
            answer_terms = set(word.lower() for word in task.gold_answer.split() if len(word) > 3)
            features['answer_terms'] = list(answer_terms)
        
        # Extract variable information
        if task.variables:
            features['variable_keys'] = list(task.variables.keys())
            features['variable_content'] = []
            for key, value in task.variables.items():
                if isinstance(value, str):
                    terms = set(word.lower() for word in value.split() if len(word) > 3)
                    features['variable_content'].extend(list(terms))
        
        # Extract subgraph information if available
        if hasattr(task, 'subgraph_nodes') and task.subgraph_nodes:
            features['node_count'] = len(task.subgraph_nodes)
            # subgraph_nodes is a List[str] of node IDs, not a dict
            if isinstance(task.subgraph_nodes, list):
                features['node_types'] = []  # We don't have node type info from just IDs
            else:
                # Fallback for dict format (shouldn't happen with current TaskInstance)
                features['node_types'] = [str(node.node_type) for node in task.subgraph_nodes.values()]
        
        # Content length features
        features['prompt_length'] = len(task.prompt)
        features['answer_length'] = len(task.gold_answer) if task.gold_answer else 0
        
        return features
    
    def _calculate_coverage_scores(self, candidates: List[TextTaskCandidate]):
        """Calculate coverage scores for text tasks"""
        for candidate in candidates:
            coverage_score = 0.0
            
            # Task type coverage (30%) - ensure no None values
            task_type_coverage = self._calculate_task_type_coverage(candidate.task_type) or 0.0
            coverage_score += 0.3 * task_type_coverage
            
            # Difficulty coverage (25%) - ensure no None values
            difficulty_coverage = self._calculate_difficulty_coverage(candidate.difficulty) or 0.0
            coverage_score += 0.25 * difficulty_coverage
            
            # Template coverage (20%) - ensure no None values
            template_coverage = self._calculate_template_coverage(candidate.template_id) or 0.0
            coverage_score += 0.2 * template_coverage
            
            # Variable coverage (15%) - ensure no None values
            variable_coverage = self._calculate_variable_coverage(candidate.variables) or 0.0
            coverage_score += 0.15 * variable_coverage
            
            # Content length coverage (10%) - ensure no None values
            length_coverage = self._calculate_length_coverage(candidate) or 0.0
            coverage_score += 0.1 * length_coverage
            
            candidate.coverage_score = coverage_score
    
    def _calculate_novelty_scores(self, candidates: List[TextTaskCandidate]):
        """Calculate novelty scores for text tasks"""
        if len(candidates) <= 1:
            for candidate in candidates:
                candidate.novelty_score = 1.0
            return
        
        # Use concurrent similarity detection for novelty calculation
        task_objects = []
        for candidate in candidates:
            task_obj = type('Task', (), {
                'task_id': candidate.task_id,
                'prompt': candidate.prompt,
                'description': '',
                'task_steps': [{'action_description': candidate.prompt}],
                'diversity_features': candidate.diversity_features
            })()
            task_objects.append(task_obj)
        
        # Calculate similarity matrix
        similarity_matrix = self.similarity_analyzer.similarity_detector.get_similarity_matrix(task_objects)
        
        # Calculate novelty scores based on similarity
        for candidate in candidates:
            if candidate.task_id in similarity_matrix:
                similarities = [sim for tid, sim in similarity_matrix[candidate.task_id].items() if tid != candidate.task_id]
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    candidate.novelty_score = 1.0 - avg_similarity
                else:
                    candidate.novelty_score = 1.0
            else:
                candidate.novelty_score = 1.0
            
            # Combine with quality score - ensure no None values
            coverage_score = candidate.coverage_score or 0.0
            novelty_score = candidate.novelty_score or 0.0
            quality_score = candidate.quality_score or 0.0
            
            candidate.combined_score = (
                0.4 * coverage_score +
                0.4 * novelty_score +
                0.2 * quality_score
            )
    
    def _calculate_task_type_coverage(self, task_type: str) -> float:
        """Calculate task type coverage"""
        if task_type in self.task_type_counts:
            return 0.0  # Existing type
        else:
            return 1.0  # New type
    
    def _calculate_difficulty_coverage(self, difficulty: str) -> float:
        """Calculate difficulty coverage"""
        if difficulty in self.difficulty_counts:
            return 0.0  # Existing difficulty
        else:
            return 1.0  # New difficulty
    
    def _calculate_template_coverage(self, template_id: str) -> float:
        """Calculate template coverage"""
        if template_id in self.template_counts:
            return 0.0  # Existing template
        else:
            return 1.0  # New template
    
    def _calculate_variable_coverage(self, variables: Dict[str, Any]) -> float:
        """Calculate variable coverage"""
        if not variables:
            return 0.0
        
        new_keys = 0
        for key in variables.keys():
            if key not in self.variable_key_counts:
                new_keys += 1
        
        return new_keys / len(variables) if variables else 0.0
    
    def _calculate_length_coverage(self, candidate: TextTaskCandidate) -> float:
        """Calculate content length coverage"""
        prompt_length = len(candidate.prompt)
        answer_length = len(candidate.gold_answer) if candidate.gold_answer else 0
        
        # Categorize lengths
        prompt_range = self._categorize_length(prompt_length)
        answer_range = self._categorize_length(answer_length)
        
        new_ranges = 0
        if prompt_range not in self.content_length_ranges:
            new_ranges += 1
        if answer_range not in self.content_length_ranges:
            new_ranges += 1
        
        return new_ranges / 2.0  # Normalize by number of length features
    
    def _categorize_length(self, length: int) -> str:
        """Categorize content length"""
        if length < 50:
            return "short"
        elif length < 200:
            return "medium"
        else:
            return "long"
    
    def _select_optimal_tasks(self, candidates: List[TextTaskCandidate], target_count: int) -> List[TextTaskCandidate]:
        """Select optimal tasks using MMR-like algorithm"""
        if len(candidates) <= target_count:
            return candidates
        
        # Sort by combined score - handle None values
        candidates.sort(key=lambda x: x.combined_score or 0.0, reverse=True)
        
        selected = []
        remaining = candidates.copy()
        
        # Select first task (highest score)
        if remaining:
            selected.append(remaining[0])
            remaining = remaining[1:]
        
        # Iteratively select remaining tasks
        while len(selected) < target_count and remaining:
            best_candidate = None
            best_score = -1
            
            for candidate in remaining:
                # Calculate diversity score
                diversity_score = self._calculate_diversity_score(candidate, selected)
                
                # Combined score: quality + diversity - handle None values
                candidate_combined_score = candidate.combined_score or 0.0
                diversity_score = diversity_score or 0.0
                
                combined_score = (
                    self.config.mmr_lambda * candidate_combined_score +
                    (1 - self.config.mmr_lambda) * diversity_score
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                
                # Update coverage tracking
                self._update_coverage_tracking(best_candidate)
            else:
                break
        
        return selected
    
    def _calculate_diversity_score(self, candidate: TextTaskCandidate, selected: List[TextTaskCandidate]) -> float:
        """Calculate diversity score for a candidate against selected tasks"""
        if not selected:
            return 1.0
        
        # Calculate minimum similarity to selected tasks
        min_similarity = 1.0
        for selected_candidate in selected:
            similarity = self._calculate_text_similarity(candidate, selected_candidate)
            min_similarity = min(min_similarity, similarity)
        
        return 1.0 - min_similarity
    
    def _calculate_text_similarity(self, candidate1: TextTaskCandidate, candidate2: TextTaskCandidate) -> float:
        """Calculate similarity between two text task candidates"""
        # Use the similarity detector
        task1 = type('Task', (), {
            'task_id': candidate1.task_id,
            'prompt': candidate1.prompt,
            'description': '',
            'task_steps': [{'action_description': candidate1.prompt}],
            'diversity_features': candidate1.diversity_features
        })()
        
        task2 = type('Task', (), {
            'task_id': candidate2.task_id,
            'prompt': candidate2.prompt,
            'description': '',
            'task_steps': [{'action_description': candidate2.prompt}],
            'diversity_features': candidate2.diversity_features
        })()
        
        result = self.similarity_analyzer.similarity_detector._calculate_similarity_single(
            task1, task2, SimilarityAlgorithm.HYBRID
        )
        
        if result and hasattr(result, 'similarity_score'):
            similarity_score = result.similarity_score
            if similarity_score is None:
                return 0.0
            return float(similarity_score)
        else:
            return 0.0
    
    def _update_coverage_tracking(self, candidate: TextTaskCandidate):
        """Update coverage tracking with selected candidate"""
        self.task_type_counts[candidate.task_type] += 1
        self.difficulty_counts[candidate.difficulty] += 1
        self.template_counts[candidate.template_id] += 1
        
        for key in candidate.variables.keys():
            self.variable_key_counts[key] += 1
        
        prompt_range = self._categorize_length(len(candidate.prompt))
        answer_range = self._categorize_length(len(candidate.gold_answer) if candidate.gold_answer else 0)
        self.content_length_ranges[prompt_range] += 1
        self.content_length_ranges[answer_range] += 1
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            **self.optimization_stats,
            'coverage_summary': {
                'task_types': len(self.task_type_counts),
                'difficulties': len(self.difficulty_counts),
                'templates': len(self.template_counts),
                'variable_keys': len(self.variable_key_counts),
                'length_ranges': len(self.content_length_ranges)
            },
            'similarity_stats': self.similarity_analyzer.get_cache_stats()
        }
    
    def export_optimization_report(self, output_path: str):
        """Export optimization report"""
        report = {
            'optimization_stats': self.get_optimization_stats(),
            'coverage_details': {
                'task_type_distribution': dict(self.task_type_counts),
                'difficulty_distribution': dict(self.difficulty_counts),
                'template_distribution': dict(self.template_counts),
                'variable_key_distribution': dict(self.variable_key_counts),
                'length_range_distribution': dict(self.content_length_ranges)
            },
            'config': {
                'coverage_weights': {k.value: v for k, v in self.config.coverage_weights.items()},
                'similarity_threshold': self.similarity_config.similarity_threshold,
                'mmr_lambda': self.config.mmr_lambda
            }
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"🎯 Text task optimization report exported to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting text task optimization report: {e}")