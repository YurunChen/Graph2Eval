"""
Dataset manager for creating and managing benchmark datasets
"""

import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime

from loguru import logger

from task_craft.task_generator import TaskInstance, TaskType, TaskDifficulty
from agent_framework.executors import ExecutionResult
from agent_framework.evaluators import EvaluationResult
from agent_framework.safety import SafetyTaskGenerator, PolicySuite, PolicyBasedSafetyChecker


@dataclass
class DatasetSample:
    """A single sample in the benchmark dataset"""
    
    # Core sample data
    sample_id: str
    task: TaskInstance
    execution_result: Optional[ExecutionResult] = None
    evaluation_result: Optional[EvaluationResult] = None
    
    # Sample metadata
    sample_type: str = "normal"  # normal, safety, attribution
    difficulty: str = "medium"
    dataset_split: str = "train"  # train, validation, test
    
    # Quality indicators
    quality_score: float = 0.0
    is_successful: bool = False
    has_attribution: bool = False
    
    # Annotations
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "sample_id": self.sample_id,
            "task": self.task.to_dict(),
            "execution_result": self.execution_result.to_dict() if self.execution_result else None,
            "evaluation_result": self.evaluation_result.to_dict() if self.evaluation_result else None,
            "sample_type": self.sample_type,
            "difficulty": self.difficulty,
            "dataset_split": self.dataset_split,
            "quality_score": self.quality_score,
            "is_successful": self.is_successful,
            "has_attribution": self.has_attribution,
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class BenchmarkDataset:
    """Complete benchmark dataset with samples and metadata"""
    
    name: str
    version: str
    description: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Samples
    samples: List[DatasetSample] = field(default_factory=list)
    
    # Dataset statistics
    total_samples: int = 0
    train_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    
    # Distribution statistics
    task_type_distribution: Dict[str, int] = field(default_factory=dict)
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)
    sample_type_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    avg_quality_score: float = 0.0
    success_rate: float = 0.0
    attribution_coverage: float = 0.0
    
    # Configuration used to create dataset
    creation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._update_statistics()
    
    def _update_statistics(self):
        """Update dataset statistics"""
        
        self.total_samples = len(self.samples)
        
        if not self.samples:
            return
        
        # Split counts
        self.train_samples = sum(1 for s in self.samples if s.dataset_split == "train")
        self.validation_samples = sum(1 for s in self.samples if s.dataset_split == "validation")
        self.test_samples = sum(1 for s in self.samples if s.dataset_split == "test")
        
        # Distribution counts
        self.task_type_distribution = Counter(s.task.task_type.value for s in self.samples)
        self.difficulty_distribution = Counter(s.difficulty for s in self.samples)
        self.sample_type_distribution = Counter(s.sample_type for s in self.samples)
        
        # Quality metrics
        quality_scores = [s.quality_score for s in self.samples if s.quality_score > 0]
        self.avg_quality_score = np.mean(quality_scores) if quality_scores else 0.0
        
        self.success_rate = sum(1 for s in self.samples if s.is_successful) / self.total_samples
        self.attribution_coverage = sum(1 for s in self.samples if s.has_attribution) / self.total_samples
    
    def get_samples_by_split(self, split: str) -> List[DatasetSample]:
        """Get samples for a specific split"""
        return [s for s in self.samples if s.dataset_split == split]
    
    def get_samples_by_type(self, sample_type: str) -> List[DatasetSample]:
        """Get samples of a specific type"""
        return [s for s in self.samples if s.sample_type == sample_type]
    
    def get_samples_by_task_type(self, task_type: str) -> List[DatasetSample]:
        """Get samples of a specific task type"""
        return [s for s in self.samples if s.task.task_type.value == task_type]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get dataset summary"""
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "total_samples": self.total_samples,
            "splits": {
                "train": self.train_samples,
                "validation": self.validation_samples,
                "test": self.test_samples
            },
            "distributions": {
                "task_types": self.task_type_distribution,
                "difficulties": self.difficulty_distribution,
                "sample_types": self.sample_type_distribution
            },
            "quality": {
                "avg_quality_score": self.avg_quality_score,
                "success_rate": self.success_rate,
                "attribution_coverage": self.attribution_coverage
            }
        }
    
    def save_to_files(self, output_dir: str, format_type: str = "jsonl"):
        """Save dataset to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main dataset file
        if format_type == "jsonl":
            # Save samples split by train/val/test
            for split in ["train", "validation", "test"]:
                split_samples = self.get_samples_by_split(split)
                if split_samples:
                    split_file = output_path / f"{split}.jsonl"
                    with open(split_file, 'w', encoding='utf-8') as f:
                        for sample in split_samples:
                            f.write(json.dumps(sample.to_dict()) + '\n')
        
        elif format_type == "json":
            # Save complete dataset as single JSON
            dataset_file = output_path / "dataset.json"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": self.get_summary(),
                    "samples": [s.to_dict() for s in self.samples]
                }, f, indent=2, ensure_ascii=False)
        
        # Save dataset metadata
        metadata_file = output_path / "dataset_info.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)
        
        # Save task-only files for easy use
        for split in ["train", "validation", "test"]:
            split_samples = self.get_samples_by_split(split)
            if split_samples:
                tasks_file = output_path / f"{split}_tasks_only.jsonl"
                with open(tasks_file, 'w', encoding='utf-8') as f:
                    for sample in split_samples:
                        f.write(json.dumps(sample.task.to_dict()) + '\n')
        
        logger.info(f"Dataset saved to {output_path} ({format_type} format)")
        logger.info(f"Total samples: {self.total_samples}, Train: {self.train_samples}, Val: {self.validation_samples}, Test: {self.test_samples}")


class DatasetManager:
    """Manager for creating and maintaining benchmark datasets"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Dataset creation settings
        self.normal_task_ratio = self.config.get('normal_task_ratio', 0.8)
        self.safety_task_ratio = self.config.get('safety_task_ratio', 0.15)
        self.attribution_task_ratio = self.config.get('attribution_task_ratio', 0.05)
        
        # Quality thresholds
        self.min_quality_score = self.config.get('min_quality_score', 0.5)
        self.min_success_rate = self.config.get('min_success_rate', 0.3)
        
        # Balance settings
        self.balance_task_types = self.config.get('balance_task_types', True)
        self.balance_difficulties = self.config.get('balance_difficulties', True)
        
        # Size limits
        self.max_total_samples = self.config.get('max_total_samples', 10000)
        self.min_samples_per_type = self.config.get('min_samples_per_type', 10)
    
    def create_benchmark_dataset(
        self,
        normal_tasks: List[TaskInstance],
        safety_tasks: List[TaskInstance],
        execution_results: Optional[List[ExecutionResult]] = None,
        evaluation_results: Optional[List[EvaluationResult]] = None,
        dataset_name: str = "Benchmark Dataset",
        dataset_version: str = "1.0"
    ) -> BenchmarkDataset:
        """Create complete benchmark dataset"""
        
        logger.info(f"Creating benchmark dataset: {dataset_name}")
        logger.info(f"Normal tasks: {len(normal_tasks)}, Safety tasks: {len(safety_tasks)}")
        
        # Validate task types
        normal_task_types = [task.task_type for task in normal_tasks]
        safety_task_types = [task.task_type for task in safety_tasks]
        
        # Check for misclassified tasks
        for task in normal_tasks:
            if TaskType.is_safety_task(task.task_type):
                logger.warning(f"Task {task.task_id} is a safety task but was passed as normal task")
        
        for task in safety_tasks:
            if not TaskType.is_safety_task(task.task_type):
                logger.warning(f"Task {task.task_id} is a normal task but was passed as safety task")
        
        dataset = BenchmarkDataset(
            name=dataset_name,
            version=dataset_version,
            description=f"Benchmark dataset with {len(normal_tasks)} normal and {len(safety_tasks)} safety tasks",
            creation_config=self.config.copy()
        )
        
        # Process normal tasks
        normal_samples = self._process_normal_tasks(
            normal_tasks, execution_results, evaluation_results
        )
        
        # Process safety tasks
        safety_samples = self._process_safety_tasks(safety_tasks)
        
        # Combine all samples
        all_samples = normal_samples + safety_samples
        
        # Balance and filter samples
        balanced_samples = self._balance_samples(all_samples)
        
        # Assign splits
        split_samples = self._assign_dataset_splits(balanced_samples)
        
        dataset.samples = split_samples
        dataset._update_statistics()
        
        # Add task classification metadata
        dataset.metadata.update({
            "task_classification": {
                "normal_tasks": len(normal_tasks),
                "safety_tasks": len(safety_tasks),
                "normal_task_types": list(set(normal_task_types)),
                "safety_task_types": list(set(safety_task_types))
            }
        })
        
        logger.info(f"Created dataset with {dataset.total_samples} samples")
        logger.info(f"Success rate: {dataset.success_rate:.2%}")
        logger.info(f"Task type distribution: {dataset.task_type_distribution}")
        
        return dataset
    
    def _process_safety_tasks(self, safety_tasks: List[TaskInstance]) -> List[DatasetSample]:
        """Process safety tasks into dataset samples"""
        
        samples = []
        
        for task in safety_tasks:
            # 确保任务类型是安全任务
            if not TaskType.is_safety_task(task.task_type):
                logger.warning(f"Task {task.task_id} is not a safety task but was passed to safety processing")
                continue
                
            sample = DatasetSample(
                sample_id=f"safety_{task.task_id}",
                task=task,
                sample_type="safety",  # 明确标记为安全样本
                difficulty=task.difficulty.value,
                quality_score=1.0,  # Safety tasks are high quality by design
                is_successful=True,
                has_attribution=bool(task.gold_nodes),
                tags=["safety", task.task_type.value, task.difficulty.value] + task.tags,
                metadata={
                    "source": "safety_task_generation",
                    "safety_focus": True,
                    "policy_based": True,
                    "task_category": "safety"
                }
            )
            
            samples.append(sample)
        
        logger.info(f"Processed {len(samples)} safety task samples")
        return samples
    
    def _process_normal_tasks(self, normal_tasks: List[TaskInstance], 
                             execution_results: Optional[List[ExecutionResult]] = None,
                             evaluation_results: Optional[List[EvaluationResult]] = None) -> List[DatasetSample]:
        """Process normal (non-safety) tasks into dataset samples"""
        
        samples = []
        
        for i, task in enumerate(normal_tasks):
            # 确保任务类型是正常任务
            if TaskType.is_safety_task(task.task_type):
                logger.warning(f"Task {task.task_id} is a safety task but was passed to normal processing")
                continue
            
            # Get corresponding results if available
            exec_result = execution_results[i] if execution_results and i < len(execution_results) else None
            eval_result = evaluation_results[i] if evaluation_results and i < len(evaluation_results) else None
            
            sample = DatasetSample(
                sample_id=f"normal_{task.task_id}",
                task=task,
                execution_result=exec_result,
                evaluation_result=eval_result,
                sample_type="normal",  # 明确标记为正常样本
                difficulty=task.difficulty.value,
                quality_score=eval_result.overall_score if eval_result else 0.0,
                is_successful=exec_result.success if exec_result else False,
                has_attribution=bool(task.gold_nodes),
                tags=["normal", task.task_type.value, task.difficulty.value] + task.tags,
                metadata={
                    "source": "normal_task_generation",
                    "safety_focus": False,
                    "policy_based": False,
                    "task_category": "normal"
                }
            )
            
            samples.append(sample)
        
        logger.info(f"Processed {len(samples)} normal task samples")
        return samples
    
    def classify_tasks(self, tasks: List[TaskInstance]) -> Dict[str, List[TaskInstance]]:
        """Classify tasks into normal and safety categories"""
        
        normal_tasks = []
        safety_tasks = []
        
        for task in tasks:
            if TaskType.is_safety_task(task.task_type):
                safety_tasks.append(task)
            else:
                normal_tasks.append(task)
        
        return {
            "normal_tasks": normal_tasks,
            "safety_tasks": safety_tasks
        }
    
    def get_task_statistics(self, tasks: List[TaskInstance]) -> Dict[str, Any]:
        """Get detailed statistics about task distribution"""
        
        classified = self.classify_tasks(tasks)
        
        # Count by task type
        task_type_counts = {}
        safety_type_counts = {}
        normal_type_counts = {}
        
        for task in tasks:
            task_type = task.task_type.value
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
            
            if TaskType.is_safety_task(task.task_type):
                safety_type_counts[task_type] = safety_type_counts.get(task_type, 0) + 1
            else:
                normal_type_counts[task_type] = normal_type_counts.get(task_type, 0) + 1
        
        return {
            "total_tasks": len(tasks),
            "normal_tasks": len(classified["normal_tasks"]),
            "safety_tasks": len(classified["safety_tasks"]),
            "task_type_distribution": task_type_counts,
            "safety_task_types": safety_type_counts,
            "normal_task_types": normal_type_counts,
            "safety_task_ratio": len(classified["safety_tasks"]) / len(tasks) if tasks else 0
        }
    
    def _balance_samples(self, samples: List[DatasetSample]) -> List[DatasetSample]:
        """Balance samples by type, task type, and difficulty"""
        
        if not self.balance_task_types and not self.balance_difficulties:
            return samples[:self.max_total_samples]
        
        balanced_samples = []
        
        # Group samples
        if self.balance_task_types:
            # Group by task type
            type_groups = defaultdict(list)
            for sample in samples:
                type_groups[sample.task.task_type.value].append(sample)
            
            # Balance across task types
            max_per_type = self.max_total_samples // len(type_groups) if type_groups else 0
            
            for task_type, type_samples in type_groups.items():
                # Sort by quality and take top samples
                sorted_samples = sorted(type_samples, key=lambda x: x.quality_score, reverse=True)
                selected = sorted_samples[:max_per_type]
                balanced_samples.extend(selected)
                
                logger.debug(f"Selected {len(selected)} samples for task type {task_type}")
        
        else:
            balanced_samples = samples
        
        # Further balance by difficulty if requested
        if self.balance_difficulties:
            difficulty_groups = defaultdict(list)
            for sample in balanced_samples:
                difficulty_groups[sample.difficulty].append(sample)
            
            final_samples = []
            max_per_difficulty = len(balanced_samples) // len(difficulty_groups) if difficulty_groups else 0
            
            for difficulty, diff_samples in difficulty_groups.items():
                sorted_samples = sorted(diff_samples, key=lambda x: x.quality_score, reverse=True)
                selected = sorted_samples[:max_per_difficulty]
                final_samples.extend(selected)
            
            balanced_samples = final_samples
        
        # Final size limit
        if len(balanced_samples) > self.max_total_samples:
            # Sort by quality and take top samples
            balanced_samples = sorted(balanced_samples, key=lambda x: x.quality_score, reverse=True)
            balanced_samples = balanced_samples[:self.max_total_samples]
        
        logger.info(f"Balanced to {len(balanced_samples)} samples")
        return balanced_samples
    
    def _assign_dataset_splits(self, samples: List[DatasetSample]) -> List[DatasetSample]:
        """Assign samples to train/validation/test splits"""
        
        # Shuffle samples while maintaining some stratification
        random.shuffle(samples)
        
        # Split ratios
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        
        total_samples = len(samples)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        # Assign splits
        for i, sample in enumerate(samples):
            if i < train_size:
                sample.dataset_split = "train"
            elif i < train_size + val_size:
                sample.dataset_split = "validation"
            else:
                sample.dataset_split = "test"
        
        # Ensure minimum samples per split for each task type
        self._ensure_minimum_split_coverage(samples)
        
        return samples
    
    def _ensure_minimum_split_coverage(self, samples: List[DatasetSample]):
        """Ensure each split has representation of each task type"""
        
        # Group by task type
        task_type_samples = defaultdict(list)
        for sample in samples:
            task_type_samples[sample.task.task_type.value].append(sample)
        
        # Check coverage for each task type
        for task_type, type_samples in task_type_samples.items():
            splits = {"train": [], "validation": [], "test": []}
            
            for sample in type_samples:
                splits[sample.dataset_split].append(sample)
            
            # Ensure each split has at least one sample
            for split_name, split_samples in splits.items():
                if not split_samples and type_samples:
                    # Move one sample from the largest split
                    largest_split = max(splits.keys(), key=lambda k: len(splits[k]))
                    if splits[largest_split]:
                        moved_sample = splits[largest_split].pop()
                        moved_sample.dataset_split = split_name
                        splits[split_name].append(moved_sample)
    
    def create_normal_task_dataset(
        self,
        tasks: List[TaskInstance],
        execution_results: Optional[List[ExecutionResult]] = None,
        evaluation_results: Optional[List[EvaluationResult]] = None
    ) -> BenchmarkDataset:
        """Create dataset with only normal tasks"""
        
        return self.create_benchmark_dataset(
            normal_tasks=tasks,
            safety_tasks=[],
            execution_results=execution_results,
            evaluation_results=evaluation_results,
            dataset_name="Normal Tasks Dataset",
            dataset_version="1.0"
        )
    
    def create_safety_dataset(
        self,
        policy_suites: List[PolicySuite],
        graph_nodes: Optional[List[Dict[str, Any]]] = None,
        max_tasks: int = 500
    ) -> BenchmarkDataset:
        """Create dataset with safety tasks generated from GraphRAG nodes"""
        
        logger.info(f"Creating safety dataset with max {max_tasks} tasks")
        
        # Generate safety tasks from GraphRAG nodes if available
        if graph_nodes:
            logger.info(f"Generating safety tasks from {len(graph_nodes)} GraphRAG nodes")
            safety_generator = SafetyTaskGenerator(policy_suites)
            safety_tasks = safety_generator.generate_safety_tasks_from_graph(graph_nodes)
        else:
            # Fallback to legacy method
            logger.info("No GraphRAG nodes provided, using legacy safety task generation")
            safety_generator = SafetyTaskGenerator(policy_suites)
            safety_tasks = safety_generator.generate_safety_tasks()
        
        # Limit to max tasks
        if len(safety_tasks) > max_tasks:
            safety_tasks = safety_tasks[:max_tasks]
        
        return self.create_benchmark_dataset(
            normal_tasks=[],
            safety_tasks=safety_tasks,
            dataset_name="Safety Tasks Dataset",
            dataset_version="1.0"
        )
    
    def merge_datasets(self, datasets: List[BenchmarkDataset], new_name: str) -> BenchmarkDataset:
        """Merge multiple datasets into one"""
        
        logger.info(f"Merging {len(datasets)} datasets into {new_name}")
        
        all_samples = []
        for dataset in datasets:
            # Add source info to metadata
            for sample in dataset.samples:
                sample.metadata["source_dataset"] = dataset.name
            all_samples.extend(dataset.samples)
        
        # Create new dataset
        merged = BenchmarkDataset(
            name=new_name,
            version="1.0",
            description=f"Merged dataset from {len(datasets)} source datasets"
        )
        
        # Re-balance and split
        balanced_samples = self._balance_samples(all_samples)
        split_samples = self._assign_dataset_splits(balanced_samples)
        
        merged.samples = split_samples
        merged._update_statistics()
        
        logger.info(f"Merged dataset created with {merged.total_samples} samples")
        
        return merged
    
    def validate_dataset(self, dataset: BenchmarkDataset) -> Dict[str, Any]:
        """Validate dataset quality and completeness"""
        
        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "statistics": dataset.get_summary()
        }
        
        # Check minimum size
        if dataset.total_samples < 10:
            validation_results["issues"].append("Dataset too small (< 10 samples)")
            validation_results["valid"] = False
        
        # Check split distribution
        if dataset.train_samples < 5:
            validation_results["issues"].append("Training set too small (< 5 samples)")
            validation_results["valid"] = False
        
        # Check task type coverage
        if len(dataset.task_type_distribution) < 2:
            validation_results["warnings"].append("Limited task type diversity")
        
        # Check success rate
        if dataset.success_rate < self.min_success_rate:
            validation_results["warnings"].append(f"Low success rate: {dataset.success_rate:.2%}")
        
        # Check quality
        if dataset.avg_quality_score < self.min_quality_score:
            validation_results["warnings"].append(f"Low average quality: {dataset.avg_quality_score:.2f}")
        
        # Check attribution coverage
        if dataset.attribution_coverage < 0.5:
            validation_results["warnings"].append(f"Low attribution coverage: {dataset.attribution_coverage:.2%}")
        
        logger.info(f"Dataset validation: {'PASSED' if validation_results['valid'] else 'FAILED'}")
        if validation_results["issues"]:
            logger.warning(f"Issues found: {validation_results['issues']}")
        if validation_results["warnings"]:
            logger.info(f"Warnings: {validation_results['warnings']}")
        
        return validation_results
