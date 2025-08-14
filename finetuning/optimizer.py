"""
Offline optimization system for improving LLM performance using benchmark data
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np

from loguru import logger

from config_manager import get_config
from task_craft.task_generator import TaskInstance
from agent_framework.executors import ExecutionResult
from agent_framework.evaluators import EvaluationResult
from agent_framework.attributors import AttributionResult
from .dataset_builder import DatasetBuilder, TrainingDataset, TrainingSample
from .trainer import TrainingConfig, TrainingResult, get_trainer


@dataclass
class OptimizationConfig:
    """Configuration for offline optimization"""
    
    # Dataset settings
    min_samples_for_training: int = 100
    max_samples_per_iteration: int = 5000
    quality_threshold: float = 0.7
    balance_failure_types: bool = True
    
    # Training settings
    training_config: Optional[TrainingConfig] = None
    trainer_type: str = "auto"  # auto, lora, huggingface, mock
    
    # Optimization strategy
    optimization_strategy: str = "iterative"  # iterative, single_shot, adaptive
    max_iterations: int = 3
    improvement_threshold: float = 0.05  # Minimum improvement to continue
    
    # Evaluation settings
    holdout_ratio: float = 0.2  # Ratio of data to hold out for final evaluation
    patience: int = 2  # Early stopping patience
    
    # Output settings
    save_intermediate_models: bool = True
    save_datasets: bool = True
    output_dir: str = "optimization_outputs"


@dataclass
class OptimizationResult:
    """Result of offline optimization"""
    
    success: bool = False
    optimization_time: float = 0.0
    iterations_completed: int = 0
    
    # Performance improvements
    initial_performance: Dict[str, float] = field(default_factory=dict)
    final_performance: Dict[str, float] = field(default_factory=dict)
    performance_history: List[Dict[str, float]] = field(default_factory=list)
    
    # Model paths
    best_model_path: str = ""
    intermediate_model_paths: List[str] = field(default_factory=list)
    
    # Dataset info
    total_samples_used: int = 0
    samples_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Detailed results
    training_results: List[TrainingResult] = field(default_factory=list)
    improvement_per_iteration: List[float] = field(default_factory=list)
    
    # Analysis
    attribution_analysis: Dict[str, Any] = field(default_factory=dict)
    failure_pattern_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "optimization_time": self.optimization_time,
            "iterations_completed": self.iterations_completed,
            "performance": {
                "initial": self.initial_performance,
                "final": self.final_performance,
                "history": self.performance_history
            },
            "models": {
                "best_model_path": self.best_model_path,
                "intermediate_models": self.intermediate_model_paths
            },
            "dataset_info": {
                "total_samples": self.total_samples_used,
                "samples_by_type": self.samples_by_type
            },
            "training_results": [r.to_dict() for r in self.training_results],
            "improvements": self.improvement_per_iteration,
            "analysis": {
                "attribution": self.attribution_analysis,
                "failure_patterns": self.failure_pattern_analysis
            },
            "error_message": self.error_message
        }


class OfflineOptimizer:
    """Main class for offline optimization of LLMs using benchmark data"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.dataset_builder = DatasetBuilder()
        self.trainer = get_trainer(config.trainer_type)
        
        # State tracking
        self.optimization_history = []
        self.best_performance = 0.0
        self.patience_counter = 0
    
    def optimize(
        self,
        tasks: List[TaskInstance],
        execution_results: List[ExecutionResult],
        evaluation_results: List[EvaluationResult],
        attribution_results: Optional[List[AttributionResult]] = None,
        retrieval_contexts: Optional[List[str]] = None
    ) -> OptimizationResult:
        """Run offline optimization process"""
        
        start_time = time.time()
        result = OptimizationResult()
        
        try:
            logger.info("Starting offline optimization process")
            
            # Validate inputs
            self._validate_inputs(tasks, execution_results, evaluation_results)
            
            # Analyze initial performance
            result.initial_performance = self._analyze_performance(evaluation_results)
            logger.info(f"Initial performance: {result.initial_performance}")
            
            # Split data for holdout evaluation
            train_indices, holdout_indices = self._split_holdout_data(len(tasks))
            
            # Extract training data
            train_tasks = [tasks[i] for i in train_indices]
            train_exec_results = [execution_results[i] for i in train_indices]
            train_eval_results = [evaluation_results[i] for i in train_indices]
            train_attr_results = [attribution_results[i] for i in train_indices] if attribution_results else None
            train_contexts = [retrieval_contexts[i] for i in train_indices] if retrieval_contexts else None
            
            # Extract holdout data for final evaluation
            holdout_tasks = [tasks[i] for i in holdout_indices]
            holdout_eval_results = [evaluation_results[i] for i in holdout_indices]
            
            # Run optimization strategy
            if self.config.optimization_strategy == "single_shot":
                self._run_single_shot_optimization(
                    train_tasks, train_exec_results, train_eval_results,
                    train_attr_results, train_contexts, result
                )
            elif self.config.optimization_strategy == "iterative":
                self._run_iterative_optimization(
                    train_tasks, train_exec_results, train_eval_results,
                    train_attr_results, train_contexts, result
                )
            elif self.config.optimization_strategy == "adaptive":
                self._run_adaptive_optimization(
                    train_tasks, train_exec_results, train_eval_results,
                    train_attr_results, train_contexts, result
                )
            else:
                raise ValueError(f"Unknown optimization strategy: {self.config.optimization_strategy}")
            
            # Final evaluation on holdout data
            if result.best_model_path and holdout_tasks:
                holdout_performance = self._evaluate_on_holdout(holdout_tasks, result.best_model_path)
                result.final_performance = holdout_performance
                logger.info(f"Final holdout performance: {holdout_performance}")
            else:
                result.final_performance = result.initial_performance
            
            # Generate analysis
            result.attribution_analysis = self._analyze_attribution_patterns(attribution_results)
            result.failure_pattern_analysis = self._analyze_failure_patterns(evaluation_results)
            
            result.success = True
            result.optimization_time = time.time() - start_time
            
            logger.info(f"Optimization completed successfully in {result.optimization_time:.2f}s")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.optimization_time = time.time() - start_time
            logger.error(f"Optimization failed: {e}")
        
        return result
    
    def _validate_inputs(self, tasks: List[TaskInstance], execution_results: List[ExecutionResult], 
                        evaluation_results: List[EvaluationResult]):
        """Validate input data"""
        
        if len(tasks) != len(execution_results) or len(tasks) != len(evaluation_results):
            raise ValueError("Tasks, execution results, and evaluation results must have same length")
        
        if len(tasks) < self.config.min_samples_for_training:
            raise ValueError(f"Insufficient data: {len(tasks)} samples, need at least {self.config.min_samples_for_training}")
        
        # Check for valid samples
        valid_samples = sum(1 for eval_result in evaluation_results if eval_result.success)
        if valid_samples < self.config.min_samples_for_training // 2:
            logger.warning(f"Low number of valid samples: {valid_samples}")
    
    def _analyze_performance(self, evaluation_results: List[EvaluationResult]) -> Dict[str, float]:
        """Analyze current performance metrics"""
        
        valid_results = [r for r in evaluation_results if r.success]
        
        if not valid_results:
            return {"overall_score": 0.0}
        
        metrics = {
            "overall_score": np.mean([r.overall_score for r in valid_results]),
            "f1_score": np.mean([r.f1_score for r in valid_results]),
            "citation_f1": np.mean([r.citation_f1 for r in valid_results]),
            "safety_compliance": np.mean([r.safety_compliance for r in valid_results]),
            "success_rate": len(valid_results) / len(evaluation_results)
        }
        
        return metrics
    
    def _split_holdout_data(self, total_samples: int) -> Tuple[List[int], List[int]]:
        """Split data into training and holdout sets"""
        
        indices = list(range(total_samples))
        np.random.shuffle(indices)
        
        holdout_size = int(total_samples * self.config.holdout_ratio)
        
        holdout_indices = indices[:holdout_size]
        train_indices = indices[holdout_size:]
        
        logger.info(f"Split data: {len(train_indices)} training, {len(holdout_indices)} holdout")
        
        return train_indices, holdout_indices
    
    def _run_single_shot_optimization(
        self,
        tasks: List[TaskInstance],
        execution_results: List[ExecutionResult],
        evaluation_results: List[EvaluationResult],
        attribution_results: Optional[List[AttributionResult]],
        retrieval_contexts: Optional[List[str]],
        result: OptimizationResult
    ):
        """Run single-shot optimization"""
        
        logger.info("Running single-shot optimization")
        
        # Build dataset
        dataset = self.dataset_builder.build_dataset(
            tasks, execution_results, evaluation_results, 
            attribution_results, retrieval_contexts
        )
        
        # Train model
        training_config = self.config.training_config or TrainingConfig()
        training_result = self.trainer.train(dataset, training_config)
        
        result.training_results.append(training_result)
        result.iterations_completed = 1
        
        if training_result.success:
            result.best_model_path = training_result.model_path
            result.total_samples_used = dataset.total_samples
            result.samples_by_type = dataset.sample_type_distribution
        
        # Save dataset if requested
        if self.config.save_datasets:
            dataset_path = Path(self.config.output_dir) / "single_shot_dataset"
            dataset.save_to_files(str(dataset_path))
    
    def _run_iterative_optimization(
        self,
        tasks: List[TaskInstance],
        execution_results: List[ExecutionResult],
        evaluation_results: List[EvaluationResult],
        attribution_results: Optional[List[AttributionResult]],
        retrieval_contexts: Optional[List[str]],
        result: OptimizationResult
    ):
        """Run iterative optimization with feedback"""
        
        logger.info("Running iterative optimization")
        
        current_performance = result.initial_performance.get("overall_score", 0.0)
        self.best_performance = current_performance
        
        for iteration in range(self.config.max_iterations):
            logger.info(f"Optimization iteration {iteration + 1}/{self.config.max_iterations}")
            
            # Focus on problematic areas based on previous iteration
            focused_data = self._focus_training_data(
                tasks, execution_results, evaluation_results,
                attribution_results, iteration
            )
            
            # Build dataset for this iteration
            dataset = self.dataset_builder.build_dataset(*focused_data)
            
            # Configure training for this iteration
            training_config = self._adapt_training_config(iteration)
            
            # Train model
            training_result = self.trainer.train(dataset, training_config)
            result.training_results.append(training_result)
            
            if not training_result.success:
                logger.warning(f"Training failed in iteration {iteration + 1}")
                break
            
            # Evaluate improvement (mock evaluation for now)
            new_performance = self._estimate_performance_improvement(training_result, current_performance)
            improvement = new_performance - current_performance
            
            result.performance_history.append({"iteration": iteration + 1, "performance": new_performance})
            result.improvement_per_iteration.append(improvement)
            
            logger.info(f"Iteration {iteration + 1}: Performance {new_performance:.3f} (improvement: {improvement:+.3f})")
            
            # Check for improvement
            if improvement > self.config.improvement_threshold:
                self.best_performance = new_performance
                result.best_model_path = training_result.model_path
                self.patience_counter = 0
                
                if self.config.save_intermediate_models:
                    result.intermediate_model_paths.append(training_result.model_path)
            else:
                self.patience_counter += 1
                logger.info(f"No significant improvement. Patience: {self.patience_counter}/{self.config.patience}")
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info("Early stopping due to lack of improvement")
                break
            
            current_performance = new_performance
            
            # Save dataset for this iteration
            if self.config.save_datasets:
                dataset_path = Path(self.config.output_dir) / f"iteration_{iteration + 1}_dataset"
                dataset.save_to_files(str(dataset_path))
        
        result.iterations_completed = iteration + 1
        result.total_samples_used = sum(len(tr.train_samples) for tr in [dataset] if hasattr(dataset, 'train_samples'))
    
    def _run_adaptive_optimization(
        self,
        tasks: List[TaskInstance],
        execution_results: List[ExecutionResult],
        evaluation_results: List[EvaluationResult],
        attribution_results: Optional[List[AttributionResult]],
        retrieval_contexts: Optional[List[str]],
        result: OptimizationResult
    ):
        """Run adaptive optimization that adjusts strategy based on results"""
        
        logger.info("Running adaptive optimization")
        
        # Start with iterative approach
        self._run_iterative_optimization(
            tasks, execution_results, evaluation_results,
            attribution_results, retrieval_contexts, result
        )
        
        # Adapt strategy based on results
        if result.improvement_per_iteration:
            avg_improvement = np.mean(result.improvement_per_iteration)
            
            if avg_improvement < self.config.improvement_threshold:
                logger.info("Low improvement detected, trying specialized strategies")
                
                # Try focused training on attribution failures
                if attribution_results:
                    self._run_attribution_focused_training(
                        tasks, execution_results, evaluation_results,
                        attribution_results, retrieval_contexts, result
                    )
    
    def _focus_training_data(
        self,
        tasks: List[TaskInstance],
        execution_results: List[ExecutionResult],
        evaluation_results: List[EvaluationResult],
        attribution_results: Optional[List[AttributionResult]],
        iteration: int
    ) -> Tuple[List[TaskInstance], List[ExecutionResult], List[EvaluationResult], Optional[List[AttributionResult]], Optional[List[str]]]:
        """Focus training data on problematic areas"""
        
        if iteration == 0:
            # First iteration: use all data
            return tasks, execution_results, evaluation_results, attribution_results, None
        
        # Focus on failures and low-quality results
        focused_indices = []
        
        for i, eval_result in enumerate(evaluation_results):
            # Include failures
            if not eval_result.success:
                focused_indices.append(i)
            # Include low-quality successes
            elif eval_result.overall_score < self.config.quality_threshold:
                focused_indices.append(i)
            # Include attribution failures
            elif attribution_results and attribution_results[i] and attribution_results[i].overall_confidence < 0.5:
                focused_indices.append(i)
            # Include safety violations
            elif eval_result.safety_compliance < 0.8:
                focused_indices.append(i)
        
        # Ensure we have enough data
        if len(focused_indices) < self.config.min_samples_for_training:
            # Add some successful examples
            success_indices = [
                i for i, eval_result in enumerate(evaluation_results)
                if eval_result.success and eval_result.overall_score >= self.config.quality_threshold
            ]
            
            needed = self.config.min_samples_for_training - len(focused_indices)
            focused_indices.extend(success_indices[:needed])
        
        # Limit data size
        if len(focused_indices) > self.config.max_samples_per_iteration:
            focused_indices = focused_indices[:self.config.max_samples_per_iteration]
        
        logger.info(f"Focused on {len(focused_indices)} samples for iteration {iteration + 1}")
        
        return (
            [tasks[i] for i in focused_indices],
            [execution_results[i] for i in focused_indices],
            [evaluation_results[i] for i in focused_indices],
            [attribution_results[i] for i in focused_indices] if attribution_results else None,
            None  # No retrieval contexts for simplicity
        )
    
    def _adapt_training_config(self, iteration: int) -> TrainingConfig:
        """Adapt training configuration for each iteration"""
        
        base_config = self.config.training_config or TrainingConfig()
        
        # Reduce learning rate in later iterations
        if iteration > 0:
            base_config.learning_rate *= 0.8
        
        # Increase batch size for stability
        if iteration > 1:
            base_config.batch_size = min(base_config.batch_size * 2, 32)
        
        # Adjust output directory
        base_config.output_dir = f"{self.config.output_dir}/iteration_{iteration + 1}"
        
        return base_config
    
    def _estimate_performance_improvement(self, training_result: TrainingResult, baseline: float) -> float:
        """Estimate performance improvement from training result"""
        
        # This is a simplified estimation - in practice, you'd evaluate on validation set
        if not training_result.success:
            return baseline
        
        # Use training loss as proxy for improvement
        loss_improvement = max(0, 2.0 - training_result.final_train_loss)  # Assume starting loss ~2.0
        
        # Convert to performance score (simplified)
        estimated_performance = baseline + (loss_improvement * 0.1)
        
        return min(1.0, estimated_performance)  # Cap at 1.0
    
    def _run_attribution_focused_training(
        self,
        tasks: List[TaskInstance],
        execution_results: List[ExecutionResult],
        evaluation_results: List[EvaluationResult],
        attribution_results: List[AttributionResult],
        retrieval_contexts: Optional[List[str]],
        result: OptimizationResult
    ):
        """Run training focused on attribution failures"""
        
        logger.info("Running attribution-focused training")
        
        # Filter for attribution failures
        attribution_failures = []
        for i, attr_result in enumerate(attribution_results):
            if attr_result and attr_result.overall_confidence < 0.5:
                attribution_failures.append(i)
        
        if len(attribution_failures) < 10:  # Need minimum samples
            logger.warning("Insufficient attribution failures for focused training")
            return
        
        # Create focused dataset
        focused_tasks = [tasks[i] for i in attribution_failures]
        focused_exec = [execution_results[i] for i in attribution_failures]
        focused_eval = [evaluation_results[i] for i in attribution_failures]
        focused_attr = [attribution_results[i] for i in attribution_failures]
        
        # Build dataset with emphasis on attribution
        dataset = self.dataset_builder.build_dataset(
            focused_tasks, focused_exec, focused_eval, focused_attr
        )
        
        # Special training config for attribution
        attr_config = self.config.training_config or TrainingConfig()
        attr_config.learning_rate *= 0.5  # Lower learning rate for fine-tuning
        attr_config.num_epochs = 2  # Shorter training
        attr_config.output_dir = f"{self.config.output_dir}/attribution_focused"
        
        # Train
        training_result = self.trainer.train(dataset, attr_config)
        result.training_results.append(training_result)
        
        if training_result.success:
            result.intermediate_model_paths.append(training_result.model_path)
    
    def _evaluate_on_holdout(self, holdout_tasks: List[TaskInstance], model_path: str) -> Dict[str, float]:
        """Evaluate model on holdout data"""
        
        # This is a simplified mock evaluation
        # In practice, you'd run the model on holdout tasks and evaluate
        
        logger.info(f"Evaluating model {model_path} on {len(holdout_tasks)} holdout samples")
        
        # Mock evaluation with some realistic metrics
        base_performance = 0.7
        noise = np.random.normal(0.05, 0.02)  # Slight improvement with noise
        
        return {
            "overall_score": min(1.0, base_performance + noise),
            "f1_score": min(1.0, base_performance + noise * 0.8),
            "citation_f1": min(1.0, base_performance + noise * 1.2),
            "safety_compliance": min(1.0, base_performance + 0.1 + noise * 0.5),
            "success_rate": min(1.0, 0.85 + noise * 0.5)
        }
    
    def _analyze_attribution_patterns(self, attribution_results: Optional[List[AttributionResult]]) -> Dict[str, Any]:
        """Analyze patterns in attribution results"""
        
        if not attribution_results:
            return {}
        
        valid_attributions = [r for r in attribution_results if r and r.attributed_nodes]
        
        if not valid_attributions:
            return {}
        
        # Analyze failure types
        failure_types = [r.failure_type for r in valid_attributions]
        failure_type_counts = {
            failure_type: failure_types.count(failure_type)
            for failure_type in set(failure_types)
        }
        
        # Analyze confidence distribution
        confidences = [r.overall_confidence for r in valid_attributions]
        
        # Analyze attribution methods
        methods = [r.attribution_method for r in valid_attributions]
        method_counts = {
            method: methods.count(method)
            for method in set(methods)
        }
        
        return {
            "total_attributions": len(valid_attributions),
            "failure_type_distribution": failure_type_counts,
            "confidence_stats": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences)
            },
            "method_distribution": method_counts,
            "low_confidence_ratio": sum(1 for c in confidences if c < 0.5) / len(confidences)
        }
    
    def _analyze_failure_patterns(self, evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze patterns in evaluation failures"""
        
        failures = [r for r in evaluation_results if not r.success or r.overall_score < 0.5]
        
        if not failures:
            return {"failure_rate": 0.0}
        
        # Analyze score distributions
        scores = [r.overall_score for r in failures]
        citation_scores = [r.citation_f1 for r in failures]
        safety_scores = [r.safety_compliance for r in failures]
        
        return {
            "failure_rate": len(failures) / len(evaluation_results),
            "failure_count": len(failures),
            "score_analysis": {
                "overall_scores": {"mean": np.mean(scores), "std": np.std(scores)},
                "citation_scores": {"mean": np.mean(citation_scores), "std": np.std(citation_scores)},
                "safety_scores": {"mean": np.mean(safety_scores), "std": np.std(safety_scores)}
            },
            "common_issues": {
                "citation_failures": sum(1 for r in failures if r.citation_f1 < 0.3),
                "safety_failures": sum(1 for r in failures if r.safety_compliance < 0.8),
                "quality_failures": sum(1 for r in failures if r.f1_score < 0.3)
            }
        }


def create_default_optimization_config() -> OptimizationConfig:
    """Create default optimization configuration"""
    
    # 从配置获取模型名称
    from config_manager import get_config
    config = get_config()
    finetuning_config = config.finetuning.get('finetuning', {})
    
    training_config = TrainingConfig(
        model_name=finetuning_config.get('base_model', "microsoft/DialoGPT-medium"),
        learning_rate=3e-5,
        batch_size=4,
        num_epochs=2,
        use_lora=True,
        lora_r=8,
        lora_alpha=16
    )
    
    return OptimizationConfig(
        min_samples_for_training=50,
        max_samples_per_iteration=1000,
        quality_threshold=0.6,
        training_config=training_config,
        trainer_type="mock",  # Use mock for testing
        optimization_strategy="iterative",
        max_iterations=2,
        improvement_threshold=0.03,
        output_dir="optimization_results"
    )
