"""
Dataset builder for creating fine-tuning datasets from benchmark results
"""

import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

from loguru import logger

from ..task_craft.task_generator import TaskInstance
from ..agent_framework.executors import ExecutionResult
from ..agent_framework.evaluators import EvaluationResult
from ..agent_framework.attributors import AttributionResult


@dataclass
class TrainingSample:
    """A single training sample for fine-tuning"""
    
    # Input components
    prompt: str
    context: str = ""
    
    # Target outputs
    target_answer: str = ""
    target_citations: List[str] = field(default_factory=list)
    target_reasoning: List[str] = field(default_factory=list)
    
    # Sample metadata
    sample_id: str = ""
    task_type: str = ""
    difficulty: str = ""
    sample_type: str = "success"  # success, failure_corrected, attribution
    
    # Quality indicators
    quality_score: float = 1.0
    attribution_confidence: float = 0.0
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "context": self.context,
            "target_answer": self.target_answer,
            "target_citations": self.target_citations,
            "target_reasoning": self.target_reasoning,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "sample_type": self.sample_type,
            "quality_score": self.quality_score,
            "attribution_confidence": self.attribution_confidence,
            "metadata": self.metadata
        }
    
    def to_training_format(self, format_type: str = "instruction") -> Dict[str, str]:
        """Convert to specific training format"""
        
        if format_type == "instruction":
            # Standard instruction-following format
            instruction = self.prompt
            if self.context:
                instruction += f"\n\nContext:\n{self.context}"
            
            response_parts = [self.target_answer]
            
            if self.target_citations:
                response_parts.append(f"Citations: {', '.join(self.target_citations)}")
            
            if self.target_reasoning:
                response_parts.append(f"Reasoning: {' -> '.join(self.target_reasoning)}")
            
            return {
                "instruction": instruction,
                "response": "\n\n".join(response_parts)
            }
        
        elif format_type == "chat":
            # Chat format for conversational models
            return {
                "messages": [
                    {"role": "user", "content": self.prompt},
                    {"role": "assistant", "content": self.target_answer}
                ]
            }
        
        elif format_type == "alpaca":
            # Alpaca format
            return {
                "instruction": self.prompt,
                "input": self.context,
                "output": self.target_answer
            }
        
        else:
            raise ValueError(f"Unknown format type: {format_type}")


@dataclass
class TrainingDataset:
    """Complete training dataset with splits"""
    
    train_samples: List[TrainingSample] = field(default_factory=list)
    validation_samples: List[TrainingSample] = field(default_factory=list)
    test_samples: List[TrainingSample] = field(default_factory=list)
    
    # Dataset metadata
    total_samples: int = 0
    task_type_distribution: Dict[str, int] = field(default_factory=dict)
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)
    sample_type_distribution: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        self._update_statistics()
    
    def _update_statistics(self):
        """Update dataset statistics"""
        all_samples = self.train_samples + self.validation_samples + self.test_samples
        self.total_samples = len(all_samples)
        
        self.task_type_distribution = Counter(sample.task_type for sample in all_samples)
        self.difficulty_distribution = Counter(sample.difficulty for sample in all_samples)
        self.sample_type_distribution = Counter(sample.sample_type for sample in all_samples)
    
    def save_to_files(self, output_dir: str, format_type: str = "jsonl"):
        """Save dataset to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each split
        splits = {
            "train": self.train_samples,
            "validation": self.validation_samples,
            "test": self.test_samples
        }
        
        for split_name, samples in splits.items():
            if not samples:
                continue
                
            if format_type == "jsonl":
                file_path = output_path / f"{split_name}.jsonl"
                with open(file_path, 'w', encoding='utf-8') as f:
                    for sample in samples:
                        f.write(json.dumps(sample.to_dict()) + '\n')
            
            elif format_type == "json":
                file_path = output_path / f"{split_name}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([sample.to_dict() for sample in samples], f, indent=2, ensure_ascii=False)
            
            elif format_type in ["instruction", "chat", "alpaca"]:
                file_path = output_path / f"{split_name}_{format_type}.jsonl"
                with open(file_path, 'w', encoding='utf-8') as f:
                    for sample in samples:
                        formatted = sample.to_training_format(format_type)
                        f.write(json.dumps(formatted) + '\n')
        
        # Save dataset metadata
        metadata = {
            "total_samples": self.total_samples,
            "train_samples": len(self.train_samples),
            "validation_samples": len(self.validation_samples),
            "test_samples": len(self.test_samples),
            "task_type_distribution": self.task_type_distribution,
            "difficulty_distribution": self.difficulty_distribution,
            "sample_type_distribution": self.sample_type_distribution,
            "format_type": format_type
        }
        
        with open(output_path / "dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved dataset to {output_path} ({format_type} format)")
        logger.info(f"Train: {len(self.train_samples)}, Val: {len(self.validation_samples)}, Test: {len(self.test_samples)}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get dataset summary"""
        return {
            "total_samples": self.total_samples,
            "splits": {
                "train": len(self.train_samples),
                "validation": len(self.validation_samples),
                "test": len(self.test_samples)
            },
            "distributions": {
                "task_types": self.task_type_distribution,
                "difficulties": self.difficulty_distribution,
                "sample_types": self.sample_type_distribution
            },
            "quality_stats": {
                "avg_quality_score": np.mean([s.quality_score for s in self.train_samples + self.validation_samples + self.test_samples]) if self.total_samples > 0 else 0,
                "min_quality_score": np.min([s.quality_score for s in self.train_samples + self.validation_samples + self.test_samples]) if self.total_samples > 0 else 0,
                "max_quality_score": np.max([s.quality_score for s in self.train_samples + self.validation_samples + self.test_samples]) if self.total_samples > 0 else 0
            }
        }


class DatasetBuilder:
    """Builds training datasets from benchmark execution results"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Quality thresholds
        self.min_success_quality = self.config.get('min_success_quality', 0.7)
        self.min_attribution_confidence = self.config.get('min_attribution_confidence', 0.5)
        
        # Sample limits
        self.max_samples_per_type = self.config.get('max_samples_per_type', 1000)
        self.balance_task_types = self.config.get('balance_task_types', True)
        
        # Dataset splits
        self.train_ratio = self.config.get('train_ratio', 0.7)
        self.val_ratio = self.config.get('val_ratio', 0.15)
        self.test_ratio = self.config.get('test_ratio', 0.15)
    
    def build_dataset(
        self,
        tasks: List[TaskInstance],
        execution_results: List[ExecutionResult],
        evaluation_results: List[EvaluationResult],
        attribution_results: Optional[List[AttributionResult]] = None,
        retrieval_contexts: Optional[List[str]] = None
    ) -> TrainingDataset:
        """Build complete training dataset from benchmark results"""
        
        logger.info(f"Building dataset from {len(tasks)} tasks")
        
        # Validate input lengths
        if len(tasks) != len(execution_results) or len(tasks) != len(evaluation_results):
            raise ValueError("Tasks, execution results, and evaluation results must have same length")
        
        if attribution_results and len(attribution_results) != len(tasks):
            logger.warning("Attribution results length mismatch, will skip attribution samples")
            attribution_results = None
        
        if retrieval_contexts and len(retrieval_contexts) != len(tasks):
            logger.warning("Retrieval contexts length mismatch, will use empty contexts")
            retrieval_contexts = [""]*len(tasks)
        
        all_samples = []
        
        # Process each task result
        for i, (task, exec_result, eval_result) in enumerate(zip(tasks, execution_results, evaluation_results)):
            context = retrieval_contexts[i] if retrieval_contexts else ""
            attribution = attribution_results[i] if attribution_results else None
            
            # Generate samples from this task
            task_samples = self._process_task_result(task, exec_result, eval_result, attribution, context)
            all_samples.extend(task_samples)
        
        logger.info(f"Generated {len(all_samples)} total samples")
        
        # Balance and filter samples
        filtered_samples = self._filter_and_balance_samples(all_samples)
        
        # Split into train/val/test
        dataset = self._split_dataset(filtered_samples)
        
        return dataset
    
    def _process_task_result(
        self,
        task: TaskInstance,
        exec_result: ExecutionResult,
        eval_result: EvaluationResult,
        attribution: Optional[AttributionResult],
        context: str
    ) -> List[TrainingSample]:
        """Process a single task result to generate training samples"""
        
        samples = []
        
        # Success sample (if execution was successful and quality is good)
        if exec_result.success and eval_result.overall_score >= self.min_success_quality:
            success_sample = self._create_success_sample(task, exec_result, eval_result, context)
            samples.append(success_sample)
        
        # Failure correction sample (if we have gold answer to correct with)
        elif not exec_result.success or eval_result.overall_score < self.min_success_quality:
            if task.gold_answer:
                corrected_sample = self._create_corrected_sample(task, exec_result, eval_result, context)
                samples.append(corrected_sample)
        
        # Attribution sample (if attribution is available and confident)
        if (attribution and 
            attribution.overall_confidence >= self.min_attribution_confidence and
            attribution.attributed_nodes):
            
            attribution_sample = self._create_attribution_sample(task, exec_result, attribution, context)
            samples.append(attribution_sample)
        
        # Safety correction sample (if safety violations detected)
        if eval_result.safety_compliance < 0.8:
            safety_sample = self._create_safety_sample(task, exec_result, eval_result, context)
            samples.append(safety_sample)
        
        return samples
    
    def _create_success_sample(
        self,
        task: TaskInstance,
        exec_result: ExecutionResult,
        eval_result: EvaluationResult,
        context: str
    ) -> TrainingSample:
        """Create training sample from successful execution"""
        
        return TrainingSample(
            sample_id=f"success_{task.task_id}",
            prompt=task.prompt,
            context=context,
            target_answer=exec_result.answer,
            target_citations=exec_result.citations,
            target_reasoning=exec_result.reasoning_path,
            task_type=task.task_type.value,
            difficulty=task.difficulty.value,
            sample_type="success",
            quality_score=eval_result.overall_score,
            metadata={
                "original_task_id": task.task_id,
                "execution_time": exec_result.execution_time,
                "evaluation_scores": eval_result.to_dict()
            }
        )
    
    def _create_corrected_sample(
        self,
        task: TaskInstance,
        exec_result: ExecutionResult,
        eval_result: EvaluationResult,
        context: str
    ) -> TrainingSample:
        """Create training sample with corrected response"""
        
        # Use gold answer as target, but improve it with citations and reasoning
        target_citations = task.gold_nodes if task.gold_nodes else []
        
        # Generate improved reasoning if task requires it
        target_reasoning = []
        if task.requires_reasoning_path:
            target_reasoning = self._generate_reasoning_path(task.prompt, task.gold_answer)
        
        return TrainingSample(
            sample_id=f"corrected_{task.task_id}",
            prompt=task.prompt,
            context=context,
            target_answer=task.gold_answer,
            target_citations=target_citations,
            target_reasoning=target_reasoning,
            task_type=task.task_type.value,
            difficulty=task.difficulty.value,
            sample_type="failure_corrected",
            quality_score=1.0,  # Perfect since using gold answer
            metadata={
                "original_task_id": task.task_id,
                "failure_type": eval_result.details.get("failure_type", "unknown"),
                "correction_applied": True
            }
        )
    
    def _create_attribution_sample(
        self,
        task: TaskInstance,
        exec_result: ExecutionResult,
        attribution: AttributionResult,
        context: str
    ) -> TrainingSample:
        """Create training sample focusing on attribution"""
        
        # Create a prompt that emphasizes citation requirements
        attribution_prompt = f"{task.prompt}\n\nIMPORTANT: Provide specific citations for all information used in your response."
        
        # Use attributed nodes as target citations
        target_citations = [node_id for node_id, confidence in attribution.attributed_nodes if confidence > 0.5]
        
        # Use gold answer if available, otherwise the original answer
        target_answer = task.gold_answer if task.gold_answer else exec_result.answer
        
        return TrainingSample(
            sample_id=f"attribution_{task.task_id}",
            prompt=attribution_prompt,
            context=context,
            target_answer=target_answer,
            target_citations=target_citations,
            target_reasoning=exec_result.reasoning_path,
            task_type=task.task_type.value,
            difficulty=task.difficulty.value,
            sample_type="attribution",
            quality_score=0.8,  # High quality for attribution training
            attribution_confidence=attribution.overall_confidence,
            metadata={
                "original_task_id": task.task_id,
                "attribution_method": attribution.attribution_method,
                "failure_type": attribution.failure_type,
                "primary_cause": attribution.primary_cause
            }
        )
    
    def _create_safety_sample(
        self,
        task: TaskInstance,
        exec_result: ExecutionResult,
        eval_result: EvaluationResult,
        context: str
    ) -> TrainingSample:
        """Create training sample for safety improvement"""
        
        # Create a safe response that acknowledges the safety concern
        safe_answer = f"I understand your question, but I need to be careful about safety considerations. "
        
        if task.gold_answer:
            # Use gold answer if it's safe
            if eval_result.safety_compliance > 0.8:
                safe_answer += task.gold_answer
            else:
                safe_answer += "I'd recommend consulting with appropriate professionals for this type of question."
        else:
            safe_answer += "I can help with related topics that don't raise safety concerns."
        
        return TrainingSample(
            sample_id=f"safety_{task.task_id}",
            prompt=task.prompt,
            context=context,
            target_answer=safe_answer,
            target_citations=task.gold_nodes if task.gold_nodes else [],
            target_reasoning=["Consider safety implications", "Provide responsible response"],
            task_type=task.task_type.value,
            difficulty=task.difficulty.value,
            sample_type="safety_corrected",
            quality_score=1.0,  # High quality for safety
            metadata={
                "original_task_id": task.task_id,
                "safety_issues": eval_result.details.get("safety", {}),
                "original_safety_score": eval_result.safety_compliance
            }
        )
    
    def _generate_reasoning_path(self, prompt: str, answer: str) -> List[str]:
        """Generate reasoning steps for a prompt and answer (simplified)"""
        
        # This is a simplified reasoning generator
        # In practice, this could be more sophisticated
        
        reasoning_steps = [
            "Analyze the question and identify key information needed",
            "Review available context and sources",
            f"Synthesize information to form answer: {answer[:50]}...",
            "Verify answer completeness and accuracy"
        ]
        
        return reasoning_steps
    
    def _filter_and_balance_samples(self, samples: List[TrainingSample]) -> List[TrainingSample]:
        """Filter and balance samples by quality and type"""
        
        # Filter by quality
        quality_filtered = [s for s in samples if s.quality_score >= 0.5]
        
        logger.info(f"Quality filtered: {len(quality_filtered)} from {len(samples)}")
        
        if not self.balance_task_types:
            return quality_filtered[:self.max_samples_per_type * 4]  # Rough limit
        
        # Balance by task type and sample type
        balanced_samples = []
        
        # Group by task type and sample type
        grouped = defaultdict(lambda: defaultdict(list))
        for sample in quality_filtered:
            grouped[sample.task_type][sample.sample_type].append(sample)
        
        # Balance within each group
        max_per_group = max(1, self.max_samples_per_type // len(grouped))
        
        for task_type, type_groups in grouped.items():
            for sample_type, type_samples in type_groups.items():
                # Sort by quality and take top samples
                sorted_samples = sorted(type_samples, key=lambda x: x.quality_score, reverse=True)
                selected = sorted_samples[:max_per_group]
                balanced_samples.extend(selected)
        
        logger.info(f"Balanced samples: {len(balanced_samples)}")
        return balanced_samples
    
    def _split_dataset(self, samples: List[TrainingSample]) -> TrainingDataset:
        """Split samples into train/validation/test sets"""
        
        # Shuffle samples
        random.shuffle(samples)
        
        # Calculate split sizes
        total_samples = len(samples)
        train_size = int(total_samples * self.train_ratio)
        val_size = int(total_samples * self.val_ratio)
        
        # Split samples
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        dataset = TrainingDataset(
            train_samples=train_samples,
            validation_samples=val_samples,
            test_samples=test_samples
        )
        
        logger.info(f"Dataset split - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
        
        return dataset
    
    def augment_dataset(self, dataset: TrainingDataset, augmentation_factor: float = 1.5) -> TrainingDataset:
        """Augment dataset with variations"""
        
        logger.info(f"Augmenting dataset with factor {augmentation_factor}")
        
        augmented_train = list(dataset.train_samples)  # Copy existing
        
        # Generate variations of existing samples
        target_size = int(len(dataset.train_samples) * augmentation_factor)
        samples_to_generate = target_size - len(dataset.train_samples)
        
        for _ in range(samples_to_generate):
            # Select random sample to augment
            original = random.choice(dataset.train_samples)
            
            # Create variation
            augmented = self._create_sample_variation(original)
            augmented_train.append(augmented)
        
        return TrainingDataset(
            train_samples=augmented_train,
            validation_samples=dataset.validation_samples,
            test_samples=dataset.test_samples
        )
    
    def _create_sample_variation(self, original: TrainingSample) -> TrainingSample:
        """Create a variation of an existing sample"""
        
        # Simple variations - in practice could be more sophisticated
        variations = [
            self._paraphrase_prompt,
            self._add_instruction_emphasis,
            self._modify_context_order
        ]
        
        variation_func = random.choice(variations)
        return variation_func(original)
    
    def _paraphrase_prompt(self, sample: TrainingSample) -> TrainingSample:
        """Create paraphrased version of prompt"""
        
        # Simple paraphrasing by adding request variations
        paraphrase_prefixes = [
            "Please analyze the following and ",
            "Based on the information provided, ",
            "Considering the given context, ",
            "I need you to "
        ]
        
        prefix = random.choice(paraphrase_prefixes)
        new_prompt = prefix + sample.prompt.lower()
        
        # Copy sample with modified prompt
        new_sample = TrainingSample(
            sample_id=f"{sample.sample_id}_para",
            prompt=new_prompt,
            context=sample.context,
            target_answer=sample.target_answer,
            target_citations=sample.target_citations,
            target_reasoning=sample.target_reasoning,
            task_type=sample.task_type,
            difficulty=sample.difficulty,
            sample_type=f"{sample.sample_type}_augmented",
            quality_score=sample.quality_score * 0.9,  # Slightly lower quality
            metadata={**sample.metadata, "augmentation": "paraphrase"}
        )
        
        return new_sample
    
    def _add_instruction_emphasis(self, sample: TrainingSample) -> TrainingSample:
        """Add emphasis to important instructions"""
        
        emphasis_additions = [
            "\n\nPlease be thorough in your analysis.",
            "\n\nMake sure to cite all sources used.",
            "\n\nProvide step-by-step reasoning.",
            "\n\nEnsure your response is accurate and complete."
        ]
        
        addition = random.choice(emphasis_additions)
        new_prompt = sample.prompt + addition
        
        new_sample = TrainingSample(
            sample_id=f"{sample.sample_id}_emph",
            prompt=new_prompt,
            context=sample.context,
            target_answer=sample.target_answer,
            target_citations=sample.target_citations,
            target_reasoning=sample.target_reasoning,
            task_type=sample.task_type,
            difficulty=sample.difficulty,
            sample_type=f"{sample.sample_type}_augmented",
            quality_score=sample.quality_score * 0.95,
            metadata={**sample.metadata, "augmentation": "emphasis"}
        )
        
        return new_sample
    
    def _modify_context_order(self, sample: TrainingSample) -> TrainingSample:
        """Modify the order of context information"""
        
        if not sample.context or len(sample.context.split('\n')) < 2:
            return sample  # Can't modify if insufficient context
        
        # Split context into parts and shuffle
        context_parts = sample.context.split('\n\n')
        if len(context_parts) > 1:
            random.shuffle(context_parts)
            new_context = '\n\n'.join(context_parts)
        else:
            new_context = sample.context
        
        new_sample = TrainingSample(
            sample_id=f"{sample.sample_id}_reorder",
            prompt=sample.prompt,
            context=new_context,
            target_answer=sample.target_answer,
            target_citations=sample.target_citations,
            target_reasoning=sample.target_reasoning,
            task_type=sample.task_type,
            difficulty=sample.difficulty,
            sample_type=f"{sample.sample_type}_augmented",
            quality_score=sample.quality_score * 0.9,
            metadata={**sample.metadata, "augmentation": "context_reorder"}
        )
        
        return new_sample
