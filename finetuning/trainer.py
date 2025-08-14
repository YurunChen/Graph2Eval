"""
Model training components for fine-tuning LLMs
"""

import os
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import numpy as np

from loguru import logger

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - training functionality will be limited")

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        DataCollatorForSeq2Seq, EarlyStoppingCallback
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - HuggingFace training disabled")

try:
    from peft import (
        LoraConfig, TaskType, get_peft_model, 
        prepare_model_for_kbit_training, PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available - LoRA training disabled")

from .dataset_builder import TrainingDataset, TrainingSample


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    
    # Model settings
    model_name: str = "microsoft/DialoGPT-medium"
    model_type: str = "causal_lm"  # causal_lm, seq2seq
    tokenizer_name: Optional[str] = None
    
    # Training hyperparameters
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data settings
    max_seq_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    
    # Optimization settings
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    dataloader_num_workers: int = 0
    
    # Evaluation settings
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    
    # LoRA settings (if using LoRA)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    
    # Output settings
    output_dir: str = "models/fine_tuned"
    logging_dir: str = "logs/training"
    logging_steps: int = 100
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    def to_training_arguments(self) -> "TrainingArguments":
        """Convert to HuggingFace TrainingArguments"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
        
        return TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=self.fp16,
            dataloader_num_workers=self.dataloader_num_workers,
            evaluation_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
            report_to=None,  # Disable wandb by default
            resume_from_checkpoint=self.resume_from_checkpoint
        )


@dataclass
class TrainingResult:
    """Result of model training"""
    
    success: bool = False
    model_path: str = ""
    training_time: float = 0.0
    
    # Training metrics
    final_train_loss: float = 0.0
    final_eval_loss: float = 0.0
    best_eval_loss: float = float('inf')
    
    # Training history
    train_loss_history: List[float] = field(default_factory=list)
    eval_loss_history: List[float] = field(default_factory=list)
    learning_rate_history: List[float] = field(default_factory=list)
    
    # Model info
    model_size: int = 0
    num_parameters: int = 0
    
    # Training details
    total_steps: int = 0
    epochs_completed: float = 0.0
    early_stopped: bool = False
    
    # Error info
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "model_path": self.model_path,
            "training_time": self.training_time,
            "metrics": {
                "final_train_loss": self.final_train_loss,
                "final_eval_loss": self.final_eval_loss,
                "best_eval_loss": self.best_eval_loss
            },
            "training_history": {
                "train_loss": self.train_loss_history,
                "eval_loss": self.eval_loss_history,
                "learning_rate": self.learning_rate_history
            },
            "model_info": {
                "model_size": self.model_size,
                "num_parameters": self.num_parameters
            },
            "training_details": {
                "total_steps": self.total_steps,
                "epochs_completed": self.epochs_completed,
                "early_stopped": self.early_stopped
            },
            "error_message": self.error_message
        }


class TrainingDataset(Dataset):
    """PyTorch Dataset for training"""
    
    def __init__(self, samples: List[TrainingSample], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Create input text
        input_text = sample.prompt
        if sample.context:
            input_text += f"\n\nContext: {sample.context}"
        
        # Create target text
        target_text = sample.target_answer
        if sample.target_citations:
            target_text += f"\nCitations: {', '.join(sample.target_citations)}"
        if sample.target_reasoning:
            target_text += f"\nReasoning: {' -> '.join(sample.target_reasoning)}"
        
        # Tokenize
        full_text = f"{input_text}\n\nResponse: {target_text}"
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


class LLMTrainer(ABC):
    """Abstract base class for LLM trainers"""
    
    @abstractmethod
    def train(self, dataset: TrainingDataset, config: TrainingConfig) -> TrainingResult:
        """Train the model"""
        pass
    
    @abstractmethod
    def evaluate(self, dataset: TrainingDataset, model_path: str) -> Dict[str, float]:
        """Evaluate the trained model"""
        pass


class HuggingFaceTrainer(LLMTrainer):
    """Trainer using HuggingFace Transformers"""
    
    def __init__(self):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("HuggingFace Transformers not available")
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
    
    def train(self, training_dataset: TrainingDataset, config: TrainingConfig) -> TrainingResult:
        """Train model using HuggingFace Trainer"""
        
        start_time = time.time()
        result = TrainingResult()
        
        try:
            # Load tokenizer and model
            tokenizer = self._load_tokenizer(config)
            model = self._load_model(config)
            
            # Prepare datasets
            train_dataset = TrainingDataset(
                training_dataset.train_samples, 
                tokenizer, 
                config.max_seq_length
            )
            
            eval_dataset = None
            if training_dataset.validation_samples:
                eval_dataset = TrainingDataset(
                    training_dataset.validation_samples,
                    tokenizer,
                    config.max_seq_length
                )
            
            # Setup data collator
            if config.model_type == "causal_lm":
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False
                )
            else:
                data_collator = DataCollatorForSeq2Seq(
                    tokenizer=tokenizer
                )
            
            # Setup training arguments
            training_args = config.to_training_arguments()
            
            # Setup callbacks
            callbacks = []
            if config.early_stopping_patience > 0:
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=config.early_stopping_patience,
                        early_stopping_threshold=config.early_stopping_threshold
                    )
                )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks
            )
            
            # Train the model
            logger.info("Starting training...")
            train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
            
            # Save the model
            output_path = Path(config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            trainer.save_model(str(output_path))
            tokenizer.save_pretrained(str(output_path))
            
            # Extract training metrics
            result.success = True
            result.model_path = str(output_path)
            result.training_time = time.time() - start_time
            result.total_steps = train_result.global_step
            result.epochs_completed = train_result.epoch
            
            # Get loss history from logs
            if hasattr(train_result, 'log_history'):
                for log_entry in train_result.log_history:
                    if 'train_loss' in log_entry:
                        result.train_loss_history.append(log_entry['train_loss'])
                    if 'eval_loss' in log_entry:
                        result.eval_loss_history.append(log_entry['eval_loss'])
                        result.best_eval_loss = min(result.best_eval_loss, log_entry['eval_loss'])
                    if 'learning_rate' in log_entry:
                        result.learning_rate_history.append(log_entry['learning_rate'])
            
            # Final losses
            if result.train_loss_history:
                result.final_train_loss = result.train_loss_history[-1]
            if result.eval_loss_history:
                result.final_eval_loss = result.eval_loss_history[-1]
            
            # Model info
            result.num_parameters = sum(p.numel() for p in model.parameters())
            
            logger.info(f"Training completed successfully in {result.training_time:.2f}s")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.training_time = time.time() - start_time
            logger.error(f"Training failed: {e}")
        
        return result
    
    def evaluate(self, dataset: TrainingDataset, model_path: str) -> Dict[str, float]:
        """Evaluate trained model"""
        
        try:
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Create test dataset
            test_dataset = TrainingDataset(
                dataset.test_samples,
                tokenizer,
                512
            )
            
            # Setup trainer for evaluation
            trainer = Trainer(
                model=model,
                eval_dataset=test_dataset,
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            )
            
            # Evaluate
            eval_results = trainer.evaluate()
            
            return {
                "eval_loss": eval_results.get("eval_loss", float('inf')),
                "perplexity": eval_results.get("eval_perplexity", float('inf'))
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}
    
    def _load_tokenizer(self, config: TrainingConfig):
        """Load tokenizer"""
        tokenizer_name = config.tokenizer_name or config.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Add special tokens if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def _load_model(self, config: TrainingConfig):
        """Load model"""
        if config.model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(config.model_name)
        elif config.model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        return model


class LoRATrainer(HuggingFaceTrainer):
    """Trainer using LoRA (Low-Rank Adaptation)"""
    
    def __init__(self):
        super().__init__()
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library not available for LoRA training")
    
    def train(self, training_dataset: TrainingDataset, config: TrainingConfig) -> TrainingResult:
        """Train model using LoRA"""
        
        if not config.use_lora:
            logger.warning("LoRA not enabled in config, falling back to standard training")
            return super().train(training_dataset, config)
        
        start_time = time.time()
        result = TrainingResult()
        
        try:
            # Load tokenizer and model
            tokenizer = self._load_tokenizer(config)
            model = self._load_model(config)
            
            # Prepare model for LoRA
            model = prepare_model_for_kbit_training(model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules or self._get_default_target_modules(model),
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM if config.model_type == "causal_lm" else TaskType.SEQ_2_SEQ_LM
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Prepare datasets
            train_dataset = TrainingDataset(
                training_dataset.train_samples,
                tokenizer,
                config.max_seq_length
            )
            
            eval_dataset = None
            if training_dataset.validation_samples:
                eval_dataset = TrainingDataset(
                    training_dataset.validation_samples,
                    tokenizer,
                    config.max_seq_length
                )
            
            # Setup data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # Setup training arguments
            training_args = config.to_training_arguments()
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator
            )
            
            # Train the model
            logger.info("Starting LoRA training...")
            train_result = trainer.train()
            
            # Save the model
            output_path = Path(config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save LoRA adapters
            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))
            
            # Save LoRA config
            with open(output_path / "lora_config.json", 'w') as f:
                json.dump({
                    "r": config.lora_r,
                    "alpha": config.lora_alpha,
                    "dropout": config.lora_dropout,
                    "target_modules": config.lora_target_modules,
                    "base_model": config.model_name
                }, f, indent=2)
            
            # Fill result
            result.success = True
            result.model_path = str(output_path)
            result.training_time = time.time() - start_time
            result.total_steps = train_result.global_step
            result.epochs_completed = train_result.epoch
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            result.num_parameters = trainable_params
            
            logger.info(f"LoRA training completed successfully in {result.training_time:.2f}s")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.training_time = time.time() - start_time
            logger.error(f"LoRA training failed: {e}")
        
        return result
    
    def _get_default_target_modules(self, model) -> List[str]:
        """Get default target modules for LoRA based on model architecture"""
        
        # Common target modules for different architectures
        model_name = model.__class__.__name__.lower()
        
        if "gpt" in model_name or "opt" in model_name:
            return ["c_attn", "c_proj"]
        elif "llama" in model_name:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "t5" in model_name:
            return ["q", "v", "k", "o"]
        else:
            # Default to common attention modules
            return ["q_proj", "v_proj"]
    
    def load_lora_model(self, base_model_name: str, lora_path: str):
        """Load a trained LoRA model"""
        
        try:
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            
            # Load LoRA adapters
            model = PeftModel.from_pretrained(base_model, lora_path)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(lora_path)
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load LoRA model: {e}")
            raise


class MockTrainer(LLMTrainer):
    """Mock trainer for testing without actual model training"""
    
    def train(self, dataset: TrainingDataset, config: TrainingConfig) -> TrainingResult:
        """Mock training that simulates the training process"""
        
        logger.info("Running mock training (no actual model training)")
        
        start_time = time.time()
        
        # Simulate training time
        training_time = min(30.0, len(dataset.train_samples) * 0.1)  # Cap at 30 seconds
        time.sleep(min(5.0, training_time))  # Actually wait a bit for realism
        
        # Create mock result
        result = TrainingResult(
            success=True,
            model_path=f"{config.output_dir}/mock_model",
            training_time=training_time,
            final_train_loss=1.5 - np.random.exponential(0.3),  # Decreasing loss
            final_eval_loss=1.8 - np.random.exponential(0.2),
            best_eval_loss=1.6,
            train_loss_history=[2.0, 1.8, 1.6, 1.5],
            eval_loss_history=[2.2, 2.0, 1.8, 1.8],
            total_steps=len(dataset.train_samples) // config.batch_size * config.num_epochs,
            epochs_completed=config.num_epochs,
            num_parameters=350_000_000  # Mock parameter count
        )
        
        # Create mock output directory
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save mock model info
        with open(output_path / "training_info.json", 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Mock training completed in {result.training_time:.2f}s")
        
        return result
    
    def evaluate(self, dataset: TrainingDataset, model_path: str) -> Dict[str, float]:
        """Mock evaluation"""
        
        logger.info("Running mock evaluation")
        
        # Simulate some evaluation metrics
        return {
            "eval_loss": 1.5 + np.random.normal(0, 0.1),
            "perplexity": 4.5 + np.random.normal(0, 0.5),
            "accuracy": 0.75 + np.random.uniform(-0.1, 0.1)
        }


def get_trainer(trainer_type: str = "auto") -> LLMTrainer:
    """Get appropriate trainer based on availability and type"""
    
    if trainer_type == "mock":
        return MockTrainer()
    
    if trainer_type == "lora" or (trainer_type == "auto" and PEFT_AVAILABLE):
        if PEFT_AVAILABLE:
            return LoRATrainer()
        else:
            logger.warning("LoRA requested but PEFT not available, falling back to HuggingFace")
            trainer_type = "huggingface"
    
    if trainer_type == "huggingface" or trainer_type == "auto":
        if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            return HuggingFaceTrainer()
        else:
            logger.warning("HuggingFace/PyTorch not available, using mock trainer")
            return MockTrainer()
    
    raise ValueError(f"Unknown trainer type: {trainer_type}")


if __name__ == "__main__":
    # Test training system
    print("Testing Training System")
    print("=" * 30)
    
    # Test with mock trainer
    trainer = get_trainer("mock")
    
    # Create mock training dataset
    from .dataset_builder import TrainingSample, TrainingDataset
    
    mock_samples = [
        TrainingSample(
            sample_id="sample_1",
            prompt="What is the capital of France?",
            target_answer="The capital of France is Paris.",
            task_type="qa",
            difficulty="easy",
            quality_score=0.9
        ),
        TrainingSample(
            sample_id="sample_2", 
            prompt="Explain photosynthesis",
            target_answer="Photosynthesis is the process by which plants convert sunlight into energy.",
            task_type="explanation",
            difficulty="medium",
            quality_score=0.8
        )
    ]
    
    dataset = TrainingDataset(train_samples=mock_samples)
    
    # Test training
    config = TrainingConfig(
        model_name="gpt2",
        batch_size=2,
        num_epochs=1,
        output_dir="test_output"
    )
    
    result = trainer.train(dataset, config)
    print(f"Training result: {result.success}")
    print(f"Training time: {result.training_time:.2f}s")
    
    if result.success:
        eval_result = trainer.evaluate(dataset, result.model_path)
        print(f"Evaluation: {eval_result}")
    
    print("✅ Training system test completed")
