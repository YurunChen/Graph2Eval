"""
Text Task Types - Separate from Web Task Types
This file contains only text-based task types and templates
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

class TextTaskType(Enum):
    """Types of text-based tasks that can be generated"""
    # Basic comprehension
    EXTRACTION = "extraction"  # Extract specific information
    SUMMARIZATION = "summarization"  # Summarize content
    COMPREHENSION = "comprehension"  # Answer questions about content
    
    # Analysis and reasoning
    COMPARISON = "comparison"  # Compare multiple pieces of information
    ANALYSIS = "analysis"  # Analyze relationships or patterns
    REASONING = "reasoning"  # Multi-step reasoning tasks
    
    # Aggregation and synthesis
    AGGREGATION = "aggregation"  # Combine information from multiple sources
    SYNTHESIS = "synthesis"  # Create new insights from existing information
    
    # Specialized tasks
    TABLE_QA = "table_qa"  # Question answering over tables
    FIGURE_INTERPRETATION = "figure_interpretation"  # Interpret figures/charts
    CROSS_REFERENCE = "cross_reference"  # Follow cross-references
    
    # Safety and compliance tasks
    FACT_VERIFICATION = "fact_verification"  # Verify factual claims
    
    # Dynamic threat types (for LLM-based safety tasks)
    CONTENT_INJECTION = "content_injection"  # Content injection threat
    PROMPT_MANIPULATION = "prompt_manipulation"  # Prompt manipulation threat
    CONTEXT_SWITCHING = "context_switching"  # Context switching threat
    INDIRECT_REFERENCE = "indirect_reference"  # Indirect reference threat
    
    @classmethod
    def is_safety_task(cls, task_type) -> bool:
        """Check if a task type is a safety-related task"""
        safety_types = {
            # Dynamic threat types
            cls.CONTENT_INJECTION,
            cls.PROMPT_MANIPULATION,
            cls.CONTEXT_SWITCHING,
            cls.INDIRECT_REFERENCE,
        }
        return task_type in safety_types
    
    @classmethod
    def is_normal_task(cls, task_type) -> bool:
        """Check if a task type is a normal (non-safety) task"""
        return not cls.is_safety_task(task_type)
    
    @classmethod
    def from_strategy(cls, strategy: str) -> 'TextTaskType':
        """Convert embedding strategy to TextTaskType"""
        strategy_mapping = {
            'content_injection': cls.CONTENT_INJECTION,
            'prompt_manipulation': cls.PROMPT_MANIPULATION,
            'context_switching': cls.CONTEXT_SWITCHING,
            'indirect_reference': cls.INDIRECT_REFERENCE,
            
            # Text task strategies
            'extraction': cls.EXTRACTION,
            'summarization': cls.SUMMARIZATION,
            'comprehension': cls.COMPREHENSION,
            'comparison': cls.COMPARISON,
            'analysis': cls.ANALYSIS,
            'reasoning': cls.REASONING,
            'aggregation': cls.AGGREGATION,
            'synthesis': cls.SYNTHESIS,
            'table_qa': cls.TABLE_QA,
            'figure_interpretation': cls.FIGURE_INTERPRETATION,
            'cross_reference': cls.CROSS_REFERENCE,
            'fact_verification': cls.FACT_VERIFICATION,
        }
        return strategy_mapping.get(strategy, cls.COMPREHENSION)  # Default to comprehension

@dataclass
class TextTaskTemplate:
    """Template for text-based tasks"""
    template_id: str
    task_type: TextTaskType
    prompt_template: str
    answer_template: str
    required_node_types: List[str] = field(default_factory=list)
    required_edge_types: List[str] = field(default_factory=list)
    difficulty: str = "MEDIUM"
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "task_type": self.task_type.value,
            "prompt_template": self.prompt_template,
            "answer_template": self.answer_template,
            "required_node_types": self.required_node_types,
            "required_edge_types": self.required_edge_types,
            "difficulty": self.difficulty,
            "description": self.description
        }



