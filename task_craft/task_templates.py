"""
Task templates for TaskCraft-style automatic task generation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Set
from enum import Enum
import json
import re
from jinja2 import Template


class TaskType(Enum):
    """Types of tasks that can be generated"""
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
    
    # Web task types
    SEARCH = "Search"  # Search functionality, filtering
    FORM_FILLING = "Form Filling"  # Complete forms, login, registration
    NAVIGATION = "Navigation"  # Multi-page navigation, link following
    DATA_EXTRACTION = "Data Extraction"  # Extract information from pages
    E_COMMERCE = "E-commerce"  # Product browsing, shopping cart
    CONTENT_BROWSING = "Content Browsing"  # Reading articles, exploring content
    
    # Dynamic threat types (for LLM-based safety tasks)
    CONTENT_INJECTION = "content_injection"  # Content injection threat
    PROMPT_MANIPULATION = "prompt_manipulation"  # Prompt manipulation threat
    CONTEXT_SWITCHING = "context_switching"  # Context switching threat
    INDIRECT_REFERENCE = "indirect_reference"  # Indirect reference threat
    
    # Web safety task types
    WEB_MALICIOUS_INPUT = "web_malicious_input"  # Malicious input detection
    WEB_PHISHING_DETECTION = "web_phishing_detection"  # Phishing detection
    WEB_DATA_PRIVACY = "web_data_privacy"  # Data privacy protection
    WEB_ACCESS_CONTROL = "web_access_control"  # Access control validation
    WEB_CONTENT_MODERATION = "web_content_moderation"  # Content moderation
    WEB_FORM_VALIDATION = "web_form_validation"  # Form validation
    WEB_NAVIGATION_SAFETY = "web_navigation_safety"  # Navigation safety
    
    @classmethod
    def is_safety_task(cls, task_type) -> bool:
        """Check if a task type is a safety-related task"""
        safety_types = {
            # Dynamic threat types
            cls.CONTENT_INJECTION,
            cls.PROMPT_MANIPULATION,
            cls.CONTEXT_SWITCHING,
            cls.INDIRECT_REFERENCE,
            
            # Web safety types
            cls.WEB_MALICIOUS_INPUT,
            cls.WEB_PHISHING_DETECTION,
            cls.WEB_DATA_PRIVACY,
            cls.WEB_ACCESS_CONTROL,
            cls.WEB_CONTENT_MODERATION,
            cls.WEB_FORM_VALIDATION,
            cls.WEB_NAVIGATION_SAFETY
        }
        return task_type in safety_types
    
    @classmethod
    def is_normal_task(cls, task_type) -> bool:
        """Check if a task type is a normal (non-safety) task"""
        return not cls.is_safety_task(task_type)
    
    @classmethod
    def from_strategy(cls, strategy: str) -> 'TaskType':
        """Convert embedding strategy to TaskType"""
        strategy_mapping = {
            'content_injection': cls.CONTENT_INJECTION,
            'prompt_manipulation': cls.PROMPT_MANIPULATION,
            'context_switching': cls.CONTEXT_SWITCHING,
            'indirect_reference': cls.INDIRECT_REFERENCE,
            
            # Web safety strategies
            'malicious_input': cls.WEB_MALICIOUS_INPUT,
            'phishing_detection': cls.WEB_PHISHING_DETECTION,
            'data_privacy': cls.WEB_DATA_PRIVACY,
            'access_control': cls.WEB_ACCESS_CONTROL,
            'content_moderation': cls.WEB_CONTENT_MODERATION,
            'form_validation': cls.WEB_FORM_VALIDATION,
            'navigation_safety': cls.WEB_NAVIGATION_SAFETY,
            
            # Web task strategies
            'search': cls.SEARCH,
            'form_filling': cls.FORM_FILLING,
            'navigation': cls.NAVIGATION,
            'data_extraction': cls.DATA_EXTRACTION,
            'e_commerce': cls.E_COMMERCE,
            'content_browsing': cls.CONTENT_BROWSING,
            # Also support the actual values used by web task generator
            'Search': cls.SEARCH,
            'Form Filling': cls.FORM_FILLING,
            'Navigation': cls.NAVIGATION,
            'Data Extraction': cls.DATA_EXTRACTION,
            'E-commerce': cls.E_COMMERCE,
            'Content Browsing': cls.CONTENT_BROWSING
        }
        return strategy_mapping.get(strategy, cls.CONTENT_INJECTION)  # Default to content_injection


class TaskDifficulty(Enum):
    """Task difficulty levels"""
    EASY = "easy"      # Single-hop, direct information
    MEDIUM = "medium"  # 2-3 hops, some reasoning required
    HARD = "hard"      # Multi-hop, complex reasoning
    EXPERT = "expert"  # Domain expertise required


class RequiredCapability(Enum):
    """Capabilities required to solve tasks"""
    READING_COMPREHENSION = "reading_comprehension"
    INFORMATION_EXTRACTION = "information_extraction"
    LOGICAL_REASONING = "logical_reasoning"
    NUMERICAL_COMPUTATION = "numerical_computation"
    TEMPORAL_REASONING = "temporal_reasoning"
    SPATIAL_REASONING = "spatial_reasoning"
    CAUSAL_REASONING = "causal_reasoning"
    MULTI_HOP_REASONING = "multi_hop_reasoning"
    CROSS_MODAL_UNDERSTANDING = "cross_modal_understanding"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    COMMON_SENSE = "common_sense"
    SAFETY_AWARENESS = "safety_awareness"
    # Web-specific capabilities
    FORM_FILLING = "form_filling"
    NAVIGATION = "navigation"
    SEARCH = "search"
    FILTERING = "filtering"
    CONTENT_BROWSING = "content_browsing"
    E_COMMERCE = "e_commerce"


@dataclass
class TaskTemplate:
    """Template for generating tasks from graph patterns"""
    
    template_id: str
    name: str
    description: str
    task_type: TaskType
    difficulty: TaskDifficulty
    required_capabilities: List[RequiredCapability]
    
    # Template content
    prompt_template: str  # Jinja2 template for the task prompt
    gold_answer_template: Optional[str] = None  # Template for expected answer
    
    # Graph pattern requirements
    required_node_types: List[str] = field(default_factory=list)
    required_edge_types: List[str] = field(default_factory=list)
    min_nodes: int = 1
    max_nodes: int = 10
    max_hops: int = 2
    
    # Evaluation criteria
    evaluation_metrics: List[str] = field(default_factory=list)
    requires_exact_match: bool = False
    requires_citations: bool = True
    requires_reasoning_path: bool = False
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    created_by: str = "system"
    
    def __post_init__(self):
        """Validate template after initialization"""
        if not self.prompt_template:
            raise ValueError("prompt_template cannot be empty")
        
        # Set default evaluation metrics based on task type
        if not self.evaluation_metrics:
            self.evaluation_metrics = self._get_default_metrics()
    
    def _get_default_metrics(self) -> List[str]:
        """Get default evaluation metrics for task type"""
        metrics_map = {
            TaskType.EXTRACTION: ["exact_match", "f1_score"],
            TaskType.SUMMARIZATION: ["rouge_score", "bert_score"],
            TaskType.COMPREHENSION: ["exact_match", "f1_score"],
            TaskType.COMPARISON: ["accuracy", "f1_score"],
            TaskType.ANALYSIS: ["accuracy", "reasoning_quality"],
            TaskType.REASONING: ["accuracy", "reasoning_path_match"],
            TaskType.AGGREGATION: ["completeness", "accuracy"],
            TaskType.SYNTHESIS: ["novelty", "coherence", "accuracy"],
            TaskType.TABLE_QA: ["exact_match", "numerical_accuracy"],
            TaskType.FIGURE_INTERPRETATION: ["accuracy", "completeness"],
            TaskType.CROSS_REFERENCE: ["citation_f1", "link_accuracy"],
            TaskType.FACT_VERIFICATION: ["accuracy", "confidence"],
            
            # Web task types
            TaskType.SEARCH: ["accuracy", "completeness"],
            TaskType.FORM_FILLING: ["accuracy", "completeness"],
            TaskType.NAVIGATION: ["accuracy", "completeness"],
            TaskType.DATA_EXTRACTION: ["exact_match", "f1_score"],
            TaskType.E_COMMERCE: ["accuracy", "completeness"],
            TaskType.CONTENT_BROWSING: ["accuracy", "completeness"],
            
            # Dynamic threat types
            TaskType.CONTENT_INJECTION: ["precision", "recall"],
            TaskType.PROMPT_MANIPULATION: ["precision", "recall"],
            TaskType.CONTEXT_SWITCHING: ["precision", "recall"],
            TaskType.INDIRECT_REFERENCE: ["precision", "recall"],
            
            # Web safety types
            TaskType.WEB_MALICIOUS_INPUT: ["precision", "recall"],
            TaskType.WEB_PHISHING_DETECTION: ["precision", "recall"],
            TaskType.WEB_DATA_PRIVACY: ["precision", "recall"],
            TaskType.WEB_ACCESS_CONTROL: ["precision", "recall"],
            TaskType.WEB_CONTENT_MODERATION: ["precision", "recall"],
            TaskType.WEB_FORM_VALIDATION: ["precision", "recall"],
            TaskType.WEB_NAVIGATION_SAFETY: ["precision", "recall"]
        }
        return metrics_map.get(self.task_type, ["accuracy"])
    
    def can_apply_to_subgraph(self, node_types: List[str], edge_types: List[str], node_count: int) -> bool:
        """Check if template can be applied to a subgraph"""
        # Check node count constraints
        if not (self.min_nodes <= node_count <= self.max_nodes):
            return False
        
        # Check required node types
        available_node_types = set(node_types)
        required_node_types = set(self.required_node_types)
        if not required_node_types.issubset(available_node_types):
            return False
        
        # Check required edge types
        available_edge_types = set(edge_types)
        required_edge_types = set(self.required_edge_types)
        if not required_edge_types.issubset(available_edge_types):
            return False
        
        return True
    
    def render_prompt(self, variables: Dict[str, Any]) -> str:
        """Render the prompt template with variables"""
        template = Template(self.prompt_template)
        return template.render(**variables)
    
    def render_gold_answer(self, variables: Dict[str, Any]) -> Optional[str]:
        """Render the gold answer template with variables"""
        if not self.gold_answer_template:
            return None
        
        template = Template(self.gold_answer_template)
        return template.render(**variables)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary"""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type.value,
            "difficulty": self.difficulty.value,
            "required_capabilities": [cap.value for cap in self.required_capabilities],
            "prompt_template": self.prompt_template,
            "gold_answer_template": self.gold_answer_template,
            "required_node_types": self.required_node_types,
            "required_edge_types": self.required_edge_types,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "max_hops": self.max_hops,
            "evaluation_metrics": self.evaluation_metrics,
            "requires_exact_match": self.requires_exact_match,
            "requires_citations": self.requires_citations,
            "requires_reasoning_path": self.requires_reasoning_path,
            "tags": self.tags,
            "version": self.version,
            "created_by": self.created_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskTemplate':
        """Create template from dictionary"""
        return cls(
            template_id=data["template_id"],
            name=data["name"],
            description=data["description"],
            task_type=TaskType(data["task_type"]),
            difficulty=TaskDifficulty(data["difficulty"]),
            required_capabilities=[RequiredCapability(cap) for cap in data["required_capabilities"]],
            prompt_template=data["prompt_template"],
            gold_answer_template=data.get("gold_answer_template"),
            required_node_types=data.get("required_node_types", []),
            required_edge_types=data.get("required_edge_types", []),
            min_nodes=data.get("min_nodes", 1),
            max_nodes=data.get("max_nodes", 10),
            max_hops=data.get("max_hops", 2),
            evaluation_metrics=data.get("evaluation_metrics", []),
            requires_exact_match=data.get("requires_exact_match", False),
            requires_citations=data.get("requires_citations", True),
            requires_reasoning_path=data.get("requires_reasoning_path", False),
            tags=data.get("tags", []),
            version=data.get("version", "1.0"),
            created_by=data.get("created_by", "system")
        )


class TaskTemplateLibrary:
    """Library of predefined task templates"""
    
    def __init__(self):
        self.templates = {}
        self._load_default_templates()
    
    def add_template(self, template: TaskTemplate):
        """Add a template to the library"""
        self.templates[template.template_id] = template
    
    def get_template(self, template_id: str) -> Optional[TaskTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)
    
    def get_templates_by_type(self, task_type: TaskType) -> List[TaskTemplate]:
        """Get all templates of a specific type"""
        return [t for t in self.templates.values() if t.task_type == task_type]
    
    def get_templates_by_difficulty(self, difficulty: TaskDifficulty) -> List[TaskTemplate]:
        """Get all templates of a specific difficulty"""
        return [t for t in self.templates.values() if t.difficulty == difficulty]
    
    def find_applicable_templates(
        self, 
        node_types: List[str], 
        edge_types: List[str], 
        node_count: int,
        task_types: Optional[List[TaskType]] = None,
        difficulties: Optional[List[TaskDifficulty]] = None
    ) -> List[TaskTemplate]:
        """Find templates that can be applied to a subgraph"""
        applicable = []
        
        for template in self.templates.values():
            # Filter by task type if specified
            if task_types and template.task_type not in task_types:
                continue
            
            # Filter by difficulty if specified
            if difficulties and template.difficulty not in difficulties:
                continue
            
            # Check if template can apply to subgraph
            if template.can_apply_to_subgraph(node_types, edge_types, node_count):
                applicable.append(template)
        
        return applicable
    
    def save_to_file(self, file_path: str):
        """Save template library to JSON file"""
        data = {
            "templates": [template.to_dict() for template in self.templates.values()]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, file_path: str):
        """Load template library from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.templates.clear()
        for template_data in data["templates"]:
            template = TaskTemplate.from_dict(template_data)
            self.add_template(template)
    
    def _load_default_templates(self):
        """Load default task templates"""
        # Basic extraction template
        self.add_template(TaskTemplate(
            template_id="basic_extraction",
            name="Basic Information Extraction",
            description="Extract specific information from a single node",
            task_type=TaskType.EXTRACTION,
            difficulty=TaskDifficulty.EASY,
            required_capabilities=[RequiredCapability.READING_COMPREHENSION, RequiredCapability.INFORMATION_EXTRACTION],
            prompt_template="Extract the following information from the given text:\n\n{{ content }}\n\nQuestion: {{ question }}",
            gold_answer_template="{{ answer }}",
            required_node_types=["paragraph"],
            min_nodes=1,
            max_nodes=1,
            requires_citations=True
        ))
        
        # Table QA template
        self.add_template(TaskTemplate(
            template_id="table_qa",
            name="Table Question Answering",
            description="Answer questions about tabular data",
            task_type=TaskType.TABLE_QA,
            difficulty=TaskDifficulty.MEDIUM,
            required_capabilities=[RequiredCapability.READING_COMPREHENSION, RequiredCapability.NUMERICAL_COMPUTATION],
            prompt_template="Based on the following table:\n\n{{ table_content }}\n\nAnswer the question: {{ question }}\n\nProvide your answer and cite the specific cells or rows used.",
            gold_answer_template="{{ answer }}",
            required_node_types=["table"],
            min_nodes=1,
            max_nodes=3,
            evaluation_metrics=["exact_match", "numerical_accuracy", "citation_f1"],
            requires_citations=True
        ))
        
        # Multi-hop reasoning template
        self.add_template(TaskTemplate(
            template_id="multi_hop_reasoning",
            name="Multi-hop Reasoning",
            description="Reasoning task requiring information from multiple connected nodes",
            task_type=TaskType.REASONING,
            difficulty=TaskDifficulty.HARD,
            required_capabilities=[RequiredCapability.MULTI_HOP_REASONING, RequiredCapability.LOGICAL_REASONING],
            prompt_template="Based on the following information:\n\n{% for node in nodes %}{{ loop.index }}. {{ node.content }}\n\n{% endfor %}Answer the question by connecting information from multiple sources: {{ question }}\n\nProvide your reasoning steps and cite the sources used.",
            gold_answer_template="{{ answer }}",
            required_node_types=["paragraph"],
            required_edge_types=["semantic_sim", "refers_to"],
            min_nodes=2,
            max_nodes=5,
            max_hops=3,
            evaluation_metrics=["accuracy", "reasoning_path_match", "citation_f1"],
            requires_citations=True,
            requires_reasoning_path=True
        ))
        
        # Comparison template
        self.add_template(TaskTemplate(
            template_id="comparison_analysis",
            name="Comparison Analysis",
            description="Compare information from different sources",
            task_type=TaskType.COMPARISON,
            difficulty=TaskDifficulty.MEDIUM,
            required_capabilities=[RequiredCapability.LOGICAL_REASONING, RequiredCapability.READING_COMPREHENSION],
            prompt_template="Compare the following pieces of information:\n\n{% for item in comparison_items %}{{ loop.index }}. {{ item.content }}\n\n{% endfor %}{{ question }}\n\nProvide a detailed comparison and cite your sources.",
            gold_answer_template="{{ answer }}",
            required_node_types=["paragraph"],
            required_edge_types=["semantic_sim"],
            min_nodes=2,
            max_nodes=4,
            evaluation_metrics=["accuracy", "completeness", "citation_f1"],
            requires_citations=True
        ))
        
        # Summarization template
        self.add_template(TaskTemplate(
            template_id="multi_source_summary",
            name="Multi-source Summarization",
            description="Summarize information from multiple related sources",
            task_type=TaskType.SUMMARIZATION,
            difficulty=TaskDifficulty.MEDIUM,
            required_capabilities=[RequiredCapability.READING_COMPREHENSION, RequiredCapability.LOGICAL_REASONING],
            prompt_template="Summarize the key points from the following related information:\n\n{% for node in nodes %}{{ loop.index }}. {{ node.content }}\n\n{% endfor %}Provide a comprehensive summary that captures the main themes and important details. Include citations to the source materials.",
            gold_answer_template="{{ summary }}",
            required_node_types=["paragraph"],
            required_edge_types=["semantic_sim", "sequence"],
            min_nodes=2,
            max_nodes=6,
            evaluation_metrics=["rouge_score", "bert_score", "citation_f1"],
            requires_citations=True
        ))
        
        # Cross-reference template
        self.add_template(TaskTemplate(
            template_id="cross_reference_follow",
            name="Cross-reference Following",
            description="Follow cross-references to find related information",
            task_type=TaskType.CROSS_REFERENCE,
            difficulty=TaskDifficulty.MEDIUM,
            required_capabilities=[RequiredCapability.READING_COMPREHENSION, RequiredCapability.LOGICAL_REASONING],
            prompt_template="Starting from the following text:\n\n{{ source_content }}\n\nFollow the reference to {{ reference_target }} and answer: {{ question }}\n\nProvide the answer and explain how you followed the reference.",
            gold_answer_template="{{ answer }}",
            required_node_types=["paragraph"],
            required_edge_types=["refers_to"],
            min_nodes=2,
            max_nodes=3,
            evaluation_metrics=["accuracy", "link_accuracy", "citation_f1"],
            requires_citations=True
        ))
        
        # Figure interpretation template
        self.add_template(TaskTemplate(
            template_id="figure_interpretation",
            name="Figure Interpretation",
            description="Interpret figures with textual context",
            task_type=TaskType.FIGURE_INTERPRETATION,
            difficulty=TaskDifficulty.MEDIUM,
            required_capabilities=[RequiredCapability.CROSS_MODAL_UNDERSTANDING, RequiredCapability.READING_COMPREHENSION],
            prompt_template="Based on the figure: {{ figure_description }}\n\nAnd the accompanying text:\n{{ context_text }}\n\nAnswer: {{ question }}",
            gold_answer_template="{{ answer }}",
            required_node_types=["figure", "paragraph"],
            required_edge_types=["figure_context"],
            min_nodes=2,
            max_nodes=3,
            evaluation_metrics=["accuracy", "completeness"],
            requires_citations=True
        ))
        
        # Aggregation template
        self.add_template(TaskTemplate(
            template_id="information_aggregation",
            name="Information Aggregation",
            description="Aggregate information from multiple sources",
            task_type=TaskType.AGGREGATION,
            difficulty=TaskDifficulty.HARD,
            required_capabilities=[RequiredCapability.INFORMATION_EXTRACTION, RequiredCapability.LOGICAL_REASONING],
            prompt_template="Aggregate information about {{ topic }} from the following sources:\n\n{% for source in sources %}{{ loop.index }}. {{ source.content }}\n\n{% endfor %}Provide a comprehensive aggregation that includes all relevant details. Cite all sources used.",
            gold_answer_template="{{ aggregated_answer }}",
            required_node_types=["paragraph"],
            required_edge_types=["semantic_sim"],
            min_nodes=3,
            max_nodes=8,
            evaluation_metrics=["completeness", "accuracy", "citation_f1"],
            requires_citations=True
        ))
        
        # Information verification template
        self.add_template(TaskTemplate(
            template_id="information_verification",
            name="Information Verification",
            description="Verify the accuracy and reliability of information",
            task_type=TaskType.FACT_VERIFICATION,
            difficulty=TaskDifficulty.MEDIUM,
            required_capabilities=[RequiredCapability.READING_COMPREHENSION, RequiredCapability.LOGICAL_REASONING],
            prompt_template="Verify the accuracy of the following information:\n\n{{ content }}\n\nQuestion: {{ question }}\n\nAnalyze the provided information and determine if it's accurate, reliable, and well-supported. Consider the source, evidence, and logical consistency.",
            gold_answer_template="{{ verification_result }}",
            required_node_types=["paragraph"],
            min_nodes=1,
            max_nodes=3,
            evaluation_metrics=["accuracy", "reliability", "evidence_quality"],
            requires_exact_match=False,
            requires_citations=True
        ))
        
        # Numerical reasoning template
        self.add_template(TaskTemplate(
            template_id="numerical_reasoning",
            name="Numerical Reasoning",
            description="Perform calculations and numerical reasoning",
            task_type=TaskType.REASONING,
            difficulty=TaskDifficulty.MEDIUM,
            required_capabilities=[RequiredCapability.NUMERICAL_COMPUTATION, RequiredCapability.LOGICAL_REASONING],
            prompt_template="Based on the following data:\n\n{{ data_content }}\n\nSolve: {{ question }}\n\nShow your calculation steps and cite the specific data points used.",
            gold_answer_template="{{ calculation_result }}",
            required_node_types=["table", "paragraph"],
            min_nodes=1,
            max_nodes=3,
            evaluation_metrics=["numerical_accuracy", "reasoning_quality", "citation_f1"],
            requires_citations=True,
            requires_reasoning_path=True
        ))
        
        # Context chain reasoning template
        self.add_template(TaskTemplate(
            template_id="context_chain_reasoning",
            name="Context Chain Reasoning",
            description="Follow a chain of context to reach a conclusion",
            task_type=TaskType.REASONING,
            difficulty=TaskDifficulty.HARD,
            required_capabilities=[RequiredCapability.MULTI_HOP_REASONING, RequiredCapability.LOGICAL_REASONING, RequiredCapability.CAUSAL_REASONING],
            prompt_template="Follow this chain of reasoning through the provided context:\n\n{% for step in reasoning_chain %}{{ loop.index }}. {{ step.question }}\nContext: {{ step.context }}\n\n{% endfor %}Final question: {{ final_question }}\n\nTrace your reasoning through each step and cite the specific context used at each stage.",
            gold_answer_template="{{ final_answer }}",
            required_node_types=["paragraph", "heading"],
            required_edge_types=["sequence", "causal", "semantic_sim"],
            min_nodes=3,
            max_nodes=8,
            max_hops=4,
            evaluation_metrics=["accuracy", "reasoning_path_match", "context_utilization", "citation_f1"],
            requires_citations=True,
            requires_reasoning_path=True
        ))
        
        # Context contradiction detection template
        self.add_template(TaskTemplate(
            template_id="context_contradiction_detection",
            name="Context Contradiction Detection",
            description="Identify contradictions or inconsistencies across context",
            task_type=TaskType.ANALYSIS,
            difficulty=TaskDifficulty.HARD,
            required_capabilities=[RequiredCapability.LOGICAL_REASONING, RequiredCapability.READING_COMPREHENSION],
            prompt_template="Analyze the following information for contradictions or inconsistencies:\n\n{% for node in nodes %}{{ loop.index }}. {{ node.content }}\n\n{% endfor %}Question: {{ question }}\n\nIdentify any contradictions, inconsistencies, or conflicting information. Explain your reasoning and cite the specific conflicting statements.",
            gold_answer_template="{{ contradiction_analysis }}",
            required_node_types=["paragraph"],
            required_edge_types=["semantic_sim", "refers_to"],
            min_nodes=2,
            max_nodes=6,
            evaluation_metrics=["accuracy", "contradiction_detection_rate", "explanation_quality", "citation_f1"],
            requires_citations=True,
            requires_reasoning_path=True
        ))
        
        # Context gap identification template
        self.add_template(TaskTemplate(
            template_id="context_gap_identification",
            name="Context Gap Identification",
            description="Identify missing information or gaps in the provided context",
            task_type=TaskType.ANALYSIS,
            difficulty=TaskDifficulty.MEDIUM,
            required_capabilities=[RequiredCapability.LOGICAL_REASONING, RequiredCapability.READING_COMPREHENSION],
            prompt_template="Given the following context:\n\n{% for node in nodes %}{{ loop.index }}. {{ node.content }}\n\n{% endfor %}Question: {{ question }}\n\nIdentify what information is missing or unclear that would be needed to fully answer this question. Explain why this information is important and cite the specific parts of the context that indicate these gaps.",
            gold_answer_template="{{ gap_analysis }}",
            required_node_types=["paragraph"],
            min_nodes=1,
            max_nodes=4,
            evaluation_metrics=["gap_identification_accuracy", "explanation_quality", "citation_f1"],
            requires_citations=True,
            requires_reasoning_path=True
        ))
        
        # Context hierarchy understanding template
        self.add_template(TaskTemplate(
            template_id="context_hierarchy_understanding",
            name="Context Hierarchy Understanding",
            description="Understand hierarchical relationships in the context",
            task_type=TaskType.COMPREHENSION,
            difficulty=TaskDifficulty.MEDIUM,
            required_capabilities=[RequiredCapability.READING_COMPREHENSION, RequiredCapability.LOGICAL_REASONING],
            prompt_template="Based on the hierarchical structure of the following information:\n\n{% for level in hierarchy %}{{ level.indent }}{{ level.content }}\n\n{% endfor %}Question: {{ question }}\n\nAnswer the question by understanding the hierarchical relationships. Cite the specific levels and relationships you used.",
            gold_answer_template="{{ hierarchical_answer }}",
            required_node_types=["heading", "paragraph"],
            required_edge_types=["contains", "sequence"],
            min_nodes=2,
            max_nodes=5,
            evaluation_metrics=["accuracy", "hierarchy_understanding", "citation_f1"],
            requires_citations=True
        ))
        
        # Context temporal reasoning template
        self.add_template(TaskTemplate(
            template_id="context_temporal_reasoning",
            name="Context Temporal Reasoning",
            description="Reason about temporal relationships in context",
            task_type=TaskType.REASONING,
            difficulty=TaskDifficulty.MEDIUM,
            required_capabilities=[RequiredCapability.TEMPORAL_REASONING, RequiredCapability.LOGICAL_REASONING],
            prompt_template="Based on the temporal information in the following context:\n\n{% for event in events %}{{ loop.index }}. {{ event.description }} ({{ event.temporal_info }})\n\n{% endfor %}Question: {{ question }}\n\nAnswer by understanding the temporal relationships and sequence of events. Cite the specific temporal information used.",
            gold_answer_template="{{ temporal_answer }}",
            required_node_types=["paragraph"],
            required_edge_types=["temporal", "sequence"],
            min_nodes=2,
            max_nodes=6,
            evaluation_metrics=["temporal_accuracy", "sequence_understanding", "citation_f1"],
            requires_citations=True,
            requires_reasoning_path=True
        ))
        
        # Context spatial reasoning template
        self.add_template(TaskTemplate(
            template_id="context_spatial_reasoning",
            name="Context Spatial Reasoning",
            description="Reason about spatial relationships in context",
            task_type=TaskType.REASONING,
            difficulty=TaskDifficulty.MEDIUM,
            required_capabilities=[RequiredCapability.SPATIAL_REASONING, RequiredCapability.READING_COMPREHENSION],
            prompt_template="Based on the spatial information in the following context:\n\n{% for location in locations %}{{ loop.index }}. {{ location.description }} ({{ location.spatial_info }})\n\n{% endfor %}Question: {{ question }}\n\nAnswer by understanding the spatial relationships and positioning. Cite the specific spatial information used.",
            gold_answer_template="{{ spatial_answer }}",
            required_node_types=["paragraph"],
            required_edge_types=["spatial", "contains"],
            min_nodes=2,
            max_nodes=5,
            evaluation_metrics=["spatial_accuracy", "relationship_understanding", "citation_f1"],
            requires_citations=True,
            requires_reasoning_path=True
        ))
        
        # Context causal reasoning template
        self.add_template(TaskTemplate(
            template_id="context_causal_reasoning",
            name="Context Causal Reasoning",
            description="Understand cause-and-effect relationships in context",
            task_type=TaskType.REASONING,
            difficulty=TaskDifficulty.HARD,
            required_capabilities=[RequiredCapability.CAUSAL_REASONING, RequiredCapability.LOGICAL_REASONING],
            prompt_template="Based on the causal relationships in the following context:\n\n{% for relationship in causal_relationships %}{{ loop.index }}. Cause: {{ relationship.cause }}\nEffect: {{ relationship.effect }}\n\n{% endfor %}Question: {{ question }}\n\nAnswer by understanding the causal chain and relationships. Trace the cause-and-effect connections and cite the specific causal information used.",
            gold_answer_template="{{ causal_answer }}",
            required_node_types=["paragraph"],
            required_edge_types=["causal", "sequence"],
            min_nodes=2,
            max_nodes=6,
            evaluation_metrics=["causal_accuracy", "chain_understanding", "citation_f1"],
            requires_citations=True,
            requires_reasoning_path=True
        ))
        
        # Context synthesis template
        self.add_template(TaskTemplate(
            template_id="context_synthesis",
            name="Context Synthesis",
            description="Synthesize new insights from multiple context pieces",
            task_type=TaskType.SYNTHESIS,
            difficulty=TaskDifficulty.EXPERT,
            required_capabilities=[RequiredCapability.LOGICAL_REASONING, RequiredCapability.DOMAIN_KNOWLEDGE, RequiredCapability.COMMON_SENSE],
            prompt_template="Based on the following diverse information:\n\n{% for piece in context_pieces %}{{ loop.index }}. {{ piece.content }}\n\n{% endfor %}Synthesis question: {{ question }}\n\nCreate a new insight or conclusion by synthesizing the provided information. Explain your synthesis process and cite the specific pieces of information that led to your conclusion.",
            gold_answer_template="{{ synthesis_result }}",
            required_node_types=["paragraph"],
            required_edge_types=["semantic_sim", "refers_to"],
            min_nodes=3,
            max_nodes=8,
            evaluation_metrics=["synthesis_quality", "novelty", "coherence", "citation_f1"],
            requires_citations=True,
            requires_reasoning_path=True
        ))
        
        # Context verification template
        self.add_template(TaskTemplate(
            template_id="context_verification",
            name="Context Verification",
            description="Verify claims against provided context",
            task_type=TaskType.FACT_VERIFICATION,
            difficulty=TaskDifficulty.MEDIUM,
            required_capabilities=[RequiredCapability.READING_COMPREHENSION, RequiredCapability.LOGICAL_REASONING],
            prompt_template="Verify the following claim against the provided context:\n\nClaim: {{ claim }}\n\nContext:\n{% for node in nodes %}{{ loop.index }}. {{ node.content }}\n\n{% endfor %}Provide a verification with: 1) Whether the claim is supported, contradicted, or cannot be determined, 2) Specific evidence from the context, 3) Your reasoning process. Cite all relevant context pieces.",
            gold_answer_template="{{ verification_result }}",
            required_node_types=["paragraph"],
            min_nodes=1,
            max_nodes=4,
            evaluation_metrics=["verification_accuracy", "evidence_quality", "reasoning_quality", "citation_f1"],
            requires_citations=True,
            requires_reasoning_path=True
        ))


# Create default template library instance
DEFAULT_TEMPLATE_LIBRARY = TaskTemplateLibrary()
