"""
Multi-Agent System for Graph-based Reasoning
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
import time
import json
import re
from enum import Enum
from loguru import logger

from .agent import AgentConfig, AgentResponse
from .retrievers import HybridRetriever, RetrievalConfig, RetrievalResult
from .executors import LLMExecutor, ExecutionConfig

from task_craft.task_generator import TaskInstance
from graph_rag.graph_builder import DocumentGraph


class AgentRole(Enum):
    """Agent role enumeration"""
    PLANNER = "planner"
    RETRIEVER = "retriever"
    REASONER = "reasoner"
    VERIFIER = "verifier"
    SUMMARIZER = "summarizer"


@dataclass
class ReasoningStep:
    """Reasoning step"""
    step_id: str
    description: str
    source_nodes: List[str]
    target_nodes: List[str]
    edges: List[str]
    reasoning: str
    confidence: float
    step_type: str  # "retrieval", "inference", "synthesis"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "source_nodes": self.source_nodes,
            "target_nodes": self.target_nodes,
            "edges": self.edges,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "step_type": self.step_type
        }


@dataclass
class SubgraphInfo:
    """Subgraph information"""
    nodes: List[str]
    edges: List[str]
    relevance_score: float
    coverage_score: float
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "relevance_score": self.relevance_score,
            "coverage_score": self.coverage_score,
            "reasoning": self.reasoning
        }


@dataclass
class VerificationResult:
    """Verification result"""
    is_valid: bool
    issues: List[str]
    suggestions: List[str]
    confidence: float
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


@dataclass
class MultiAgentResponse:
    """Multi-agent system response"""
    task_id: str
    final_answer: str
    success: bool
    
    # Outputs from each agent
    planner_output: Dict[str, Any] = field(default_factory=dict)
    retriever_output: SubgraphInfo = field(default_factory=lambda: SubgraphInfo([], [], 0.0, 0.0, ""))
    reasoner_output: List[ReasoningStep] = field(default_factory=list)
    verifier_output: VerificationResult = field(default_factory=lambda: VerificationResult(True, [], [], 1.0, ""))
    summarizer_output: Dict[str, Any] = field(default_factory=dict)
    
    # Execution information
    execution_time: float = 0.0
    total_tokens: int = 0
    model_used: str = ""
    
    # Error information
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "final_answer": self.final_answer,
            "success": self.success,
            "planner_output": self.planner_output,
            "retriever_output": {
                "nodes": self.retriever_output.nodes,
                "edges": self.retriever_output.edges,
                "relevance_score": self.retriever_output.relevance_score,
                "coverage_score": self.retriever_output.coverage_score,
                "reasoning": self.retriever_output.reasoning
            },
            "reasoner_output": [
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "source_nodes": step.source_nodes,
                    "target_nodes": step.target_nodes,
                    "edges": step.edges,
                    "reasoning": step.reasoning,
                    "confidence": step.confidence,
                    "step_type": step.step_type
                }
                for step in self.reasoner_output
            ],
            "verifier_output": {
                "is_valid": self.verifier_output.is_valid,
                "issues": self.verifier_output.issues,
                "suggestions": self.verifier_output.suggestions,
                "confidence": self.verifier_output.confidence,
                "reasoning": self.verifier_output.reasoning
            },
            "summarizer_output": self.summarizer_output,
            "execution_time": self.execution_time,
            "total_tokens": self.total_tokens,
            "model_used": self.model_used,
            "error_type": self.error_type,
            "error_message": self.error_message
        }


class BaseAgent(ABC):
    """Base agent abstract class"""
    
    def __init__(self, role: AgentRole, config: AgentConfig, executor: Optional[LLMExecutor] = None):
        self.role = role
        self.config = config
        # Use provided executor or shared singleton executor
        if executor is not None:
            self.executor = executor
            self.logger = logger.bind(agent=role.value)
        else:
            # Use shared singleton executor instead of creating new instance
            self.executor = LLMExecutor.get_instance(config.execution_config)
            self.logger = logger.bind(agent=role.value)
    
    @abstractmethod
    def execute(self, task: TaskInstance, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent task"""
        pass
    
    def _log_execution(self, message: str, **kwargs):
        """Log execution message"""
        self.logger.info(f"[{self.role.value.upper()}] {message}", **kwargs)


class PlannerAgent(BaseAgent):
    """Planning agent - Research director role"""
    
    def __init__(self, config: AgentConfig, executor: Optional[LLMExecutor] = None):
        super().__init__(AgentRole.PLANNER, config, executor)
    
    def execute(self, task: TaskInstance, context: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose task into multi-hop reasoning steps"""
        self._log_execution(f"Starting planning task: {task.task_id}")
        
                            # Build planning prompt with JSON format requirement
        prompt = self._build_planning_prompt(task, context)
        
        try:
            # Create planning task
            from task_craft.task_generator import TaskInstance
            from task_craft.task_templates import TaskType, TaskDifficulty
            
            planning_task = TaskInstance(
                task_id=f"{task.task_id}_planner",
                template_id="planner_template",
                task_type=TaskType.COMPREHENSION,
                difficulty=TaskDifficulty.MEDIUM,
                prompt=prompt,
                gold_answer="",
                gold_nodes=[],
                gold_edges=[],
                subgraph_nodes=[],
                subgraph_edges=[],
                requires_citations=False,  # Planning stage doesn't need citations
                requires_reasoning_path=True  # Planning stage needs reasoning path
            )
            
            # Get task-related context information
            graph = context.get('graph')
            task = context.get('task')
            
            # Perform initial retrieval for planning stage
            initial_retrieval = self._perform_initial_retrieval(task, graph)
            
            # Use shared singleton executor instead of creating new instance
            temp_executor = LLMExecutor.get_instance()
            
            # Use temporary executor to execute planning task
            response = temp_executor.execute(planning_task, initial_retrieval)
            

            
            # Parse planning result from LLMExecutor's parsed response
            plan = self._parse_planning_result_from_executor(response)
            

            
            # Safely get step count
            steps = plan.get('reasoning_steps', [])
            self._log_execution(f"Planning completed, generated {len(steps)} reasoning steps")

            
            return {
                "plan": plan,
                "reasoning": response.answer,
                "confidence": response.confidence,
                "tokens_used": response.tokens_used
            }
            
        except Exception as e:
            self._log_execution(f"Planning failed: {str(e)}")
            raise
    

    def _build_planning_prompt(self, task: TaskInstance, context: Dict[str, Any]) -> str:
        """Build planning prompt with dynamic format based on execution_config"""
        # Get response format from execution config
        response_format = self.config.execution_config.response_format if hasattr(self.config, 'execution_config') else "json"
        
        if response_format == "json":
            return f"""
You are a research director who needs to decompose complex tasks into multi-hop reasoning steps.

Task Description: {task.prompt}

CRITICAL: You MUST respond with a valid JSON object in the following format:

{{
    "answer": "Clear description of the expected final output and planning conclusion",
    "reasoning": [
        "Step 1: First specific reasoning step description",
        "Step 2: Second specific reasoning step description", 
        "Step 3: Third specific reasoning step description",
        "Step 4: Fourth specific reasoning step description",
        "Step 5: Fifth specific reasoning step description"
    ],
    "citations": ["List of relevant node IDs or references"],
    "confidence": 0.85
}}

IMPORTANT: 
1. Your response must be a valid JSON object
2. Include at least 3-5 reasoning steps in the "reasoning" array
3. Each step should be specific and actionable
4. Do not include any text before or after the JSON object
5. Ensure all JSON syntax is correct (proper quotes, commas, brackets)
6. The "answer" field should contain your main planning conclusion
7. The "reasoning" field should be an array of reasoning step strings
"""
        else:
            # Structured format
            return f"""
You are a research director who needs to decompose complex tasks into multi-hop reasoning steps.

Task Description: {task.prompt}

CRITICAL: You MUST respond in the EXACT format specified below. Do not deviate from this format.

Your task is to:
1. Analyze the given task
2. Break it down into specific numbered reasoning steps
3. Provide a structured response

RESPONSE FORMAT (MANDATORY):

Answer: [Provide your main planning conclusion and expected output here]

Reasoning: [Provide detailed reasoning including:
1. Task analysis: [your analysis of the task]
2. Required information: [what information needs to be retrieved]
3. Reasoning steps:
   1. [First specific reasoning step]
   2. [Second specific reasoning step]
   3. [Third specific reasoning step]
   ...]

Citations: [List any relevant references, separated by commas]

Confidence: [Provide a number between 0.0-1.0]

EXAMPLE FORMAT:
Answer: The task requires analyzing the relationship between edge types and graph construction elements.

Reasoning: 
1. Task analysis: This task involves understanding how different edge types contribute to graph construction
2. Required information: Information about edge types, graph construction methods, and their relationships
3. Reasoning steps:
   1. Identify the main edge types mentioned in the context
   2. Analyze how each edge type contributes to graph construction
   3. Examine the relationships between different construction elements
   4. Synthesize findings into a comprehensive understanding

Citations: node_1, node_2, edge_1

Confidence: 0.85

REMEMBER: You MUST follow this exact format with the section headers (Answer:, Reasoning:, Citations:, Confidence:).
"""
    
    def _perform_initial_retrieval(self, task: TaskInstance, graph) -> RetrievalResult:
        """Perform initial retrieval for planning stage"""
        from .retrievers import HybridRetriever
        
        if graph is None:
            # If no graph, return empty retrieval result
            return RetrievalResult(
                nodes=[],
                edges=[],
                scores={},
                retrieval_method="no_graph",
                total_nodes_considered=0
            )
        
        try:
            # Create retriever for initial retrieval
            retriever = HybridRetriever(self.config.retrieval_config)
            
            # Execute retrieval to get basic context information
            retrieval_result = retriever.retrieve(task, graph)
            
            self._log_execution(f"Initial retrieval completed, obtained {len(retrieval_result.nodes)} nodes")
            
            return retrieval_result
            
        except Exception as e:
            self._log_execution(f"Initial retrieval failed: {str(e)}")
            # Return empty retrieval result
            return RetrievalResult(
                nodes=[],
                edges=[],
                scores={},
                retrieval_method="initial_retrieval_failed",
                total_nodes_considered=0
            )
    
    def _parse_planning_result_from_executor(self, response) -> Dict[str, Any]:
        """Parse planning result from LLMExecutor's parsed response"""
        try:
            # Convert LLMExecutor format to our internal format
            result = {
                "task_analysis": response.answer,  # Use answer as task analysis
                "required_info": response.citations,  # Use citations as required info
                "reasoning_steps": [],
                "expected_output": response.answer  # Use answer as expected output
            }
            
            # Convert reasoning_path array to reasoning_steps format
            if response.reasoning_path:
                for i, step_text in enumerate(response.reasoning_path):
                    result["reasoning_steps"].append({
                        "step_id": f"step_{i+1}",
                        "description": step_text,
                        "input_nodes": [],
                        "output_nodes": [],
                        "reasoning_type": "inference",
                        "dependencies": []
                    })
            
            return result
            
        except Exception as e:
            self._log_execution(f"Executor parsing failed: {e}, returning basic structure")
            return {
                "task_analysis": response.answer if hasattr(response, 'answer') else "",
                "required_info": response.citations if hasattr(response, 'citations') else [],
                "reasoning_steps": [],
                "expected_output": response.answer if hasattr(response, 'answer') else ""
            }
    
    def _parse_planning_result(self, response: str) -> Dict[str, Any]:
        """Parse planning result"""
        try:
            # Try to parse as JSON first
            try:
                # Clean the response to extract JSON
                response = response.strip()
                
                # Find JSON object
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    # Ensure all required fields exist
                    if "reasoning_steps" not in result:
                        result["reasoning_steps"] = []
                    if "required_info" not in result:
                        result["required_info"] = []
                    if "task_analysis" not in result:
                        result["task_analysis"] = ""
                    if "expected_output" not in result:
                        result["expected_output"] = ""
                    
                    return result
                else:
                    pass
                    
            except json.JSONDecodeError as e:
                self._log_execution("JSON parsing failed, returning basic structure")
            
            # Fallback to structured text parsing if JSON fails
            result = {
                "task_analysis": "",
                "required_info": [],
                "reasoning_steps": [],
                "expected_output": ""
            }
            
            # Extract Answer section
            answer_match = re.search(r"Answer:\s*(.+?)(?=\n(?:Reasoning|Citations|Confidence):|$)", response, re.DOTALL | re.IGNORECASE)
            if answer_match:
                result["expected_output"] = answer_match.group(1).strip()
            
            # Extract Reasoning section
            reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=\n(?:Answer|Citations|Confidence):|$)", response, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                reasoning_text = reasoning_match.group(1).strip()
                result["task_analysis"] = reasoning_text
                
                # Try to extract reasoning steps from reasoning text
                step_patterns = [
                    r'\d+\.\s*(.+?)(?=\n\d+\.|$)',
                    r'[-*]\s*(.+?)(?=\n[-*]|$)',
                    r'Step\s*\d+[:\s]*(.+?)(?=\nStep\s*\d+|$)'
                ]
                
                for pattern in step_patterns:
                    steps = re.findall(pattern, reasoning_text, re.DOTALL | re.IGNORECASE)
                    if steps:
                        for i, step in enumerate(steps):
                            # Handle different pattern formats
                            if isinstance(step, tuple):
                                step_text = step[1] if len(step) > 1 else step[0]
                            else:
                                step_text = step
                            
                            result["reasoning_steps"].append({
                                "step_id": f"step_{i+1}",
                                "description": step_text.strip(),
                                "input_nodes": [],
                                "output_nodes": [],
                                "reasoning_type": "inference",
                                "dependencies": []
                            })
                        break
            
            # If no structured content found, use entire response
            if not result["task_analysis"]:
                result["task_analysis"] = response.strip()
                result["expected_output"] = response.strip()
            
            return result
            
        except Exception as e:
            return {
                "task_analysis": response,
                "required_info": [],
                "reasoning_steps": [],
                "expected_output": response
            }


class RetrieverAgent(BaseAgent):
    """Retrieval agent - Information retrieval specialist"""
    
    def __init__(self, config: AgentConfig, executor: Optional[LLMExecutor] = None):
        super().__init__(AgentRole.RETRIEVER, config, executor)
        self.retriever = HybridRetriever(config.retrieval_config)
    
    def execute(self, task: TaskInstance, context: Dict[str, Any]) -> SubgraphInfo:
        """Retrieve relevant subgraph from large knowledge graph"""
        self._log_execution(f"Starting retrieval task: {task.task_id}")
        
        # Get planning result
        planner_output = context.get('planner_output', {})
        plan = planner_output.get('plan', {})
        
        # Build retrieval query
        query = self._build_retrieval_query(task, plan)
        
        try:
            # Execute retrieval
            graph = context.get('graph')
            if not graph:
                # If no graph, return empty subgraph info
                self._log_execution("No graph provided, returning empty subgraph info")
                return SubgraphInfo(
                    nodes=[],
                    edges=[],
                    relevance_score=0.0,
                    coverage_score=0.0,
                    reasoning="No graph provided, cannot perform retrieval"
                )
            
            retrieval_result = self.retriever.retrieve(task, graph)
            
            # Build subgraph information
            subgraph = self._build_subgraph_info(retrieval_result, task, plan)
            
            self._log_execution(f"Retrieval completed, found {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")
            
            return subgraph
            
        except Exception as e:
            self._log_execution(f"Retrieval failed: {str(e)}")
            raise
    
    def _build_retrieval_query(self, task: TaskInstance, plan: Dict[str, Any]) -> str:
        """Build retrieval query"""
        # Combine task description and required information from planning
        required_info = plan.get('required_info', [])
        info_str = ", ".join(required_info) if required_info else ""
        
        return f"""
Task: {task.prompt}

Required information: {info_str}

Please retrieve all important nodes and edges related to the task.
"""
    
    def _build_subgraph_info(self, retrieval_result, task: TaskInstance, plan: Dict[str, Any]) -> SubgraphInfo:
        """Build subgraph information"""
        nodes = []
        edges = []
        
        # Extract nodes and edges from retrieval result
        for node in retrieval_result.nodes:
            nodes.append(node.node_id)
        
        for edge in retrieval_result.edges:
            # Store the actual edge object or edge_id for proper graph queries
            try:
                if hasattr(edge, 'edge_id'):
                    # Store the unique edge ID for proper graph queries
                    edges.append(edge.edge_id)
                elif hasattr(edge, 'id'):
                    # Alternative edge ID attribute
                    edges.append(edge.id)
                else:
                    # Fallback: store the edge object itself for later processing
                    edges.append(edge)
            except Exception:
                # If all else fails, store the edge object
                edges.append(edge)
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(retrieval_result, task)
        coverage_score = self._calculate_coverage_score(retrieval_result, plan)
        
        reasoning = f"Retrieved {len(nodes)} relevant nodes and {len(edges)} relevant edges, covering the key information required for the task"
        
        return SubgraphInfo(
            nodes=nodes,
            edges=edges,
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            reasoning=reasoning
        )
    
    def _calculate_relevance_score(self, retrieval_result, task: TaskInstance) -> float:
        """Calculate relevance score"""
        # Calculate based on similarity scores from retrieval result
        if hasattr(retrieval_result, 'scores') and retrieval_result.scores:
            return sum(retrieval_result.scores.values()) / len(retrieval_result.scores)
        return 0.8  # Default score
    
    def _calculate_coverage_score(self, retrieval_result, plan: Dict[str, Any]) -> float:
        """Calculate coverage score"""
        # Calculate coverage based on required information from planning
        required_info = plan.get('required_info', [])
        if not required_info:
            return 0.8
        
        # Simple coverage calculation
        covered_info = 0
        for info in required_info:
            # Check if retrieval result contains relevant information
            if any(info.lower() in node.content.lower() for node in retrieval_result.nodes):
                covered_info += 1
        
        return covered_info / len(required_info) if required_info else 0.8


class ReasonerAgent(BaseAgent):
    """Reasoning agent - Data scientist role"""
    
    def __init__(self, config: AgentConfig, executor: Optional[LLMExecutor] = None):
        super().__init__(AgentRole.REASONER, config, executor)
    
    def execute(self, task: TaskInstance, context: Dict[str, Any]) -> List[ReasoningStep]:
        """Perform reasoning based on subgraph"""
        self._log_execution(f"Starting reasoning task: {task.task_id}")
        
        # Get planning result and subgraph information
        planner_output = context.get('planner_output', {})
        plan = planner_output.get('plan', {})
        subgraph_info = context.get('retriever_output')
        
        if not subgraph_info:
            raise ValueError("Subgraph information not provided")
        
        try:
            # Execute reasoning steps
            reasoning_steps = []
            steps = plan.get('reasoning_steps', [])
            
            for i, step_plan in enumerate(steps):
                step = self._execute_reasoning_step(step_plan, subgraph_info, task, i, context)
                reasoning_steps.append(step)
            
            self._log_execution(f"Reasoning completed, executed {len(reasoning_steps)} reasoning steps")
            
            return reasoning_steps
            
        except Exception as e:
            self._log_execution(f"Reasoning failed: {str(e)}")
            raise
    
    def _execute_reasoning_step(self, step_plan: Dict[str, Any], subgraph_info: SubgraphInfo, 
                              task: TaskInstance, step_index: int, context: Dict[str, Any]) -> ReasoningStep:
        """Execute single reasoning step"""
        # Build reasoning prompt
        prompt = self._build_reasoning_prompt(step_plan, subgraph_info, task)
        
        # Create reasoning task
        from task_craft.task_generator import TaskInstance
        from task_craft.task_templates import TaskType, TaskDifficulty
        
        reasoning_task = TaskInstance(
            task_id=f"{task.task_id}_reasoner_step_{step_index + 1}",
            template_id="reasoner_template",
            task_type=TaskType.COMPREHENSION,
            difficulty=TaskDifficulty.MEDIUM,
            prompt=prompt,
            gold_answer="",
            gold_nodes=[],
            gold_edges=[],
            subgraph_nodes=[],
            subgraph_edges=[],
            requires_citations=True,  # Reasoning stage needs citations
            requires_reasoning_path=True  # Reasoning stage needs reasoning path
        )
        
        # Create context with actual subgraph information
        from .retrievers import RetrievalResult
        from graph_rag.node_types import Node
        from graph_rag.edge_types import Edge
        
        # Convert subgraph info to actual nodes and edges
        context_nodes = []
        context_edges = []
        context_scores = {}
        
        # Get actual node content from graph
        graph = context.get('graph')
        if graph and hasattr(subgraph_info, 'nodes'):
            for node_id in subgraph_info.nodes:
                actual_node = graph.storage.get_node(node_id)
                if actual_node:
                    context_nodes.append(actual_node)
                    context_scores[node_id] = 1.0  # Default score
        
        # Get actual edge information from graph
        if graph and hasattr(subgraph_info, 'edges'):
            for edge_item in subgraph_info.edges:
                # Handle different edge formats: edge_id string, edge object, or edge reference
                if isinstance(edge_item, str):
                    # edge_item is an edge_id string
                    actual_edge = graph.storage.get_edge(edge_item)
                    if actual_edge:
                        context_edges.append(actual_edge)
                elif hasattr(edge_item, 'edge_id'):
                    # edge_item is an edge object with edge_id
                    actual_edge = graph.storage.get_edge(edge_item.edge_id)
                    if actual_edge:
                        context_edges.append(actual_edge)
                elif hasattr(edge_item, 'id'):
                    # edge_item is an edge object with id
                    actual_edge = graph.storage.get_edge(edge_item.id)
                    if actual_edge:
                        context_edges.append(actual_edge)
                else:
                    # edge_item is already an edge object
                    context_edges.append(edge_item)
        
        reasoning_context = RetrievalResult(
            nodes=context_nodes,
            edges=context_edges,
            scores=context_scores,
            retrieval_method="reasoning_context",
            total_nodes_considered=len(context_nodes)
        )
        
        # Use shared singleton executor instead of creating new instance
        temp_executor = LLMExecutor.get_instance()
        
        # Use temporary executor to execute reasoning task
        response = temp_executor.execute(reasoning_task, reasoning_context)
        
        # Parse reasoning result from executor response
        reasoning_result = self._parse_reasoning_result_from_executor(response)
        
        return ReasoningStep(
            step_id=f"step_{step_index + 1}",
            description=step_plan.get('description', f"Reasoning step {step_index + 1}"),
            source_nodes=reasoning_result.get('source_nodes', []),
            target_nodes=reasoning_result.get('target_nodes', []),
            edges=reasoning_result.get('edges', []),
            reasoning=reasoning_result.get('reasoning', response.answer),
            confidence=response.confidence,
            step_type=step_plan.get('reasoning_type', 'inference')
        )
    
    def _format_edge_display(self, edges, max_edges=10):
        """Format edges for display in prompts"""
        formatted_edges = []
        for edge in edges[:max_edges]:
            if isinstance(edge, str):
                formatted_edges.append(edge)
            elif hasattr(edge, 'edge_id'):
                formatted_edges.append(edge.edge_id)
            elif hasattr(edge, 'id'):
                formatted_edges.append(edge.id)
            else:
                formatted_edges.append(str(edge)[:20])  # Truncate long edge representations
        return ', '.join(formatted_edges)
    
    def _build_reasoning_prompt(self, step_plan: Dict[str, Any], subgraph_info: SubgraphInfo, 
                              task: TaskInstance) -> str:
        """Build reasoning prompt with dynamic format based on execution_config"""
        # Get response format from execution config
        response_format = self.config.execution_config.response_format if hasattr(self.config, 'execution_config') else "json"
        
        if response_format == "json":
            return f"""
You are an analyst who needs to perform reasoning based on given subgraph information.

Task: {task.prompt}

Current reasoning step: {step_plan.get('description', '')}

Available nodes: {', '.join(subgraph_info.nodes[:10])}  # Show first 10 nodes
Available edges: {self._format_edge_display(subgraph_info.edges)}  # Show first 10 edges

Reasoning type: {step_plan.get('reasoning_type', 'inference')}

CRITICAL: You MUST respond with a valid JSON object in the following format:

{{
    "answer": "Your reasoning conclusion and findings",
    "reasoning": ["Step 1: Analyze available nodes and edges", "Step 2: Identify relevant information", "Step 3: Perform reasoning operation", "Step 4: Draw conclusions"],
    "citations": ["node1", "node2"],
    "confidence": 0.85
}}

IMPORTANT: 
1. Your response must be a valid JSON object
2. Do not include any text before or after the JSON object
3. Ensure all JSON syntax is correct (proper quotes, commas, brackets)
4. The "answer" field should contain your reasoning conclusion
5. The "reasoning" field should be an array of reasoning steps
6. The "citations" field should be an array of node IDs
7. The "confidence" field should be a number between 0.0 and 1.0
"""
        else:
            # Structured format
            return f"""
You are an analyst who needs to perform reasoning based on given subgraph information.

Task: {task.prompt}

Current reasoning step: {step_plan.get('description', '')}

Available nodes: {', '.join(subgraph_info.nodes[:10])}  # Show first 10 nodes
Available edges: {self._format_edge_display(subgraph_info.edges)}  # Show first 10 edges

Reasoning type: {step_plan.get('reasoning_type', 'inference')}

CRITICAL: You MUST respond in the EXACT format specified below. Do not deviate from this format.

RESPONSE FORMAT (MANDATORY):

Answer: [Your reasoning conclusion and findings]

Reasoning: [Provide detailed reasoning including:
1. Analysis of available nodes and edges
2. Identification of relevant information
3. Performance of reasoning operation
4. Drawing of conclusions]

Citations: [List relevant node IDs, separated by commas]

Confidence: [Provide a number between 0.0-1.0]

EXAMPLE FORMAT:
Answer: Based on the available nodes, I can see that the graph construction process involves multiple edge types.

Reasoning: 
1. Analysis of available nodes and edges: Examined the provided node and edge information
2. Identification of relevant information: Found key nodes related to graph construction
3. Performance of reasoning operation: Analyzed the relationships between different components
4. Drawing of conclusions: The graph construction process is well-structured with clear edge type definitions

Citations: node_1, node_2, edge_1

Confidence: 0.85

REMEMBER: You MUST follow this exact format with the section headers (Answer:, Reasoning:, Citations:, Confidence:).
"""
    
    def _parse_reasoning_result_from_executor(self, response) -> Dict[str, Any]:
        """Parse reasoning result from LLMExecutor's parsed response"""
        try:
            # Convert LLMExecutor format to our internal format
            answer = response.answer if hasattr(response, 'answer') else ""
            reasoning_steps = response.reasoning_path if hasattr(response, 'reasoning_path') else []
            citations = response.citations if hasattr(response, 'citations') else []
            
            # Convert reasoning steps to single reasoning text
            reasoning_text = "\n".join(reasoning_steps) if reasoning_steps else answer
            
            return {
                "source_nodes": citations,  # Use citations as source nodes
                "target_nodes": citations,  # Use citations as target nodes for now
                "edges": citations,  # Use citations as edges for now
                "reasoning": reasoning_text
            }
            
        except Exception as e:
            return {
                "source_nodes": [],
                "target_nodes": [],
                "edges": [],
                "reasoning": f"Parsing failed: {e}"
            }
    
    def _parse_reasoning_result(self, response: str) -> Dict[str, Any]:
        """Parse reasoning result"""
        try:
            # Parse structured text format
            result = {
                "source_nodes": [],
                "target_nodes": [],
                "edges": [],
                "reasoning": response
            }
            
            # Extract Answer section
            answer_match = re.search(r"Answer:\s*(.+?)(?=\n(?:Reasoning|Citations|Confidence):|$)", response, re.DOTALL | re.IGNORECASE)
            if answer_match:
                # Try to extract node/edge references from answer
                answer_text = answer_match.group(1).strip()
                # Look for node patterns in answer
                node_patterns = [r'node[_\s]*(\w+)', r'para[_\s]*(\w+)', r'table[_\s]*(\w+)']
                for pattern in node_patterns:
                    nodes = re.findall(pattern, answer_text, re.IGNORECASE)
                    result["target_nodes"].extend(nodes)
            
            # Extract Reasoning section
            reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=\n(?:Answer|Citations|Confidence):|$)", response, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                result["reasoning"] = reasoning_match.group(1).strip()
            
            # Extract Citations section
            citations_match = re.search(r"Citations:\s*(.+?)(?=\n(?:Answer|Reasoning|Confidence):|$)", response, re.DOTALL | re.IGNORECASE)
            if citations_match:
                citations_text = citations_match.group(1).strip()
                # Parse comma-separated citations
                citations = [c.strip() for c in citations_text.split(",") if c.strip()]
                result["source_nodes"] = citations
                result["edges"] = citations  # Treat citations as both nodes and edges for now
            
            return result
            
        except Exception as e:
            return {
                "source_nodes": [],
                "target_nodes": [],
                "edges": [],
                "reasoning": f"Parsing failed: {e}. Original response: {response}"
            }


class VerifierAgent(BaseAgent):
    """Verification agent - Quality assurance specialist"""
    
    def __init__(self, config: AgentConfig, executor: Optional[LLMExecutor] = None):
        super().__init__(AgentRole.VERIFIER, config, executor)
    
    def execute(self, task: TaskInstance, context: Dict[str, Any]) -> VerificationResult:
        """Validate the reasonableness of the reasoning chain"""
        self._log_execution(f"Starting verification task: {task.task_id}")
        
        # Get outputs from each agent
        planner_output = context.get('planner_output', {})
        retriever_output = context.get('retriever_output')
        reasoner_output = context.get('reasoner_output', [])
        graph = context.get('graph')  # Get graph
        
        try:
            # Build verification prompt
            prompt = self._build_verification_prompt(task, planner_output, retriever_output, reasoner_output)
            
            # Create verification task
            from task_craft.task_generator import TaskInstance
            from task_craft.task_templates import TaskType, TaskDifficulty
            
            verification_task = TaskInstance(
                task_id=f"{task.task_id}_verifier",
                template_id="verifier_template",
                task_type=TaskType.COMPREHENSION,
                difficulty=TaskDifficulty.MEDIUM,
                prompt=prompt,
                gold_answer="",
                gold_nodes=[],
                gold_edges=[],
                subgraph_nodes=[],
                subgraph_edges=[],
                requires_citations=False,  # Verification stage doesn't need citations
                requires_reasoning_path=True  # But needs reasoning path
            )
            
            # Create context with actual graph data
            from .retrievers import RetrievalResult
            context_nodes = []
            context_edges = []
            context_scores = {}
            
            # Get actual node content from graph
            if graph and retriever_output and hasattr(retriever_output, 'nodes'):
                for node_id in retriever_output.nodes:
                    # Find actual node from graph
                    actual_node = graph.storage.get_node(node_id)
                    if actual_node:
                        context_nodes.append(actual_node)
                        context_scores[node_id] = retriever_output.relevance_score
            
            # Get actual edge information from graph
            if graph and retriever_output and hasattr(retriever_output, 'edges'):
                for edge_item in retriever_output.edges:
                    # Handle different edge formats: edge_id string, edge object, or edge reference
                    if isinstance(edge_item, str):
                        # edge_item is an edge_id string
                        actual_edge = graph.storage.get_edge(edge_item)
                        if actual_edge:
                            context_edges.append(actual_edge)
                    elif hasattr(edge_item, 'edge_id'):
                        # edge_item is an edge object with edge_id
                        actual_edge = graph.storage.get_edge(edge_item.edge_id)
                        if actual_edge:
                            context_edges.append(actual_edge)
                    elif hasattr(edge_item, 'id'):
                        # edge_item is an edge object with id
                        actual_edge = graph.storage.get_edge(edge_item.id)
                        if actual_edge:
                            context_edges.append(actual_edge)
                    else:
                        # edge_item is already an edge object
                        context_edges.append(edge_item)
            
            verification_context = RetrievalResult(
                nodes=context_nodes,
                edges=context_edges,
                scores=context_scores,
                retrieval_method="verification_context",
                total_nodes_considered=len(context_nodes)
            )
            
            # Use shared singleton executor instead of creating new instance
            temp_executor = LLMExecutor.get_instance()
            
            # Use temporary executor to execute verification task
            response = temp_executor.execute(verification_task, verification_context)
            
            # Parse verification result from executor response
            verification_result = self._parse_verification_result_from_executor(response)
            
            self._log_execution(f"Verification completed, result: {'Passed' if verification_result.is_valid else 'Issues Found'}")
            
            return verification_result
            
        except Exception as e:
            self._log_execution(f"Verification failed: {str(e)}")
            raise
    
    def _build_verification_prompt(self, task: TaskInstance, planner_output: Dict[str, Any], 
                                 retriever_output: SubgraphInfo, reasoner_output: List[ReasoningStep]) -> str:
        """Build verification prompt with dynamic format based on execution_config"""
        # Get response format from execution config
        response_format = self.config.execution_config.response_format if hasattr(self.config, 'execution_config') else "json"
        
        # Build reasoning steps summary
        steps_summary = []
        for step in reasoner_output:
            steps_summary.append(f"- {step.description}: {step.reasoning}")
        
        steps_text = "\n".join(steps_summary)
        
        if response_format == "json":
            return f"""
You are a reviewer who needs to validate the reasonableness of the reasoning chain.

Task: {task.prompt}

Planning Result: {planner_output.get('plan', {}).get('task_analysis', '')}

Retrieved Subgraph: {len(retriever_output.nodes)} nodes, {len(retriever_output.edges)} edges
Relevance Score: {retriever_output.relevance_score:.2f}
Coverage Score: {retriever_output.coverage_score:.2f}

Reasoning Steps:
{steps_text}

Please validate:
1. Whether the reasoning chain is complete
2. Whether each step is reasonable
3. Whether any key steps are missing
4. Whether the conclusion is reliable

CRITICAL: You MUST respond with a valid JSON object in the following format:

{{
    "answer": "Your main validation conclusion - whether the reasoning is valid or not",
    "reasoning": ["Step 1: Check reasoning chain completeness", "Step 2: Validate reasoning steps", "Step 3: Identify missing steps", "Step 4: Assess conclusion reliability"],
    "citations": ["node1", "node2"],
    "confidence": 0.85
}}

IMPORTANT: 
1. Your response must be a valid JSON object
2. Do not include any text before or after the JSON object
3. Ensure all JSON syntax is correct (proper quotes, commas, brackets)
4. The "answer" field should contain your validation conclusion
5. The "reasoning" field should be an array of validation steps
6. The "citations" field should be an array of node IDs
7. The "confidence" field should be a number between 0.0 and 1.0
"""
        else:
            # Structured format
            return f"""
You are a reviewer who needs to validate the reasonableness of the reasoning chain.

Task: {task.prompt}

Planning Result: {planner_output.get('plan', {}).get('task_analysis', '')}

Retrieved Subgraph: {len(retriever_output.nodes)} nodes, {len(retriever_output.edges)} edges
Relevance Score: {retriever_output.relevance_score:.2f}
Coverage Score: {retriever_output.coverage_score:.2f}

Reasoning Steps:
{steps_text}

Please validate:
1. Whether the reasoning chain is complete
2. Whether each step is reasonable
3. Whether any key steps are missing
4. Whether the conclusion is reliable

CRITICAL: You MUST respond in the EXACT format specified below. Do not deviate from this format.

RESPONSE FORMAT (MANDATORY):

Answer: [Your main validation conclusion - whether the reasoning is valid or not]

Reasoning: [Provide detailed validation including:
1. Check reasoning chain completeness
2. Validate reasoning steps
3. Identify missing steps
4. Assess conclusion reliability]

Citations: [List relevant node IDs, separated by commas]

Confidence: [Provide a number between 0.0-1.0]

EXAMPLE FORMAT:
Answer: The reasoning chain is valid and complete, with all necessary steps properly executed.

Reasoning: 
1. Check reasoning chain completeness: All required reasoning steps have been executed
2. Validate reasoning steps: Each step is logical and well-founded
3. Identify missing steps: No critical steps appear to be missing
4. Assess conclusion reliability: The final conclusion is supported by the evidence

Citations: node_1, node_2, edge_1

Confidence: 0.85

REMEMBER: You MUST follow this exact format with the section headers (Answer:, Reasoning:, Citations:, Confidence:).
"""
    
    def _parse_verification_result_from_executor(self, response) -> VerificationResult:
        """Parse verification result from LLMExecutor's parsed response"""
        try:
            # Convert LLMExecutor format to VerificationResult
            answer_text = response.answer if hasattr(response, 'answer') else ""
            reasoning_steps = response.reasoning_path if hasattr(response, 'reasoning_path') else []
            confidence = response.confidence if hasattr(response, 'confidence') else 0.8
            
            # Determine if validation passed based on answer content
            is_valid = "valid" in answer_text.lower() or "pass" in answer_text.lower() or "complete" in answer_text.lower()
            
            # Convert reasoning steps to single reasoning text
            reasoning_text = "\n".join(reasoning_steps) if reasoning_steps else answer_text
            
            return VerificationResult(
                is_valid=is_valid,
                issues=[] if is_valid else [answer_text],
                suggestions=[],
                confidence=confidence,
                reasoning=reasoning_text
            )
            
        except Exception as e:
            return VerificationResult(
                is_valid=True,
                issues=[],
                suggestions=[],
                confidence=0.8,
                reasoning=f"Parsing failed: {e}. Original response: {response}"
            )
    
    def _parse_verification_result(self, response: str) -> VerificationResult:
        """Parse verification result"""
        try:
            # Parse structured text format
            result = {
                "is_valid": True,
                "issues": [],
                "suggestions": [],
                "confidence": 0.8,
                "reasoning": response
            }
            
            # Extract Answer section
            answer_match = re.search(r"Answer:\s*(.+?)(?=\n(?:Reasoning|Citations|Confidence):|$)", response, re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                # Determine if validation passed based on answer content
                result["is_valid"] = "valid" in answer_text.lower() or "pass" in answer_text.lower() or "complete" in answer_text.lower()
                if not result["is_valid"]:
                    result["issues"].append(answer_text)
            
            # Extract Reasoning section
            reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=\n(?:Answer|Citations|Confidence):|$)", response, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                result["reasoning"] = reasoning_match.group(1).strip()
            
            # Extract Confidence section
            confidence_match = re.search(r"Confidence:\s*(.+?)(?=\n(?:Answer|Reasoning|Citations):|$)", response, re.DOTALL | re.IGNORECASE)
            if confidence_match:
                try:
                    result["confidence"] = float(confidence_match.group(1).strip())
                except ValueError:
                    result["confidence"] = 0.8
            
            return VerificationResult(
                is_valid=result["is_valid"],
                issues=result["issues"],
                suggestions=result["suggestions"],
                confidence=result["confidence"],
                reasoning=result["reasoning"]
            )
            
        except Exception as e:
            # If parsing fails, return fallback
            return VerificationResult(
                is_valid=True,
                issues=[],
                suggestions=[],
                confidence=0.8,
                reasoning=f"Parsing failed: {e}. Original response: {response}"
            )


class SummarizerAgent(BaseAgent):
    """Summarization agent - Communication specialist"""
    
    def __init__(self, config: AgentConfig, executor: Optional[LLMExecutor] = None):
        super().__init__(AgentRole.SUMMARIZER, config, executor)
    
    def execute(self, task: TaskInstance, context: Dict[str, Any]) -> Dict[str, Any]:
        """Organize reasoning process and answer"""
        self._log_execution(f"Starting summarization task: {task.task_id}")
        
        # Get outputs from each agent
        planner_output = context.get('planner_output', {})
        retriever_output = context.get('retriever_output')
        reasoner_output = context.get('reasoner_output', [])
        verifier_output = context.get('verifier_output')
        graph = context.get('graph')  # Get graph
        
        try:
            # Build summarization prompt
            prompt = self._build_summarization_prompt(task, planner_output, retriever_output, 
                                                    reasoner_output, verifier_output)
            
            # Create summarization task
            from task_craft.task_generator import TaskInstance
            from task_craft.task_templates import TaskType, TaskDifficulty
            
            summarization_task = TaskInstance(
                task_id=f"{task.task_id}_summarizer",
                template_id="summarizer_template",
                task_type=TaskType.COMPREHENSION,
                difficulty=TaskDifficulty.MEDIUM,
                prompt=prompt,
                gold_answer="",
                gold_nodes=[],
                gold_edges=[],
                subgraph_nodes=[],
                subgraph_edges=[],
                requires_citations=True,  # Summarization stage needs citations
                requires_reasoning_path=True  # Summarization stage needs reasoning path
            )
            
            # Create context with actual graph data
            from .retrievers import RetrievalResult
            context_nodes = []
            context_edges = []
            context_scores = {}
            
            # Get actual node content from graph
            if graph and retriever_output and hasattr(retriever_output, 'nodes'):
                for node_id in retriever_output.nodes:
                    # Find actual node from graph
                    actual_node = graph.storage.get_node(node_id)
                    if actual_node:
                        context_nodes.append(actual_node)
                        context_scores[node_id] = retriever_output.relevance_score
            
            # Get actual edge information from graph
            if graph and retriever_output and hasattr(retriever_output, 'edges'):
                for edge_item in retriever_output.edges:
                    # Handle different edge formats: edge_id string, edge object, or edge reference
                    if isinstance(edge_item, str):
                        # edge_item is an edge_id string
                        actual_edge = graph.storage.get_edge(edge_item)
                        if actual_edge:
                            context_edges.append(actual_edge)
                    elif hasattr(edge_item, 'edge_id'):
                        # edge_item is an edge object with edge_id
                        actual_edge = graph.storage.get_edge(edge_item.edge_id)
                        if actual_edge:
                            context_edges.append(actual_edge)
                    elif hasattr(edge_item, 'id'):
                        # edge_item is an edge object with id
                        actual_edge = graph.storage.get_edge(edge_item.id)
                        if actual_edge:
                            context_edges.append(actual_edge)
                    else:
                        # edge_item is already an edge object
                        context_edges.append(edge_item)
            
            summarization_context = RetrievalResult(
                nodes=context_nodes,
                edges=context_edges,
                scores=context_scores,
                retrieval_method="summarization_context",
                total_nodes_considered=len(context_nodes)
            )
            
            # Use shared singleton executor instead of creating new instance
            temp_executor = LLMExecutor.get_instance()
            
            # Use temporary executor to execute summarization task
            response = temp_executor.execute(summarization_task, summarization_context)
            
            # Parse summarization result from executor response
            summary_result = self._parse_summarization_result_from_executor(response)
            
            self._log_execution("Summarization completed")
            
            return summary_result
            
        except Exception as e:
            self._log_execution(f"Summarization failed: {str(e)}")
            raise
    
    def _build_summarization_prompt(self, task: TaskInstance, planner_output: Dict[str, Any], 
                                  retriever_output: SubgraphInfo, reasoner_output: List[ReasoningStep], 
                                  verifier_output: VerificationResult) -> str:
        """Build summary prompt with dynamic format based on execution_config"""
        # Get response format from execution config
        response_format = self.config.execution_config.response_format if hasattr(self.config, 'execution_config') else "json"
        
        # Build reasoning steps summary
        steps_summary = []
        for step in reasoner_output:
            steps_summary.append(f"Step {step.step_id}: {step.description}\nReasoning: {step.reasoning}")
        
        steps_text = "\n\n".join(steps_summary)
        
        if response_format == "json":
            return f"""
You are a writing editor who needs to organize the reasoning process and final answer.

Task: {task.prompt}

Planning Analysis: {planner_output.get('plan', {}).get('task_analysis', '')}

Retrieved Information: Retrieved {len(retriever_output.nodes)} relevant nodes from the graph, relevance score {retriever_output.relevance_score:.2f}

Reasoning Process:
{steps_text}

Verification Result: {'Passed' if verifier_output.is_valid else 'Issues Found'}
Verification Details: {verifier_output.reasoning}

CRITICAL: You MUST respond with a valid JSON object in the following format:

{{
    "answer": "Clear final answer to the task",
    "reasoning": ["Step 1: Summary of planning findings", "Step 2: Summary of retrieved information", "Step 3: Summary of reasoning process", "Step 4: Final conclusion"],
    "citations": ["node1", "node2"],
    "confidence": 0.85
}}

IMPORTANT: 
1. Your response must be a valid JSON object
2. Do not include any text before or after the JSON object
3. Ensure all JSON syntax is correct (proper quotes, commas, brackets)
4. The "answer" field should contain your final answer
5. The "reasoning" field should be an array of summary steps
6. The "citations" field should be an array of node IDs
7. The "confidence" field should be a number between 0.0 and 1.0
"""
        else:
            # Structured format
            return f"""
You are a writing editor who needs to organize the reasoning process and final answer.

Task: {task.prompt}

Planning Analysis: {planner_output.get('plan', {}).get('task_analysis', '')}

Retrieved Information: Retrieved {len(retriever_output.nodes)} relevant nodes from the graph, relevance score {retriever_output.relevance_score:.2f}

Reasoning Process:
{steps_text}

Verification Result: {'Passed' if verifier_output.is_valid else 'Issues Found'}
Verification Details: {verifier_output.reasoning}

CRITICAL: You MUST respond in the EXACT format specified below. Do not deviate from this format.

RESPONSE FORMAT (MANDATORY):

Answer: [Clear final answer to the task]

Reasoning: [Provide detailed summary including:
1. Summary of planning findings
2. Summary of retrieved information
3. Summary of reasoning process
4. Final conclusion]

Citations: [List relevant node IDs, separated by commas]

Confidence: [Provide a number between 0.0-1.0]

EXAMPLE FORMAT:
Answer: Based on the comprehensive analysis, the graph construction process involves multiple edge types that contribute to different aspects of the knowledge representation.

Reasoning: 
1. Summary of planning findings: The task was successfully decomposed into manageable reasoning steps
2. Summary of retrieved information: Retrieved relevant nodes covering edge types and graph construction methods
3. Summary of reasoning process: Analyzed the relationships between different components systematically
4. Final conclusion: The graph construction process is well-structured with clear edge type definitions

Citations: node_1, node_2, edge_1

Confidence: 0.85

REMEMBER: You MUST follow this exact format with the section headers (Answer:, Reasoning:, Citations:, Confidence:).
"""
    
    def _parse_summarization_result_from_executor(self, response) -> Dict[str, Any]:
        """Parse summarization result from LLMExecutor's parsed response"""
        try:
            # Convert LLMExecutor format to our internal format
            answer = response.answer if hasattr(response, 'answer') else ""
            reasoning_steps = response.reasoning_path if hasattr(response, 'reasoning_path') else []
            citations = response.citations if hasattr(response, 'citations') else []
            confidence = response.confidence if hasattr(response, 'confidence') else 0.8
            
            # Convert reasoning steps to single reasoning text
            reasoning_summary = "\n".join(reasoning_steps) if reasoning_steps else answer
            
            return {
                "final_answer": answer,
                "reasoning_summary": reasoning_summary,
                "citations": citations,
                "confidence": confidence
            }
            
        except Exception as e:
            return {
                "final_answer": f"Parsing failed: {e}",
                "reasoning_summary": f"Parsing failed: {e}",
                "citations": [],
                "confidence": 0.8
            }
    
    def _parse_summarization_result(self, response: str) -> Dict[str, Any]:
        """Parse summarization result"""
        try:
            # Parse structured text format
            result = {
                "final_answer": "",
                "reasoning_summary": "",
                "citations": [],
                "confidence": 0.8
            }
            
            # Extract Answer section
            answer_match = re.search(r"Answer:\s*(.+?)(?=\n(?:Reasoning|Citations|Confidence):|$)", response, re.DOTALL | re.IGNORECASE)
            if answer_match:
                result["final_answer"] = answer_match.group(1).strip()
            
            # Extract Reasoning section
            reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=\n(?:Answer|Citations|Confidence):|$)", response, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                result["reasoning_summary"] = reasoning_match.group(1).strip()
            
            # Extract Citations section
            citations_match = re.search(r"Citations:\s*(.+?)(?=\n(?:Answer|Reasoning|Confidence):|$)", response, re.DOTALL | re.IGNORECASE)
            if citations_match:
                citations_text = citations_match.group(1).strip()
                # Parse comma-separated citations
                citations = [c.strip() for c in citations_text.split(",") if c.strip()]
                result["citations"] = citations
            
            # Extract Confidence section
            confidence_match = re.search(r"Confidence:\s*(.+?)(?=\n(?:Answer|Reasoning|Citations):|$)", response, re.DOTALL | re.IGNORECASE)
            if confidence_match:
                try:
                    result["confidence"] = float(confidence_match.group(1).strip())
                except ValueError:
                    result["confidence"] = 0.8
            
            # If no structured content found, use entire response as answer
            if not result["final_answer"]:
                result["final_answer"] = response.strip()
                result["reasoning_summary"] = response.strip()
            
            return result
            
        except Exception as e:
            # If parsing fails, return fallback
            return {
                "final_answer": f"Parsing failed: {e}. Original response: {response}",
                "reasoning_summary": f"Parsing failed: {e}. Original response: {response}",
                "citations": [],
                "confidence": 0.8
            }


@dataclass
class MultiAgentSystemConfig:
    """Multi-agent system configuration"""
    
    # Agent configurations with individual LLM models
    planner_config: AgentConfig = field(default_factory=AgentConfig)
    retriever_config: AgentConfig = field(default_factory=AgentConfig)
    reasoner_config: AgentConfig = field(default_factory=AgentConfig)
    verifier_config: AgentConfig = field(default_factory=AgentConfig)
    summarizer_config: AgentConfig = field(default_factory=AgentConfig)
    
    # Individual LLM model configurations for each agent
    planner_model: str = "gpt-4o-mini"
    retriever_model: str = "gpt-4o-mini"
    reasoner_model: str = "gpt-4o-mini"
    verifier_model: str = "gpt-4o-mini"
    summarizer_model: str = "gpt-4o-mini"
    
    # System configuration
    max_iterations: int = 3
    confidence_threshold: float = 0.7
    
    # Logging configuration
    verbose: bool = False


class MultiAgentSystem:
    """Multi-agent system - Coordinate multiple specialized agents to complete complex reasoning tasks"""
    
    def __init__(self, config: MultiAgentSystemConfig, executor: Optional[LLMExecutor] = None):
        self.config = config
        self.logger = logger.bind(system="multi_agent")
        
        # Initialize agents with individual LLM configurations
        self.agents = {}
        
        if executor is not None:
            # Use provided shared executor for all agents
            self.logger.info(f"Using provided shared executor: {executor.config.model_name}")
            self.actual_models = {
                'planner': executor.config.model_name,
                'retriever': executor.config.model_name,
                'reasoner': executor.config.model_name,
                'verifier': executor.config.model_name,
                'summarizer': executor.config.model_name
            }
            
            # Initialize agents with shared executor
            self.agents = {
                AgentRole.PLANNER: PlannerAgent(config.planner_config, executor),
                AgentRole.REASONER: ReasonerAgent(config.reasoner_config, executor),
                AgentRole.VERIFIER: VerifierAgent(config.verifier_config, executor),
                AgentRole.SUMMARIZER: SummarizerAgent(config.summarizer_config, executor)
            }
            
            # Conditionally add retriever agent based on configuration
            # Check if retriever should be enabled by looking at the config
            if hasattr(config, 'retriever_config') and config.retriever_config:
                self.agents[AgentRole.RETRIEVER] = RetrieverAgent(config.retriever_config, executor)
            
            self.logger.info(f"Multi-agent system initialized with shared executor: {executor.config.model_name}")
        else:
            # Create individual executors for each agent with their specific models
            planner_executor = self._create_agent_executor(config.planner_model, config.planner_config)
            retriever_executor = self._create_agent_executor(config.retriever_model, config.retriever_config)
            reasoner_executor = self._create_agent_executor(config.reasoner_model, config.reasoner_config)
            verifier_executor = self._create_agent_executor(config.verifier_model, config.verifier_config)
            summarizer_executor = self._create_agent_executor(config.summarizer_model, config.summarizer_config)
            
            # Store actual model names used for logging
            self.actual_models = {
                'planner': planner_executor.config.model_name,
                'reasoner': reasoner_executor.config.model_name,
                'verifier': verifier_executor.config.model_name,
                'summarizer': summarizer_executor.config.model_name
            }
            
            # Conditionally add retriever model if retriever agent is enabled
            if hasattr(config, 'retriever_config') and config.retriever_config:
                self.actual_models['retriever'] = retriever_executor.config.model_name
            else:
                # Remove retriever from actual_models if it was added earlier
                self.actual_models.pop('retriever', None)
            
            # Initialize agents with their specific executors
            self.agents = {
                AgentRole.PLANNER: PlannerAgent(config.planner_config, planner_executor),
                AgentRole.REASONER: ReasonerAgent(config.reasoner_config, reasoner_executor),
                AgentRole.VERIFIER: VerifierAgent(config.verifier_config, verifier_executor),
                AgentRole.SUMMARIZER: SummarizerAgent(config.summarizer_config, summarizer_executor)
            }
            
            # Conditionally add retriever agent based on configuration
            if hasattr(config, 'retriever_config') and config.retriever_config:
                self.agents[AgentRole.RETRIEVER] = RetrieverAgent(config.retriever_config, retriever_executor)
            
            self.logger.info(f"Multi-agent system initialized with individual LLM models:")
            self.logger.info(f"  Planner: {self.actual_models['planner']}")
            if 'retriever' in self.actual_models:
                self.logger.info(f"  Retriever: {self.actual_models['retriever']}")
            else:
                self.logger.info(f"  Retriever: Disabled (no_rag mode)")
            self.logger.info(f"  Reasoner: {self.actual_models['reasoner']}")
            self.logger.info(f"  Verifier: {self.actual_models['verifier']}")
            self.logger.info(f"  Summarizer: {self.actual_models['summarizer']}")
    
    def _create_agent_executor(self, model_name: str, agent_config: AgentConfig) -> LLMExecutor:
        """Create individual executor for each agent with specific model"""
        # Priority logic:
        # 1. If model_name is explicitly set (not default), use it
        # 2. Otherwise, use the model from agent_config.execution_config
        if model_name != "gpt-4o-mini":
            # Use explicitly set model name
            final_model_name = model_name
        else:
            # Use model from agent_config
            final_model_name = agent_config.execution_config.model_name
        
        # Create execution config with the specific model
        execution_config = ExecutionConfig(
            model_name=final_model_name,
            temperature=agent_config.execution_config.temperature,
            max_tokens=agent_config.execution_config.max_tokens,
            timeout=agent_config.execution_config.timeout,
            max_retries=agent_config.execution_config.max_retries,
            response_format=agent_config.execution_config.response_format
        )
        
        # Create new executor instance for this agent
        return LLMExecutor(execution_config)
    
    def execute_task(self, task: TaskInstance, graph: DocumentGraph = None) -> MultiAgentResponse:
        """Execute task"""
        start_time = time.time()
        total_tokens = 0
        
        self.logger.info(f"Starting task execution: {task.task_id}")
        
        try:
            # Initialize context
            context = {
                'task': task,
                'graph': graph,
                'iteration': 0
            }
            
            # Execute agent collaboration pipeline
            response = self._execute_agent_pipeline(context)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Build final response
            multi_agent_response = MultiAgentResponse(
                task_id=task.task_id,
                final_answer=response['summarizer_output'].get('final_answer', ''),
                success=True,
                planner_output=response.get('planner_output', {}),
                retriever_output=response.get('retriever_output'),
                reasoner_output=response.get('reasoner_output', []),
                verifier_output=response.get('verifier_output'),
                summarizer_output=response.get('summarizer_output', {}),
                execution_time=execution_time,
                total_tokens=total_tokens,
                model_used=f"Multi-Agent({self.actual_models['planner']},{self.actual_models.get('retriever', 'disabled')},{self.actual_models['reasoner']},{self.actual_models['verifier']},{self.actual_models['summarizer']})"
            )
            
            self.logger.info(f"Task execution completed: {task.task_id}, time taken: {execution_time:.2f}s")
            
            return multi_agent_response
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {task.task_id}, error: {str(e)}")
            
            return MultiAgentResponse(
                task_id=task.task_id,
                final_answer="",
                success=False,
                execution_time=time.time() - start_time,
                error_type=type(e).__name__,
                error_message=str(e)
            )
    
    def _execute_agent_pipeline(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent collaboration pipeline"""
        max_iterations = self.config.max_iterations
        confidence_threshold = self.config.confidence_threshold
        
        for iteration in range(max_iterations):
            context['iteration'] = iteration
            self.logger.info(f"Starting iteration {iteration + 1}")
            
            # 1. Planning stage
            planner_output = self._execute_planner(context)
            context['planner_output'] = planner_output
            
            # 2. Retrieval stage
            retriever_output = self._execute_retriever(context)
            context['retriever_output'] = retriever_output
            
            # 3. Reasoning stage
            reasoner_output = self._execute_reasoner(context)
            context['reasoner_output'] = reasoner_output
            
            # 4. Verification stage
            verifier_output = self._execute_verifier(context)
            context['verifier_output'] = verifier_output
            
            # Check if iteration is needed
            if verifier_output.is_valid and verifier_output.confidence >= confidence_threshold:
                self.logger.info(f"Verification passed, confidence: {verifier_output.confidence:.2f}")
                break
            
            # If verification fails, adjust based on suggestions
            if verifier_output.suggestions:
                self.logger.info(f"Verification failed, adjusting based on suggestions: {verifier_output.suggestions}")
                context = self._apply_verification_suggestions(context, verifier_output.suggestions)
        
        # 5. Summarization stage
        summarizer_output = self._execute_summarizer(context)
        context['summarizer_output'] = summarizer_output
        
        return context
    
    def _execute_planner(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute planning agent"""
        task = context['task']
        return self.agents[AgentRole.PLANNER].execute(task, context)
    
    def _execute_retriever(self, context: Dict[str, Any]) -> SubgraphInfo:
        """Execute retrieval agent"""
        task = context['task']
        graph = context.get('graph')
        
        # Check if retriever agent exists
        if AgentRole.RETRIEVER in self.agents:
            return self.agents[AgentRole.RETRIEVER].execute(task, {'graph': graph, 'planner_output': context.get('planner_output', {})})
        else:
            # Return empty subgraph info if retriever agent is disabled
            from .retrievers import RetrievalResult
            return SubgraphInfo(
                nodes=[],
                edges=[],
                relevance_score=0.0,
                coverage_score=0.0,
                reasoning="Retriever agent is disabled (no_rag mode)"
            )
    
    def _execute_reasoner(self, context: Dict[str, Any]) -> List[ReasoningStep]:
        """Execute reasoning agent"""
        task = context['task']
        return self.agents[AgentRole.REASONER].execute(task, context)
    
    def _execute_verifier(self, context: Dict[str, Any]) -> VerificationResult:
        """Execute verification agent"""
        task = context['task']
        graph = context.get('graph')
        return self.agents[AgentRole.VERIFIER].execute(task, {'graph': graph, 'planner_output': context.get('planner_output', {}), 'retriever_output': context.get('retriever_output'), 'reasoner_output': context.get('reasoner_output', [])})
    
    def _execute_summarizer(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute summarization agent"""
        task = context['task']
        graph = context.get('graph')
        return self.agents[AgentRole.SUMMARIZER].execute(task, {'graph': graph, 'planner_output': context.get('planner_output', {}), 'retriever_output': context.get('retriever_output'), 'reasoner_output': context.get('reasoner_output', []), 'verifier_output': context.get('verifier_output')})
    
    def _apply_verification_suggestions(self, context: Dict[str, Any], suggestions: List[str]) -> Dict[str, Any]:
        """Adjust context based on verification suggestions"""
        # Here you can implement context adjustment logic based on suggestions
        # For example: re-retrieval, adjust reasoning steps, etc.
        self.logger.info(f"Applying verification suggestions: {suggestions}")
        return context
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status"""
        agent_status = {}
        for role in AgentRole:
            if role in self.agents:
                agent_status[role.value] = "active"
            else:
                agent_status[role.value] = "disabled"
        
        return {
            "agents": agent_status,
            "config": {
                "max_iterations": self.config.max_iterations,
                "confidence_threshold": self.config.confidence_threshold
            }
        }
    
    def update_config(self, new_config: MultiAgentSystemConfig):
        """Update system configuration"""
        self.config = new_config
        self.logger.info("System configuration updated")
    
    def reset(self):
        """Reset system state"""
        self.logger.info("System state reset")


# Convenient factory function
def create_multi_agent_system(
    multi_agent_config: Dict[str, Any] = None,
    system_config: Dict[str, Any] = None,
    agent_type: str = "rag",
    executor: Optional[LLMExecutor] = None
) -> MultiAgentSystem:
    """Create multi-agent system from unified configuration
    
    Args:
        multi_agent_config: Multi-agent system configuration dictionary
        system_config: Common system configuration dictionary
        agent_type: "rag" or "no_rag" - controls whether retrieval is enabled
        executor: Optional shared executor (if provided, all agents will use this executor; if None, each agent gets its own executor)
    
    Example:
        # Use unified configuration
        system = create_multi_agent_system(
            multi_agent_config=agent_config['multi_agent'],
            system_config=agent_config['system'],
            agent_type=agent_config['agent_type']
        )
    """
    
    # Use default configs if not provided
    multi_agent_config = multi_agent_config or {}
    system_config = system_config or {}
    
    # Get system-level settings
    max_iterations = multi_agent_config.get('max_iterations', 3)
    confidence_threshold = multi_agent_config.get('confidence_threshold', 0.7)
    verbose = multi_agent_config.get('verbose', system_config.get('verbose', False))
    
    # Get agent configurations
    agents_config = multi_agent_config.get('agents', {})
    
    # Create agent configs from unified configuration
    agent_configs = {}
    for role in ['planner', 'retriever', 'reasoner', 'verifier', 'summarizer']:
        role_config = agents_config.get(role, {})
        
        # Skip retriever agent if agent_type is "no_rag"
        if role == 'retriever' and agent_type == 'no_rag':
            logger.info("🔄 Skipping retriever agent (agent_type is 'no_rag')")
            continue
        
        # Create execution config for this agent
        execution_config = ExecutionConfig(
            model_name=role_config.get('model_name', 'gpt-4o-mini'),
            temperature=role_config.get('temperature', 0.1),
            max_tokens=role_config.get('max_tokens', 1000),
            timeout=role_config.get('timeout', 30),
            max_retries=role_config.get('max_retries', 3),
            response_format=role_config.get('response_format', 'json')
        )
        
        # Create retrieval config for this agent
        # Disable retrieval for all agents if agent_type is "no_rag"
        retrieval_enabled = (agent_type == 'rag')
        retrieval_config = RetrievalConfig(
            max_nodes=role_config.get('retrieval', {}).get('max_nodes', 10),
            max_hops=role_config.get('retrieval', {}).get('max_hops', 3),
            similarity_threshold=role_config.get('retrieval', {}).get('similarity_threshold', 0.7)
        )
        
        # Create agent config
        agent_configs[role] = AgentConfig(
            execution_config=execution_config,
            retrieval_config=retrieval_config,
            enable_evaluation=role_config.get('enable_evaluation', system_config.get('enable_evaluation', True)),
            verbose=role_config.get('verbose', system_config.get('verbose', False))
        )
    
    # Create multi-agent system configuration
    # Use default config for retriever if it was skipped
    if 'retriever' not in agent_configs:
        # Create a default retriever config for no_rag mode
        default_retriever_config = AgentConfig(
            execution_config=ExecutionConfig(
                model_name='gpt-4o-mini',
                temperature=0.1,
                max_tokens=1000,
                timeout=30,
                max_retries=3,
                response_format='json'
            ),
            retrieval_config=RetrievalConfig(
                max_nodes=10,
                max_hops=3,
                similarity_threshold=0.7
            ),
            enable_evaluation=system_config.get('enable_evaluation', True),
            verbose=system_config.get('verbose', False)
        )
        agent_configs['retriever'] = default_retriever_config
    
    mas_config = MultiAgentSystemConfig(
        planner_config=agent_configs['planner'],
        retriever_config=agent_configs['retriever'],
        reasoner_config=agent_configs['reasoner'],
        verifier_config=agent_configs['verifier'],
        summarizer_config=agent_configs['summarizer'],
        max_iterations=max_iterations,
        confidence_threshold=confidence_threshold,
        verbose=verbose
    )
    
    return MultiAgentSystem(mas_config, executor)


def create_multi_agent_system_from_config(
    agent_configs: Dict[str, AgentConfig],
    system_config: Optional[MultiAgentSystemConfig] = None,
    executor: Optional[LLMExecutor] = None
) -> MultiAgentSystem:
    """Create multi-agent system from individual agent configurations
    
    Args:
        agent_configs: Dictionary with keys 'planner', 'retriever', 'reasoner', 'verifier', 'summarizer'
        system_config: Optional system-level configuration
        executor: Optional shared executor (if provided, all agents will use this executor; if None, each agent gets its own executor)
    
    Example:
        # Create individual agent configs
        planner_config = AgentConfig(
            execution_config=ExecutionConfig(model_name="gpt-4", temperature=0.1)
        )
        retriever_config = AgentConfig(
            execution_config=ExecutionConfig(model_name="gpt-3.5-turbo", temperature=0.0)
        )
        
        # Create system
        system = create_multi_agent_system_from_config({
            'planner': planner_config,
            'retriever': retriever_config,
            # Other agents will use default config
        })
    """
    
    # Create default base configuration
    default_config = AgentConfig(
        execution_config=ExecutionConfig(
            model_name="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1000,
            timeout=30,
            max_retries=3
        ),
        retrieval_config=RetrievalConfig(
            max_nodes=10,
            max_hops=3,
            similarity_threshold=0.7
        ),
        enable_evaluation=True,
        verbose=False
    )
    
    # Use provided configs or defaults
    planner_config = agent_configs.get('planner', default_config)
    retriever_config = agent_configs.get('retriever', default_config)
    reasoner_config = agent_configs.get('reasoner', default_config)
    verifier_config = agent_configs.get('verifier', default_config)
    summarizer_config = agent_configs.get('summarizer', default_config)
    
    # Create system config
    if system_config is None:
        system_config = MultiAgentSystemConfig(
            max_iterations=3,
            confidence_threshold=0.7,
            verbose=False
        )
    
    # Update system config with agent configs
    system_config.planner_config = planner_config
    system_config.retriever_config = retriever_config
    system_config.reasoner_config = reasoner_config
    system_config.verifier_config = verifier_config
    system_config.summarizer_config = summarizer_config
    
    # Set model names from agent configs
    system_config.planner_model = planner_config.execution_config.model_name
    system_config.retriever_model = retriever_config.execution_config.model_name
    system_config.reasoner_model = reasoner_config.execution_config.model_name
    system_config.verifier_model = verifier_config.execution_config.model_name
    system_config.summarizer_model = summarizer_config.execution_config.model_name
    
    return MultiAgentSystem(system_config, executor)