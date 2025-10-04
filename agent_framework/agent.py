"""
RAG Agent - Intelligent agent with retrieval-augmented generation capabilities
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Callable
import time
from loguru import logger

from .retrievers import Retriever, RetrievalResult, RetrievalConfig, HybridRetriever, ContextualRetriever
from .executors import TaskExecutor, ExecutionResult, ExecutionConfig, LLMExecutor, MultiStepExecutor
from .evaluators import TaskEvaluator
from task_craft.task_generator import TaskInstance
from graph_rag.graph_builder import DocumentGraph
from config_manager import get_config


@dataclass
class AgentConfig:
    """Configuration for RAG Agent"""
    
    # Core components
    retriever_type: str = "hybrid"  # hybrid, contextual, subgraph
    executor_type: str = "llm"  # llm, multistep
    enable_evaluation: bool = True
    
    # Retrieval settings
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    # Execution settings
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    # Memory and context
    max_context_length: int = 4000
    enable_memory: bool = True
    memory_size: int = 10
    
    # Debug and logging
    verbose: bool = False
    log_intermediate: bool = False
    
    @classmethod
    def from_config(cls):
        """从配置文件创建Agent配置"""
        config = get_config()
        agent_config = config.agent
        
        return cls(
            retriever_type=agent_config.get('retriever_type', 'hybrid'),
            executor_type=agent_config.get('executor_type', 'llm'),
            enable_evaluation=agent_config.get('enable_evaluation', True),
            retrieval_config=RetrievalConfig.from_config(),
            execution_config=ExecutionConfig.from_config(),
            max_context_length=agent_config.get('max_context_length', 4000),
            enable_memory=agent_config.get('enable_memory', True),
            memory_size=agent_config.get('memory_size', 10),
            verbose=agent_config.get('verbose', False),
            log_intermediate=agent_config.get('log_intermediate', False)
        )


@dataclass
class AgentResponse:
    """Response from RAG Agent"""
    
    task_id: str
    answer: str
    success: bool
    
    # Retrieval information
    retrieved_nodes: List[str] = field(default_factory=list)
    retrieval_method: str = ""
    retrieval_scores: Dict[str, float] = field(default_factory=dict)
    
    # Execution information
    execution_time: float = 0.0
    model_used: str = ""
    tokens_used: int = 0
    
    # Safety and evaluation
    safety_passed: bool = True
    safety_issues: List[str] = field(default_factory=list)
    evaluation_score: float = 0.0
    evaluation_details: Dict[str, Any] = field(default_factory=dict)
    
    # Citations and reasoning
    citations: List[str] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    # Error information
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "answer": self.answer,
            "success": self.success,
            "retrieved_nodes": self.retrieved_nodes,
            "retrieval_method": self.retrieval_method,
            "retrieval_scores": self.retrieval_scores,
            "execution_time": self.execution_time,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "safety_passed": self.safety_passed,
            "safety_issues": self.safety_issues,
            "evaluation_score": self.evaluation_score,
            "evaluation_details": self.evaluation_details,
            "citations": self.citations,
            "reasoning_path": self.reasoning_path,
            "confidence": self.confidence,
            "error_type": self.error_type,
            "error_message": self.error_message
        }


class RAGAgent:
    """RAG Agent with retrieval-augmented generation capabilities"""
    
    def __init__(self, config: Optional[AgentConfig] = None, executor: Optional[TaskExecutor] = None):
        self.config = config or AgentConfig()
        self._setup_components(executor)
        self.memory = []  # Task history for context
        self.graph: Optional[DocumentGraph] = None
    
    def _setup_components(self, executor: Optional[TaskExecutor] = None):
        """Setup agent components"""
        logger.info("Setting up RAG Agent components...")
        
        # Setup retriever
        self.retriever = self._create_retriever()
        
        # Setup executor - use provided executor or create new one
        if executor is not None:
            self.executor = executor
            logger.info(f"Using provided executor: {type(executor).__name__}")
        else:
            self.executor = self._create_executor()
        
        # Setup evaluator
        if self.config.enable_evaluation:
            from .evaluators import MultiDimensionalEvaluator
            self.evaluator = MultiDimensionalEvaluator()
        else:
            self.evaluator = None
        
        logger.info("RAG Agent components setup complete")
    
    def _create_retriever(self) -> Retriever:
        """Create retriever based on configuration"""
        if self.config.retriever_type == "hybrid":
            return HybridRetriever(self.config.retrieval_config)
        elif self.config.retriever_type == "contextual":
            return ContextualRetriever(self.config.retrieval_config)
        elif self.config.retriever_type == "subgraph":
            from .retrievers import SubgraphRetriever
            return SubgraphRetriever(self.config.retrieval_config)
        else:
            logger.warning(f"Unknown retriever type: {self.config.retriever_type}, using hybrid")
            return HybridRetriever(self.config.retrieval_config)
    
    def _create_executor(self) -> TaskExecutor:
        """Create executor based on configuration"""
        if self.config.executor_type == "llm":
            return LLMExecutor(self.config.execution_config)
        elif self.config.executor_type == "multistep":
            return MultiStepExecutor(self.config.execution_config)
        else:
            logger.warning(f"Unknown executor type: {self.config.executor_type}, using llm")
            return LLMExecutor(self.config.execution_config)
    
    def set_graph(self, graph: DocumentGraph):
        """Set the knowledge graph for the agent"""
        self.graph = graph
        stats = graph.storage.get_stats()
        logger.info(f"Agent knowledge graph set with {stats['total_nodes']} nodes")
    
    def execute_task(self, task: TaskInstance) -> AgentResponse:
        """Execute a task using RAG capabilities"""
        
        if self.graph is None:
            return AgentResponse(
                task_id=task.task_id,
                answer="",
                success=False,
                error_type="NoGraphError",
                error_message="No knowledge graph available for retrieval"
            )
        
        start_time = time.time()
        
        try:
            if self.config.verbose:
                logger.info(f"🔄 Executing task: {task.task_id}")
                logger.info(f"   Task type: {task.task_type.value}")
                logger.info(f"   Prompt: {task.prompt[:100]}...")
            
            # Step 1: Retrieve relevant context
            retrieval_result = self._retrieve_context(task)
            
            if self.config.verbose:
                logger.info(f"📚 Retrieved {len(retrieval_result.nodes)} nodes using {retrieval_result.retrieval_method}")
            
            # Step 2: Execute task
            execution_result = self.executor.execute(task, retrieval_result)
            
            # Step 3: Evaluation (if enabled)
            evaluation_result = self._evaluate_response(task, execution_result, retrieval_result)
            
            # Step 4: Build response
            response = self._build_response(
                task, execution_result, retrieval_result, 
                evaluation_result, time.time() - start_time
            )
            
            # Step 5: Update memory
            self._update_memory(task, response)
            
            if self.config.verbose:
                logger.info(f"✅ Task {task.task_id} completed successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Task {task.task_id} failed: {e}")
            
            return AgentResponse(
                task_id=task.task_id,
                answer="",
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _retrieve_context(self, task: TaskInstance) -> RetrievalResult:
        """Retrieve relevant context for the task"""
        if self.config.log_intermediate:
            logger.debug(f"Retrieving context for task {task.task_id}")
        
        retrieval_result = self.retriever.retrieve(task, self.graph)
        
        if self.config.log_intermediate:
            logger.debug(f"Retrieved {len(retrieval_result.nodes)} nodes")
            for node in retrieval_result.nodes[:3]:  # Log top 3 nodes
                score = retrieval_result.scores.get(node.node_id, 0.0)
                logger.debug(f"  - {node.node_id}: {score:.3f} ({node.node_type.value})")
        
        return retrieval_result
    
    def _evaluate_response(
        self, 
        task: TaskInstance, 
        execution_result: ExecutionResult, 
        retrieval_result: RetrievalResult
    ) -> Dict[str, Any]:
        """Evaluate the response quality"""
        if not self.config.enable_evaluation or self.evaluator is None:
            return {"score": 0.0, "details": {}}
        
        try:
            evaluation_result = self.evaluator.evaluate(task, execution_result, retrieval_result)
            
            return {
                "score": evaluation_result.overall_score,
                "details": evaluation_result.details
            }
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return {"score": 0.0, "details": {"error": str(e)}}
    
    def _build_response(
        self,
        task: TaskInstance,
        execution_result: ExecutionResult,
        retrieval_result: RetrievalResult,
        evaluation_result: Dict[str, Any],
        total_time: float
    ) -> AgentResponse:
        """Build comprehensive response"""
        
        return AgentResponse(
            task_id=task.task_id,
            success=execution_result.success,
            answer=execution_result.answer,
            citations=execution_result.citations,
            reasoning_path=execution_result.reasoning_path,
            confidence=execution_result.confidence,
            execution_time=total_time,
            model_used=execution_result.model_used,
            tokens_used=execution_result.tokens_used,
            evaluation_score=evaluation_result["score"],
            evaluation_details=evaluation_result["details"],
            retrieved_nodes=[node.node_id for node in retrieval_result.nodes] if retrieval_result else [],
            retrieval_method=retrieval_result.retrieval_method if retrieval_result else "",
            retrieval_scores=retrieval_result.scores if retrieval_result else {},
            error_type=execution_result.error_type,
            error_message=execution_result.error_message
        )
    
    def _update_memory(self, task: TaskInstance, response: AgentResponse):
        """Update agent memory with task and response"""
        if not self.config.enable_memory:
            return
        
        memory_entry = {
            "task": task,
            "response": response,
            "timestamp": time.time()
        }
        
        self.memory.append(memory_entry)
        
        # Keep memory size limited
        if len(self.memory) > self.config.memory_size:
            self.memory.pop(0)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of agent memory"""
        if not self.memory:
            return {"total_tasks": 0, "success_rate": 0.0, "avg_confidence": 0.0}
        
        total_tasks = len(self.memory)
        successful_tasks = sum(1 for entry in self.memory if entry["response"].success)
        avg_confidence = sum(entry["response"].confidence for entry in self.memory) / total_tasks
        
        return {
            "total_tasks": total_tasks,
            "success_rate": successful_tasks / total_tasks,
            "avg_confidence": avg_confidence,
            "recent_tasks": [
                {
                    "task_id": entry["task"].task_id,
                    "task_type": entry["task"].task_type.value,
                    "success": entry["response"].success,
                    "confidence": entry["response"].confidence
                }
                for entry in self.memory[-5:]  # Last 5 tasks
            ]
        }
    
    def clear_memory(self):
        """Clear agent memory"""
        self.memory.clear()
        logger.info("Agent memory cleared")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        if not self.memory:
            return {}
        
        retrieval_methods = {}
        avg_nodes_retrieved = 0
        
        for entry in self.memory:
            method = entry["response"].retrieval_method
            retrieval_methods[method] = retrieval_methods.get(method, 0) + 1
            avg_nodes_retrieved += len(entry["response"].retrieved_nodes)
        
        if self.memory:
            avg_nodes_retrieved /= len(self.memory)
        
        return {
            "retrieval_methods": retrieval_methods,
            "avg_nodes_retrieved": avg_nodes_retrieved,
            "total_retrieval_operations": len(self.memory)
        }


class ConversationalRAGAgent(RAGAgent):
    """RAG Agent with conversational capabilities"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.conversation_history = []
        self.max_conversation_length = 20
    
    def chat(self, message: str, graph: Optional[DocumentGraph] = None) -> AgentResponse:
        """Chat with the agent using natural language"""
        
        if graph:
            self.set_graph(graph)
        
        if self.graph is None:
            return AgentResponse(
                task_id="chat",
                answer="I don't have access to any knowledge base. Please provide a document graph first.",
                success=False,
                error_type="NoGraphError",
                error_message="No knowledge graph available"
            )
        
        # Create a simple task from the message
        task = TaskInstance(
            task_id=f"chat_{int(time.time())}",
            template_id="conversational",
            task_type="extraction",  # Default type for chat
            difficulty="medium",
            prompt=message,
            subgraph_nodes=[],
            requires_citations=True,
            requires_reasoning_path=True
        )
        
        # Add conversation context
        if self.conversation_history:
            context = self._build_conversation_context()
            task.prompt = f"{context}\n\nCurrent question: {message}"
        
        # Execute the task
        response = self.execute_task(task)
        
        # Update conversation history
        self.conversation_history.append({
            "user": message,
            "agent": response.answer,
            "timestamp": time.time()
        })
        
        # Keep conversation history manageable
        if len(self.conversation_history) > self.max_conversation_length:
            self.conversation_history.pop(0)
        
        return response
    
    def _build_conversation_context(self) -> str:
        """Build context from conversation history"""
        if not self.conversation_history:
            return ""
        
        context_parts = ["Previous conversation:"]
        
        # Include last few exchanges
        recent_history = self.conversation_history[-3:]  # Last 3 exchanges
        
        for exchange in recent_history:
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['agent']}")
        
        return "\n".join(context_parts)
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
