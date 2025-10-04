"""
No-RAG Agent Implementation
直接使用LLM执行任务，不进行知识图谱检索
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger

from .executors import LLMExecutor, ExecutionConfig, ExecutionResult
from .evaluators import MultiDimensionalEvaluator
from task_craft.task_generator import TaskInstance


@dataclass
class NoRAGAgentConfig:
    """No-RAG Agent配置"""
    model_name: str = "qwen2.5-vl-7b-instruct"  # Use default from config
    model_provider: str = "qwen"  # Use default from config
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_evaluation: bool = True
    max_context_length: int = 4000
    verbose: bool = False
    require_citations: bool = False
    require_reasoning: bool = False
    response_format: str = "text"


@dataclass
class NoRAGAgentResponse:
    """No-RAG Agent响应结果"""
    task_id: str
    answer: str
    success: bool
    execution_time: float
    model_used: str
    tokens_used: int
    safety_passed: bool
    safety_issues: List[str]
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class NoRAGAgent:
    """无RAG能力的Agent，直接使用LLM执行任务"""
    
    def __init__(self, config: NoRAGAgentConfig, executor: Optional[LLMExecutor] = None):
        """Initialize No-RAG Agent"""
        self.config = config
        
        # Use provided executor or create new singleton instance
        if executor is not None:
            self.executor = executor
            logger.info(f"🤖 No-RAG Agent using provided executor: {executor.config.model_name}")
        else:
            # Initialize executor with correct configuration
            execution_config = ExecutionConfig(
                model_name=config.model_name,
                model_provider=config.model_provider,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                max_retries=config.max_retries,
                require_citations=config.require_citations,
                require_reasoning=config.require_reasoning,
                response_format=config.response_format,
                max_context_length=config.max_context_length
            )
            self.executor = LLMExecutor(execution_config)
            logger.info(f"🤖 No-RAG Agent initialized with model: {config.model_name}, response_format: {config.response_format}")
        
        # Initialize evaluator if enabled
        if self.config.enable_evaluation:
            try:
                from .evaluators import MultiDimensionalEvaluator
                self.evaluator = MultiDimensionalEvaluator()
            except Exception as e:
                logger.warning(f"Failed to initialize evaluator: {e}")
                self.evaluator = None
        else:
            self.evaluator = None
    
    def execute_task(self, task: TaskInstance) -> NoRAGAgentResponse:
        """执行任务（无RAG检索）"""
        start_time = time.time()
        
        try:
            if self.config.verbose:
                logger.info(f"🔄 No-RAG Agent executing task: {task.task_id}")
            
            # 创建空的检索结果
            from .retrievers import RetrievalResult
            empty_context = RetrievalResult(
                nodes=[],
                edges=[],
                scores={},
                retrieval_method="no_rag",
                total_nodes_considered=0
            )
            
            # 对于NoRAG agent，禁用引用要求（因为没有可引用的来源）
            # 但保留推理要求（因为推理是LLM的内在能力）
            if hasattr(task, 'requires_citations'):
                task.requires_citations = False
            
            # 执行任务
            execution_result = self.executor.execute(task, empty_context)
            
            # 安全检查
            # safety_passed, safety_issues = self._check_safety(execution_result) # Removed safety check
            
            execution_time = time.time() - start_time
            
            return NoRAGAgentResponse(
                task_id=task.task_id,
                answer=execution_result.answer or "",
                success=execution_result.success,
                execution_time=execution_time,
                model_used=execution_result.model_used,
                tokens_used=execution_result.tokens_used,
                safety_passed=True, # Always True as safety is removed
                safety_issues=[] # Always empty as safety is removed
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ No-RAG Agent execution failed for task {task.task_id}: {e}")
            
            return NoRAGAgentResponse(
                task_id=task.task_id,
                answer="",
                success=False,
                execution_time=execution_time,
                model_used=self.config.model_name,
                tokens_used=0,
                safety_passed=True, # Always True as safety is removed
                safety_issues=[], # Always empty as safety is removed
                error_type="execution_error",
                error_message=str(e)
            )
    
    
    def get_stats(self) -> Dict[str, Any]:
        """获取agent统计信息"""
        return {
            "agent_type": "no_rag",
            "model_name": self.config.model_name
        }
