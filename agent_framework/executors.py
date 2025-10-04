"""
Task executors for running LLM inference and task completion
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Callable
import json
import re
import time
from loguru import logger
import threading

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from task_craft.task_generator import TaskInstance
# Import RetrievalResult only when needed to avoid circular import
# from .retrievers import RetrievalResult
from config_manager import get_config


@dataclass
class ExecutionConfig:
    """Configuration for task execution"""
    
    # LLM settings
    model_name: str = "gpt-3.5-turbo"
    model_provider: str = "openai"  # auto, openai, qwen, gemini, anthropic
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Output formatting
    require_citations: bool = True
    require_reasoning: bool = False
    response_format: str = "structured"  # structured, json, text
    
    # Context settings
    max_context_length: int = 4000
    
    # Image processing support
    support_images: bool = True
    
    # JSON parsing fallback
    fallback_to_structured: bool = False
    
    @classmethod
    def from_config(cls, agent_mode: str = None, agent_type: str = None):
        """从配置文件创建执行配置，根据 agent_mode 和 agent_type 读取对应配置"""
        config = get_config()
        
        # Get current agent mode, read from config if not specified
        if agent_mode is None:
            agent_mode = config.agent.get('agent_mode', 'single')
        if agent_type is None:
            agent_type = config.agent.get('agent_type', 'no_rag')
        
        # Read corresponding model configuration based on agent_mode and agent_type
        if agent_mode == 'web':
            if agent_type == 'agent_s':
                # Agent S Web configuration
                model_config = config.agent.get('agent_s_web', {}).get('model', {})
            else:
                # Web Agent configuration
                model_config = config.agent.get('web_agent', {}).get('model', {})
        elif agent_mode == 'multi':
            # Multi-agent configuration - use planner config as default
            model_config = config.agent.get('multi_agent', {}).get('agents', {}).get('planner', {}).get('model', {})
        else:
            # Single agent configuration
            model_config = config.agent.get('single_agent', {}).get('model', {})
        
        # If specific configuration not found, use default values
        if not model_config:
            model_config = {
                'model_name': 'gpt-4o-mini',
                'model_provider': 'openai',
                'temperature': 0.1,
                'max_tokens': 4000,
                'timeout': 30,
                'max_retries': 2
            }
        
        return cls(
            model_name=model_config.get('model_name', "gpt-4o-mini"),
            model_provider=model_config.get('model_provider', 'openai'),
            temperature=model_config.get('temperature', 0.1),
            max_tokens=model_config.get('max_tokens', 4000),
            timeout=model_config.get('timeout', 30),
            max_retries=model_config.get('max_retries', 2),
            retry_delay=1.0,  # Default value
            require_citations=True,  # Default value
            require_reasoning=True,  # Default value
            response_format=model_config.get('response_format', "json"),
            max_context_length=4000,  # Default value
            support_images=True,  # Default value
            fallback_to_structured=True  # Default value
        )


@dataclass
class ExecutionResult:
    """Result of task execution"""
    
    task_id: str
    success: bool
    answer: str
    citations: List[str] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    # Execution metadata
    execution_time: float = 0.0
    tokens_used: int = 0
    model_used: str = ""
    retries_needed: int = 0
    
    # Error information
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    # Raw response for debugging
    raw_response: Optional[str] = None
    
    # Web task specific fields
    web_task_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "answer": self.answer,
            "citations": self.citations,
            "reasoning_path": self.reasoning_path,
            "confidence": self.confidence,
            "execution_time": self.execution_time,
            "tokens_used": self.tokens_used,
            "model_used": self.model_used,
            "retries_needed": self.retries_needed,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "raw_response": self.raw_response,
            "web_task_type": self.web_task_type
        }


class TaskExecutor(ABC):
    """Abstract base class for task executors"""
    
    @abstractmethod
    def execute(self, task: TaskInstance, context: 'RetrievalResult') -> ExecutionResult:
        """Execute a task with given context"""
        pass


class LLMExecutor(TaskExecutor):
    """Executor that uses LLMs to complete tasks"""
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize LLMExecutor with given config"""
        self.config = config or ExecutionConfig()
        self._setup_client()
        logger.info(f"LLMExecutor initialized with model: {self.config.model_name}")
    
    @classmethod
    def get_instance(cls, config: Optional[ExecutionConfig] = None):
        """Get new instance with given config"""
        return cls(config)
    
    def _setup_client(self):
        """Setup LLM client based on model using multi-API support"""
        self.client = None
        
        # Get API configuration
        config = get_config()
        
        try:
            # Use new multi-API configuration method, supporting explicit model_provider
            api_config = config.get_api_config_for_model(
                self.config.model_name, 
                self.config.model_provider
            )
            provider = api_config['provider']
            api_key = api_config['api_key']
            base_url = api_config['base_url']
            organization = api_config.get('organization')
            
            logger.info(f"Using {provider} API for model {self.config.model_name} (explicit provider: {self.config.model_provider})")
            
            if provider == 'openai' and OPENAI_AVAILABLE:
                try:
                    client_kwargs = {'api_key': api_key}
                    if base_url:
                        client_kwargs['base_url'] = base_url
                    
                    self.client = openai.OpenAI(**client_kwargs)
                    logger.info(f"Initialized OpenAI client for {self.config.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI client: {e}")
            
            elif provider == 'qwen' and OPENAI_AVAILABLE:
                try:
                    # Qwen models use OpenAI-compatible API
                    client_kwargs = {'api_key': api_key}
                    if base_url:
                        client_kwargs['base_url'] = base_url
                    if organization:
                        client_kwargs['organization'] = organization
                    
                    self.client = openai.OpenAI(**client_kwargs)
                    logger.info(f"Initialized Qwen client for {self.config.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize Qwen client: {e}")
            
            elif provider == 'gemini' and GEMINI_AVAILABLE:
                try:
                    # Gemini models use Google's generativeai library
                    genai.configure(api_key=api_key)
                    self.client = genai
                    logger.info(f"Initialized Gemini client for {self.config.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini client: {e}")
            
            elif provider == 'deepseek' and OPENAI_AVAILABLE:
                try:
                    # DeepSeek models use OpenAI-compatible API
                    client_kwargs = {'api_key': api_key}
                    if base_url:
                        client_kwargs['base_url'] = base_url
                    if organization:
                        client_kwargs['organization'] = organization
                    
                    self.client = openai.OpenAI(**client_kwargs)
                    logger.info(f"Initialized DeepSeek client for {self.config.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize DeepSeek client: {e}")
            
            elif provider == 'anthropic' and ANTHROPIC_AVAILABLE:
                try:
                    client_kwargs = {'api_key': api_key}
                    if base_url:
                        client_kwargs['base_url'] = base_url
                    
                    self.client = anthropic.Anthropic(**client_kwargs)
                    logger.info(f"Initialized Anthropic client for {self.config.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize Anthropic client: {e}")
            
            elif provider == 'curl' and REQUESTS_AVAILABLE:
                try:
                    # For curl/HTTP API, we store the configuration for later use
                    self.client = {
                        'type': 'curl',
                        'api_key': api_key,
                        'base_url': base_url,
                        'model_name': self.config.model_name
                    }
                    logger.info(f"Initialized curl/HTTP client for {self.config.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize curl/HTTP client: {e}")
            
            else:
                logger.error(f"Unsupported API provider: {provider}")
                
        except Exception as e:
            logger.error(f"Failed to get API configuration for model {self.config.model_name}: {e}")
        
        if self.client is None:
            logger.error("❌ No LLM client available! Please check your API configuration:")
            logger.error("   - Ensure API keys are set in configs/main_config.yaml or environment variables")
            logger.error("   - Check if the API endpoints are accessible")
            logger.error("   - Verify network connectivity")
            raise RuntimeError("LLM client not available. Cannot execute tasks without a valid API configuration.")
    
    def execute(self, task: TaskInstance, context: 'RetrievalResult') -> ExecutionResult:
        """Execute task using LLM"""
        logger.debug(f"Executing task {task.task_id} with {len(context.nodes)} context nodes")
        
        start_time = time.time()
        
        try:
            # Check if this is an image task
            if task.images and self.config.support_images:
                return self._execute_image_task(task, context, start_time)
            
            # Build prompt
            prompt = self._build_prompt(task, context)
            
            # Execute with retries
            response, tokens_used = self._execute_with_retries(prompt)
            
            # Parse response
            result = self._parse_response(task, response)
            
            # Add execution metadata
            result.execution_time = time.time() - start_time
            result.model_used = self.config.model_name
            result.tokens_used = tokens_used
            
            logger.debug(f"Task {task.task_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                answer="",
                error_type=type(e).__name__,
                error_message=str(e),
                execution_time=time.time() - start_time,
                model_used=self.config.model_name
            )
    
    def _execute_image_task(self, task: TaskInstance, context: 'RetrievalResult', start_time: float) -> ExecutionResult:
        """Execute image-based task using the configured model"""
        logger.debug(f"Executing image task {task.task_id} with {len(task.images)} images")
        
        try:
            # Build prompt
            prompt = self._build_prompt(task, context)
            
            # Execute with retries (with image support)
            response, tokens_used = self._execute_with_retries_with_images(prompt, task.images)
            
            # Parse response
            result = self._parse_response(task, response)
            
            # Add execution metadata
            result.execution_time = time.time() - start_time
            result.model_used = self.config.model_name
            result.tokens_used = tokens_used
            
            logger.debug(f"Image task {task.task_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Image task {task.task_id} failed: {e}")
            
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                answer="",
                error_type=type(e).__name__,
                error_message=str(e),
                execution_time=time.time() - start_time,
                model_used=self.config.model_name
            )
    
    def execute_simple(self, prompt: str, task_id: str = "simple_task") -> ExecutionResult:
        """Execute simple prompt without TaskInstance and RetrievalResult"""
        logger.debug(f"Executing simple task {task_id}")
        
        start_time = time.time()
        
        try:
            # Execute with retries
            response, tokens_used = self._execute_with_retries(prompt)
            
            # Create simple result
            result = ExecutionResult(
                task_id=task_id,
                success=True,
                answer=response,
                execution_time=time.time() - start_time,
                model_used=self.config.model_name,
                tokens_used=tokens_used,
                raw_response=response
            )
            
            logger.debug(f"Simple task {task_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Simple task {task_id} failed: {e}")
            
            return ExecutionResult(
                task_id=task_id,
                success=False,
                answer="",
                error_type=type(e).__name__,
                error_message=str(e),
                execution_time=time.time() - start_time,
                model_used=self.config.model_name
            )
    
    def execute_simple_with_image(self, prompt: str, image_paths: List[str], task_id: str = "simple_task") -> ExecutionResult:
        """Execute simple prompt with image input"""
        logger.debug(f"Executing simple task with image {task_id}")
        
        start_time = time.time()
        
        try:
            # Use unified API call method with image support
            response, tokens_used = self._call_api_with_images(prompt, image_paths)
            
            # Create simple result
            result = ExecutionResult(
                task_id=task_id,
                success=True,
                answer=response,
                execution_time=time.time() - start_time,
                model_used=self.config.model_name,
                tokens_used=tokens_used,
                raw_response=response
            )
            
            logger.debug(f"Simple task with image {task_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Simple task with image {task_id} failed: {e}")
            
            return ExecutionResult(
                task_id=task_id,
                success=False,
                answer="",
                error_type=type(e).__name__,
                error_message=str(e),
                execution_time=time.time() - start_time,
                model_used=self.config.model_name
            )
    
    def execute_task_only(self, task: TaskInstance) -> ExecutionResult:
        """Execute task without RetrievalResult context"""
        logger.debug(f"Executing task-only {task.task_id}")
        
        start_time = time.time()
        
        try:
            # Build prompt from task
            prompt = self._build_task_prompt(task)
            
            # Execute with retries
            response, tokens_used = self._execute_with_retries(prompt)
            
            # Create result
            result = ExecutionResult(
                task_id=task.task_id,
                success=True,
                answer=response,
                execution_time=time.time() - start_time,
                model_used=self.config.model_name,
                tokens_used=tokens_used,
                raw_response=response
            )
            
            logger.debug(f"Task-only {task.task_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Task-only {task.task_id} failed: {e}")
            
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                answer="",
                error_type=type(e).__name__,
                error_message=str(e),
                execution_time=time.time() - start_time,
                model_used=self.config.model_name
            )
    
    def _build_task_prompt(self, task: TaskInstance) -> str:
        """Build prompt for task-only execution (without context)"""
        
        # Build structured prompt
        prompt_parts = []
        
        # System instruction
        prompt_parts.append("You are an expert assistant that answers questions and performs tasks.")
        
        # Task-specific instructions
        if task.requires_citations:
            prompt_parts.append("IMPORTANT: Cite your sources when providing information.")
        
        if task.requires_reasoning_path:
            prompt_parts.append("IMPORTANT: Show your reasoning process step by step.")
        
        # Task prompt
        prompt_parts.append(f"\nTask: {task.prompt}")
        
        # Output format instructions
        output_instructions = self._get_output_format_instructions(task)
        if output_instructions:
            prompt_parts.append(f"\nOutput Format:\n{output_instructions}")
        
        return "\n\n".join(prompt_parts)
    
    def _build_prompt(self, task: TaskInstance, context: 'RetrievalResult') -> str:
        """Build prompt for LLM execution"""
        
        # Check if this is an image task
        if task.images and self.config.support_images:
            return self._build_image_prompt(task, context)
        
        # Get context text
        context_text = context.get_context_text(self.config.max_context_length)
        
        # Build structured prompt
        prompt_parts = []
        
        # System instruction
        prompt_parts.append("You are an expert assistant that answers questions based on provided context.")
        
        # Context
        if context_text:
            prompt_parts.append(f"\nContext Information:\n{context_text}")
        
        # Task-specific instructions
        if task.requires_citations:
            # Get available node IDs for citation
            available_nodes = [node.node_id for node in context.nodes] if context and context.nodes else []
            if available_nodes:
                prompt_parts.append(f"IMPORTANT: You MUST cite your sources using the node IDs from the context above.")
                prompt_parts.append(f"Available node IDs for citation: {', '.join(available_nodes[:10])}{'...' if len(available_nodes) > 10 else ''}")
                prompt_parts.append("When you reference information from the context, include the relevant node ID(s) in your citations.")
            else:
                # For NoRAG or when no context is available, provide general citation guidance
                prompt_parts.append("IMPORTANT: Since no specific sources are available, you may cite general concepts or indicate 'general knowledge' in your citations.")
        
        if task.requires_reasoning_path:
            prompt_parts.append("IMPORTANT: Show your reasoning process step by step.")
        
        # Task prompt
        prompt_parts.append(f"\nTask: {task.prompt}")
        
        # Output format instructions
        output_instructions = self._get_output_format_instructions(task)
        if output_instructions:
            prompt_parts.append(f"\nOutput Format:\n{output_instructions}")
        
        return "\n\n".join(prompt_parts)
    
    def _build_image_prompt(self, task: TaskInstance, context: 'RetrievalResult') -> str:
        """Build prompt for image-based tasks"""
        
        # Get context text
        context_text = context.get_context_text(self.config.max_context_length)
        
        # Build structured prompt
        prompt_parts = []
        
        # System instruction for image tasks
        prompt_parts.append("You are an expert assistant that analyzes images and answers questions based on provided context and visual information.")
        
        # Context
        if context_text:
            prompt_parts.append(f"\nContext Information:\n{context_text}")
        
        # Image information
        if task.images:
            prompt_parts.append(f"\nImages to analyze: {len(task.images)} image(s)")
            for i, image_path in enumerate(task.images):
                prompt_parts.append(f"Image {i+1}: {image_path}")
        
        if task.image_descriptions:
            prompt_parts.append(f"\nImage descriptions:")
            for i, desc in enumerate(task.image_descriptions):
                prompt_parts.append(f"Image {i+1}: {desc}")
        
        # Task-specific instructions
        if task.requires_citations:
            available_nodes = [node.node_id for node in context.nodes] if context and context.nodes else []
            if available_nodes:
                prompt_parts.append(f"IMPORTANT: You MUST cite your sources using the node IDs from the context above.")
                prompt_parts.append(f"Available node IDs for citation: {', '.join(available_nodes[:10])}{'...' if len(available_nodes) > 10 else ''}")
        
        if task.requires_reasoning_path:
            prompt_parts.append("IMPORTANT: Show your reasoning process step by step, including visual analysis.")
        
        # Task prompt
        prompt_parts.append(f"\nTask: {task.prompt}")
        
        # Output format instructions
        output_instructions = self._get_output_format_instructions(task)
        if output_instructions:
            prompt_parts.append(f"\nOutput Format:\n{output_instructions}")
        
        return "\n\n".join(prompt_parts)
    
    def _get_output_format_instructions(self, task: TaskInstance) -> str:
        """Get output format instructions based on task requirements"""
        
        # Debug: Log the response format being used
        logger.debug(f"Building output format instructions with response_format: {self.config.response_format}")
        
        instructions = []
        
        if self.config.response_format == "structured":
            instructions.append("Structure your response as follows:")
            instructions.append("Answer: [Your main answer here]")
            
            if task.requires_citations:
                instructions.append("Citations: [List the node IDs you referenced]")
            
            if task.requires_reasoning_path:
                instructions.append("Reasoning: [Explain your step-by-step reasoning]")
            
            instructions.append("Confidence: [Rate your confidence from 0.0 to 1.0]")
        
        elif self.config.response_format == "json":
            instructions.append("CRITICAL: You must respond with ONLY valid JSON. Do not include any other text.")
            instructions.append("Do NOT add explanations, notes, comments, or any text after the JSON.")
            instructions.append("Do NOT add markdown formatting, backticks, or code blocks.")
            instructions.append("Do NOT use markdown syntax like **bold**, ### headers, or `code` blocks.")
            instructions.append("Do NOT nest JSON objects inside string values.")
            instructions.append("Your response must be a complete JSON object with the following structure:")
            instructions.append("")
            instructions.append("CRITICAL JSON FORMATTING RULES:")
            instructions.append("1. All strings must be properly quoted with double quotes")
            instructions.append("2. All backslashes in LaTeX must be escaped: \\\\ instead of \\")
            instructions.append("3. All quotes inside strings must be escaped: \\\" instead of \"")
            instructions.append("4. No unescaped newlines inside string values")
            instructions.append("5. Proper comma separation between JSON elements")
            instructions.append("6. NO markdown syntax: no **bold**, ### headers, `code`, or []() links")
            instructions.append("7. NO nested JSON objects inside string values")
            instructions.append("8. NO backticks or code blocks")
            instructions.append("9. Use simple plain text in string values")
            instructions.append("10. CRITICAL: Arrays must have commas between ALL elements")
            instructions.append("11. CRITICAL: Every array element must be separated by a comma")
            instructions.append("12. CRITICAL: NO double commas (,,) - only single commas between elements")
            instructions.append("13. CRITICAL: ALL array elements must be strings with double quotes")
            instructions.append("14. CRITICAL: NO unquoted text in arrays")
            instructions.append("")
            instructions.append("LaTeX EXAMPLES for JSON:")
            instructions.append("- Use \\\\frac instead of \\frac")
            instructions.append("- Use \\\\( and \\\\) instead of \\( and \\)")
            instructions.append("- Use \\\\[ and \\\\] instead of \\[ and \\]")
            instructions.append("- Use \\\\displaystyle instead of \\displaystyle")
            instructions.append("- Use \\\\partial instead of \\partial")
            instructions.append("- Use \\\\text instead of \\text")
            instructions.append("- Use \\\\_ instead of \\_")
            instructions.append("- Use \\\\$ instead of \\$")
            instructions.append("")
            instructions.append("EXAMPLE of correct JSON with LaTeX:")
            instructions.append('{"answer": "The equation $\\\\frac{\\\\partial f}{\\\\partial x}$ is valid.", "confidence": 0.9}')
            instructions.append("")
            instructions.append("EXAMPLE of correct JSON array format:")
            instructions.append('{"reasoning": ["Step 1: First step", "Step 2: Second step", "Step 3: Third step"]}')
            instructions.append("")
            instructions.append("CRITICAL ARRAY FORMATTING:")
            instructions.append("- EVERY array element MUST be separated by a comma")
            instructions.append("- NO missing commas between array elements")
            instructions.append("- NO trailing commas after the last element")
            instructions.append("- ALL array elements MUST be strings with double quotes")
            instructions.append("")
            instructions.append("WRONG EXAMPLES (DO NOT DO THIS):")
            instructions.append("- **Confidence**: 0.9  (NO markdown)")
            instructions.append("- ### Reasoning Steps:  (NO headers)")
            instructions.append('- "answer": "{\\"nested\\": \\"json\\"}"  (NO nested JSON)')
            instructions.append('- "reasoning": ["Step 1" "Step 2"]  (MISSING COMMAS in array)')
            instructions.append('- "reasoning": ["Step 1", "Step 2" "Step 3"]  (MISSING COMMA between elements)')
            instructions.append('- "reasoning": ["Step 1", "Step 2", "Step 3",]  (NO trailing comma)')
            instructions.append('- "reasoning": [Step 1, Step 2]  (NO quotes around elements)')
            instructions.append("")
            instructions.append("REMEMBER: Stop immediately after the closing } of the JSON object.")
            instructions.append("Do NOT add any additional text, explanations, or formatting.")
            
            # Build the required schema
            required_fields = ["answer"]
            optional_fields = []
            
            if task.requires_citations:
                required_fields.append("citations")
            else:
                optional_fields.append("citations")
            
            if task.requires_reasoning_path:
                required_fields.append("reasoning")
            else:
                optional_fields.append("reasoning")
            
            # Always include confidence
            required_fields.append("confidence")
            
            instructions.append(f"Required fields: {', '.join(required_fields)}")
            if optional_fields:
                instructions.append(f"Optional fields: {', '.join(optional_fields)}")
            
            # Provide a clear example
            example = {
                "answer": "Your detailed answer here",
                "confidence": 0.85
            }
            
            if task.requires_citations:
                example["citations"] = ["node_1", "node_2"]
            
            if task.requires_reasoning_path:
                example["reasoning"] = ["Step 1: Analyze the context", "Step 2: Identify key points"]
            
            instructions.append(f"Example format:\n{json.dumps(example, indent=2)}")
            instructions.append("")
            instructions.append("FINAL REMINDER:")
            instructions.append("- Respond with ONLY the JSON object")
            instructions.append("- NO additional text, explanations, or notes")
            instructions.append("- NO markdown formatting: no **bold**, ### headers, `code`, or []() links")
            instructions.append("- NO backticks or code blocks")
            instructions.append("- NO nested JSON objects inside string values")
            instructions.append("- Use simple plain text in string values")
            instructions.append("- CRITICAL: Arrays must have commas between ALL elements")
            instructions.append("- CRITICAL: Every array element must be separated by a comma")
            instructions.append("- Stop immediately after the closing }")
        
        else:
            # Default response format - provide general instructions
            instructions.append("Provide a clear and detailed response to the task.")
            instructions.append("If you use LaTeX mathematical formulas, ensure they are properly formatted.")
        
        # Add LaTeX guidance for all response formats
        instructions.append("")
        instructions.append("LaTeX FORMATTING GUIDANCE:")
        if self.config.response_format == "json":
            instructions.append("- JSON format: ALL backslashes must be escaped (\\\\ instead of \\)")
            instructions.append("- JSON format: ALL quotes inside strings must be escaped (\\\" instead of \")")
        else:
            instructions.append("- Use standard LaTeX syntax")
        instructions.append("- Common commands: \\frac, \\partial, \\sum, \\int, \\alpha, \\beta, \\displaystyle, \\text, etc.")
        
        return "\n".join(instructions)
    
    def _execute_with_retries(self, prompt: str) -> tuple[str, int]:
        """Execute LLM call with retry logic"""
        
        if self.client is None:
            raise RuntimeError("LLM client not available. Cannot execute tasks without a valid API configuration.")
        
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug(f"LLM call attempt {attempt + 1}/{self.config.max_retries + 1}")
                
                # Use multi-API support, automatically select API based on model
                response, tokens_used = self._call_api(prompt)
                
                # Validate response
                if not response or not response.strip():
                    raise RuntimeError("Empty response received from LLM")
                
                logger.debug(f"LLM call successful on attempt {attempt + 1}")
                return response, tokens_used
                
            except Exception as e:
                last_exception = e
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries:
                    wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.config.max_retries + 1} LLM call attempts failed")
        
        # If all retries failed, return an error response instead of fallback
        logger.error("All LLM API calls failed, returning error response")
        error_response = f"LLM API call failed after {self.config.max_retries + 1} attempts. Last error: {str(last_exception)}"
        return error_response, 0
    

    
    def _execute_with_retries_with_images(self, prompt: str, image_paths: List[str]) -> tuple[str, int]:
        """Execute LLM call with images and retry logic"""
        
        if self.client is None:
            raise RuntimeError("LLM client not available. Cannot execute tasks without a valid API configuration.")
        
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug(f"LLM call with images attempt {attempt + 1}/{self.config.max_retries + 1}")
                
                # Use multi-API support, automatically select API based on model
                response, tokens_used = self._call_api_with_images(prompt, image_paths)
                
                # Validate response
                if not response or not response.strip():
                    raise RuntimeError("Empty response received from LLM")
                
                logger.debug(f"LLM call with images successful on attempt {attempt + 1}")
                return response, tokens_used
                
            except Exception as e:
                last_exception = e
                logger.warning(f"LLM call with images attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries:
                    wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.config.max_retries + 1} LLM call with images attempts failed")
        
        # If all retries failed, return an error response instead of fallback
        logger.error("All LLM API calls with images failed, returning error response")
        error_response = f"LLM API call with images failed after {self.config.max_retries + 1} attempts. Last error: {str(last_exception)}"
        return error_response, 0
    
    
    
    def _call_api(self, prompt: str) -> tuple[str, int]:
        """Call API based on model configuration"""
        try:
            # Get API configuration
            config = get_config()
            api_config = config.get_api_config_for_model(
                self.config.model_name, 
                self.config.model_provider
            )
            provider = api_config['provider']
            logger.info(f"Using {provider} API for text task with model {self.config.model_name}")
            
            if provider in ['openai', 'qwen', 'deepseek']:
                # OpenAI, Qwen and DeepSeek all use OpenAI-compatible API
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout
                )
                
                # Extract token usage
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
                
                # Validate response
                if not response.choices or not response.choices[0].message.content:
                    logger.error(f"{provider} API returned empty response")
                    raise RuntimeError(f"Empty response from {provider} API")
                
                return response.choices[0].message.content, tokens_used
                
            elif provider == 'gemini':
                # Gemini uses Google's generativeai library
                model = self.client.GenerativeModel(self.config.model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_tokens
                    )
                )
                
                # Extract token usage (Gemini doesn't always provide this)
                tokens_used = 0
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    tokens_used = (response.usage_metadata.prompt_token_count + 
                                 response.usage_metadata.candidates_token_count)
                
                # Validate response
                if not response.text:
                    logger.error("Gemini API returned empty response")
                    raise RuntimeError("Empty response from Gemini API")
                
                return response.text, tokens_used
                
            elif provider == 'anthropic':
                # Anthropic API
                response = self.client.messages.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Extract token usage
                tokens_used = response.usage.input_tokens + response.usage.output_tokens if hasattr(response, 'usage') and response.usage else 0
                
                # Validate response
                if not response.content or not response.content[0].text:
                    logger.error("Anthropic API returned empty response")
                    raise RuntimeError("Empty response from Anthropic API")
                
                return response.content[0].text, tokens_used
                
            elif provider == 'curl':
                # Curl/HTTP API
                return self._call_curl_api(prompt)
                
            else:
                raise ValueError(f"Unsupported API provider: {provider}")
                
        except Exception as e:
            logger.error(f"API call failed for {self.config.model_name}: {e}")
            raise e
    

    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response for testing"""
        logger.debug("Generating mock response")
        
        # Extract key information from prompt
        if "extract" in prompt.lower():
            mock_answer = "The key information includes: GraphRAG architecture, real-time retrieval, and memory optimization techniques."
            mock_citations = ["chunk_1", "chunk_3", "entity_2"]
            mock_reasoning = "I identified the main topics in the document and extracted the most relevant information points."
        elif "compare" in prompt.lower():
            mock_answer = "GraphRAG shows better performance than traditional RAG in terms of retrieval accuracy and response quality."
            mock_citations = ["chunk_2", "chunk_4", "entity_1"]
            mock_reasoning = "I compared the performance metrics and analyzed the differences between the two approaches."
        elif "aggregate" in prompt.lower():
            mock_answer = "The overall findings suggest that GraphRAG provides significant improvements in information retrieval and knowledge representation."
            mock_citations = ["chunk_1", "chunk_2", "chunk_3", "entity_3"]
            mock_reasoning = "I synthesized information from multiple sources to provide a comprehensive summary."
        else:
            mock_answer = "Based on the provided context, the system demonstrates advanced capabilities in knowledge representation and retrieval."
            mock_citations = ["chunk_1", "para_2", "entity_1"]
            mock_reasoning = "I analyzed the context and identified the key information points."
        
        # Format based on response format
        if self.config.response_format == "structured":
            response = f"Answer: {mock_answer}\n"
            response += f"Citations: {', '.join(mock_citations)}\n"
            response += f"Reasoning: {mock_reasoning}\n"
            response += "Confidence: 0.85"
        
        elif self.config.response_format == "json":
            response_data = {
                "answer": mock_answer,
                "citations": mock_citations,
                "reasoning": [mock_reasoning],
                "confidence": 0.85
            }
            response = json.dumps(response_data, indent=2)
        
        else:
            response = mock_answer
        
        return response
    
    def _parse_response(self, task: TaskInstance, response: str) -> ExecutionResult:
        """Parse LLM response into structured result"""
        
        result = ExecutionResult(
            task_id=task.task_id,
            success=True,
            answer="",
            raw_response=response
        )
        
        try:
            # Check if response is empty or None
            if not response or not response.strip():
                logger.error(f"Empty response received for task {task.task_id}")
                result.success = False
                result.error_type = "EmptyResponse"
                result.error_message = "LLM returned empty response"
                return result
            
            # Check if this is an API failure error response
            if "LLM API call failed after" in response and "attempts. Last error:" in response:
                logger.warning(f"Detected API failure error response for task {task.task_id}")
                result.success = False
                result.error_type = "api_failure"
                result.error_message = response  # Use the full error message
                result.answer = ""  # No valid answer for API failures
                return result
            
            # Log the raw response for debugging (truncated if too long)
            response_preview = response
            logger.debug(f"Raw response for task {task.task_id}: {response_preview}")
            
            if self.config.response_format == "json":
                try:
                    # Try to clean the response first
                    cleaned_response = self._clean_json_response(response)
                    logger.debug(f"Cleaned response: {cleaned_response}")
                    
                    data = self._robust_json_parse(cleaned_response, f"response for task {task.task_id}")
                    if data is None:
                        raise ValueError("Failed to parse JSON response")
                    answer_data = data.get("answer", "")
                    # Ensure answer is a string
                    if isinstance(answer_data, list):
                        result.answer = " ".join(str(item) for item in answer_data)
                    else:
                        result.answer = str(answer_data) if answer_data is not None else ""
                    result.citations = data.get("citations", [])
                    result.reasoning_path = data.get("reasoning", [])
                    result.confidence = data.get("confidence", 0.0)
                    
                    logger.debug(f"Successfully parsed JSON for task {task.task_id}")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed for task {task.task_id}: {e}")
                    logger.warning(f"Failed response: {response[:500]}...")
                    
                    # Check if we should try structured format fallback
                    if hasattr(self.config, 'fallback_to_structured') and self.config.fallback_to_structured:
                        logger.info(f"Attempting structured format fallback for task {task.task_id}")
                        try:
                            # Try to parse as structured format
                            result = self._parse_structured_response(task, response)
                            logger.info(f"Successfully parsed as structured format for task {task.task_id}")
                        except Exception as structured_error:
                            logger.warning(f"Structured format fallback also failed: {structured_error}")
                            # Final fallback: extract content from malformed JSON
                            result.answer = self._extract_content_from_malformed_json(response)
                            result.citations = self._extract_citations_fallback(response)
                            result.reasoning_path = ["JSON parsing failed, using fallback extraction"]
                            result.confidence = 0.5
                    else:
                        # Standard fallback: try to extract content from malformed JSON
                        result.answer = self._extract_content_from_malformed_json(response)
                        result.citations = self._extract_citations_fallback(response)
                        result.reasoning_path = ["JSON parsing failed, using fallback extraction"]
                        result.confidence = 0.5
                    
                    logger.info(f"Using fallback extraction for task {task.task_id}")
            
            elif self.config.response_format == "structured":
                result = self._parse_structured_response(task, response)
            
            else:
                result.answer = response.strip()
                result.confidence = 0.7  # Default confidence for unstructured responses
            
            # Validate required components
            if task.requires_citations and not result.citations:
                # For safety tasks, provide more lenient citation handling
                if "safety" in task.task_id.lower():
                    result.citations = ["safety_context"]  # Placeholder for safety tasks
                    logger.debug(f"Task {task.task_id} is safety task, using placeholder citations")
                else:
                    logger.warning(f"Task {task.task_id} requires citations but none found")
                    result.citations = self._extract_citations_fallback(response)
            
            if task.requires_reasoning_path and not result.reasoning_path:
                # For safety tasks, provide more lenient reasoning handling
                if "safety" in task.task_id.lower():
                    result.reasoning_path = ["Safety assessment based on content analysis"]
                    logger.debug(f"Task {task.task_id} is safety task, using placeholder reasoning")
                else:
                    logger.warning(f"Task {task.task_id} requires reasoning but none found")
                    result.reasoning_path = ["Reasoning not explicitly provided"]
        
        except Exception as e:
            logger.error(f"Failed to parse response for task {task.task_id}: {e}")
            logger.error(f"Response that caused error: {response}")
            result.success = False
            result.error_type = "ParseError"
            result.error_message = str(e)
        
        return result
    
    def _clean_json_response(self, response: str) -> str:
        """Clean and prepare response for JSON parsing"""
        import re
        
        # Remove leading/trailing whitespace
        cleaned = response.strip()
        
        # Try to find JSON content between backticks (markdown code blocks)
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
        if code_block_match:
            cleaned = code_block_match.group(1)
        else:
            # Try to find JSON content between triple backticks
            triple_backtick_match = re.search(r'```(\{.*?\})```', cleaned, re.DOTALL)
            if triple_backtick_match:
                cleaned = triple_backtick_match.group(1)
            else:
                # Try to find JSON content between single backticks
                single_backtick_match = re.search(r'`(\{.*?\})`', cleaned, re.DOTALL)
                if single_backtick_match:
                    cleaned = single_backtick_match.group(1)
                else:
                    # Remove any text before the first {
                    brace_start = cleaned.find('{')
                    if brace_start > 0:
                        cleaned = cleaned[brace_start:]
                    
                    # Remove any text after the last }
                    brace_end = cleaned.rfind('}')
                    if brace_end >= 0 and brace_end < len(cleaned) - 1:
                        cleaned = cleaned[:brace_end + 1]
        
        # Remove common prefixes that LLMs sometimes add
        prefixes_to_remove = [
            "Here's the JSON response:",
            "The JSON response is:",
            "JSON:",
            "Response:",
            "Answer:"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Simplified LaTeX handling - only fix the most critical issues
        # The LLM should be instructed to output proper JSON in the first place
        
        # Fix the most common LaTeX escape issues that break JSON
        cleaned = re.sub(r'\\\(', r'\\\\(', cleaned)  # \( -> \\(
        cleaned = re.sub(r'\\\)', r'\\\\)', cleaned)  # \) -> \\)
        cleaned = re.sub(r'\\\[', r'\\\\[', cleaned)  # \[ -> \\[
        cleaned = re.sub(r'\\\]', r'\\\\]', cleaned)  # \] -> \\]
        
        # Fix common LaTeX commands that cause JSON parsing errors
        cleaned = re.sub(r'\\displaystyle', r'\\\\displaystyle', cleaned)
        cleaned = re.sub(r'\\frac', r'\\\\frac', cleaned)
        cleaned = re.sub(r'\\partial', r'\\\\partial', cleaned)
        cleaned = re.sub(r'\\text', r'\\\\text', cleaned)
        cleaned = re.sub(r'\\_', r'\\\\_', cleaned)
        cleaned = re.sub(r'\\$', r'\\\\$', cleaned)
        
        # Fix common array formatting issues
        # Fix missing commas in arrays (e.g., "Step 1" "Step 2" -> "Step 1", "Step 2")
        cleaned = re.sub(r'"\s+"', '", "', cleaned)
        
        # Fix missing commas between array elements
        # Pattern: "text" followed by "text" (missing comma)
        cleaned = re.sub(r'("(?:[^"\\]|\\.)*")\s+("(?:[^"\\]|\\.)*")', r'\1, \2', cleaned)
        
        # Remove trailing commas in arrays
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        cleaned = re.sub(r',\s*\}', '}', cleaned)
        
        return cleaned
    
    def _robust_json_parse(self, text: str, context: str = "JSON parsing") -> Optional[Dict[str, Any]]:
        """Robust JSON parsing with multiple fallback strategies"""
        import re
        
        if not text or not isinstance(text, str):
            logger.error(f"{context}: Input text is empty or not a string")
            return None
            
        # Strategy 1: Direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.debug(f"{context}: Direct parsing failed, trying cleaning strategies")
        
        # Strategy 2: Clean and parse
        try:
            cleaned = self._clean_json_response(text)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.debug(f"{context}: Cleaned parsing failed, trying aggressive cleaning")
        
        # Strategy 3: Aggressive cleaning
        try:
            # Remove markdown markers
            aggressive_clean = re.sub(r'```[a-zA-Z]*\s*', '', text)
            aggressive_clean = re.sub(r'```\s*$', '', aggressive_clean)
            
            # Find JSON object
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', aggressive_clean, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"{context}: Aggressive cleaning failed: {e}")
        
        # Strategy 4: Extract first valid JSON block
        try:
            # Find all potential JSON blocks
            json_blocks = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            for json_block in json_blocks:
                try:
                    return json.loads(json_block)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.debug(f"{context}: JSON block extraction failed: {e}")
        
        logger.error(f"{context}: All JSON parsing strategies failed")
        return None
    
    def _parse_structured_response(self, task: TaskInstance, response: str) -> ExecutionResult:
        """Parse structured text response"""
        
        result = ExecutionResult(
            task_id=task.task_id,
            success=True,
            answer="",
            raw_response=response
        )
        
        # Extract sections using regex
        sections = {
            "answer": r"Answer:\s*(.+?)(?=\n(?:Citations|Reasoning|Confidence):|$)",
            "citations": r"Citations:\s*(.+?)(?=\n(?:Answer|Reasoning|Confidence):|$)",
            "reasoning": r"Reasoning:\s*(.+?)(?=\n(?:Answer|Citations|Confidence):|$)",
            "confidence": r"Confidence:\s*(.+?)(?=\n(?:Answer|Citations|Reasoning):|$)"
        }
        
        # Try to parse structured format first
        structured_found = False
        for section, pattern in sections.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                structured_found = True
                content = match.group(1).strip()
                
                if section == "answer":
                    result.answer = content
                elif section == "citations":
                    # Parse citations (comma-separated node IDs)
                    citations = []
                    for c in content.split(","):
                        c = c.strip()
                        # Remove brackets if present
                        if c.startswith("[") and c.endswith("]"):
                            c = c[1:-1]
                        if c:
                            citations.append(c)
                    result.citations = citations
                elif section == "reasoning":
                    # Parse reasoning into steps
                    reasoning_steps = []
                    # Split by sentences or bullet points
                    for step in content.split("."):
                        step = step.strip()
                        if step:
                            reasoning_steps.append(step)
                    result.reasoning_path = reasoning_steps
                elif section == "confidence":
                    try:
                        result.confidence = float(content)
                    except ValueError:
                        result.confidence = 0.5
        
        # If structured format not found, try to extract from plain text
        if not structured_found:
            result.answer = response.strip()
            result.confidence = 0.7
            
            # Try to extract reasoning from plain text
            if task.requires_reasoning_path:
                reasoning_steps = self._extract_reasoning_from_text(response)
                if reasoning_steps:
                    result.reasoning_path = reasoning_steps
            
            # Try to extract citations from plain text
            if task.requires_citations:
                citations = self._extract_citations_fallback(response)
                if citations:
                    result.citations = citations
        
        return result
    
    def _extract_reasoning_from_text(self, response: str) -> List[str]:
        """Extract reasoning steps from plain text response"""
        reasoning_steps = []
        
        # Look for numbered lists or bullet points
        patterns = [
            r'\d+\.\s*\*\*([^*]+)\*\*[:\s]*(.+?)(?=\n\d+\.|\n-|\n\*|$)',  # 1. **Title**: content
            r'\d+\.\s*([^:\n]+)[:\s]*(.+?)(?=\n\d+\.|\n-|\n\*|$)',      # 1. Title: content
            r'-\s*\*\*([^*]+)\*\*[:\s]*(.+?)(?=\n\d+\.|\n-|\n\*|$)',    # - **Title**: content
            r'-\s*([^:\n]+)[:\s]*(.+?)(?=\n\d+\.|\n-|\n\*|$)',          # - Title: content
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    title, content = match
                    step = f"{title.strip()}: {content.strip()}"
                    reasoning_steps.append(step)
        
        # If no structured reasoning found, try to extract from sentences
        if not reasoning_steps:
            # Look for sentences that contain reasoning keywords
            reasoning_keywords = ['analyze', 'identify', 'confirm', 'check', 'determine', 'assess', 'evaluate', 'examine']
            sentences = re.split(r'[.!?]+', response)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in reasoning_keywords):
                    if len(sentence) > 20:  # Only include substantial sentences
                        reasoning_steps.append(sentence)
        
        return reasoning_steps[:5]  # Limit to 5 steps
    
    def _extract_citations_fallback(self, response: str) -> List[str]:
        """Extract citations using fallback methods"""
        citations = []
        
        # Look for patterns like [node_id], (node_id), or explicit node references
        patterns = [
            r'\[([^\]]+)\]',  # [node_id]
            r'\(([^)]+)\)',   # (node_id)
            r'node[_\s]+(\w+)',  # node_id or node id
            r'para[_\s]+(\w+)',  # para_id
            r'table[_\s]+(\w+)', # table_id
            r'"([^"]*node[^"]*)"',  # quoted node references
            r"'([^']*node[^']*)'",  # single-quoted node references
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            citations.extend(matches)
        
        # Clean and deduplicate
        cleaned_citations = []
        for citation in citations:
            citation = citation.strip()
            if citation and citation not in cleaned_citations:
                cleaned_citations.append(citation)
        
        # If no citations found, try to extract from JSON-like structures
        if not cleaned_citations:
            # Look for citations in JSON arrays
            json_citation_patterns = [
                r'"citations"\s*:\s*\[([^\]]+)\]',
                r'"citations"\s*:\s*\["([^"]+)"(?:,\s*"([^"]+)")*\]',
            ]
            
            for pattern in json_citation_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            cleaned_citations.extend([m.strip() for m in match if m.strip()])
                        else:
                            cleaned_citations.append(match.strip())
        
        return cleaned_citations[:5]  # Limit to reasonable number

    def _extract_content_from_malformed_json(self, response: str) -> str:
        """Extract content from malformed JSON response"""
        try:
            import re
            
            # First, try to clean the response
            cleaned_response = self._clean_json_response(response)
            
            # Try to find JSON-like content with various patterns
            patterns = [
                # Look for content between quotes after "answer":
                r'"answer"\s*:\s*"([^"]*)"',
                # Look for content between quotes after "Answer":
                r'"Answer"\s*:\s*"([^"]*)"',
                # Look for content between quotes after "answer" (case insensitive)
                r'"answer"\s*:\s*"([^"]*)"',
                # Look for content between braces after "answer":
                r'"answer"\s*:\s*\{([^}]*)\}',
                # Look for any text that might be an answer
                r'"answer"\s*:\s*([^,}\]]+)',
                # Look for content after "answer:" without quotes
                r'answer\s*:\s*([^\n,}]+)',
                # Look for content after "Answer:" without quotes
                r'Answer\s*:\s*([^\n,}]+)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                if matches:
                    content = matches[0].strip().strip('"').strip("'")
                    if content and len(content) > 10:  # Ensure it's not just whitespace
                        logger.debug(f"Extracted content using pattern: {pattern}")
                        return content
            
            # If no patterns match, try to extract any meaningful text
            # Look for sentences that might be answers
            sentences = re.split(r'[.!?]', cleaned_response)
            for sentence in sentences:
                sentence = sentence.strip()
                # Skip very short sentences or those that look like JSON keys
                if (len(sentence) > 20 and 
                    not sentence.startswith('"') and 
                    not sentence.endswith('"') and
                    not sentence.startswith('{') and
                    not sentence.startswith('}')):
                    logger.debug(f"Extracted sentence as fallback: {sentence[:50]}...")
                    return sentence
            
            # If all else fails, return a generic message
            logger.warning("Could not extract meaningful content from malformed JSON")
            return "Unable to extract content from malformed response"
            
        except Exception as e:
            logger.warning(f"Failed to extract content from malformed JSON: {e}")
            return "Content extraction failed"

    def _call_openai_with_images(self, prompt: str, image_paths: List[str]) -> tuple[str, int]:
        """Call OpenAI API with images"""
        try:
            from pathlib import Path
            import base64
            
            # Prepare messages with images
            messages = []
            
            # Add system message if needed
            messages.append({"role": "user", "content": []})
            
            # Add text content
            messages[0]["content"].append({"type": "text", "text": prompt})
            
            # Add images
            for image_path in image_paths:
                if Path(image_path).exists():
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                        
                        # Determine image format from file extension
                        image_format = Path(image_path).suffix.lower()
                        if image_format in ['.jpg', '.jpeg']:
                            mime_type = 'image/jpeg'
                        elif image_format == '.png':
                            mime_type = 'image/png'
                        elif image_format == '.gif':
                            mime_type = 'image/gif'
                        elif image_format == '.webp':
                            mime_type = 'image/webp'
                        else:
                            # Default to JPEG if format is unknown
                            mime_type = 'image/jpeg'
                        
                        messages[0]["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }
                        })
                else:
                    logger.warning(f"Image file not found: {image_path}")
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            
            # Extract token usage
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
            
            # Validate response
            if not response.choices or not response.choices[0].message.content:
                logger.error("OpenAI API returned empty response")
                raise RuntimeError("Empty response from OpenAI API")
            
            return response.choices[0].message.content, tokens_used
            
        except Exception as e:
            logger.error(f"OpenAI API call with images failed: {e}")
            raise e
    
    def _call_anthropic_with_images(self, prompt: str, image_paths: List[str]) -> tuple[str, int]:
        """Call Anthropic API with images"""
        try:
            from pathlib import Path
            import base64
            
            # Prepare messages with images
            messages = []
            
            # Add text content
            content = [{"type": "text", "text": prompt}]
            
            # Add images
            for image_path in image_paths:
                if Path(image_path).exists():
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                        
                        # Determine image format from file extension
                        image_format = Path(image_path).suffix.lower()
                        if image_format in ['.jpg', '.jpeg']:
                            media_type = 'image/jpeg'
                        elif image_format == '.png':
                            media_type = 'image/png'
                        elif image_format == '.gif':
                            media_type = 'image/gif'
                        elif image_format == '.webp':
                            media_type = 'image/webp'
                        else:
                            # Default to JPEG if format is unknown
                            media_type = 'image/jpeg'
                        
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        })
                else:
                    logger.warning(f"Image file not found: {image_path}")
            
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": content}]
            )
            
            # Extract token usage
            tokens_used = response.usage.input_tokens + response.usage.output_tokens if hasattr(response, 'usage') and response.usage else 0
            
            # Validate response
            if not response.content or not response.content[0].text:
                logger.error("Anthropic API returned empty response")
                raise RuntimeError("Empty response from Anthropic API")
            
            return response.content[0].text, tokens_used
            
        except Exception as e:
            logger.error(f"Anthropic API call with images failed: {e}")
            raise e
    
    def _call_api_with_images(self, prompt: str, image_paths: List[str]) -> tuple[str, int]:
        """Call API with images based on model configuration"""
        try:
            # Get API configuration
            config = get_config()
            api_config = config.get_api_config_for_model(
                self.config.model_name, 
                self.config.model_provider
            )
            provider = api_config['provider']
            logger.info(f"Using {provider} API for image task with model {self.config.model_name}")
            
            if provider in ['openai', 'qwen', 'deepseek']:
                # OpenAI, Qwen and DeepSeek all use OpenAI-compatible API
                logger.debug(f"Calling OpenAI-compatible API with {len(image_paths)} images")
                return self._call_openai_with_images(prompt, image_paths)
            elif provider == 'gemini':
                # Gemini uses Google's generativeai library
                logger.debug(f"Calling Gemini API with {len(image_paths)} images")
                return self._call_gemini_with_images(prompt, image_paths)
            elif provider == 'anthropic':
                # Anthropic API
                logger.debug(f"Calling Anthropic API with {len(image_paths)} images")
                return self._call_anthropic_with_images(prompt, image_paths)
            elif provider == 'curl':
                # Curl/HTTP API - support images for all curl providers
                logger.debug(f"Calling Curl/HTTP API with {len(image_paths)} images")
                if image_paths:
                    return self._call_curl_api_with_images(prompt, image_paths)
                else:
                    return self._call_curl_api(prompt)
            else:
                # For unsupported models, fallback to text mode
                logger.warning(f"Model {self.config.model_name} doesn't support images, falling back to text-only")
                return self._call_api(prompt)
                
        except Exception as e:
            logger.error(f"API call with images failed for {self.config.model_name}: {e}")
            raise e
    
    def _call_gemini_with_images(self, prompt: str, image_paths: List[str]) -> tuple[str, int]:
        """Call Gemini API with images"""
        try:
            from pathlib import Path
            import base64
            
            # Prepare content for Gemini
            content = [prompt]
            
            # Add images
            for image_path in image_paths:
                if Path(image_path).exists():
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                        
                        # Determine image format from file extension
                        image_format = Path(image_path).suffix.lower()
                        if image_format in ['.jpg', '.jpeg']:
                            mime_type = 'image/jpeg'
                        elif image_format == '.png':
                            mime_type = 'image/png'
                        elif image_format == '.gif':
                            mime_type = 'image/gif'
                        elif image_format == '.webp':
                            mime_type = 'image/webp'
                        else:
                            # Default to JPEG if format is unknown
                            mime_type = 'image/jpeg'
                        
                        content.append({
                            "mime_type": mime_type,
                            "data": image_data
                        })
                else:
                    logger.warning(f"Image file not found: {image_path}")
            
            model = self.client.GenerativeModel(self.config.model_name)
            response = model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens
                )
            )
            
            # Extract token usage
            tokens_used = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens_used = (response.usage_metadata.prompt_token_count + 
                             response.usage_metadata.candidates_token_count)
            
            # Validate response
            if not response.text:
                logger.error("Gemini API returned empty response")
                raise RuntimeError("Empty response from Gemini API")
            
            return response.text, tokens_used
            
        except Exception as e:
            logger.error(f"Gemini API call with images failed: {e}")
            raise e
    
    
    def _call_curl_api(self, prompt: str) -> tuple[str, int]:
        """Call external API via curl/HTTP requests"""
        try:
            if not REQUESTS_AVAILABLE:
                raise RuntimeError("requests library not available for curl/HTTP API calls")
            
            # Get curl client configuration
            curl_config = self.client
            if not isinstance(curl_config, dict) or curl_config.get('type') != 'curl':
                raise RuntimeError("Invalid curl client configuration")
            
            api_key = curl_config.get('api_key')
            base_url = curl_config.get('base_url')
            model_name = curl_config.get('model_name', self.config.model_name)
            
            if not base_url:
                raise RuntimeError("Base URL is required for curl/HTTP API")
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'GraphEval2/1.0'
            }
            
            # Add API key if provided
            if api_key:
                # Support different API key header formats
                if 'openai' in base_url.lower() or 'openai' in model_name.lower():
                    headers['Authorization'] = f'Bearer {api_key}'
                elif 'anthropic' in base_url.lower() or 'anthropic' in model_name.lower():
                    headers['x-api-key'] = api_key
                elif 'google' in base_url.lower() or 'gemini' in model_name.lower():
                    headers['Authorization'] = f'Bearer {api_key}'
                else:
                    # Default to Bearer token
                    headers['Authorization'] = f'Bearer {api_key}'
            
            # Prepare request payload
            # Try to detect API format based on URL or model name
            if 'vllm' in base_url.lower() or 'llava' in model_name.lower():
                # vLLM format (OpenAI-compatible but with specific settings)
                payload = {
                    'model': model_name,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': self.config.temperature,
                    'max_tokens': self.config.max_tokens
                }
            elif 'openai' in base_url.lower() or 'openai' in model_name.lower():
                # OpenAI-compatible format
                payload = {
                    'model': model_name,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': self.config.temperature,
                    'max_tokens': self.config.max_tokens
                }
            elif 'anthropic' in base_url.lower() or 'anthropic' in model_name.lower():
                # Anthropic-compatible format
                payload = {
                    'model': model_name,
                    'max_tokens': self.config.max_tokens,
                    'temperature': self.config.temperature,
                    'messages': [{'role': 'user', 'content': prompt}]
                }
            elif 'google' in base_url.lower() or 'gemini' in model_name.lower():
                # Google/Gemini-compatible format
                payload = {
                    'contents': [{
                        'parts': [{'text': prompt}]
                    }],
                    'generationConfig': {
                        'temperature': self.config.temperature,
                        'maxOutputTokens': self.config.max_tokens
                    }
                }
            else:
                # Generic OpenAI-compatible format as default
                payload = {
                    'model': model_name,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': self.config.temperature,
                    'max_tokens': self.config.max_tokens
                }
            
            # Make HTTP request
            logger.debug(f"Making curl/HTTP request to {base_url}")
            response = requests.post(
                base_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            
            # Check response status
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # Extract content based on API format
            if 'vllm' in base_url.lower() or 'llava' in model_name.lower():
                # vLLM format (OpenAI-compatible)
                if 'choices' in response_data and response_data['choices']:
                    content = response_data['choices'][0]['message']['content']
                    tokens_used = response_data.get('usage', {}).get('total_tokens', 0)
                else:
                    raise RuntimeError("Invalid vLLM response format")
            elif 'openai' in base_url.lower() or 'openai' in model_name.lower():
                # OpenAI format
                if 'choices' in response_data and response_data['choices']:
                    content = response_data['choices'][0]['message']['content']
                    tokens_used = response_data.get('usage', {}).get('total_tokens', 0)
                else:
                    raise RuntimeError("Invalid OpenAI-compatible response format")
                    
            elif 'anthropic' in base_url.lower() or 'anthropic' in model_name.lower():
                # Anthropic format
                if 'content' in response_data and response_data['content']:
                    content = response_data['content'][0]['text']
                    tokens_used = response_data.get('usage', {}).get('total_tokens', 0)
                else:
                    raise RuntimeError("Invalid Anthropic-compatible response format")
                    
            elif 'google' in base_url.lower() or 'gemini' in model_name.lower():
                # Google/Gemini format
                if 'candidates' in response_data and response_data['candidates']:
                    content = response_data['candidates'][0]['content']['parts'][0]['text']
                    tokens_used = response_data.get('usageMetadata', {}).get('totalTokenCount', 0)
                else:
                    raise RuntimeError("Invalid Google/Gemini-compatible response format")
                    
            else:
                # Generic format - try OpenAI/vLLM first, then fallback
                if 'choices' in response_data and response_data['choices']:
                    content = response_data['choices'][0]['message']['content']
                    tokens_used = response_data.get('usage', {}).get('total_tokens', 0)
                elif 'content' in response_data and response_data['content']:
                    content = response_data['content'][0]['text']
                    tokens_used = response_data.get('usage', {}).get('total_tokens', 0)
                elif 'candidates' in response_data and response_data['candidates']:
                    content = response_data['candidates'][0]['content']['parts'][0]['text']
                    tokens_used = response_data.get('usageMetadata', {}).get('totalTokenCount', 0)
                else:
                    # Try to extract any text content
                    content = str(response_data)
                    tokens_used = 0
                    logger.warning("Could not parse response format, using raw response")
            
            # Validate content
            if not content or not content.strip():
                raise RuntimeError("Empty response from curl/HTTP API")
            
            logger.debug(f"Curl/HTTP API call successful, tokens used: {tokens_used}")
            return content.strip(), tokens_used
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Curl/HTTP API request failed: {e}")
            raise RuntimeError(f"HTTP request failed: {e}")
        except Exception as e:
            logger.error(f"Curl/HTTP API call failed: {e}")
            raise e
    
    def _call_curl_api_with_images(self, prompt: str, image_paths: List[str]) -> tuple[str, int]:
        """Call external API via curl/HTTP requests with image support (vLLM format)"""
        try:
            if not REQUESTS_AVAILABLE:
                raise RuntimeError("requests library not available for curl/HTTP API calls")
            
            # Get curl client configuration
            curl_config = self.client
            if not isinstance(curl_config, dict) or curl_config.get('type') != 'curl':
                raise RuntimeError("Invalid curl client configuration")
            
            api_key = curl_config.get('api_key')
            base_url = curl_config.get('base_url')
            model_name = curl_config.get('model_name', self.config.model_name)
            
            if not base_url:
                raise RuntimeError("Base URL is required for curl/HTTP API")
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'GraphEval2/1.0'
            }
            
            # Add API key if provided
            if api_key:
                # Support different API key header formats
                if 'openai' in base_url.lower() or 'openai' in model_name.lower():
                    headers['Authorization'] = f'Bearer {api_key}'
                elif 'anthropic' in base_url.lower() or 'anthropic' in model_name.lower():
                    headers['x-api-key'] = api_key
                elif 'google' in base_url.lower() or 'gemini' in model_name.lower():
                    headers['Authorization'] = f'Bearer {api_key}'
                else:
                    # Default to Bearer token for most services
                    headers['Authorization'] = f'Bearer {api_key}'
            
            # Prepare request payload based on API format
            if 'anthropic' in base_url.lower() or 'anthropic' in model_name.lower():
                # Anthropic format
                content = [{"type": "text", "text": prompt}]
                
                # Add images for Anthropic
                for image_path in image_paths:
                    from pathlib import Path
                    if Path(image_path).exists():
                        import base64
                        with open(image_path, "rb") as image_file:
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                            
                            # Determine image format from file extension
                            image_format = Path(image_path).suffix.lower()
                            if image_format in ['.jpg', '.jpeg']:
                                media_type = 'image/jpeg'
                            elif image_format == '.png':
                                media_type = 'image/png'
                            elif image_format == '.gif':
                                media_type = 'image/gif'
                            elif image_format == '.webp':
                                media_type = 'image/webp'
                            else:
                                # Default to JPEG if format is unknown
                                media_type = 'image/jpeg'
                            
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data
                                }
                            })
                    else:
                        logger.warning(f"Image file not found: {image_path}")
                
                payload = {
                    'model': model_name,
                    'max_tokens': self.config.max_tokens,
                    'temperature': self.config.temperature,
                    'messages': [{"role": "user", "content": content}]
                }
                
            elif 'google' in base_url.lower() or 'gemini' in model_name.lower():
                # Google/Gemini format
                content = [prompt]
                
                # Add images for Gemini
                for image_path in image_paths:
                    from pathlib import Path
                    if Path(image_path).exists():
                        import base64
                        with open(image_path, "rb") as image_file:
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                            
                            # Determine image format from file extension
                            image_format = Path(image_path).suffix.lower()
                            if image_format in ['.jpg', '.jpeg']:
                                mime_type = 'image/jpeg'
                            elif image_format == '.png':
                                mime_type = 'image/png'
                            elif image_format == '.gif':
                                mime_type = 'image/gif'
                            elif image_format == '.webp':
                                mime_type = 'image/webp'
                            else:
                                # Default to JPEG if format is unknown
                                mime_type = 'image/jpeg'
                            
                            content.append({
                                "mime_type": mime_type,
                                "data": image_data
                            })
                    else:
                        logger.warning(f"Image file not found: {image_path}")
                
                payload = {
                    'contents': [{
                        'parts': content
                    }],
                    'generationConfig': {
                        'temperature': self.config.temperature,
                        'maxOutputTokens': self.config.max_tokens
                    }
                }
                
            else:
                # OpenAI-compatible format (default for most services including vLLM)
                content = [{'type': 'text', 'text': prompt}]
                
                # Add images in OpenAI-compatible format
                for image_path in image_paths:
                    from pathlib import Path
                    if Path(image_path).exists():
                        import base64
                        with open(image_path, "rb") as image_file:
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                            
                            # Determine image format from file extension
                            image_format = Path(image_path).suffix.lower()
                            if image_format in ['.jpg', '.jpeg']:
                                mime_type = 'image/jpeg'
                            elif image_format == '.png':
                                mime_type = 'image/png'
                            elif image_format == '.gif':
                                mime_type = 'image/gif'
                            elif image_format == '.webp':
                                mime_type = 'image/webp'
                            else:
                                # Default to JPEG if format is unknown
                                mime_type = 'image/jpeg'
                            
                            content.append({
                                'type': 'image_url',
                                'image_url': {
                                    'url': f'data:{mime_type};base64,{image_data}'
                                }
                            })
                    else:
                        logger.warning(f"Image file not found: {image_path}")
                
                payload = {
                    'model': model_name,
                    'temperature': self.config.temperature,
                    'messages': [{
                        'role': 'user',
                        'content': content
                    }],
                    'max_tokens': self.config.max_tokens
                }
            
            # Make HTTP request
            logger.debug(f"Making curl/HTTP request with images to {base_url}")
            response = requests.post(
                base_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            
            # Check response status
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # Extract content based on API format
            if 'anthropic' in base_url.lower() or 'anthropic' in model_name.lower():
                # Anthropic format
                if 'content' in response_data and response_data['content']:
                    content = response_data['content'][0]['text']
                    tokens_used = response_data.get('usage', {}).get('total_tokens', 0)
                else:
                    raise RuntimeError("Invalid Anthropic response format")
                    
            elif 'google' in base_url.lower() or 'gemini' in model_name.lower():
                # Google/Gemini format
                if 'candidates' in response_data and response_data['candidates']:
                    content = response_data['candidates'][0]['content']['parts'][0]['text']
                    tokens_used = response_data.get('usageMetadata', {}).get('totalTokenCount', 0)
                else:
                    raise RuntimeError("Invalid Google/Gemini response format")
                    
            else:
                # OpenAI-compatible format (default for most services including vLLM)
                if 'choices' in response_data and response_data['choices']:
                    content = response_data['choices'][0]['message']['content']
                    tokens_used = response_data.get('usage', {}).get('total_tokens', 0)
                else:
                    raise RuntimeError("Invalid OpenAI-compatible response format")
            
            # Validate content
            if not content or not content.strip():
                raise RuntimeError("Empty response from curl/HTTP API")
            
            logger.debug(f"Curl/HTTP API call with images successful, tokens used: {tokens_used}")
            return content.strip(), tokens_used
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Curl/HTTP API request with images failed: {e}")
            raise RuntimeError(f"HTTP request failed: {e}")
        except Exception as e:
            logger.error(f"Curl/HTTP API call with images failed: {e}")
            raise e


class MultiStepExecutor(TaskExecutor):
    """Executor for complex multi-step tasks"""
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.base_executor = LLMExecutor(self.config)
    
    def execute(self, task: TaskInstance, context: 'RetrievalResult') -> ExecutionResult:
        """Execute multi-step task"""
        
        if task.task_type.value not in ["reasoning", "aggregation", "synthesis"]:
            # Use base executor for simple tasks
            return self.base_executor.execute(task, context)
        
        logger.debug(f"Multi-step execution for task {task.task_id}")
        
        try:
            # Step 1: Break down the task
            subtasks = self._decompose_task(task)
            
            # Step 2: Execute each subtask
            subtask_results = []
            for subtask in subtasks:
                result = self.base_executor.execute(subtask, context)
                subtask_results.append(result)
            
            # Step 3: Synthesize results
            final_result = self._synthesize_results(task, subtask_results)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Multi-step execution failed for task {task.task_id}: {e}")
            
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                answer="",
                error_type=type(e).__name__,
                error_message=str(e)
            )
    
    def _decompose_task(self, task: TaskInstance) -> List[TaskInstance]:
        """Break down complex task into subtasks"""
        
        # Simple decomposition based on task type
        subtasks = []
        
        if task.task_type.value == "reasoning":
            # Break into: information gathering -> analysis -> conclusion
            subtasks = [
                TaskInstance(
                    task_id=f"{task.task_id}_info",
                    template_id=task.template_id,
                    task_type=task.task_type,
                    difficulty=task.difficulty,
                    prompt=f"Extract key information relevant to: {task.prompt}",
                    subgraph_nodes=task.subgraph_nodes,
                    requires_citations=True
                ),
                TaskInstance(
                    task_id=f"{task.task_id}_analysis",
                    template_id=task.template_id,
                    task_type=task.task_type,
                    difficulty=task.difficulty,
                    prompt=f"Analyze the relationships and patterns in the information to answer: {task.prompt}",
                    subgraph_nodes=task.subgraph_nodes,
                    requires_reasoning_path=True
                )
            ]
        
        elif task.task_type.value == "aggregation":
            # Break into: collection -> organization -> synthesis
            subtasks = [
                TaskInstance(
                    task_id=f"{task.task_id}_collect",
                    template_id=task.template_id,
                    task_type=task.task_type,
                    difficulty=task.difficulty,
                    prompt=f"Collect all relevant information about the topic in: {task.prompt}",
                    subgraph_nodes=task.subgraph_nodes,
                    requires_citations=True
                ),
                TaskInstance(
                    task_id=f"{task.task_id}_organize",
                    template_id=task.template_id,
                    task_type=task.task_type,
                    difficulty=task.difficulty,
                    prompt=f"Organize and structure the collected information to answer: {task.prompt}",
                    subgraph_nodes=task.subgraph_nodes
                )
            ]
        
        else:
            # Default: use original task
            subtasks = [task]
        
        return subtasks
    
    def _synthesize_results(self, original_task: TaskInstance, subtask_results: List[ExecutionResult]) -> ExecutionResult:
        """Synthesize subtask results into final answer"""
        
        # Combine answers
        combined_answer_parts = []
        all_citations = []
        all_reasoning = []
        
        for result in subtask_results:
            if result.success and result.answer:
                combined_answer_parts.append(result.answer)
                all_citations.extend(result.citations)
                all_reasoning.extend(result.reasoning_path)
        
        # Create final result
        final_result = ExecutionResult(
            task_id=original_task.task_id,
            success=len(combined_answer_parts) > 0,
            answer=" ".join(combined_answer_parts),
            citations=list(set(all_citations)),  # Deduplicate
            reasoning_path=all_reasoning,
            confidence=sum(r.confidence for r in subtask_results if r.success) / len(subtask_results) if subtask_results else 0.0
        )
        
        # Aggregate execution metadata
        final_result.execution_time = sum(r.execution_time for r in subtask_results)
        final_result.tokens_used = sum(r.tokens_used for r in subtask_results)
        final_result.retries_needed = sum(r.retries_needed for r in subtask_results)
        
        return final_result
    

