"""
Agent S 2.5 Web - Complete implementation based on official Agent S 2.5 design
Implements the core components: ACI, Worker, Reflection, and Procedural Memory
Compatible with Graph2Eval framework
"""

import asyncio
import time
import json
import os
import re
import base64
import textwrap
import inspect
import tempfile
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger
from abc import ABC, abstractmethod
import hashlib
import pickle
from datetime import datetime

from task_craft.task_generator import WebTaskInstance, WebTaskStep
from agent_framework.evaluators import WebTaskEvaluator, WebTaskExecutionResult
from agent_framework.executors import ExecutionResult, LLMExecutor, ExecutionConfig
from config_manager import get_config


def split_thinking_response(full_response: str) -> Tuple[str, str]:
    """Split thinking response into answer and thoughts"""
    try:
        # Extract thoughts section
        thoughts_match = re.search(
            r"<thoughts>(.*?)</thoughts>", full_response, re.DOTALL
        )
        thoughts = thoughts_match.group(1).strip() if thoughts_match else ""
        
        # Extract answer section
        answer_match = re.search(r"<answer>(.*?)</answer>", full_response, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else full_response
        
        return answer, thoughts
    except Exception as e:
        return full_response, ""


def parse_single_code_from_string(input_string):
    """Parse single code from string - simplified version"""
    input_string = input_string.strip()
    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return input_string.strip()

    # Match code blocks
    pattern = r"```(?:\w+\s+)?(.*?)```"
    matches = re.findall(pattern, input_string, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Fallback: look for agent. function calls
    agent_match = re.search(r'agent\.\w+\([^)]*\)', input_string)
    if agent_match:
        return agent_match.group(0)
    
    return "agent.wait(1.0)"


def sanitize_code(code):
    """Sanitize code - simplified version"""
    return code.strip()


def extract_first_agent_function(code):
    """Extract first agent function - simplified version"""
    # Look for agent. function calls
    agent_match = re.search(r'agent\.\w+\([^)]*\)', code)
    if agent_match:
        return agent_match.group(0)
    
    # Fallback
    if code.strip() in ["WAIT", "DONE", "FAIL"]:
        return f"agent.{code.strip().lower()}()"
    
    return "agent.wait(1.0)"

try:
    from playwright.async_api import async_playwright, Browser, Page, ElementHandle
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available, falling back to simulation mode")


@dataclass
class WebExecutionTrajectory:
    """Records the execution trajectory of a web task (compatible with Graph2Eval)"""
    task_id: str
    start_time: float
    end_time: float = 0.0
    actions_executed: List[Dict[str, Any]] = field(default_factory=list)
    action_results: List[Dict[str, Any]] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    page_info_files: List[str] = field(default_factory=list)
    current_url: str = ""
    success: bool = False
    total_tokens_used: int = 0
    error_message: str = ""
    final_state: Dict[str, Any] = field(default_factory=dict)
    
    def add_action(self, action: Dict[str, Any], result: Dict[str, Any]):
        self.actions_executed.append(action)
        self.action_results.append(result)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "actions_executed": self.actions_executed,
            "action_results": self.action_results,
            "screenshots": self.screenshots,
            "page_info_files": self.page_info_files,
            "current_url": self.current_url,
            "success": self.success,
            "error_message": self.error_message,
            "final_state": self.final_state
        }


class LMMAgent:
    """Multimodal Language Model Agent - Simplified version of official LMMAgent"""
    
    def __init__(self, engine_params: Dict[str, Any], system_prompt: str = None):
        self.engine_params = engine_params
        self.messages = []
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.reset()
    
    def reset(self):
        """Reset agent state"""
        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]
    
    def add_system_prompt(self, system_prompt: str):
        """Add or update system prompt"""
        self.system_prompt = system_prompt
        if len(self.messages) > 0:
            self.messages[0] = {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        else:
            self.messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
    
    def encode_image(self, image_content):
        """Encode image to base64"""
        if isinstance(image_content, str):
            with open(image_content, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        else:
            return base64.b64encode(image_content).decode("utf-8")
    
    def add_message(
        self,
        text_content: str,
        image_content=None,
        role=None,
        image_detail="high",
        put_text_last=False,
    ):
        """Add a new message to the conversation"""
        # Infer role from previous message
        if role != "user":
            if self.messages[-1]["role"] == "system":
                role = "user"
            elif self.messages[-1]["role"] == "user":
                role = "assistant"
            elif self.messages[-1]["role"] == "assistant":
                role = "user"
        
        message = {
            "role": role,
            "content": [{"type": "text", "text": text_content}],
        }
        
        if image_content:
            base64_image = self.encode_image(image_content)
            message["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": image_detail,
                    },
                }
            )
        
        # Rotate text to be the last message if desired
        if put_text_last and len(message["content"]) > 1:
            text_content = message["content"].pop(0)
            message["content"].append(text_content)
        
        self.messages.append(message)
    
    async def get_response(self, temperature: float = 0.0, use_validation_model: bool = False, task_id: str = None, use_thinking: bool = False, **kwargs) -> Tuple[str, int]:
        """Generate response using LLM executor with multimodal support"""
        # Check if we have images in the messages
        has_images = False
        image_paths = []
        
        # Check if text-only mode is enabled
        global_config = get_config()
        agent_config = global_config.agent.get('agent_s_web', {}).get('agent', {})
        text_only_mode = agent_config.get('text_only_mode', False)
        
        if not text_only_mode:
            # Process all images in messages (following original Agent S 2.5 design)
            for message in self.messages:
                for content_item in message["content"]:
                    if content_item["type"] == "image_url":
                        has_images = True
                        # Extract base64 image data and save to temporary file
                        image_url = content_item["image_url"]["url"]
                        if image_url.startswith("data:image/png;base64,"):
                            base64_data = image_url.split(",")[1]
                            image_data = base64.b64decode(base64_data)
                            
                            # Save to temporary file
                            import tempfile
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            temp_file.write(image_data)
                            temp_file.close()
                            image_paths.append(temp_file.name)
        else:
            logger.info("📝 Text-only mode enabled - skipping image processing")
        
        # Use LLM executor with configurable model
        if use_validation_model:
            # Use validation model from config
            global_config = get_config()
            validation_config = global_config.agent.get('agent_s_web', {}).get('validation', {})
            
            execution_config = ExecutionConfig(
                model_name=validation_config.get('model_name', 'gpt-4o-mini'),
                model_provider=validation_config.get('model_provider', 'openai'),
                temperature=validation_config.get('temperature', 0.0),
                max_tokens=validation_config.get('max_tokens', 1000),
                response_format=validation_config.get('response_format', 'json'),
                timeout=30,
                max_retries=2,
                retry_delay=1.0,
                require_citations=False,
                require_reasoning=True,
                max_context_length=4000,
                support_images=True,
                fallback_to_structured=True
            )
            logger.info(f"🔍 Using VALIDATION model: {execution_config.model_name} (provider: {execution_config.model_provider})")
        else:
            # Use execution model (default behavior)
            base_config = ExecutionConfig.from_config()
            execution_config = ExecutionConfig(
                model_name=base_config.model_name,
                model_provider=base_config.model_provider,
                temperature=temperature,
                max_tokens=2000,
                timeout=base_config.timeout,
                max_retries=base_config.max_retries,
                retry_delay=base_config.retry_delay,
                require_citations=False,
                require_reasoning=True,
                response_format="text",
                max_context_length=base_config.max_context_length,
                support_images=True,
                fallback_to_structured=base_config.fallback_to_structured
            )
        
        # Create LLMExecutor instance
        llm_executor = LLMExecutor(execution_config)
        
        if has_images and image_paths:
            # Use multimodal execution with images
            logger.info(f"🖼️ Using multimodal execution with {len(image_paths)} images")
            
            # Convert messages to text prompt
            prompt_parts = []
            for message in self.messages:
                if message["role"] == "system":
                    continue  # Skip system message as it's handled separately
                
                role = "User" if message["role"] == "user" else "Assistant"
                content = ""
                
                for content_item in message["content"]:
                    if content_item["type"] == "text":
                        content += content_item["text"]
                    elif content_item["type"] == "image_url":
                        content += "\n[Image provided]"
                
                prompt_parts.append(f"{role}: {content}")
            
            full_prompt = f"System: {self.system_prompt}\n\n" + "\n\n".join(prompt_parts)
            
            # Use execute_simple_with_image method
            actual_task_id = task_id or "agent_s2_5_multimodal"
            response = llm_executor.execute_simple_with_image(
                prompt=full_prompt,
                image_paths=image_paths,
                task_id=actual_task_id
            )
            
            # Clean up temporary files
            for temp_file in image_paths:
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        else:
            # Use text-only execution
            logger.info("📝 Using text-only execution")
            
            # Convert messages to prompt format for LLM executor
            prompt_parts = []
            for message in self.messages:
                if message["role"] == "system":
                    continue  # Skip system message as it's handled separately
                
                role = "User" if message["role"] == "user" else "Assistant"
                content = ""
                
                for content_item in message["content"]:
                    if content_item["type"] == "text":
                        content += content_item["text"]
                    elif content_item["type"] == "image_url":
                        content += "\n[Image provided]"
                
                prompt_parts.append(f"{role}: {content}")
            
            full_prompt = f"System: {self.system_prompt}\n\n" + "\n\n".join(prompt_parts)
            actual_task_id = task_id or "agent_s2_5_text"
            response = llm_executor.execute_simple(full_prompt, actual_task_id)
        
        # Debug logging
        logger.debug(f"LLM response: {response}")
        logger.debug(f"Response answer: {response.answer}")
        logger.debug(f"Response answer type: {type(response.answer)}")
        logger.debug(f"Response answer length: {len(response.answer) if response.answer else 0}")
        logger.debug(f"Tokens used: {response.tokens_used}")
        
        if not response.answer or response.answer.strip() == "":
            logger.warning("Empty response from LLM, using fallback")
            return "agent.wait(1.0)", 0  # Fallback action with 0 tokens
        
        return response.answer, response.tokens_used


class BaseModule:
    """Base module for Agent S 2.5 components"""
    
    def __init__(self, engine_params: Dict[str, Any], platform: str = "web"):
        self.engine_params = engine_params
        self.platform = platform
    
    def _create_agent(self, system_prompt: str = None, engine_params: Optional[Dict] = None) -> LMMAgent:
        """Create a new LMMAgent instance"""
        agent = LMMAgent(engine_params or self.engine_params)
        if system_prompt:
            agent.add_system_prompt(system_prompt)
        return agent


class WebACI:
    """Web Agent-Computer Interface - Simplified version of official ACI"""
    
    def __init__(self):
        self.notes: List[str] = []
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.use_browser = PLAYWRIGHT_AVAILABLE
    
    async def initialize(self, headless: bool = True):
        """Initialize browser"""
        if not self.use_browser:
            logger.info("Browser automation disabled, using simulation mode")
            return
        
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=headless)
            self.page = await self.browser.new_page()
            await self.page.set_viewport_size({"width": 1280, "height": 720})
            logger.info("Web ACI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Web ACI: {e}")
            self.use_browser = False
    
    async def close(self):
        """Close browser"""
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
    
    async def take_screenshot(self) -> bytes:
        """Take screenshot of current page"""
        if not self.use_browser or not self.page:
            # Return empty bytes for simulation mode
            return b""
        
        try:
            screenshot = await self.page.screenshot()
            return screenshot
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return b""
    
    async def get_page_info(self) -> Dict[str, Any]:
        """Get current page information"""
        if not self.use_browser or not self.page:
            return {"url": "", "title": "", "elements": []}
        
        try:
            url = self.page.url
            title = await self.page.title()
            
            # Get basic page elements
            elements = await self.page.evaluate("""
                () => {
                    const elements = [];
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_ELEMENT,
                        {
                            acceptNode: function(node) {
                                if (node.offsetParent !== null) {
                                    return NodeFilter.FILTER_ACCEPT;
                                }
                                return NodeFilter.FILTER_SKIP;
                            }
                        }
                    );
                    
                    let node;
                    while (node = walker.nextNode()) {
                        const rect = node.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            elements.push({
                                tag: node.tagName.toLowerCase(),
                                text: node.textContent?.trim() || '',
                                id: node.id || '',
                                className: node.className || '',
                                role: node.getAttribute('role') || '',
                                rect: {
                                    x: rect.x,
                                    y: rect.y,
                                    width: rect.width,
                                    height: rect.height
                                }
                            });
                        }
                    }
                    return elements;
                }
            """)
            
            return {"url": url, "title": title, "elements": elements}
        except Exception as e:
            logger.error(f"Failed to get page info: {e}")
            return {"url": "", "title": "", "elements": []}
    
    async def execute_action(self, action_code: str) -> Dict[str, Any]:
        """Execute action code"""
        if not self.use_browser or not self.page:
            return {"success": False, "error": "Browser not available"}
        
        try:
            # Parse and execute action
            if "done()" in action_code or action_code.strip() == "done()":
                return {"success": True, "action": "done()", "result": "DONE"}
            elif "fail()" in action_code or action_code.strip() == "fail()":
                return {"success": True, "action": "fail()", "result": "FAIL"}
            elif "click" in action_code:
                return await self._execute_click(action_code)
            elif "type" in action_code:
                return await self._execute_type(action_code)
            elif "navigate" in action_code:
                return await self._execute_navigate(action_code)
            elif "wait" in action_code:
                return await self._execute_wait(action_code)
            elif "scroll" in action_code:
                return await self._execute_scroll(action_code)
            else:
                return {"success": False, "error": f"Unknown action: {action_code}"}
        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_click(self, action_code: str) -> Dict[str, Any]:
        """Execute click action"""
        try:
            # Extract element description from action code
            desc_match = re.search(r'click\(["\']([^"\']+)["\']', action_code)
            if desc_match:
                element_description = desc_match.group(1)
                # For now, use a simple coordinate mapping based on description
                # In a real implementation, this would use a grounding model
                coords = await self._get_coordinates_for_element(element_description)
                if coords:
                    x, y = coords
                    await self.page.mouse.click(x, y)
                    return {"success": True, "action": f"click({element_description}) -> ({x}, {y})"}
                else:
                    return {"success": False, "error": f"Could not find element: {element_description}"}
            else:
                # Fallback to coordinate parsing for backward compatibility
                coords_match = re.search(r'click\((\d+),\s*(\d+)', action_code)
                if coords_match:
                    x, y = int(coords_match.group(1)), int(coords_match.group(2))
                    await self.page.mouse.click(x, y)
                    return {"success": True, "action": f"click({x}, {y})"}
                else:
                    return {"success": False, "error": "Could not parse click action"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_type(self, action_code: str) -> Dict[str, Any]:
        """Execute type action"""
        try:
            # Extract text from action code
            text_match = re.search(r"write\(['\"]([^'\"]*)['\"]", action_code)
            if text_match:
                text = text_match.group(1)
                await self.page.keyboard.type(text)
                return {"success": True, "action": f"type('{text}')"}
            else:
                return {"success": False, "error": "Could not parse text to type"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_navigate(self, action_code: str) -> Dict[str, Any]:
        """Execute navigate action"""
        try:
            # Extract URL from action code
            url_match = re.search(r"goto\(['\"]([^'\"]*)['\"]", action_code)
            if url_match:
                url = url_match.group(1)
                await self.page.goto(url)
                return {"success": True, "action": f"navigate('{url}')"}
            else:
                return {"success": False, "error": "Could not parse URL"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_wait(self, action_code: str) -> Dict[str, Any]:
        """Execute wait action"""
        try:
            # Extract time from action code
            time_match = re.search(r'sleep\(([\d.]+)', action_code)
            if time_match:
                wait_time = float(time_match.group(1))
                await asyncio.sleep(wait_time)
                return {"success": True, "action": f"wait({wait_time})"}
            else:
                return {"success": False, "error": "Could not parse wait time"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_scroll(self, action_code: str) -> Dict[str, Any]:
        """Execute scroll action"""
        try:
            # Extract scroll amount from action code
            scroll_match = re.search(r'scroll\(([-\d]+)', action_code)
            if scroll_match:
                scroll_amount = int(scroll_match.group(1))
                await self.page.mouse.wheel(0, scroll_amount)
                return {"success": True, "action": f"scroll({scroll_amount})"}
            else:
                return {"success": False, "error": "Could not parse scroll amount"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _get_coordinates_for_element(self, element_description: str) -> Optional[Tuple[int, int]]:
        """Get coordinates for an element based on its description using LLM grounding"""
        try:
            # Check if text-only mode is enabled
            global_config = get_config()
            agent_config = global_config.agent.get('agent_s_web', {}).get('agent', {})
            text_only_mode = agent_config.get('text_only_mode', False)
            
            if text_only_mode:
                # In text-only mode, use simple coordinate estimation
                logger.info("📝 Text-only mode: Using simple coordinate estimation for grounding")
                return (400, 300)  # Default center coordinates
            
            # Get current screenshot and page info
            screenshot = await self.page.screenshot()
            page_info = await self.get_page_info()
            elements = page_info.get('elements', [])
            
            # Create element descriptions for LLM
            element_descriptions = []
            for i, element in enumerate(elements):
                text = element.get('text', '').strip()
                tag = element.get('tagName', '').lower()
                rect = element.get('rect', {})
                if text and rect:
                    x = int(rect.get('x', 0) + rect.get('width', 0) / 2)
                    y = int(rect.get('y', 0) + rect.get('height', 0) / 2)
                    element_descriptions.append(f"{i}: '{text}' ({tag}) at ({x}, {y})")
            
            # Use LLM to find the best matching element with execution model
            from agent_framework.executors import ExecutionConfig
            base_config = ExecutionConfig.from_config()
            
            # Create execution config for grounding (same as main execution)
            grounding_config = ExecutionConfig(
                model_name=base_config.model_name,
                model_provider=base_config.model_provider,
                temperature=base_config.temperature,
                max_tokens=500,  # Smaller for grounding
                response_format="text",
                timeout=30,
                max_retries=2,
                retry_delay=1.0,
                require_citations=False,
                require_reasoning=True,
                max_context_length=4000,
                support_images=True,
                fallback_to_structured=True
            )
            logger.info(f"🎯 Using EXECUTION model for grounding: {grounding_config.model_name} (provider: {grounding_config.model_provider})")
            
            prompt = f"""Given the user's description: "{element_description}"

Available elements on the page:
{chr(10).join(element_descriptions)}

Please identify which element best matches the user's description. Return only the element number (e.g., "5") or "none" if no match is found."""

            # Create new LLMExecutor instance for grounding to use validation model
            llm_executor = LLMExecutor(grounding_config)
            response = llm_executor.execute_simple(prompt, "grounding")
            
            if response.answer and response.answer.strip().lower() != "none":
                try:
                    element_index = int(response.answer.strip())
                    if 0 <= element_index < len(elements):
                        element = elements[element_index]
                        rect = element.get('rect', {})
                        if rect:
                            x = int(rect.get('x', 0) + rect.get('width', 0) / 2)
                            y = int(rect.get('y', 0) + rect.get('height', 0) / 2)
                            logger.info(f"LLM grounded '{element_description}' to element {element_index}: '{element.get('text', '')}' at ({x}, {y})")
                            return (x, y)
                except ValueError:
                    pass
            
            # Fallback to simple keyword matching
            description_lower = element_description.lower()
            for element in elements:
                element_text = element.get('text', '').lower()
                if description_lower in element_text or any(keyword in element_text for keyword in description_lower.split()):
                    rect = element.get('rect', {})
                    if rect:
                        x = int(rect.get('x', 0) + rect.get('width', 0) / 2)
                        y = int(rect.get('y', 0) + rect.get('height', 0) / 2)
                        logger.info(f"Fallback grounded '{element_description}' to '{element.get('text', '')}' at ({x}, {y})")
                        return (x, y)
            
            logger.warning(f"Could not ground element description: '{element_description}'")
            return None
            
        except Exception as e:
            logger.error(f"Error getting coordinates for element: {e}")
            return None


class ProceduralMemory:
    """Procedural Memory for Agent S 2.5 - Based on official implementation"""
    
    @staticmethod
    def construct_simple_worker_procedural_memory(agent_class, skipped_actions=None):
        """Construct procedural memory for worker agent"""
        if skipped_actions is None:
            skipped_actions = []
        
        procedural_memory = textwrap.dedent(
            f"""\
        You are an expert in web interfaces and Python code. You are responsible for executing the task: `TASK_DESCRIPTION`.
        You are working in a web browser environment.
        You are provided with:
        1. A screenshot of the current web page.
        2. The history of your previous interactions with the web interface.
        3. Access to the following class and methods to interact with the web interface:
        class WebAgent:
        """
        )
        
        # Add available methods
        for attr_name in dir(agent_class):
            if attr_name in skipped_actions:
                continue
            
            attr = getattr(agent_class, attr_name)
            if callable(attr) and hasattr(attr, "is_agent_action"):
                signature = inspect.signature(attr)
                procedural_memory += f"""
    def {attr_name}{signature}:
    '''{attr.__doc__}'''
        """
        
        procedural_memory += textwrap.dedent(
            """
        Your response should be formatted like this:
        (Previous action verification)
        Carefully analyze based on the screenshot if the previous action was successful. If the previous action was not successful, provide a reason for the failure.

        (Screenshot Analysis)
        Closely examine and describe the current state of the web page along with the visible elements and their functionality.

        (Next Action)
        Based on the current screenshot and the history of your previous interaction with the web interface, decide on the next action in natural language to accomplish the given task.

        (Grounded Action)
        Translate the next action into code using the provided API methods. Format the code like this:
        ```python
        agent.click("The MDN element in the navigation menu")
        ```
        Note for the code:
        1. Only perform one action at a time.
        2. Do not put anything other than python code in the block. You can only use one function call at a time. Do not put more than one function call in the block.
        3. You must use only the available methods provided above to interact with the web interface, do not invent new methods.
        4. Only return one code block every time. There must be a single line of code in the code block.
        5. Do not do anything other than the exact specified task. Return with `agent.done()` immediately after the task is completed or `agent.fail()` if it cannot be completed.
        6. Whenever possible, use keyboard shortcuts and efficient navigation methods.
        7. Generate agent.fail() as your grounded action if you get exhaustively stuck on the task and believe it is impossible.
        8. Generate agent.done() as your grounded action when you believe the task is fully complete.
        """
        )
        
        return procedural_memory.strip()
    
    # Reflection prompt for trajectory analysis
    REFLECTION_ON_TRAJECTORY = textwrap.dedent(
        """
    You are an expert web automation agent designed to reflect on the trajectory of a task and provide feedback on what has happened so far.
    You have access to the Task Description and the Current Trajectory of another web automation agent. The Current Trajectory is a sequence of a web page screenshot, chain-of-thought reasoning, and a web action for each time step. The last image is the web page display after the last action.
    Your task is to generate a reflection. Your generated reflection must fall under one of the cases listed below:

    Case 1. The trajectory is not going according to plan. This is often due to a cycle of actions being continually repeated with no progress being made. In this case, explicitly highlight why the current trajectory is incorrect, and encourage the web automation agent to modify their action. However, DO NOT encourage a specific action in particular.
    Case 2. The trajectory is going according to plan. In this case, simply tell the agent to continue proceeding as planned. DO NOT encourage a specific action in particular.
    Case 3. You believe the current task has been completed. In this case, tell the agent that the task has been successfully completed.
    
    To be successful, you must follow the rules below:
    - **Your output MUST be based on one of the case options above**.
    - DO NOT suggest any specific future plans or actions. Your only goal is to provide a reflection, not an actual plan or action.
    - Any response that falls under Case 1 should explain why the trajectory is not going according to plan. You should especially lookout for cycles of actions that are continually repeated with no progress.
    - Any response that falls under Case 2 should be concise, since you just need to affirm the agent to continue with the current trajectory.
    """
    )


# Agent action decorator
def agent_action(func):
    """Decorator to mark agent actions"""
    func.is_agent_action = True
    return func


class WebAgentActions:
    """Web Agent Actions - Based on official ACI actions"""
    
    def __init__(self, aci: WebACI):
        self.aci = aci
    
    @agent_action
    def click(self, element_description: str, num_clicks: int = 1, button_type: str = "left"):
        """Click on the element described by the given text
        Args:
            element_description: str, description of the element to click (e.g., "The MDN element in the navigation menu")
            num_clicks: int, number of times to click
            button_type: str, which mouse button to press
        """
        return f"import pyautogui; pyautogui.click({element_description}, clicks={num_clicks}, button='{button_type}')"
    
    @agent_action
    def type(self, text: str, overwrite: bool = False, enter: bool = False):
        """Type text into the current focused element
        Args:
            text: str, the text to type
            overwrite: bool, whether to overwrite existing text
            enter: bool, whether to press enter after typing
        """
        command = "import pyautogui; "
        if overwrite:
            command += "pyautogui.hotkey('ctrl', 'a'); pyautogui.press('backspace'); "
        command += f"pyautogui.write('{text}'); "
        if enter:
            command += "pyautogui.press('enter'); "
        return command
    
    @agent_action
    def navigate(self, url: str):
        """Navigate to a URL
        Args:
            url: str, the URL to navigate to
        """
        return f"import pyautogui; pyautogui.hotkey('ctrl', 'l'); pyautogui.write('{url}'); pyautogui.press('enter')"
    
    @agent_action
    def scroll(self, clicks: int, direction: str = "down"):
        """Scroll the page
        Args:
            clicks: int, number of scroll clicks
            direction: str, direction to scroll ('up' or 'down')
        """
        scroll_amount = clicks if direction == "down" else -clicks
        return f"import pyautogui; pyautogui.scroll({scroll_amount})"
    
    @agent_action
    def wait(self, time: float):
        """Wait for a specified amount of time
        Args:
            time: float, the amount of time to wait in seconds
        """
        return f"import time; time.sleep({time})"
    
    @agent_action
    def hotkey(self, keys: List[str]):
        """Press a hotkey combination
        Args:
            keys: List[str], the keys to press in combination
        """
        keys_str = ", ".join([f"'{key}'" for key in keys])
        return f"import pyautogui; pyautogui.hotkey({keys_str})"
    
    @agent_action
    def done(self, return_value: Optional[Any] = None):
        """End the current task with success"""
        return "DONE"
    
    @agent_action
    def fail(self):
        """End the current task with failure"""
        return "FAIL"


class Worker(BaseModule):
    """Worker Agent - Based on official Agent S 2.5 Worker"""
    
    def __init__(
        self,
        engine_params: Dict[str, Any],
        grounding_agent: WebACI,
        platform: str = "web",
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
    ):
        """Initialize Worker agent
        
        Args:
            engine_params: Parameters for the multimodal engine
            grounding_agent: The grounding agent to use
            platform: OS platform the agent runs on
            max_trajectory_length: The amount of image turns to keep
            enable_reflection: Whether to enable reflection
        """
        super().__init__(engine_params, platform)
        
        self.grounding_agent = grounding_agent
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection
        self.temperature = engine_params.get("temperature", 0.0)
        self.use_thinking = engine_params.get("model", "") in [
            "claude-3-7-sonnet-20250219"
        ]
        self.reset()
    
    def reset(self):
        """Reset worker state"""
        # Create web agent actions
        web_actions = WebAgentActions(self.grounding_agent)
        
        # Construct procedural memory
        sys_prompt = ProceduralMemory.construct_simple_worker_procedural_memory(
            web_actions, skipped_actions=[]
        ).replace("CURRENT_OS", self.platform)
        
        self.generator_agent = self._create_agent(sys_prompt)
        self.reflection_agent = self._create_agent(
            ProceduralMemory.REFLECTION_ON_TRAJECTORY
        )
        
        self.turn_count = 0
        self.worker_history = []
        self.reflections = []
        self.cost_this_turn = 0
        # 注意：移除了 screenshot_inputs，因为原版也没有实际使用
    
    def flush_messages(self):
        """Flush messages based on model context limits"""
        engine_type = self.engine_params.get("engine_type", "")
        
        # Flush strategy for long-context models: keep all text, only keep latest images
        if engine_type in ["anthropic", "openai", "gemini"]:
            max_images = self.max_trajectory_length
            for agent in [self.generator_agent, self.reflection_agent]:
                # keep latest k images
                img_count = 0
                for i in range(len(agent.messages) - 1, -1, -1):
                    for j in range(len(agent.messages[i]["content"])):
                        if "image" in agent.messages[i]["content"][j].get("type", ""):
                            img_count += 1
                            if img_count > max_images:
                                del agent.messages[i]["content"][j]
        
        # Flush strategy for non-long-context models: drop full turns
        else:
            # generator msgs are alternating [user, assistant], so 2 per round
            if len(self.generator_agent.messages) > 2 * self.max_trajectory_length + 1:
                self.generator_agent.messages.pop(1)
                self.generator_agent.messages.pop(1)
            # reflector msgs are all [(user text, user image)], so 1 per round
            if len(self.reflection_agent.messages) > self.max_trajectory_length + 1:
                self.reflection_agent.messages.pop(1)
    
    async def generate_next_action(
        self,
        instruction: str,
        obs: Dict,
        task_id: str = None,
    ) -> Tuple[Dict, List]:
        """Generate the next action based on the current observation"""
        generator_message = (
            ""
            if self.turn_count > 0
            else "The initial web page is provided. No action has been taken yet."
        )
        
        # Load the task into the system prompt
        if self.turn_count == 0:
            self.generator_agent.add_system_prompt(
                self.generator_agent.system_prompt.replace(
                    "TASK_DESCRIPTION", instruction
                )
            )
        
        # Get the per-step reflection
        reflection = None
        reflection_thoughts = None
        reflection_tokens = 0  # Initialize reflection_tokens
        if self.enable_reflection:
            # Load the initial message
            if self.turn_count == 0:
                text_content = textwrap.dedent(
                    f"""
                    Task Description: {instruction}
                    Current Trajectory below:
                    """
                )
                updated_sys_prompt = (
                    self.reflection_agent.system_prompt + "\n" + text_content
                )
                self.reflection_agent.add_system_prompt(updated_sys_prompt)
                self.reflection_agent.add_message(
                    text_content="The initial web page is provided. No action has been taken yet.",
                    image_content=obs["screenshot"],
                    role="user",
                )
            # Load the latest action
            else:
                self.reflection_agent.add_message(
                    text_content=self.worker_history[-1],
                    image_content=obs["screenshot"],
                    role="user",
                )
                full_reflection, reflection_tokens = await self.reflection_agent.get_response(
                    temperature=self.temperature,
                    use_validation_model=False,  # Use execution model for reflection
                    task_id=task_id,
                    use_thinking=self.use_thinking
                )
                reflection, reflection_thoughts = split_thinking_response(full_reflection)
                self.reflections.append(reflection)
                generator_message += f"REFLECTION: You may use this reflection on the previous action and overall trajectory:\n{reflection}\n"
                logger.info(f"REFLECTION: {reflection or 'None'}")
        
        # Add finalized message to conversation
        generator_message += f"\nCurrent Text Buffer = [{','.join(self.grounding_agent.notes)}]\n"
        self.generator_agent.add_message(
            generator_message, image_content=obs["screenshot"], role="user"
        )
        
        full_plan, plan_tokens = await self.generator_agent.get_response(
            temperature=self.temperature,
            task_id=task_id,
            use_thinking=self.use_thinking
        )
        plan, plan_thoughts = split_thinking_response(full_plan)
        # NOTE: currently dropping thinking tokens from context
        self.worker_history.append(plan)
        logger.debug(f"full_plan type: {type(full_plan)}")
        logger.debug(f"full_plan value: {repr(full_plan)}")
        logger.debug(f"plan_tokens: {plan_tokens}")
        logger.info(f"FULL PLAN:\n{full_plan or 'None'}")
        self.generator_agent.add_message(plan or "No plan generated", role="assistant")
        
        # Use the grounding agent to convert agent_action("desc") into agent_action([x, y])
        try:
            # For now, we'll use a simplified approach since we don't have full grounding
            # In the future, we could implement: self.grounding_agent.assign_coordinates(plan, obs)
            plan_code = parse_single_code_from_string(plan.split("Grounded Action")[-1] if "Grounded Action" in plan else plan)
            plan_code = sanitize_code(plan_code)
            plan_code = extract_first_agent_function(plan_code)
            
            # Convert to executable code (remove agent. prefix)
            exec_code = plan_code.replace('agent.', '') if plan_code.startswith('agent.') else plan_code
            
        except Exception as e:
            logger.error("Error in parsing plan code: %s", e)
            plan_code = "agent.wait(1.0)"
            exec_code = "wait(1.0)"
        
        # Calculate total tokens used in this step
        step_tokens = plan_tokens
        if self.enable_reflection and reflection_tokens:
            step_tokens += reflection_tokens
        
        executor_info = {
            "full_plan": full_plan,
            "executor_plan": plan,
            "plan_thoughts": plan_thoughts,
            "plan_code": plan_code,
            "reflection": reflection,
            "reflection_thoughts": reflection_thoughts,
            "tokens_used": step_tokens,
        }
        self.turn_count += 1
        
        # 注意：原版 Agent S 2.5 中的 screenshot_inputs 只是存储但从未使用
        # 真正的记忆管理是通过 LLM 消息历史实现的
        self.flush_messages()
        
        return executor_info, [exec_code]


class AgentS2_5Web:
    """Agent S 2.5 Web - Complete implementation based on official Agent S 2.5 design"""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
    ):
        """Initialize Agent S 2.5 Web
        
        Args:
            config: Configuration parameters
            output_dir: Output directory for results
            max_trajectory_length: Maximum number of image turns to keep
            enable_reflection: Whether to enable reflection agent
        """
        self.config = config or {}
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection
        
        # Load text-only mode configuration
        global_config = get_config()
        agent_config = global_config.agent.get('agent_s_web', {}).get('agent', {})
        self.text_only_mode = agent_config.get('text_only_mode', False)
        
        # Initialize engine parameters
        from agent_framework.executors import ExecutionConfig
        base_config = ExecutionConfig.from_config()
        self.engine_params = {
            "engine_type": "openai",  # Default to OpenAI
            "model": base_config.model_name,
            "temperature": 0.1,
            "max_tokens": 2000,
            "timeout": base_config.timeout,
            "max_retries": base_config.max_retries,
        }
        
        # Initialize components
        self.grounding_agent = WebACI()
        self.executor = Worker(
            engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            platform="web",
            max_trajectory_length=self.max_trajectory_length,
            enable_reflection=self.enable_reflection,
        )
        
        # Output directory for Graph2Eval compatibility
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path('output/agent_s2_5_web')
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize_browser(self):
        """Initialize browser for Graph2Eval compatibility"""
        await self.grounding_agent.initialize(
            headless=self.config.get('headless', True)
        )
        logger.info("Agent S 2.5 Web browser initialized successfully")
    
    async def close_browser(self):
        """Close browser for Graph2Eval compatibility"""
        await self.grounding_agent.close()
        logger.info("Agent S 2.5 Web browser closed")
    
    async def execute_web_task(self, task: WebTaskInstance) -> Tuple[ExecutionResult, WebExecutionTrajectory]:
        """Execute a web task - Main interface for Graph2Eval framework"""
        try:
            logger.info(f"🚀 Agent S 2.5 Web executing task: {task.task_id}")
            
            # Create execution trajectory
            trajectory = WebExecutionTrajectory(
                task_id=task.task_id,
                start_time=time.time()
            )
            
            # Navigate to start page if specified
            if hasattr(task, 'start_page') and task.start_page:
                await self.grounding_agent.page.goto(task.start_page)
                await asyncio.sleep(2)  # Wait for page load
                trajectory.current_url = task.start_page
            
            # Reset executor for new task
            self.executor.reset()
            
            # Execute task using Agent S 2.5 approach
            max_steps = 20  # Maximum steps to prevent infinite loops
            step_count = 0
            total_tokens = 0
            executor_info = {}  # Initialize executor_info
            
            while step_count < max_steps:
                try:
                    # Get current observation
                    screenshot = await self.grounding_agent.take_screenshot()
                    page_info = await self.grounding_agent.get_page_info()
                    
                    obs = {
                        "screenshot": screenshot,
                        "page_info": page_info
                    }
                    
                    # Generate next action
                    executor_info, actions = await self.executor.generate_next_action(
                        instruction=task.prompt if hasattr(task, 'prompt') else f"Complete task: {task.task_id}",
                        obs=obs,
                        task_id=task.task_id
                    )
                    
                    # Accumulate token usage
                    step_tokens = executor_info.get("tokens_used", 0)
                    total_tokens += step_tokens
                    logger.debug(f"Step {step_count} used {step_tokens} tokens, total: {total_tokens}")
                    
                    # Record action
                    action_code = actions[0] if actions else "wait(1.0)"
                    trajectory.add_action(
                        {"action": action_code, "step": step_count},
                        {"info": executor_info}
                    )
                    
                    # Check for completion
                    if "DONE" in action_code:
                        trajectory.success = True
                        break
                    elif "FAIL" in action_code:
                        trajectory.success = False
                        break
                    
                    # Execute action
                    result = await self.grounding_agent.execute_action(action_code)
                    if not result.get('success', False):
                        logger.warning(f"Action failed: {action_code}")
                    elif result.get('result') == "DONE":
                        trajectory.success = True
                        break
                    elif result.get('result') == "FAIL":
                        trajectory.success = False
                        break
                    
                    # Update trajectory
                    trajectory.current_url = page_info.get('url', '')
                    
                    step_count += 1
                    await asyncio.sleep(1)  # Wait between actions
                    
                except Exception as e:
                    logger.error(f"Step {step_count} failed: {e}")
                    trajectory.error_message = str(e)
                    break
            
            # Finalize trajectory
            trajectory.end_time = time.time()
            if step_count >= max_steps:
                trajectory.success = False
            
            # Validate task completion using validation model
            validation_result = await self._validate_task_completion(task, trajectory)
            
            # Create ExecutionResult with validation result
            execution_result = self._create_execution_result(task, trajectory, executor_info, total_tokens, validation_result, step_count, max_steps)
            
            logger.info(f"✅ Agent S 2.5 Web task completed: {task.task_id} - Success: {execution_result.success}")
            return execution_result, trajectory
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            trajectory.end_time = time.time()
            trajectory.success = False
            trajectory.error_message = str(e)
            return self._create_failed_result(task, trajectory, str(e))
    
    async def _validate_task_completion(self, task: WebTaskInstance, trajectory: WebExecutionTrajectory) -> Dict[str, Any]:
        """Validate task completion using validation model from config"""
        try:
            # Check if text-only mode is enabled
            global_config = get_config()
            agent_config = global_config.agent.get('agent_s_web', {}).get('agent', {})
            text_only_mode = agent_config.get('text_only_mode', False)
            
            # Get final screenshot and page info
            screenshot = None
            if not text_only_mode:
                screenshot = await self.grounding_agent.take_screenshot()
            page_info = await self.grounding_agent.get_page_info()
            
            # Create validation prompt with proper JSON format
            prompt = f"""Task: {task.prompt if hasattr(task, 'prompt') else f'Complete task: {task.task_id}'}

Execution Summary:
- Actions executed: {len(trajectory.actions_executed)}
- Success: {trajectory.success}
- Error message: {trajectory.error_message or 'None'}

Current page URL: {page_info.get('url', 'Unknown')}
Current page title: {page_info.get('title', 'Unknown')}

Actions executed:
{chr(10).join([f"- {action.get('action', 'unknown')}" for action in trajectory.actions_executed])}

Please evaluate if the task has been completed successfully by analyzing the current page state. Consider:
1. Whether all required actions were performed
2. Whether the final state matches the task requirements
3. Whether any errors occurred that prevent completion
4. Whether the current page content indicates task completion

Respond with valid JSON format (no markdown, no code blocks):
{{
    "task_completed": true,
    "confidence": 0.8,
    "reasoning": "explanation of your evaluation",
    "missing_actions": ["list of any missing actions"],
    "final_state_analysis": "description of current page state"
}}"""

            # Use validation model from config
            global_config = get_config()
            validation_config = global_config.agent.get('agent_s_web', {}).get('validation', {})
            
            validation_execution_config = ExecutionConfig(
                model_name=validation_config.get('model_name', 'gpt-4o-mini'),
                model_provider=validation_config.get('model_provider', 'openai'),
                temperature=validation_config.get('temperature', 0.0),
                max_tokens=validation_config.get('max_tokens', 1000),  # 增加 token 限制
                response_format=validation_config.get('response_format', 'json'),
                timeout=30,
                max_retries=2,
                retry_delay=1.0,
                require_citations=False,
                require_reasoning=True,
                max_context_length=4000,
                support_images=True,
                fallback_to_structured=True
            )
            
            logger.info(f"🔍 Using VALIDATION model for final evaluation: {validation_execution_config.model_name}")
            
            # Create a multimodal agent for validation that can process images
            from agent_framework.agent_s2_5_web import LMMAgent
            
            validation_agent = LMMAgent(
                engine_params={
                    "model": validation_execution_config.model_name,
                    "engine_type": validation_execution_config.model_provider,
                    "temperature": validation_execution_config.temperature
                },
                system_prompt="You are an expert web task evaluator. Analyze the screenshot and execution details to determine if the task was completed successfully."
            )
            
            # Add the validation prompt and screenshot (if not in text-only mode)
            if text_only_mode:
                validation_agent.add_message(
                    text_content=prompt,
                    image_content=None,
                    role="user"
                )
                logger.info("📝 Text-only mode: Validation without screenshot")
            else:
                validation_agent.add_message(
                    text_content=prompt,
                    image_content=screenshot,
                    role="user"
                )
            
            # Get response from validation agent
            response_text, tokens_used = await validation_agent.get_response(
                temperature=validation_execution_config.temperature,
                use_validation_model=True,
                task_id=task.task_id
            )
            
            # Create a mock response object for compatibility
            class MockResponse:
                def __init__(self, answer, tokens_used):
                    self.answer = answer
                    self.tokens_used = tokens_used
                    self.success = True
            
            response = MockResponse(response_text, tokens_used)
            
            # Parse validation result with improved error handling
            try:
                import json
                import re
                
                # Log the raw response for debugging
                logger.debug(f"🔍 Raw validation response: {response.answer}")
                
                # Clean the response - remove markdown code blocks if present
                answer_text = response.answer.strip()
                if answer_text.startswith('```json'):
                    answer_text = answer_text[7:]  # Remove ```json
                if answer_text.endswith('```'):
                    answer_text = answer_text[:-3]  # Remove ```
                answer_text = answer_text.strip()
                
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', answer_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    validation_result = json.loads(json_str)
                else:
                    # Try parsing the entire response
                    validation_result = json.loads(answer_text)
                
                logger.info(f"✅ Task validation completed: {validation_result.get('task_completed', False)}")
                return validation_result
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse validation result as JSON: {e}")
                logger.warning(f"Raw response was: {response.answer}")
                
                # Try to extract information from the response text
                answer_lower = response.answer.lower()
                task_completed = "completed" in answer_lower or "success" in answer_lower or "true" in answer_lower
                confidence = 0.7 if task_completed else 0.3
                
                return {
                    "task_completed": task_completed,
                    "confidence": confidence,
                    "reasoning": f"JSON parsing failed, extracted from text: {response.answer[:200]}...",
                    "missing_actions": [],
                    "final_state_analysis": "Unable to analyze due to parsing error"
                }
                
        except Exception as e:
            logger.error(f"Task validation failed: {e}")
            return {
                "task_completed": trajectory.success,
                "confidence": 0.3,
                "reasoning": f"Validation failed: {str(e)}",
                "missing_actions": [],
                "final_state_analysis": "Unable to analyze due to validation error"
            }

    def _create_execution_result(self, task: WebTaskInstance, trajectory: WebExecutionTrajectory, 
                                executor_info: Dict[str, Any], total_tokens: int = 0, 
                                validation_result: Dict[str, Any] = None, step_count: int = 0, 
                                max_steps: int = 8) -> ExecutionResult:
        """Create ExecutionResult compatible with Graph2Eval"""
        # Build reasoning path from actions
        reasoning_path = []
        for action in trajectory.actions_executed:
            reasoning_path.append(f"Executed: {action.get('action', 'unknown')}")
        
        # Add reflection if available
        if executor_info.get('reflection'):
            reasoning_path.append(f"Reflection: {executor_info['reflection']}")
        
        # Use validation result if available, otherwise fall back to trajectory success
        if validation_result:
            final_success = validation_result.get('task_completed', trajectory.success)
            confidence = validation_result.get('confidence', 1.0 if final_success else 0.0)
            reasoning = validation_result.get('reasoning', 'No reasoning provided')
            
            if final_success:
                answer = f"Task completed successfully. {reasoning}"
            else:
                missing_actions = validation_result.get('missing_actions', [])
                answer = f"Task failed. {reasoning}"
                if missing_actions:
                    answer += f" Missing actions: {', '.join(missing_actions)}"
        else:
            final_success = trajectory.success
            confidence = 1.0 if trajectory.success else 0.0
            if trajectory.success:
                answer = f"Task completed successfully. Executed {len(trajectory.actions_executed)} actions."
            else:
                answer = f"Task failed. Error: {trajectory.error_message or 'Unknown error'}"
        
        return ExecutionResult(
            task_id=task.task_id,
            success=final_success,
            answer=answer,
            citations=[],
            reasoning_path=reasoning_path,
            confidence=confidence,
            execution_time=trajectory.end_time - trajectory.start_time,
            tokens_used=total_tokens,  # Use actual token usage
            model_used=self.engine_params.get('model', 'unknown'),
            retries_needed=0,
            error_type="execution_error" if (trajectory.error_message and trajectory.error_message.strip()) else None,  # Only set if there's a non-empty error message
            error_message=trajectory.error_message if (trajectory.error_message and trajectory.error_message.strip()) else None,
            raw_response=json.dumps({
                "executor_info": executor_info,
                "actions_count": len(trajectory.actions_executed),
                "total_tokens_used": total_tokens,
                "validation_result": validation_result
            }),
            web_task_type=None  # Remove web_task_type dependency, use task_info instead
        )
    
    def _create_failed_result(self, task: WebTaskInstance, trajectory: WebExecutionTrajectory, 
                             error_message: str) -> Tuple[ExecutionResult, WebExecutionTrajectory]:
        """Create failed ExecutionResult"""
        execution_result = ExecutionResult(
            task_id=task.task_id,
            success=False,
            answer=f"Task failed: {error_message}",
            citations=[],
            reasoning_path=[f"Error: {error_message}"],
            confidence=0.0,
            execution_time=trajectory.end_time - trajectory.start_time,
            tokens_used=0,
            model_used=self.engine_params.get('model', 'unknown'),
            retries_needed=0,
            error_type="execution_error",
            error_message=error_message,
            raw_response=json.dumps({"error": error_message})
        )
        return execution_result, trajectory


# Example usage and testing
async def main():
    """Example usage of Agent S 2.5 Web"""
    config = {
        'headless': False,
        'use_browser': True
    }
    
    agent = AgentS2_5Web(config)
    await agent.initialize_browser()
    
    try:
        # Create a simple test task
        from task_craft.task_generator import WebTaskInstance
        test_task = WebTaskInstance(
            task_id="test_task_001",
            prompt="Navigate to Google and search for 'Agent S'",
            web_task_type="navigation",
            start_page="https://www.google.com"
        )
        
        result, trajectory = await agent.execute_web_task(test_task)
        print(f"Task execution result: {result.success}")
        print(f"Answer: {result.answer}")
        
    finally:
        await agent.close_browser()


if __name__ == "__main__":
    asyncio.run(main())
