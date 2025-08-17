"""
Web Agent - Main orchestrator for web-based task execution with browser automation
"""

import asyncio
import time
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

from ingestion.web_collector import WebCollector, WebPageData
from graph_rag.graph_builder import GraphBuilder, WebGraphBuildConfig, DocumentGraph
from task_craft.task_generator import TaskGenerator, WebTaskInstance, WebTaskStep
from agent_framework.evaluators import WebTaskEvaluator, WebTaskExecutionResult
from agent_framework.executors import ExecutionResult, LLMExecutor

try:
    from playwright.async_api import async_playwright, Browser, Page, ElementHandle
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available, falling back to simulation mode")


@dataclass
class WebAction:
    """Represents a web action that can be executed"""
    action_type: str  # click, type, navigate, wait, scroll, etc.
    target_selector: str = ""  # CSS selector or XPath
    target_text: str = ""  # Text to find element by
    target_mark: str = ""  # SoM mark ID (e.g., "M1", "M2")
    input_value: str = ""  # Value to input for type actions
    url: str = ""  # URL for navigation
    wait_time: float = 1.0  # Time to wait
    description: str = ""
    expected_result: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target_selector": self.target_selector,
            "target_text": self.target_text,
            "target_mark": self.target_mark,
            "input_value": self.input_value,
            "url": self.url,
            "wait_time": self.wait_time,
            "description": self.description,
            "expected_result": self.expected_result
        }


@dataclass
class WebExecutionTrajectory:
    """Records the execution trajectory of a web task"""
    task_id: str
    start_time: float
    end_time: float = 0.0
    actions_executed: List[WebAction] = field(default_factory=list)
    action_results: List[Dict[str, Any]] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    page_info_files: List[str] = field(default_factory=list)  # Added page info files
    current_url: str = ""
    success: bool = False
    error_message: str = ""
    final_state: Dict[str, Any] = field(default_factory=dict)
    
    def add_action(self, action: WebAction, result: Dict[str, Any]):
        self.actions_executed.append(action)
        self.action_results.append(result)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "actions_executed": [action.to_dict() for action in self.actions_executed],
            "action_results": self.action_results,
            "screenshots": self.screenshots,
            "page_info_files": self.page_info_files,  # Added page info files
            "current_url": self.current_url,
            "success": self.success,
            "error_message": self.error_message,
            "final_state": self.final_state
        }


class WebAgent:
    """Main orchestrator for web-based task execution with browser automation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None):
        self.config = config or {}
        
        # Initialize LLMExecutor (singleton)
        from agent_framework.executors import ExecutionConfig
        execution_config = ExecutionConfig(
            model_name=self.config.get('execution', {}).get('model_name', 'gpt-4o-mini'),
            temperature=self.config.get('execution', {}).get('temperature', 0.1),
            max_tokens=self.config.get('execution', {}).get('max_tokens', 4000)
        )
        self.llm_executor = LLMExecutor.get_instance(execution_config)
        
        # Initialize components
        self.web_collector = WebCollector(self.config.get('web_collection', {}))
        self.web_graph_builder = GraphBuilder(WebGraphBuildConfig())
        self.web_task_generator = TaskGenerator(self.config.get('web_task_generation', {}))
        self.web_evaluator = WebTaskEvaluator(self.config.get('evaluator', {}))
        
        # Browser automation
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.use_browser = PLAYWRIGHT_AVAILABLE and self.config.get('use_browser', True)
        
        # Output directory - use provided output_dir (required for benchmark integration)
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Fallback for standalone usage
            self.output_dir = Path('output/web_agent')
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.collected_pages: List[WebPageData] = []
        self.web_graph: Optional[DocumentGraph] = None
        self.generated_tasks: List[WebTaskInstance] = []
        self.execution_results: List[WebTaskExecutionResult] = []
        self.execution_trajectories: List[WebExecutionTrajectory] = []
    
    async def initialize_browser(self):
        """Initialize browser for automation"""
        if not self.use_browser:
            logger.info("Browser automation disabled, using simulation mode")
            return
        
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.config.get('headless', True),
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            self.page = await self.browser.new_page()
            
            # Set viewport and user agent
            await self.page.set_viewport_size({"width": 1280, "height": 720})
            await self.page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
            
            logger.info("Browser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            self.use_browser = False
    
    async def close_browser(self):
        """Close browser"""
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
    
    async def execute_web_task(self, task: WebTaskInstance) -> Tuple[ExecutionResult, WebExecutionTrajectory]:
        """Execute a web task using browser automation"""
        # Type check to ensure we have a WebTaskInstance
        if not isinstance(task, WebTaskInstance):
            raise TypeError(f"Expected WebTaskInstance, got {type(task).__name__}")
        """Execute a web task with browser automation"""
        
        trajectory = WebExecutionTrajectory(
            task_id=task.task_id,
            start_time=time.time()
        )
        
        try:
            # Check if browser is available
            if not self.page:
                raise Exception("Browser page is not available. Please ensure browser is properly initialized.")
            
            # Use real browser automation
            result, trajectory = await self._execute_with_browser(task, trajectory)
            
            return result, trajectory
            
        except Exception as e:
            logger.error(f"Error executing web task {task.task_id}: {e}")
            trajectory.end_time = time.time()
            trajectory.success = False
            trajectory.error_message = str(e)
            
            result = ExecutionResult(
                task_id=task.task_id,
                success=False,
                answer=f"Error: {str(e)}",
                execution_time=time.time() - trajectory.start_time,
                model_used="gpt-4o-mini",
                tokens_used=0,
                citations=[],
                reasoning_path=[],
                confidence=0.0,
                error_type="execution_error",
                error_message=str(e)
            )
            
            # Add web_task_type to the result
            result.web_task_type = getattr(task, 'web_task_type', 'Search')
            
            return result, trajectory
    
    async def _execute_with_browser(self, task: WebTaskInstance, trajectory: WebExecutionTrajectory) -> Tuple[ExecutionResult, WebExecutionTrajectory]:
        """Execute task using LLM-guided browser automation"""
        
        # Print task content before execution
        import json
        task_info = {
            "task_id": task.task_id,
                            "task_type": getattr(task, 'web_task_type', 'unknown'),
            "task_description": task.prompt,
            "task_difficulty": task.difficulty,
            "task_steps": [
                {
                    "step_number": i,
                    "action_description": step.action_description,
                    "step_type": step.step_type,
                    "expected_result": step.expected_result
                }
                for i, step in enumerate(task.task_steps, 1)
            ],
            "start_page_url": getattr(task, 'start_page_url', None),
            "user_intent": getattr(task, 'user_intent', None)
        }
        
        logger.info(f"🤖 Starting Task Execution:")
        logger.info(f"📋 Task Information:")
        logger.info(f"{'='*80}")
        logger.info(json.dumps(task_info, indent=2, ensure_ascii=False))
        logger.info(f"{'='*80}")
        
        logger.info(f"Executing task {task.task_id} with LLM-guided browser automation")
        
        try:
            total_tokens_used = 0
            reasoning_path = []
            
            # Step 1: Navigate to start page if specified
            if task.start_page_url:
                await self._execute_action(WebAction(
                    action_type="navigate",
                    url=task.start_page_url,
                    description=f"Navigate to start page: {task.start_page_url}"
                ), trajectory)
                

            
            # Step 2: LLM-guided task execution loop
            max_iterations = self.config.get('max_iterations', 20)  # Get from config
            max_actions = self.config.get('max_actions_per_task', 15)  # Get from config
            completion_threshold = self.config.get('task_completion_threshold', 0.8)  # Get from config
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"LLM-guided execution iteration {iteration}")
                
                # Check if we've exceeded maximum actions
                if len(trajectory.actions_executed) >= max_actions:
                    logger.warning(f"Reached maximum actions limit ({max_actions})")
                    break
                
                # Get current page state
                page_info = await self._get_page_info()
                
                # Check if page is loaded and has content before using SoM
                current_url = page_info.get('url', '')
                page_title = page_info.get('title', '')
                interactive_elements = page_info.get('interactive_elements', [])
                
                # Only use SoM if page is loaded and has interactive elements
                use_som = (current_url and 
                           current_url != 'about:blank' and 
                           page_title and 
                           page_title != 'Unknown' and
                           len(interactive_elements) > 0)
                
                if use_som:
                    # Create SoM screenshot with marked elements
                    som_screenshot_path = await self._create_som_screenshot()
                    if som_screenshot_path:
                        screenshot_path = som_screenshot_path
                        logger.info(f"Using SoM screenshot: {screenshot_path}")
                    else:
                        # Fallback to regular screenshot if SoM fails
                        screenshot_path = await self._take_screenshot(f"{task.task_id}_iteration_{iteration}")
                        logger.info(f"Using regular screenshot: {screenshot_path}")
                else:
                    # Use regular screenshot for blank or loading pages
                    screenshot_path = await self._take_screenshot(f"{task.task_id}_iteration_{iteration}")
                    logger.info(f"Page not ready for SoM (URL: {current_url}, Title: {page_title}, Elements: {len(interactive_elements)}), using regular screenshot: {screenshot_path}")
                
                trajectory.screenshots.append(screenshot_path)
                
                # Save page information
                page_info_path = await self._save_page_info(f"{task.task_id}_iteration_{iteration}")
                if page_info_path:
                    trajectory.page_info_files = getattr(trajectory, 'page_info_files', [])
                    trajectory.page_info_files.append(page_info_path)
                
                # Use LLM to analyze current state and determine next action or completion
                planning_result = await self._plan_next_action(task, page_info, iteration, screenshot_path)
                total_tokens_used += planning_result.get("tokens_used", 0)
                reasoning_path.append(planning_result.get("reasoning", ""))
                
                # Check if LLM determined task is completed
                if planning_result.get("task_completed", False):
                    logger.info("LLM determined task is completed")
                    break
                
                # Execute the planned action
                action = planning_result.get("action")
                if action:
                    await self._execute_action(action, trajectory)
                else:
                    logger.warning("LLM did not provide a valid action")
                    break
            
            # Final task completion validation
            final_success = await self._validate_task_completion_llm(task)
            total_tokens_used += final_success.get("tokens_used", 0)
            
            trajectory.end_time = time.time()
            trajectory.success = final_success.get("completed", False)
            trajectory.current_url = self.page.url if self.page else ""
            
            # Create execution result
            result = ExecutionResult(
                task_id=task.task_id,
                success=final_success.get("completed", False),
                answer=final_success.get("explanation", f"Task completed with {len(trajectory.actions_executed)} actions"),
                execution_time=trajectory.end_time - trajectory.start_time,
                model_used="gpt-4o-mini",
                tokens_used=total_tokens_used,
                citations=[],
                reasoning_path=reasoning_path,
                confidence=final_success.get("confidence", 0.0),
                error_type=None if final_success.get("completed", False) else "task_incomplete",
                error_message="" if final_success.get("completed", False) else "Task was not completed successfully"
            )
            
            # Add web_task_type to the result
            result.web_task_type = getattr(task, 'web_task_type', 'unknown')
            
            return result, trajectory
            
        except Exception as e:
            logger.error(f"Browser execution failed: {e}")
            trajectory.end_time = time.time()
            trajectory.success = False
            trajectory.error_message = str(e)
            
            result = ExecutionResult(
                task_id=task.task_id,
                success=False,
                answer=f"Browser execution error: {str(e)}",
                execution_time=time.time() - trajectory.start_time,
                model_used="browser_automation",
                tokens_used=0,
                citations=[],
                reasoning_path=[],
                confidence=0.0,
                error_type="browser_error",
                error_message=str(e)
            )
            
            # Add web_task_type to the result
            result.web_task_type = getattr(task, 'web_task_type', 'Search')
            
            return result, trajectory
    

    
    def _convert_step_to_action(self, step: WebTaskStep) -> WebAction:
        """Convert a task step to a web action with SoM support"""
        
        action = WebAction(
            action_type=step.step_type,
            description=step.action_description,
            expected_result=step.expected_result
        )
        
        # Extract SoM mark for all action types that might need it
        mark_match = re.search(r'M\d+', step.action_description)
        if mark_match:
            action.target_mark = mark_match.group(0)
        
        # Extract action-specific information
        if step.step_type == "navigation":
            # First try to use target_page_url, then extract URL from description
            if step.target_page_url:
                action.url = step.target_page_url
            else:
                # Extract URL from action description
                url_match = re.search(r'https?://[^\s]+', step.action_description)
                if url_match:
                    action.url = url_match.group(0)
                else:
                    action.url = ""
        elif step.step_type == "click":
            # Try to extract selector from action description
            action.target_text = self._extract_target_from_description(step.action_description)
        elif step.step_type == "input":
            # Extract input value and target
            action.input_value = step.input_data.get("value", "")
            action.target_text = self._extract_target_from_description(step.action_description)
        elif step.step_type == "scroll":
            # Extract scroll direction or target
            action.target_text = self._extract_target_from_description(step.action_description)
        elif step.step_type == "wait":
            # Extract wait time if specified
            wait_match = re.search(r'(\d+(?:\.\d+)?)', step.action_description)
            if wait_match:
                action.wait_time = float(wait_match.group(1))
        elif step.step_type == "extract":
            # Extract target for extraction
            action.target_text = self._extract_target_from_description(step.action_description)
        
        return action
    
    def _extract_target_from_description(self, description: str) -> str:
        """Extract target element from action description with improved logic"""
        
        # Look for quoted text first
        quotes = re.findall(r'"([^"]*)"', description)
        if quotes:
            return quotes[0]
        
        # Look for specific patterns with better context
        patterns = [
            # Click patterns
            r'click\s+(?:the\s+)?([^,\s]+(?:\s+[^,\s]+)*?)(?:\s+button|\s+link|\s+form)?',
            r'click\s+on\s+(?:the\s+)?([^,\s]+(?:\s+[^,\s]+)*?)',
            # Input patterns
            r'use\s+(?:the\s+)?([^,\s]+(?:\s+[^,\s]+)*?)(?:\s+bar|\s+field|\s+input)?',
            r'enter\s+(?:.*?)\s+in\s+(?:the\s+)?([^,\s]+(?:\s+[^,\s]+)*?)(?:\s+field|\s+input|\s+bar)?',
            r'type\s+(?:.*?)\s+in\s+(?:the\s+)?([^,\s]+(?:\s+[^,\s]+)*?)(?:\s+field|\s+input|\s+bar)?',
            # Selection patterns
            r'select\s+(?:.*?)\s+from\s+(?:the\s+)?([^,\s]+(?:\s+[^,\s]+)*?)',
            r'choose\s+(?:.*?)\s+from\s+(?:the\s+)?([^,\s]+(?:\s+[^,\s]+)*?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description.lower())
            if match:
                target = match.group(1).strip()
                # Clean up common words and improve target
                target = re.sub(r'\b(the|a|an|this|that|and|or|but)\b', '', target).strip()
                if target and len(target) > 1:  # Avoid single characters
                    return target
        
        # Look for specific keywords and extract meaningful context
        keyword_patterns = [
            (r'search\s+(?:button|bar|field)', 'search'),
            (r'login\s+(?:button|form|link)', 'login'),
            (r'submit\s+(?:button|form)', 'submit'),
            (r'next\s+(?:button|link|page)', 'next'),
            (r'previous\s+(?:button|link|page)', 'previous'),
            (r'back\s+(?:button|link)', 'back'),
            (r'continue\s+(?:button|link)', 'continue'),
            (r'cancel\s+(?:button|link)', 'cancel'),
            (r'ok\s+(?:button|link)', 'ok'),
            (r'yes\s+(?:button|link)', 'yes'),
            (r'no\s+(?:button|link)', 'no'),
        ]
        
        for pattern, keyword in keyword_patterns:
            if re.search(pattern, description.lower()):
                return keyword
        
        # Fallback: look for specific keywords and extract more context
        keywords = ["button", "link", "form", "input", "search", "submit", "login", "register"]
        for keyword in keywords:
            if keyword in description.lower():
                # Find words around the keyword
                words = description.lower().split()
                try:
                    keyword_index = words.index(keyword)
                    # Get words before and after the keyword
                    context_words = []
                    if keyword_index > 0:
                        context_words.append(words[keyword_index - 1])
                    if keyword_index < len(words) - 1:
                        context_words.append(words[keyword_index + 1])
                    
                    # Return the most meaningful word
                    for word in context_words:
                        if len(word) > 2 and word not in ['the', 'a', 'an', 'this', 'that', 'and', 'or', 'but']:
                            return word
                except ValueError:
                    continue
        
        # If no specific target found, try to extract any meaningful word
        words = description.lower().split()
        meaningful_words = [word for word in words if len(word) > 3 and word not in ['the', 'a', 'an', 'this', 'that', 'and', 'or', 'but', 'use', 'the', 'search', 'bar', 'enter', 'keyword', 'related', 'document', 'looking', 'for']]
        
        if meaningful_words:
            return meaningful_words[0]
        
        return ""
    
    async def _execute_action(self, action: WebAction, trajectory: WebExecutionTrajectory):
        """Execute a single web action with robust element finding"""
        
        if not self.page:
            raise Exception("Browser page not initialized")
        
        result = {"success": False, "error": "Unknown action type"}
        
        try:
            if action.action_type == "navigate":
                if not action.url or action.url.strip() == "":
                    result = {"success": False, "error": "No URL provided for navigation"}
                    logger.error("❌ Navigation failed: No URL provided")
                else:
                    await self.page.goto(action.url)
                    result = {"success": True, "url": action.url}
                    logger.info(f"✅ Navigated to: {action.url}")
                
            elif action.action_type == "click":
                logger.info(f"🔍 Executing click action: target_text='{action.target_text}', target_mark='{action.target_mark}'")
                
                try:
                    # Priority 1: Check if action has target_mark (SoM approach)
                    if hasattr(action, 'target_mark') and action.target_mark and hasattr(self, 'current_som_mapping'):
                        mark = action.target_mark
                        if mark in self.current_som_mapping:
                            element_info = self.current_som_mapping[mark]
                            # Use viewport coordinates for clicking
                            coords = (element_info["rect_viewport"]["centerX"], element_info["rect_viewport"]["centerY"])
                            
                            logger.info(f"✅ Clicking mark {mark} at coordinates {coords}")
                            logger.info(f"🔍 Element info: tag={element_info.get('tag')}, text='{element_info.get('text')}', xpath={element_info.get('xpath')}")
                            
                            # Click at the specified coordinates
                            await self.page.mouse.click(coords[0], coords[1])
                            
                            # Wait a moment for the click to take effect
                            await asyncio.sleep(1.0)
                            
                            # Verify the click had an effect (e.g., URL change, form submission, etc.)
                            try:
                                # Check if URL changed (for navigation clicks)
                                current_url = self.page.url
                                # Check if any form was submitted or page content changed
                                # This is a basic verification - could be enhanced
                                result = {"success": True, "clicked": f"clicked mark {mark} at coordinates {coords}"}
                                logger.info(f"✅ Click verification: URL is {current_url}")
                                
                                # Additional verification: check if page content changed
                                try:
                                    page_title = await self.page.title()
                                    logger.info(f"✅ Page title after click: {page_title}")
                                except Exception as title_error:
                                    logger.warning(f"⚠️ Could not get page title: {title_error}")
                                    
                            except Exception as verify_error:
                                logger.warning(f"⚠️ Could not verify click effect: {verify_error}")
                                result = {"success": True, "clicked": f"clicked mark {mark} at coordinates {coords} (verification failed)"}
                            
                            # Cleanup SoM markers after successful click
                            await self._cleanup_som_markers()
                        else:
                            result = {"success": False, "error": f"Mark {mark} not found in mapping"}
                    else:
                        # Priority 2: Fallback to traditional element finding
                        logger.info(f"🔍 No target_mark provided, using traditional element finding")
                        element = await self._find_clickable_element(action)
                        if element:
                            # Check if element is SoM mark result (fallback)
                            if isinstance(element, dict) and element.get("type") == "som_mark":
                                mark = element["mark"]
                                element_info = element["element_info"]
                                coords = (element_info["rect_viewport"]["centerX"], element_info["rect_viewport"]["centerY"])
                                
                                logger.info(f"✅ Found element by SoM analysis, clicking mark {mark} at coordinates {coords}")
                                
                                # Click at the specified coordinates
                                await self.page.mouse.click(coords[0], coords[1])
                                result = {"success": True, "clicked": f"clicked mark {mark} at coordinates {coords}"}
                                
                                # Cleanup SoM markers after successful click
                                await self._cleanup_som_markers()
                                
                            # Check if element is coordinates from visual analysis
                            elif isinstance(element, dict) and element.get("type") == "coordinates":
                                coords = element["coords"]
                                x, y = coords
                                logger.info(f"✅ Found element by visual analysis, clicking at coordinates ({x}, {y})")
                                
                                # Click at the specified coordinates
                                await self.page.mouse.click(x, y)
                                result = {"success": True, "clicked": f"clicked at coordinates ({x}, {y})"}
                                
                            else:
                                # Traditional element-based clicking
                                logger.info(f"✅ Found clickable element, attempting to click")
                                
                                # For ElementHandle objects, we can directly click
                                try:
                                    # Check if element is an ElementHandle
                                    if hasattr(element, 'click'):
                                        # Direct click on ElementHandle
                                        await element.click(timeout=10000)
                                        logger.info(f"✅ Clicked on element directly")
                                        result = {"success": True, "clicked": "element found and clicked directly"}
                                    else:
                                        # Try to get element handle and click
                                        element_handle = await element.element_handle()
                                        if element_handle:
                                            await element_handle.click(timeout=10000)
                                            logger.info(f"✅ Clicked on element via element handle")
                                            result = {"success": True, "clicked": "element found and clicked via element handle"}
                                        else:
                                            # Fallback to direct click
                                            await element.click(timeout=10000)
                                            result = {"success": True, "clicked": "element found and clicked using fallback"}
                                except Exception as click_error:
                                    # Fallback to direct click if any method fails
                                    logger.warning(f"⚠️ Click method failed: {str(click_error)}, trying direct click")
                                    try:
                                        await element.click(timeout=10000)
                                        result = {"success": True, "clicked": "element found and clicked using direct click"}
                                    except Exception as final_error:
                                        result = {"success": False, "error": f"All click methods failed: {str(final_error)}"}
                                        logger.error(f"❌ All click methods failed: {str(final_error)}")
                        else:
                            result = {"success": False, "error": f"Could not find clickable element for: {action.target_text}"}
                            logger.error(f"❌ Could not find clickable element for: {action.target_text}")
                    
                    logger.info(f"✅ Click action completed successfully")
                except Exception as e:
                    result = {"success": False, "error": f"Click failed: {str(e)}"}
                    logger.error(f"❌ Click action failed: {str(e)}")
                
            elif action.action_type == "input":
                logger.info(f"🔍 Simulating input process: '{action.input_value}'")
                
                try:
                    # Step 1: Find and click on input field first
                    logger.info(f"🔍 Step 1: Looking for input field to click")
                    
                    clicked_element = None
                    
                    # Priority 1: Try SoM mark if available
                    if hasattr(action, 'target_mark') and action.target_mark and hasattr(self, 'current_som_mapping'):
                        mark = action.target_mark
                        if mark in self.current_som_mapping:
                            element_info = self.current_som_mapping[mark]
                            coords = (element_info["rect_viewport"]["centerX"], element_info["rect_viewport"]["centerY"])
                            
                            logger.info(f"✅ Clicking input mark {mark} at coordinates {coords}")
                            
                            # Click at the specified coordinates
                            await self.page.mouse.click(coords[0], coords[1])
                            clicked_element = "som_mark"
                            
                            # Don't cleanup SoM markers immediately - wait until after input is complete
                    
                    # Priority 2: Try traditional element finding if SoM not available
                    if not clicked_element:
                        # Try to find input elements
                        input_elements = await self.page.query_selector_all('input, textarea, [contenteditable="true"]')
                        logger.info(f"🔍 Found {len(input_elements)} potential input elements")
                        
                        # Try to click on the first visible input element
                        for i, element in enumerate(input_elements):
                            try:
                                if await element.is_visible():
                                    logger.info(f"🔍 Trying to click on input element {i+1}")
                                    
                                    # Get element position
                                    bbox = await element.bounding_box()
                                    if bbox:
                                        # Click at the center of the element
                                        x = bbox['x'] + bbox['width'] / 2
                                        y = bbox['y'] + bbox['height'] / 2
                                        
                                        # Simulate mouse click
                                        await self.page.mouse.click(x, y)
                                        logger.info(f"✅ Clicked on input element {i+1} at ({x}, {y})")
                                        
                                        clicked_element = element
                                        break
                                    else:
                                        logger.warning(f"⚠️ Could not get bounding box for element {i+1}")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to click on element {i+1}: {e}")
                                continue
                    
                    # Step 2: If we found and clicked an element, now input the text
                    if clicked_element:
                        logger.info(f"🔍 Step 2: Inputting text after clicking")
                        
                        # Wait a moment for focus
                        await asyncio.sleep(0.5)
                        
                        # Clear any existing content (Ctrl+A)
                        try:
                            await self.page.keyboard.press("Control+a")
                            await asyncio.sleep(0.2)
                            logger.info(f"✅ Cleared existing content")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not clear content: {e}")
                        
                        # Type the input value
                        try:
                            await self.page.keyboard.type(action.input_value, delay=100)
                            logger.info(f"✅ Typed '{action.input_value}' into input field")
                            
                            # Don't scroll after input - keep page position stable
                            logger.info("✅ Input completed, keeping page position stable")
                            
                            # Cleanup SoM markers after input is complete
                            if clicked_element == "som_mark":
                                await self._cleanup_som_markers()
                            
                            result = {"success": True, "input": action.input_value, "method": "click_then_type"}
                        except Exception as e:
                            logger.error(f"❌ Failed to type text: {e}")
                            result = {"success": False, "error": f"Failed to type text: {str(e)}"}
                    else:
                        # Fallback: Try clicking in the center of the page and typing
                        logger.info(f"🔍 Fallback: No input elements found, trying center page click")
                        try:
                            # Get page dimensions
                            viewport = self.page.viewport_size
                            if viewport:
                                # Click in the center of the page
                                x = viewport['width'] / 2
                                y = viewport['height'] / 2
                                
                                await self.page.mouse.click(x, y)
                                logger.info(f"✅ Clicked in center of page at ({x}, {y})")
                                
                                # Wait a moment
                                await asyncio.sleep(0.5)
                                
                                # Type the input value
                                await self.page.keyboard.type(action.input_value, delay=100)
                                logger.info(f"✅ Typed '{action.input_value}' after center click")
                                
                                result = {"success": True, "input": action.input_value, "method": "center_page_click_type"}
                            else:
                                result = {"success": False, "error": "Could not get viewport size"}
                        except Exception as e:
                            logger.error(f"❌ Center page click failed: {e}")
                            result = {"success": False, "error": f"Center page click failed: {str(e)}"}
                        
                except Exception as e:
                    result = {"success": False, "error": f"Input simulation failed: {str(e)}"}
                    logger.error(f"❌ Input simulation failed: {str(e)}")
                
            elif action.action_type == "wait":
                await asyncio.sleep(action.wait_time)
                result = {"success": True, "waited": action.wait_time}
                
            elif action.action_type == "scroll":
                # Support both SoM-based scrolling and general scrolling
                if hasattr(action, 'target_mark') and action.target_mark and hasattr(self, 'current_som_mapping'):
                    mark = action.target_mark
                    if mark in self.current_som_mapping:
                        element_info = self.current_som_mapping[mark]
                        coords = (element_info["rect_viewport"]["centerX"], element_info["rect_viewport"]["centerY"])
                        
                        logger.info(f"✅ Scrolling to mark {mark} at coordinates {coords}")
                        
                        # Scroll to the element position
                        await self.page.evaluate(f"window.scrollTo(0, {coords[1] - 100})")
                        result = {"success": True, "scrolled": f"to mark {mark} at coordinates {coords}"}
                    else:
                        result = {"success": False, "error": f"Mark {mark} not found in mapping"}
                else:
                    # Default scrolling behavior
                    await self.page.evaluate("window.scrollBy(0, 500)")
                    result = {"success": True, "scrolled": "down"}
                
            elif action.action_type == "navigation":
                # Handle navigation action (same as navigate)
                if action.url:
                    await self.page.goto(action.url)
                    result = {"success": True, "url": action.url}
                else:
                    # Try to extract URL from description
                    url_match = re.search(r'https?://[^\s]+', action.description)
                    if url_match:
                        url = url_match.group(0)
                        await self.page.goto(url)
                        result = {"success": True, "url": url}
                    else:
                        result = {"success": False, "error": "No URL found for navigation"}
                
            elif action.action_type == "extract":
                # Handle extract action - try to extract information from the page
                try:
                    # Support both SoM-based extraction and general extraction
                    if hasattr(action, 'target_mark') and action.target_mark and hasattr(self, 'current_som_mapping'):
                        mark = action.target_mark
                        if mark in self.current_som_mapping:
                            element_info = self.current_som_mapping[mark]
                            xpath = element_info.get("xpath", "")
                            
                            logger.info(f"✅ Extracting content from mark {mark} using xpath: {xpath}")
                            
                            # Extract content from the specific element
                            element_content = await self.page.evaluate(f"""
                                () => {{
                                    try {{
                                        const element = document.evaluate('{xpath}', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                        if (element) {{
                                            return {{
                                                text: element.textContent?.trim() || '',
                                                tag: element.tagName?.toLowerCase() || '',
                                                value: element.value || '',
                                                href: element.href || ''
                                            }};
                                        }}
                                        return null;
                                    }} catch (e) {{
                                        return null;
                                    }}
                                }}
                            """)
                            
                            if element_content:
                                result = {"success": True, "extracted": f"Content from mark {mark}", "content": element_content}
                            else:
                                result = {"success": False, "error": f"Could not extract content from mark {mark}"}
                        else:
                            result = {"success": False, "error": f"Mark {mark} not found in mapping"}
                    else:
                        # General page content extraction
                        page_content = await self.page.content()
                        result = {"success": True, "extracted": "Page content available", "content_length": len(page_content)}
                except Exception as e:
                    result = {"success": False, "error": f"Extraction failed: {str(e)}"}
                
            else:
                raise Exception(f"Unknown action type: {action.action_type}")
            
            # Wait a bit after each action
            await asyncio.sleep(0.5)
            
        except Exception as e:
            result = {"success": False, "error": str(e)}
            logger.error(f"Action execution failed: {e}")
        
        trajectory.add_action(action, result)
    
    async def _find_clickable_element(self, action: WebAction):
        """Robustly find a clickable element using multiple strategies"""
        logger.debug(f"🔍 _find_clickable_element: target_text='{action.target_text}'")
        
        if action.target_selector:
            try:
                logger.debug(f"🔍 Trying selector: {action.target_selector}")
                element = await self.page.wait_for_selector(action.target_selector, timeout=5000)
                if element:
                    logger.debug(f"✅ Found element by selector: {action.target_selector}")
                    return element
            except Exception as e:
                logger.debug(f"❌ Selector strategy failed: {action.target_selector} - {str(e)}")
                pass
        
        if not action.target_text:
            logger.debug("❌ No target_text provided")
            return None
        
        # Strategy 1: Try Set-of-Mark (SoM) visual approach
        try:
            logger.debug(f"🔍 Strategy 1: Trying Set-of-Mark visual approach for: '{action.target_text}'")
            click_coords = await self._find_element_by_visual_analysis(action.target_text)
            if click_coords:
                logger.debug(f"✅ Strategy 1 succeeded: found element by visual analysis at {click_coords}")
                return {"type": "coordinates", "coords": click_coords}
        except Exception as e:
            logger.debug(f"❌ Strategy 1 failed: {str(e)}")
            pass
        
        # Strategy 2: Try direct CSS selectors for common buttons
        try:
            logger.debug(f"🔍 Strategy 2: Trying direct CSS selectors for: '{action.target_text}'")
            if action.target_text in ["搜索", "Search", "查找", "Go", "Submit"]:
                selectors = [
                    'button[type="submit"]',
                    'input[type="submit"]',
                    'button',
                    '[role="button"]'
                ]
                
                for selector in selectors:
                    try:
                        elements = await self.page.query_selector_all(selector)
                        for element in elements:
                            try:
                                text_content = await element.text_content()
                                if text_content:
                                    text_content = text_content.strip()
                                    logger.debug(f"🔍 Checking element with text: '{text_content}'")
                                    if action.target_text in text_content:
                                        logger.debug(f"✅ Strategy 2 succeeded: found element by selector '{selector}' with text '{text_content}'")
                                        return element
                            except:
                                continue
                    except:
                        continue
        except Exception as e:
            logger.debug(f"❌ Strategy 2 failed: {str(e)}")
            pass
        
        # Strategy 3: Try exact text match
        try:
            logger.debug(f"🔍 Strategy 3: Trying exact text match: '{action.target_text}'")
            element = self.page.get_by_text(action.target_text, exact=True).first
            if element:
                logger.debug(f"✅ Strategy 3 succeeded: found element by exact text match")
                return element
        except Exception as e:
            logger.debug(f"❌ Strategy 3 failed: {str(e)}")
            pass
        
        # Strategy 4: Try partial text match
        try:
            logger.debug(f"🔍 Strategy 4: Trying partial text match: '{action.target_text}'")
            element = self.page.get_by_text(action.target_text).first
            if element:
                logger.debug(f"✅ Strategy 4 succeeded: found element by partial text match")
                return element
        except Exception as e:
            logger.debug(f"❌ Strategy 4 failed: {str(e)}")
            pass
        
        return None
    
    async def _find_element_by_visual_analysis(self, target_text: str) -> Optional[Dict[str, Any]]:
        """Find element by visual analysis using Set-of-Mark (SoM) approach"""
        try:
            # Create SoM screenshot with marked elements
            som_screenshot_path = await self._create_som_screenshot()
            
            if not som_screenshot_path:
                logger.warning("Could not create SoM screenshot")
                return None
            
            # Create SoM prompt for LLM
            prompt = f"""You are analyzing a web page screenshot with marked elements (Set-of-Mark approach). 

The screenshot contains clickable elements marked with red circular labels (M1, M2, M3, etc.). Each label is positioned above a clickable element.

Your task is to find the element that matches the text "{target_text}".

Look at the marked elements in the screenshot and identify which one contains the text "{target_text}" or similar variations.

IMPORTANT: You must respond with valid JSON only. Do not include any other text.

Respond in this exact JSON format:
{{
    "selected_mark": "M3",
    "text": "element text",
    "confidence": 85,
    "reasoning": "why this element matches the target text"
}}

If no matching element is found, return exactly:
{{
    "selected_mark": null
}}

Do not include any explanations, markdown formatting, or additional text outside the JSON."""

            # Use LLM with SoM image to analyze the screenshot
            response = self.llm_executor.execute_simple_with_image(prompt, [som_screenshot_path])
            
            if response.success:
                try:
                    # Try to extract JSON from the response
                    json_match = re.search(r'\{.*\}', response.answer, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        json_str = response.answer
                    
                    result = json.loads(json_str)
                    selected_mark = result.get("selected_mark")
                    
                    if selected_mark and hasattr(self, 'current_som_mapping'):
                        confidence = result.get("confidence", 0)
                        text = result.get("text", "")
                        
                        logger.info(f"🔍 SoM analysis found element: '{text}' with mark {selected_mark} and confidence {confidence}%")
                        
                        if confidence >= 70:  # Only use if confidence is high enough
                            # Get element info from mapping
                            if selected_mark in self.current_som_mapping:
                                element_info = self.current_som_mapping[selected_mark]
                                logger.info(f"🔍 Using element info from mark mapping: {selected_mark}")
                                return {
                                    "type": "som_mark",
                                    "mark": selected_mark,
                                    "element_info": element_info
                                }
                            else:
                                logger.warning(f"Mark {selected_mark} not found in mapping")
                        else:
                            logger.warning(f"Confidence too low ({confidence}%) for element '{text}'")
                    else:
                        logger.info("No matching element found in SoM analysis")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse SoM analysis response: {e}")
                    logger.debug(f"Raw response: {response.answer}")
                    
                    # Try to extract mark manually if JSON parsing fails
                    try:
                        # Look for mark patterns in the response (M1, M2, etc.)
                        mark_match = re.search(r'M(\d+)', response.answer)
                        if mark_match:
                            mark = f"M{mark_match.group(1)}"
                            if hasattr(self, 'current_som_mapping') and mark in self.current_som_mapping:
                                element_info = self.current_som_mapping[mark]
                                logger.info(f"🔍 Extracted mark manually: {mark}")
                                return {
                                    "type": "som_mark",
                                    "mark": mark,
                                    "element_info": element_info
                                }
                    except:
                        pass
            else:
                logger.error(f"SoM analysis failed: {response.error_message}")
                
        except Exception as e:
            logger.error(f"SoM analysis error: {e}")
        
        return None
    
    async def _create_som_screenshot(self) -> Optional[str]:
        """Create a Set-of-Mark screenshot with DOM-injected markers"""
        try:
            # Check if page is ready for SoM
            if not self.page:
                logger.warning("No page available for SoM")
                return None
                
            # Check page state
            try:
                current_url = self.page.url
                if not current_url or current_url == 'about:blank':
                    logger.info("Page not ready for SoM - blank or loading")
                    return None
            except Exception as e:
                logger.warning(f"Could not check page URL: {e}")
                return None
            
            # Inject SoM markers into the DOM
            mark_mapping = await self._inject_som_markers()
            
            if not mark_mapping:
                logger.info("No clickable elements found for SoM - page may be loading or have no interactive elements")
                return None
            
            # Take screenshot with injected markers
            som_screenshot_path = await self._take_screenshot("som_marked")
            
            # Store the mark mapping for later use
            self.current_som_mapping = mark_mapping
            
            logger.info(f"Created SoM screenshot with {len(mark_mapping)} marked elements: {som_screenshot_path}")
            return som_screenshot_path
            
        except Exception as e:
            logger.error(f"Failed to create SoM screenshot: {e}")
            return None
    
    async def _inject_som_markers(self):
        """Inject SoM markers into DOM and return mark mapping - 精简版优化"""
        try:
            logger.info("🔍 Starting optimized SoM marker injection...")
            
            # First, remove any existing markers
            await self.page.evaluate("""
                () => {
                    const old = document.getElementById('som-overlay');
                    if (old && old.__som_cleanup__) {
                        old.__som_cleanup__();
                    }
                }
            """)
            
            # Inject CSS for optimized marker styling with color coding
            await self.page.evaluate("""
                () => {
                    if (!document.getElementById('som-styles')) {
                        const style = document.createElement('style');
                        style.id = 'som-styles';
                        style.textContent = `
                            .som-wrapper {
                                position: absolute;
                                border: 2px solid;
                                background: transparent;
                                pointer-events: none;
                                z-index: 10000;
                            }
                            .som-wrapper.clickable {
                                border-color: #ff3b30; /* 红色 - 点击类 */
                            }
                            .som-wrapper.input {
                                border-color: #007aff; /* 蓝色 - 输入类 */
                            }
                            .som-wrapper.select {
                                border-color: #34c759; /* 绿色 - 选择类 */
                            }
                            .som-wrapper.navigation {
                                border-color: #ff9500; /* 橙色 - 导航/分页类 */
                            }
                            .som-wrapper.result {
                                border-color: #af52de; /* 紫色 - 结果信息类 */
                            }
                            .som-label {
                                position: absolute;
                                top: -8px; left: -8px;
                                font-weight: 700;
                                font-size: 11px;
                                line-height: 14px;
                                padding: 2px 6px;
                                border-radius: 999px;
                                border: 2px solid rgba(255,255,255,0.9);
                                text-shadow: 0 1px 1px rgba(0,0,0,0.3);
                                pointer-events: none;
                                user-select: none;
                                min-width: 16px;
                                text-align: center;
                            }
                            .som-label.clickable {
                                background: #ff3b30;
                                color: #fff;
                            }
                            .som-label.input {
                                background: #007aff;
                                color: #fff;
                            }
                            .som-label.select {
                                background: #34c759;
                                color: #fff;
                            }
                            .som-label.navigation {
                                background: #ff9500;
                                color: #fff;
                            }
                            .som-label.result {
                                background: #af52de;
                                color: #fff;
                            }
                        `;
                        document.head.appendChild(style);
                    }
                }
            """)
            
            # Get optimized elements and inject markers
            mark_mapping = await self.page.evaluate("""
                () => {
                    const markMapping = {};
                    let markCounter = 1;
                    
                    // 1. 点击类 (红色标签 🔴) - 最关键的操作元素
                    const clickableSelectors = [
                        'button:not([type="submit"])', // 主按钮，排除提交按钮（避免重复）
                        'a[href]:not([href="#"])', // 导航/跳转链接，排除空链接
                        '[role="button"]', // 语义化按钮
                        'input[type="submit"]', // 提交按钮
                        'input[type="button"]', // 按钮类型输入
                        'button[type="submit"]', // 提交按钮
                        // 显著的可点击元素
                        'div[onclick]:not([onclick*="return false"])',
                        'span[onclick]:not([onclick*="return false"])',
                        '[data-clickable="true"]',
                        // 搜索相关按钮
                        'button[class*="search"]:not([disabled])',
                        'button[class*="submit"]:not([disabled])',
                        'button[class*="btn"]:not([disabled])'
                    ];
                    
                    // 2. 输入类 (蓝色标签 🔵) - 用户输入元素
                    const inputSelectors = [
                        'input[type="text"]',
                        'input[type="search"]',
                        'input[type="email"]',
                        'input[type="password"]',
                        'input[type="url"]',
                        'input[type="tel"]',
                        'input[type="number"]',
                        'input:not([type])', // 默认文本输入
                        'input[type=""]', // 空类型输入
                        'textarea',
                        '[contenteditable="true"]'
                    ];
                    
                    // 3. 选择类 (绿色标签 🟢) - 下拉框和选择元素
                    const selectSelectors = [
                        'select',
                        'input[type="radio"]',
                        'input[type="checkbox"]'
                    ];
                    
                    // 4. 导航/分页类 (橙色标签 🟠) - 页面导航元素
                    const navigationSelectors = [
                        'a[href*="page"]', // 分页链接
                        'a[href*="next"]',
                        'a[href*="prev"]',
                        'a[href*="previous"]',
                        'button[class*="next"]',
                        'button[class*="prev"]',
                        'button[class*="previous"]',
                        '[role="menuitem"]',
                        '[role="navigation"]',
                        'nav a',
                        '.pagination a',
                        '.pagination button'
                    ];
                    
                    // 5. 结果信息类 (紫色标签 🟣) - 关键信息元素（限制数量）
                    const resultSelectors = [
                        'h1', 'h2', 'h3', // 标题
                        '.product-title', '.item-title', '.result-title',
                        '.search-result h3', '.search-result h4',
                        'table th', 'table td:first-child', // 表格关键数据
                        '.price', '.cost', '.amount',
                        '.description', '.summary'
                    ];
                    
                    // 定义元素类型和对应的选择器
                    const elementTypes = [
                        { type: 'clickable', selectors: clickableSelectors, maxElements: 15 },
                        { type: 'input', selectors: inputSelectors, maxElements: 10 },
                        { type: 'select', selectors: selectSelectors, maxElements: 8 },
                        { type: 'navigation', selectors: navigationSelectors, maxElements: 6 },
                        { type: 'result', selectors: resultSelectors, maxElements: 5 }
                    ];
                    
                    // 处理每种类型的元素
                    elementTypes.forEach(({ type, selectors, maxElements }) => {
                        let elementCount = 0;
                        
                        selectors.forEach(selector => {
                            if (elementCount >= maxElements) return;
                            
                            document.querySelectorAll(selector).forEach(el => {
                                if (elementCount >= maxElements) return;
                                
                                const rect = el.getBoundingClientRect();
                                const style = window.getComputedStyle(el);
                                
                                // 过滤条件：可见、有尺寸、在视口内
                                if (rect.width > 0 && rect.height > 0 && 
                                    style.display !== 'none' && style.visibility !== 'hidden' &&
                                    rect.top >= 0 && rect.left >= 0 &&
                                    rect.bottom <= window.innerHeight + 100 && // 允许稍微超出视口
                                    rect.right <= window.innerWidth + 100) {
                                    
                                    // 额外的过滤条件
                                    const isVisible = rect.width >= 8 && rect.height >= 8;
                                    const isNotOverlapping = !isElementOverlapping(el, markMapping);
                                    
                                    if (isVisible && isNotOverlapping) {
                                        // Create wrapper
                                        const wrapper = document.createElement('div');
                                        wrapper.className = `som-wrapper ${type}`;
                                        wrapper.style.left = rect.left + 'px';
                                        wrapper.style.top = rect.top + 'px';
                                        wrapper.style.width = rect.width + 'px';
                                        wrapper.style.height = rect.height + 'px';
                                        
                                        // Create label
                                        const label = document.createElement('div');
                                        label.className = `som-label ${type}`;
                                        label.textContent = `M${markCounter}`;
                                        wrapper.appendChild(label);
                                        
                                        // Add to body
                                        document.body.appendChild(wrapper);
                                        
                                        // Store mapping with type information
                                        markMapping[`M${markCounter}`] = {
                                            type: type,
                                            tag: el.tagName.toLowerCase(),
                                            text: el.textContent?.trim() || el.value || el.placeholder || '',
                                            xpath: getXPath(el),
                                            rect_viewport: {
                                                x: rect.left,
                                                y: rect.top,
                                                width: rect.width,
                                                height: rect.height,
                                                centerX: Math.round(rect.left + rect.width / 2),
                                                centerY: Math.round(rect.top + rect.height / 2)
                                            }
                                        };
                                        
                                        markCounter++;
                                        elementCount++;
                                    }
                                }
                            });
                        });
                    });
                    
                    // Helper function to check if element overlaps with existing markers
                    function isElementOverlapping(element, existingMappings) {
                        const rect = element.getBoundingClientRect();
                        
                        for (const mapping of Object.values(existingMappings)) {
                            const existingRect = mapping.rect_viewport;
                            
                            // Check for overlap
                            if (rect.left < existingRect.x + existingRect.width &&
                                rect.left + rect.width > existingRect.x &&
                                rect.top < existingRect.y + existingRect.height &&
                                rect.top + rect.height > existingRect.y) {
                                return true;
                            }
                        }
                        return false;
                    }
                    
                    // Helper function to get XPath
                    function getXPath(element) {
                        if (element.id !== '') {
                            return `//*[@id="${element.id}"]`;
                        }
                        if (element === document.body) {
                            return '/html/body';
                        }
                        let ix = 0;
                        const siblings = element.parentNode.childNodes;
                        for (let sibling of siblings) {
                            if (sibling === element) {
                                return getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                            }
                            if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                                ix++;
                            }
                        }
                    }
                    
                    return markMapping;
                }
            """)
            
            logger.info(f"✅ Optimized SoM marker injection completed, found {len(mark_mapping)} elements")
            
            # Log element type distribution
            type_counts = {}
            for mapping in mark_mapping.values():
                element_type = mapping.get('type', 'unknown')
                type_counts[element_type] = type_counts.get(element_type, 0) + 1
            logger.info(f"📊 Element type distribution: {type_counts}")
            
            return mark_mapping
            
        except Exception as e:
            logger.error(f"Failed to inject optimized SoM markers: {e}")
            return {}

    
    async def _cleanup_som_markers(self):
        """Remove SoM markers from DOM"""
        try:
            await self.page.evaluate("""
                () => {
                    const overlay = document.getElementById('som-overlay');
                    if (overlay && overlay.__som_cleanup__) {
                        overlay.__som_cleanup__();
                    }
                }
            """)
        except Exception as e:
            logger.error(f"Failed to cleanup SoM markers: {e}")
    
    async def _find_input_element(self, action: WebAction):
        """Robustly find an input element using multiple strategies"""
        logger.debug(f"🔍 _find_input_element: target_text='{action.target_text}'")
        
        if action.target_selector:
            try:
                logger.debug(f"🔍 Trying selector: {action.target_selector}")
                return await self.page.wait_for_selector(action.target_selector, timeout=5000)
            except:
                logger.debug(f"❌ Selector strategy failed: {action.target_selector}")
                pass
        
        if not action.target_text:
            logger.debug("❌ No target_text provided")
            return None
        
        # Strategy 1: Try by label (exact match)
        try:
            logger.debug(f"🔍 Strategy 1: Trying label (exact): '{action.target_text}'")
            element = self.page.get_by_label(action.target_text, exact=True).first
            if element:
                # Check if the element is actually an input
                tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
                logger.debug(f"🔍 Strategy 1 found element with tag: {tag_name}")
                if tag_name in ['input', 'textarea']:
                    logger.debug(f"✅ Strategy 1 succeeded: found input element by label (exact)")
                    return element
                else:
                    logger.debug(f"⚠️ Strategy 1 found non-input element with tag: {tag_name}")
        except Exception as e:
            logger.debug(f"❌ Strategy 1 failed: {str(e)}")
            pass
        
        # Strategy 2: Try by label (partial match)
        try:
            logger.debug(f"🔍 Strategy 2: Trying label (partial): '{action.target_text}'")
            element = self.page.get_by_label(action.target_text).first
            if element:
                # Check if the element is actually an input
                tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
                logger.debug(f"🔍 Strategy 2 found element with tag: {tag_name}")
                if tag_name in ['input', 'textarea']:
                    logger.debug(f"✅ Strategy 2 succeeded: found input element by label (partial)")
                    return element
                else:
                    logger.debug(f"⚠️ Strategy 2 found non-input element with tag: {tag_name}")
        except Exception as e:
            logger.debug(f"❌ Strategy 2 failed: {str(e)}")
            pass
        
        # Strategy 3: Try by placeholder
        try:
            element = await self.page.wait_for_selector(f'[placeholder*="{action.target_text}"]', timeout=3000)
            if element:
                return element
        except:
            pass
        
        # Strategy 4: Try by name attribute
        try:
            element = await self.page.wait_for_selector(f'[name*="{action.target_text.lower()}"]', timeout=3000)
            if element:
                return element
        except:
            pass
        
        # Strategy 5: Try by ID containing the text
        try:
            element = await self.page.wait_for_selector(f'#{action.target_text.lower().replace(" ", "-")}', timeout=3000)
            if element:
                return element
        except:
            pass
        
        # Strategy 6: Try by class name containing the text
        try:
            element = await self.page.wait_for_selector(f'.{action.target_text.lower().replace(" ", "-")}', timeout=3000)
            if element:
                return element
        except:
            pass
        
        # Strategy 7: Try common input selectors with longer timeout
        common_selectors = [
            'input[type="text"]',
            'input[type="search"]',
            'textarea',
            'input:not([type="hidden"]):not([type="submit"]):not([type="button"])'
        ]
        
        for selector in common_selectors:
            try:
                logger.debug(f"🔍 Strategy 7: Trying selector: {selector}")
                elements = await self.page.query_selector_all(selector)
                if elements:
                    logger.debug(f"🔍 Found {len(elements)} elements with selector: {selector}")
                    # Return the first visible input element
                    for i, element in enumerate(elements):
                        if await element.is_visible():
                            logger.debug(f"✅ Strategy 7 succeeded: found visible element {i+1} with selector: {selector}")
                            return element
            except Exception as e:
                logger.debug(f"❌ Strategy 7 failed for selector {selector}: {str(e)}")
                continue
        
        # Strategy 8: Try search-specific selectors
        search_selectors = [
            'input[placeholder*="搜索"]',
            'input[placeholder*="search"]',
            'input[placeholder*="Search"]',
            'input[name*="search"]',
            'input[name*="q"]',
            'input[id*="search"]',
            'input[class*="search"]'
        ]
        
        for selector in search_selectors:
            try:
                logger.debug(f"🔍 Strategy 8: Trying search selector: {selector}")
                element = await self.page.wait_for_selector(selector, timeout=3000)
                if element:
                    logger.debug(f"✅ Strategy 8 succeeded: found search element with selector: {selector}")
                    return element
            except Exception as e:
                logger.debug(f"❌ Strategy 8 failed for selector {selector}: {str(e)}")
                continue
        
        # Strategy 9: Try to find any input element as last resort
        try:
            logger.debug(f"🔍 Strategy 9: Trying to find any input element")
            element = await self.page.wait_for_selector('input, textarea', timeout=5000)
            if element:
                logger.debug(f"✅ Strategy 9 succeeded: found any input element")
                return element
        except Exception as e:
            logger.debug(f"❌ Strategy 9 failed: {str(e)}")
            pass
        
        logger.debug(f"❌ All strategies failed to find input element for: '{action.target_text}'")
        return None
    
    async def _take_screenshot(self, name: str) -> str:
        """Take a screenshot and save it"""
        if not self.page:
            return ""
        
        screenshot_dir = self.output_dir / "screenshots"
        screenshot_dir.mkdir(exist_ok=True)
        
        screenshot_path = screenshot_dir / f"{name}_{int(time.time())}.png"
        await self.page.screenshot(path=str(screenshot_path))
        
        return str(screenshot_path)
    
    async def _get_page_info(self) -> Dict[str, Any]:
        """Get comprehensive page information including DOM, metadata, and state"""
        if not self.page:
            return {}
        
        try:
            # Get basic page information with error handling
            page_info = {}
            
            try:
                page_info["url"] = str(self.page.url)
            except Exception as e:
                page_info["url"] = f"Error getting URL: {str(e)}"
            
            try:
                page_info["title"] = await self.page.title()
            except Exception as e:
                page_info["title"] = f"Error getting title: {str(e)}"
            
            try:
                page_info["viewport"] = await self.page.viewport_size()
            except Exception as e:
                page_info["viewport"] = {"error": str(e)}
            
            page_info["timestamp"] = time.time()
            
            # Get page content and DOM
            page_info["html_content"] = await self.page.content()
            
            # Get page metadata
            page_info["metadata"] = await self.page.evaluate("""
                () => {
                    const meta = {};
                    document.querySelectorAll('meta').forEach(el => {
                        const name = el.getAttribute('name') || el.getAttribute('property');
                        const content = el.getAttribute('content');
                        if (name && content) {
                            meta[name] = content;
                        }
                    });
                    return meta;
                }
            """)
            
            # Get interactive elements with detailed explanations
            page_info["interactive_elements"] = await self.page.evaluate("""
                () => {
                    const elements = [];
                    const selectors = [
                        'button', 'input', 'select', 'textarea', 'a[href]',
                        '[onclick]', '[role="button"]', '[role="link"]'
                    ];
                    
                    selectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach(el => {
                            // Determine element purpose and description
                            let purpose = '';
                            let description = '';
                            
                            if (el.tagName.toLowerCase() === 'input') {
                                const type = el.type || 'text';
                                const placeholder = el.placeholder || '';
                                const value = el.value || '';
                                
                                if (type === 'text' || type === 'search') {
                                    purpose = 'text_input';
                                    description = `Text input field${placeholder ? ' with placeholder: ' + placeholder : ''}${value ? ' containing: ' + value : ''}`;
                                } else if (type === 'submit' || type === 'button') {
                                    purpose = 'button';
                                    description = `Submit/Button element${value ? ' with text: ' + value : ''}`;
                                } else if (type === 'password') {
                                    purpose = 'password_input';
                                    description = 'Password input field';
                                } else if (type === 'email') {
                                    purpose = 'email_input';
                                    description = 'Email input field';
                                } else {
                                    purpose = 'input';
                                    description = `${type} input field`;
                                }
                            } else if (el.tagName.toLowerCase() === 'textarea') {
                                purpose = 'textarea';
                                description = `Multi-line text area${el.placeholder ? ' with placeholder: ' + el.placeholder : ''}`;
                            } else if (el.tagName.toLowerCase() === 'button') {
                                purpose = 'button';
                                description = `Button element with text: ${el.textContent?.trim() || ''}`;
                            } else if (el.tagName.toLowerCase() === 'a') {
                                purpose = 'link';
                                description = `Link element${el.textContent?.trim() ? ' with text: ' + el.textContent.trim() : ''}${el.href ? ' pointing to: ' + el.href : ''}`;
                            } else if (el.tagName.toLowerCase() === 'select') {
                                purpose = 'dropdown';
                                description = `Dropdown/Select element with ${el.options?.length || 0} options`;
                            } else {
                                purpose = 'interactive';
                                description = `${el.tagName.toLowerCase()} element`;
                            }
                            
                            // Get element position for reference
                            const rect = el.getBoundingClientRect();
                            
                            elements.push({
                                tag: el.tagName.toLowerCase(),
                                text: el.textContent?.trim() || '',
                                type: el.type || '',
                                id: el.id || '',
                                class: el.className || '',
                                value: el.value || '',
                                placeholder: el.placeholder || '',
                                visible: el.offsetParent !== null,
                                selector: selector,
                                purpose: purpose,
                                description: description,
                                position: {
                                    x: Math.round(rect.left),
                                    y: Math.round(rect.top),
                                    width: Math.round(rect.width),
                                    height: Math.round(rect.height)
                                },
                                attributes: {
                                    name: el.name || '',
                                    required: el.required || false,
                                    disabled: el.disabled || false,
                                    readonly: el.readOnly || false,
                                    maxlength: el.maxLength || null,
                                    minlength: el.minLength || null
                                }
                            });
                        });
                    });
                    
                    return elements;
                }
            """)
            
            # Get form information
            page_info["forms"] = await self.page.evaluate("""
                () => {
                    const forms = [];
                    document.querySelectorAll('form').forEach(form => {
                        const inputs = [];
                        form.querySelectorAll('input, select, textarea').forEach(input => {
                            inputs.push({
                                type: input.type || input.tagName.toLowerCase(),
                                name: input.name || '',
                                id: input.id || '',
                                placeholder: input.placeholder || '',
                                value: input.value || '',
                                required: input.required || false
                            });
                        });
                        
                        forms.push({
                            id: form.id || '',
                            action: form.action || '',
                            method: form.method || 'get',
                            inputs: inputs
                        });
                    });
                    
                    return forms;
                }
            """)
            
            return page_info
            
        except Exception as e:
            logger.error(f"Failed to get page info: {e}")
            return {"error": str(e)}
    
    async def _save_page_info(self, name: str) -> str:
        """Save comprehensive page information to file"""
        if not self.page:
            return ""
        
        try:
            page_info = await self._get_page_info()
            
            # Save to JSON file in page_info subdirectory
            info_dir = self.output_dir / "page_info"
            info_dir.mkdir(exist_ok=True)
            
            info_path = info_dir / f"{name}_{int(time.time())}.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(page_info, f, indent=2, ensure_ascii=False)
            
            return str(info_path)
            
        except Exception as e:
            logger.error(f"Failed to save page info: {e}")
            return ""
    
    async def _plan_next_action(self, task: WebTaskInstance, page_info: Dict[str, Any], iteration: int, screenshot_path: str = None) -> Dict[str, Any]:
        """Use LLM to plan the next action based on current page state with SoM analysis"""
        
        # Use the provided screenshot_path if available, otherwise create one
        if not screenshot_path:
            # Create SoM screenshot with marked elements
            som_screenshot_path = await self._create_som_screenshot()
            if som_screenshot_path:
                screenshot_path = som_screenshot_path
                logger.info(f"Created SoM screenshot for LLM analysis: {screenshot_path}")
            else:
                # Fallback to regular screenshot if SoM fails
                screenshot_path = await self._take_screenshot(f"{task.task_id}_iteration_{iteration}")
                logger.info(f"Using regular screenshot for LLM analysis: {screenshot_path}")
        
        # If we have a screenshot and SoM mapping, use SoM analysis
        if screenshot_path and hasattr(self, 'current_som_mapping') and self.current_som_mapping:
            prompt = f"""You are a web automation agent using Set-of-Mark (SoM) visual analysis. You can see a screenshot of the current web page with interactive elements marked with red circular labels (M1, M2, M3, etc.).

## TASK INFORMATION
- **Task ID**: {task.task_id}
- **Task Type**: {getattr(task, 'web_task_type', 'unknown')}
- **Task Description**: {task.prompt}
- **Current Iteration**: {iteration}

## CURRENT PAGE STATE
- **URL**: {page_info.get('url', 'Unknown')}
- **Title**: {page_info.get('title', 'Unknown')}

## AVAILABLE INTERACTIVE ELEMENTS (Marked in Screenshot)
```json
{json.dumps(self.current_som_mapping, indent=2, ensure_ascii=False)}
```

## SEARCH BUTTON IDENTIFICATION GUIDE
- **Search buttons typically have**: text like "搜索", "Search", "Go", "Find", "Submit", or search icons (🔍, ⚡)
- **Look for buttons near input fields** that would execute the search
- **Avoid clicking on input fields themselves** - only click actual search/submit buttons
- **Check the element text/description** to confirm it's a search button before clicking
- **Common search button patterns**: 
  - Text: "搜索", "Search", "Go", "Find", "Submit", "查询"
  - Icons: 🔍, ⚡, or similar search symbols
  - Position: Usually next to or below search input fields
- **IMPORTANT**: If an element has empty text (`text: ''`) and is a `div`, it's likely NOT a search button
- **REAL SEARCH BUTTONS**: Must have visible text, icons, or clear button-like appearance

## CLICK EFFECTIVENESS RULES
- **IF CLICK IS INEFFECTIVE**: If clicking a button doesn't change the page (URL remains same, no new content), try a different button
- **AVOID REPEAT CLICKS**: Don't click the same button multiple times if it didn't work the first time
- **BUTTON ALTERNATIVES**: If the first search button doesn't work, try other buttons with search-related text or icons
- **VISUAL FEEDBACK**: Look for buttons with search icons, "搜索", "Search", "Submit", or similar text
- **POSITION HINTS**: Search buttons are usually near the search input field or in the header area
- **TRIAL AND ERROR**: If one button doesn't work, systematically try other potential search buttons

## CURRENT INPUT FIELD STATUS
{chr(10).join([f"- **{elem.get('tag', '')}** (type: {elem.get('type', '')}): value='{elem.get('value', '')}', placeholder='{elem.get('placeholder', '')}'" for elem in page_info.get('interactive_elements', []) if elem.get('tag') in ['input', 'textarea']])}

## INPUT FIELD ANALYSIS
{chr(10).join([f"- Search box {i+1}: '{elem.get('value', '')}' (placeholder: '{elem.get('placeholder', '')}')" for i, elem in enumerate([e for e in page_info.get('interactive_elements', []) if e.get('tag') == 'input' and e.get('type') in ['text', 'search']])])}

## AVAILABLE ACTIONS
1. **click**: Click on a marked element using its mark ID (e.g., M1, M2, M3)
2. **input**: Enter text into an input field using its mark ID
3. **scroll**: Scroll to a specific element using its mark ID
4. **wait**: Wait for a specified amount of time
5. **extract**: Extract content from a specific element using its mark ID
6. **navigate**: Navigate to a specific URL

## WEB OPERATION COMMON SENSE
- **Search Process**: After entering search terms in a search box, you MUST click the search button to execute the search
- **Form Submission**: Input fields alone don't trigger actions - you need to click submit/search buttons
- **Button Identification**: Look for buttons with text like "Search", "Submit", "Go", "Find", or search icons
- **Input Validation**: After typing, verify the text appears in the input field before proceeding
- **Navigation Flow**: Navigate → Input → Click Search → View Results → Click Result
- **CRITICAL**: Just seeing text in an input field does NOT mean the search has been executed - you must click the search button
- **Search Execution**: Only after clicking the search button will you see actual search results

## EXECUTION CONTEXT
- This is iteration {iteration} of task execution
- Previous actions have been performed to reach this state
- **Task Completion Criteria**: Look for search results, document listings, or relevant content that matches the task objectives
- **Input Field Analysis**: Check if search terms have been entered into input fields
- **Next Action Logic**: 
  - If on blank page → wait or navigate to new page
  - If search box has text but no results → click search button (look for elements with text like "搜索", "Search", "Go", "Find", "Submit")
  - If no search box text → input search terms
  - If results found → task may be complete
  - **IMPORTANT**: After inputting text, ALWAYS click the search button to execute the search
  - **SEARCH BUTTON SELECTION**: Choose the element that looks like a search/submit button, not an input field
  - **AVOID**: Elements with empty text (`text: ''`) and `tag: div` - these are likely not clickable buttons

## DECISION PROCESS
1. **Analyze Current State**: What do you see on the page?
2. **Check Task Progress**: Has the task objective been achieved?
3. **Determine Next Action**: What specific action is needed?

## IMPORTANT RULES
- **ALWAYS use mark IDs** (M1, M2, M3, etc.) for element interactions
- **NEVER use text descriptions** for clicking or inputting
- **Check input field values** to understand current state
- **Respond with valid JSON only** - no additional text

## ACTION EXAMPLES
```json
{{
    "action_type": "navigate",
    "url": "https://wenku.baidu.com",
    "description": "Navigate to Baidu Wenku homepage"
}}

{{
    "action_type": "click",
    "target_mark": "M3",
    "description": "Click the search button (M3)"
}}

{{
    "action_type": "input", 
    "target_mark": "M1",
    "input_value": "Data Science Handbook",
    "description": "Input search term into search box (M1)"
}}

{{
    "action_type": "scroll",
    "target_mark": "M5", 
    "description": "Scroll to element (M5)"
}}

{{
    "action_type": "wait",
    "wait_time": 2.0,
    "description": "Wait for page to load"
}}

{{
    "action_type": "extract",
    "target_mark": "M2",
    "description": "Extract content from element (M2)"
}}
```

{{
    "action_type": "extract",
    "target_mark": "M7",
    "description": "Extract content from element (M7)"
}}
```

## YOUR RESPONSE FORMAT
```json
{{
    "task_completed": true/false,
    "reasoning": "Clear explanation of current state and decision",
    "action": {{
        "action_type": "action_type",
        "target_mark": "M#",
        "input_value": "text_if_input",
        "wait_time": number_if_wait,
        "description": "Human-readable action description"
    }}
}}
```"""
        else:
            # Fallback to traditional analysis without SoM
            prompt = f"""You are a web automation agent analyzing page state without visual markers. Use DOM information to determine the next action.

## TASK INFORMATION
- **Task ID**: {task.task_id}
- **Task Type**: {task.web_task_type}
- **Task Description**: {task.prompt}
- **Current Iteration**: {iteration}

## CURRENT PAGE STATE
- **URL**: {page_info.get('url', 'Unknown')}
- **Title**: {page_info.get('title', 'Unknown')}
- **Interactive Elements**: {len(page_info.get('interactive_elements', []))} elements found
- **Forms**: {len(page_info.get('forms', []))} forms found

## INTERACTIVE ELEMENTS DETAILS
{chr(10).join([f"- **{elem.get('purpose', 'unknown')}**: {elem.get('description', '')} | value='{elem.get('value', '')}' | position=({elem.get('position', {}).get('x', 0)}, {elem.get('position', {}).get('y', 0)}) | visible={elem.get('visible', False)}" for elem in page_info.get('interactive_elements', [])[:15]])}

## PAGE CONTENT PREVIEW
{page_info.get('html_content', '')[:500]}...

## WEB OPERATION COMMON SENSE
- **Search Process**: After entering search terms in a search box, you MUST click the search button to execute the search
- **Form Submission**: Input fields alone don't trigger actions - you need to click submit/search buttons
- **Button Identification**: Look for buttons with text like "Search", "Submit", "Go", "Find", or search icons
- **Input Validation**: After typing, verify the text appears in the input field before proceeding
- **Navigation Flow**: Navigate → Input → Click Search → View Results → Click Result
- **CRITICAL**: Just seeing text in an input field does NOT mean the search has been executed - you must click the search button
- **Search Execution**: Only after clicking the search button will you see actual search results

## AVAILABLE ACTIONS
1. **navigate**: Navigate to a specific URL (MUST include url field)
2. **click**: Click on an element using target_text
3. **input**: Enter text into an input field
4. **select**: Select an option from a dropdown
5. **scroll**: Scroll the page
6. **wait**: Wait for a specific condition
7. **extract**: Extract information from the page

## ACTION EXAMPLES
```json
{{
    "action_type": "navigate",
    "url": "https://wenku.baidu.com",
    "description": "Navigate to Baidu Wenku homepage"
}}

{{
    "action_type": "click",
    "target_text": "搜索",
    "description": "Click the search button"
}}

{{
    "action_type": "input",
    "target_text": "search",
    "input_value": "Data Science Handbook",
    "description": "Input search term into search box"
}}

{{
    "action_type": "wait",
    "wait_time": 2.0,
    "description": "Wait for page to load"
}}
```

## ACTION PARAMETERS
- **target_text**: Element identifier (e.g., "Search", "submit button")
- **input_value**: Text to input (for input actions)
- **url**: URL to navigate to (for navigate actions)
- **description**: Human-readable action description

## TASK STEPS (Reference)
{chr(10).join([f"{i+1}. {step.action_description}" for i, step in enumerate(task.task_steps)])}

## DECISION PROCESS
1. **Analyze Current State**: What elements and content are available?
2. **Check Task Progress**: Has the task objective been achieved?
3. **Determine Next Action**: What specific action is needed?

## IMPORTANT RULES
- **Check input field values** to see if text has been entered
- **If on blank page** → navigate to Baidu Wenku (https://wenku.baidu.com)
- **If search box has text but no results** → click search button
- **If no search box text** → input search terms
- **Use target_text** for element identification
- **Respond with valid JSON only**

## YOUR RESPONSE FORMAT
```json
{{
    "task_completed": true/false,
    "reasoning": "Clear explanation of current state and decision",
    "action": {{
        "action_type": "action_type",
        "target_text": "element_identifier",
        "input_value": "text_if_input",
        "wait_time": number_if_wait,
        "description": "Human-readable action description"
    }}
}}
```
"""

        try:
            
            # Use image support if screenshot is available
            if screenshot_path and os.path.exists(screenshot_path):
                response = self.llm_executor.execute_simple_with_image(prompt, [screenshot_path])
            else:
                response = self.llm_executor.execute_simple(prompt)
            
            # Debug: Log LLM response in debug mode
            logger.debug(f"🔍 LLM Response for task {task.task_id}, iteration {iteration}:")
            logger.debug(f"📝 Raw Response:")
            logger.debug(f"{'='*80}")
            logger.debug(response.answer)
            logger.debug(f"{'='*80}")
            logger.debug(f"🔢 Tokens Used: {getattr(response, 'tokens_used', 0)}")
            
            # Parse JSON response
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response.answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response.answer
            
            try:
                result = json.loads(json_str)
                # Debug: Log parsed result
                logger.debug(f"✅ Parsed JSON Result: {json.dumps(result, indent=2)}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
                logger.error(f"Response: {response.answer}")
                # Fallback if JSON parsing fails
                result = {
                    "task_completed": False,
                    "reasoning": f"Failed to parse LLM response: {str(e)}",
                    "action": None
                }
            
            # Convert action dict to WebAction object
            action = None
            if result.get("action"):
                action_data = result["action"]
                action = WebAction(
                    action_type=action_data.get("action_type", "click"),
                    target_text=action_data.get("target_text", ""),
                    target_mark=action_data.get("target_mark", ""),
                    input_value=action_data.get("input_value", ""),
                    url=action_data.get("url", ""),
                    description=action_data.get("description", "")
                )
            
            return {
                "task_completed": result.get("task_completed", False),
                "reasoning": result.get("reasoning", ""),
                "action": action,
                "tokens_used": getattr(response, 'tokens_used', 0)
            }
            
        except Exception as e:
            logger.error(f"Error in LLM planning: {e}")
            return {
                "task_completed": False,
                "reasoning": f"Planning error: {str(e)}",
                "action": None,
                "tokens_used": 0
            }



    async def _validate_task_completion_llm(self, task: WebTaskInstance) -> Dict[str, Any]:
        """Final LLM-based validation of task completion based on objectives"""
        
        # Get final page state
        page_info = await self._get_page_info()
        
        # Extract key information for comprehensive analysis
        page_content = page_info.get('html_content', '')[:3000]  # Limit content length
        interactive_elements = page_info.get('interactive_elements', [])
        
        # Create detailed element summary
        element_summary = []
        for elem in interactive_elements[:15]:  # Limit to first 15 elements
            element_summary.append(f"- {elem.get('tag', '')}: {elem.get('text', '')[:50]} (type: {elem.get('type', '')})")
        
        prompt = f"""You are a web automation agent performing comprehensive final task completion validation. Evaluate whether the task objectives have been fully achieved.

TASK OBJECTIVES:
- Task ID: {task.task_id}
- Task Type: {task.web_task_type}
- Task Description: {task.prompt}
- User Intent: {getattr(task, 'user_intent', task.prompt)}

FINAL PAGE STATE:
- URL: {page_info.get('url', 'Unknown')}
- Page Title: {page_info.get('title', 'Unknown')}
- Page Content Preview: {page_content[:800]}...
- Interactive Elements Found: {len(interactive_elements)} elements
- Key Elements: {chr(10).join(element_summary)}

TASK STEPS (for reference):
{chr(10).join([f"{i+1}. {step.action_description}" for i, step in enumerate(task.task_steps)])}

COMPREHENSIVE EVALUATION CRITERIA:
1. **Objective Achievement**: Does the current page state fulfill the specific task objectives?
2. **Content Relevance**: Is the page content relevant to the task goals?
3. **User Intent Fulfillment**: Has the user's intent been satisfied?
4. **Expected Elements**: Are the expected elements or content present?
5. **Task Completion Indicators**: Are there clear indicators that the task is complete?

SPECIFIC EVALUATION EXAMPLES:
- For search tasks: Check if search results contain the specific terms mentioned in the task
- For download tasks: Check if download buttons, success messages, or file indicators are present
- For comparison tasks: Check if multiple items are displayed for comparison
- For form tasks: Check if form completion indicators or success messages are shown

Respond in JSON format:
{{
    "completed": true/false,
    "confidence": 0.0-1.0,
    "explanation": "comprehensive analysis of task completion based on objectives",
    "quality_score": 0.0-1.0,
    "objective_achievement": "detailed assessment of how well objectives were met",
    "content_relevance": "assessment of content relevance to task goals",
    "issues": ["list of any issues or incomplete aspects"],
    "success_criteria_met": ["list of success criteria that were met"],
    "missing_requirements": ["list of missing requirements or incomplete objectives"]
}}"""

        try:
            # Debug: Log the complete prompt in debug mode
            import logging
            if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"🤖 Final LLM Validation Prompt for task {task.task_id}:")
                logger.debug(f"📝 Complete Prompt:")
                logger.debug(f"{'='*80}")
                logger.debug(prompt)
                logger.debug(f"{'='*80}")
            
            response = self.llm_executor.execute_simple(prompt)
            
            # Debug: Log LLM response in debug mode
            logger.debug(f"🔍 Final LLM Validation Response for task {task.task_id}:")
            logger.debug(f"📝 Raw Response:")
            logger.debug(f"{'='*80}")
            logger.debug(response.answer)
            logger.debug(f"{'='*80}")
            logger.debug(f"🔢 Tokens Used: {getattr(response, 'tokens_used', 0)}")
            
            # Parse JSON response
            try:
                result = json.loads(response.answer)
                # Debug: Log parsed result 
                logger.debug(f"✅ Final Validation Parsed Result: {json.dumps(result, indent=2)}")
            except json.JSONDecodeError:
                logger.error(f"JSON parsing failed for final validation: {response.answer}")
                result = {
                    "completed": False,
                    "confidence": 0.0,
                    "explanation": "Failed to parse LLM response",
                    "quality_score": 0.0,
                    "issues": ["LLM response parsing failed"],
                    "success_criteria_met": []
                }
            
            return {
                "completed": result.get("completed", False),
                "confidence": result.get("confidence", 0.0),
                "explanation": result.get("explanation", ""),
                "quality_score": result.get("quality_score", 0.0),
                "issues": result.get("issues", []),
                "success_criteria_met": result.get("success_criteria_met", []),
                "tokens_used": getattr(response, 'tokens_used', 0)
            }
            
        except Exception as e:
            logger.error(f"Error in final LLM validation: {e}")
            return {
                "completed": False,
                "confidence": 0.0,
                "explanation": f"Final validation error: {str(e)}",
                "quality_score": 0.0,
                "issues": [f"Validation error: {str(e)}"],
                "success_criteria_met": [],
                "tokens_used": 0
            }
    

