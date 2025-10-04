"""
Web Data Collection Module
Handles web page crawling, DOM extraction, and interaction data collection
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import deque
from urllib.parse import urljoin, urlparse
import re

from loguru import logger

# Small model for description refinement
SEQ2SEQ_AVAILABLE = None  # Will be checked when actually needed

def _import_seq2seq_model():
    """Import seq2seq model when needed"""
    global SEQ2SEQ_AVAILABLE
    if SEQ2SEQ_AVAILABLE is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
            SEQ2SEQ_AVAILABLE = True
            return AutoTokenizer, AutoModelForSeq2SeqLM, torch
        except ImportError:
            SEQ2SEQ_AVAILABLE = False
            logger.warning("Transformers not available for description refinement. Install with: pip install transformers torch")
            return None, None, None
    elif SEQ2SEQ_AVAILABLE:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
            return AutoTokenizer, AutoModelForSeq2SeqLM, torch
        except ImportError:
            return None, None, None
    return None, None, None

# Try to import browser automation libraries
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available. Install with: pip install playwright && playwright install")

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("Selenium not available. Install with: pip install selenium")


class DescriptionRefiner:
    """Refines element descriptions using small seq2seq models"""
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Load the seq2seq model"""
        AutoTokenizer, AutoModelForSeq2SeqLM, torch = _import_seq2seq_model()
        if AutoTokenizer is None:
            logger.warning("Transformers not available, using rule-based descriptions only")
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.model = self.model.to(self.device)
            else:
                self.device = torch.device("cpu")
            
            logger.info(f"Loaded description refinement model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            self.tokenizer = None
            self.model = None
    
    def refine_description(self, raw_description: str) -> str:
        """Refine a raw description using the seq2seq model"""
        if self.tokenizer is None or self.model is None:
            return raw_description
        
        try:
            # Import torch here to avoid import issues
            import torch
            
            # Create prompt for refinement
            prompt = f"""Task: Refine this web element description into a clear, concise description.

Input: {raw_description}

Instructions:
- Keep the description short and clear
- Preserve the element type and key information
- Use natural language
- Do not add technical jargon
- Focus on what the element does or represents

Output:"""
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate refined description
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode output
            refined = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output
            refined = refined.strip()
            if not refined:  # Fallback if empty
                return raw_description
            
            # Check if it's a single sentence (no multiple periods or excessive length)
            if refined.count('.') > 1 or len(refined) > 200:  # Fallback if multiple sentences or too long
                return raw_description
            
            return refined
            
        except Exception as e:
            logger.warning(f"Failed to refine description '{raw_description}': {e}")
            return raw_description
    
    def batch_refine_descriptions(self, descriptions: List[str]) -> List[str]:
        """Refine multiple descriptions in batch"""
        if self.tokenizer is None or self.model is None:
            return descriptions
        
        try:
            # Import torch here to avoid import issues
            import torch
            
            # Create prompts
            prompts = [f"Refine this element description into a short, human-readable sentence: {desc}" for desc in descriptions]
            
            # Tokenize inputs
            inputs = self.tokenizer(prompts, return_tensors="pt", max_length=512, truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate refined descriptions
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode outputs
            refined_descriptions = []
            for output in outputs:
                refined = self.tokenizer.decode(output, skip_special_tokens=True).strip()
                if not refined:  # Fallback if empty
                    refined_descriptions.append(descriptions[len(refined_descriptions)])
                elif refined.count('.') > 1 or len(refined) > 200:  # Fallback if multiple sentences or too long
                    refined_descriptions.append(descriptions[len(refined_descriptions)])
                else:
                    refined_descriptions.append(refined)
            
            return refined_descriptions
            
        except Exception as e:
            logger.warning(f"Failed to batch refine descriptions: {e}")
            return descriptions


@dataclass
class WebElement:
    """Represents a web page element with its properties"""
    
    element_id: str
    element_type: str  # button, input, table, link, etc.
    tag_name: str
    text_content: str = ""
    placeholder: str = ""
    value: str = ""
    href: str = ""
    src: str = ""
    
    # Position and styling
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    
    # CSS properties
    css_classes: List[str] = field(default_factory=list)
    css_selector: str = ""
    
    # Interactive properties
    is_clickable: bool = False
    is_input: bool = False
    is_visible: bool = True
    is_enabled: bool = True
    
    # Form-specific properties
    input_type: str = ""  # text, password, email, etc.
    required: bool = False
    options: List[str] = field(default_factory=list)  # for select elements
    
    # Metadata
    attributes: Dict[str, str] = field(default_factory=dict)
    
    # SoM (Set-of-Mark) specific fields
    som_mark: str = ""  # SoM mark ID (e.g., "M1", "M2")
    som_type: str = ""  # SoM element type (clickable, input, select, navigation, result)
    
    # Description field for better element understanding
    description: str = ""  # Human-readable description of the element's purpose
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "tag_name": self.tag_name,
            "text_content": self.text_content,
            "placeholder": self.placeholder,
            "value": self.value,
            "href": self.href,
            "src": self.src,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "css_classes": self.css_classes,
            "css_selector": self.css_selector,
            "is_clickable": self.is_clickable,
            "is_input": self.is_input,
            "is_visible": self.is_visible,
            "is_enabled": self.is_enabled,
            "input_type": self.input_type,
            "required": self.required,
            "options": self.options,
            "attributes": self.attributes,
            "som_mark": self.som_mark,
            "som_type": self.som_type,
            "description": self.description
        }


@dataclass
class WebPageData:
    """Complete web page data including DOM and interaction information"""
    
    url: str
    title: str
    elements: List[WebElement] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)  # Screenshot paths
    
    # Page metadata
    load_time: float = 0.0
    page_size: int = 0
    links: List[str] = field(default_factory=list)
    
    # Interaction data
    clickable_elements: List[str] = field(default_factory=list)
    form_elements: List[str] = field(default_factory=list)
    table_elements: List[str] = field(default_factory=list)
    
    # Page classification
    page_type: str = "content"  # content, form, search, product, etc.
    
    # Website classification
    website_type: str = "unknown"  # crm, ecommerce, blog, news, etc.
    website_description: str = ""  # Description of what this website does
    
    # Exploration metadata
    exploration_depth: int = 0  # Depth level in exploration tree
    
    # LLM assessment data
    llm_assessment: Optional[Dict[str, Any]] = None  # LLM-based difficulty assessment
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "url": self.url,
            "title": self.title,
            "elements": [elem.to_dict() for elem in self.elements],
            "screenshots": self.screenshots,
            "load_time": self.load_time,
            "page_size": self.page_size,
            "links": self.links,
            "clickable_elements": self.clickable_elements,
            "form_elements": self.form_elements,
            "table_elements": self.table_elements,
            "page_type": self.page_type,
            "website_type": self.website_type,
            "website_description": self.website_description,
            "exploration_depth": self.exploration_depth,
            "llm_assessment": self.llm_assessment
        }


class WebCollector:
    """Web page collector for crawling and extracting web data"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Use the passed config directly (it's already the web_collection config)
        web_config = self.config
        
        # Collection settings
        self.max_pages = web_config.get('max_pages', 10)
        self.timeout = web_config.get('timeout', 30)
        self.delay = web_config.get('delay', 1.0)
        
        # Output settings
        self.base_output_dir = Path(self.config.get('output_dir', 'data/web_screenshots'))
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use base output directory directly (no timestamp subdirectory)
        self.output_dir = self.base_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Counter for unique page identification
        self.page_counter = 0
        
        # Browser automation settings
        self.use_playwright = web_config.get('use_playwright', True)  # Prefer Playwright over Selenium
        
        # Enhanced click simulation settings
        self.enable_click_simulation = web_config.get('enable_click_simulation', False)
        self.max_clickable_elements = web_config.get('max_clickable_elements', 15)
        self.wait_after_click = web_config.get('wait_after_click', 2.0)
        self.click_simulation_depth = web_config.get('click_simulation_depth', 2)
        
        
        # LLM-based intelligent crawling settings
        self.use_llm_crawling = web_config.get('use_llm_crawling', False)
        self.llm_model_name = web_config.get('llm_model_name', 'gpt-4o-mini')
        self.llm_temperature = web_config.get('llm_temperature', 0.3)
        self.llm_max_tokens = web_config.get('llm_max_tokens', 1000)
        self.llm_quality_threshold = web_config.get('llm_quality_threshold', 0.6)
        
        # Website type detection LLM settings (separate from crawling)
        self.use_llm_website_type_detection = web_config.get('use_llm_website_type_detection', False)  # Disable by default
        self.website_type_llm_model = web_config.get('website_type_llm_model', 'gpt-4o-mini')
        self.website_type_llm_temperature = web_config.get('website_type_llm_temperature', 0.1)  # Lower temperature for classification
        
        # Link filtering configuration
        exploration_config = web_config.get('exploration', {})
        self.skip_patterns = exploration_config.get('skip_patterns', [
            'javascript:', 'mailto:', 'tel:', 'ftp://',
            '/api/', '/admin/', '/login', '/logout', '/signup',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip',
            '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.avi'
        ])
        
        link_filtering_config = exploration_config.get('link_filtering', {})
        self.skip_anchor_only = link_filtering_config.get('skip_anchor_only', True)
        self.allow_subdomains = link_filtering_config.get('allow_subdomains', True)
        self.allowed_schemes = link_filtering_config.get('allowed_schemes', ['http', 'https', ''])
        self.max_links_per_page = link_filtering_config.get('max_links_per_page', 10)
        
        # Initialize LLM executor for website type detection (separate from crawling)
        self.llm_executor = None
        if self.use_llm_website_type_detection:
            try:
                # Lazy import to avoid circular dependencies
                from agent_framework.executors import LLMExecutor, ExecutionConfig
                execution_config = ExecutionConfig(
                    model_name=self.website_type_llm_model,
                    model_provider="openai",
                    temperature=self.website_type_llm_temperature,
                    max_tokens=1000,
                    timeout=30,
                    max_retries=3,
                    response_format="json"
                )
                self.llm_executor = LLMExecutor(execution_config)
                logger.info(f"✅ LLM website type detection enabled with model: {self.website_type_llm_model}")
            except ImportError:
                logger.warning("⚠️ LLM executor not available for website type detection. Install with: pip install openai anthropic")
                self.use_llm_website_type_detection = False
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize LLM executor for website type detection: {e}")
                self.use_llm_website_type_detection = False
        
        # Initialize LLM executor for crawling (if different from website type detection)
        self.crawling_llm_executor = None
        if self.use_llm_crawling:
            try:
                # Lazy import to avoid circular dependencies
                from agent_framework.executors import LLMExecutor, ExecutionConfig
                execution_config = ExecutionConfig(
                    model_name=self.llm_model_name,
                    model_provider="openai",
                    temperature=self.llm_temperature,
                    max_tokens=2000,
                    timeout=30,
                    max_retries=3,
                    response_format="json"
                )
                self.crawling_llm_executor = LLMExecutor(execution_config)
                logger.info(f"✅ LLM-based crawling enabled with model: {self.llm_model_name}")
            except ImportError:
                logger.warning("⚠️ LLM executor not available for crawling. Install with: pip install openai anthropic")
                self.use_llm_crawling = False
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize LLM executor for crawling: {e}")
                self.use_llm_crawling = False
        
        # Browser automation setup
        self.browser = None
        self.page = None
        self.driver = None
        
        # Check available automation tools
        if self.use_playwright and not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available, falling back to Selenium")
        
        # Initialize description refiner
        self.description_refiner = None
        self.use_description_refinement = web_config.get('use_description_refinement', False)  # Disabled by default
        if self.use_description_refinement:
            try:
                model_name = web_config.get('description_refinement_model', 'google/flan-t5-small')
                self.description_refiner = DescriptionRefiner(model_name)
                logger.info(f"✅ Description refinement enabled with model: {model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize description refiner: {e}")
                self.use_description_refinement = False
        
        if not self.use_playwright and not SELENIUM_AVAILABLE:
            logger.warning("No browser automation available")
            raise RuntimeError("No browser automation tools available (Playwright or Selenium)")
        
        logger.info(f"WebCollector initialized with max_pages={self.max_pages}, timeout={self.timeout}s")
        if self.use_llm_website_type_detection:
            logger.info(f"🔍 LLM website type detection: Enabled (model: {self.website_type_llm_model})")
        else:
            logger.info(f"🔍 Rule-based website type detection: Enabled")
        if self.use_llm_crawling:
            logger.info(f"🤖 LLM-based intelligent crawling: Enabled")
        else:
            logger.info(f"📋 Rule-based crawling: Enabled")
    
    async def collect_web_data(self, urls: List[str]) -> List[WebPageData]:
        """Collect web data from multiple URLs with quality evaluation"""
        
        logger.info(f"Starting web data collection for {len(urls)} URLs")
        
        # Use real browser automation
        if self.use_playwright and PLAYWRIGHT_AVAILABLE:
            pages = await self._collect_with_playwright(urls)
        elif SELENIUM_AVAILABLE:
            pages = await self._collect_with_selenium(urls)
        else:
            raise RuntimeError("No browser automation tools available (Playwright or Selenium)")
    
        # Apply quality evaluation and filtering
        logger.info(f"🔍 Applying quality evaluation to {len(pages)} collected pages...")
        filtered_pages = await self._filter_pages_by_quality(pages)
        
        logger.info(f"✅ Web data collection complete: {len(filtered_pages)}/{len(pages)} pages passed quality filter")
        return filtered_pages
    

    
    async def collect_web_data_with_exploration(self, urls: List[str], max_depth: int = 3, max_pages_per_depth: int = 5) -> List[WebPageData]:
        """
        Collect web data with multi-step cross-page exploration
        
        Args:
            urls: Starting URLs to explore
            max_depth: Maximum depth of exploration (how many clicks deep)
            max_pages_per_depth: Maximum pages to collect at each depth level
            
        Returns:
            List of collected web page data
        """
        logger.info(f"Starting multi-step web exploration from {len(urls)} URLs")
        logger.info(f"Exploration settings: max_depth={max_depth}, max_pages_per_depth={max_pages_per_depth}")
        
        all_pages = []
        visited_urls = set()
        
        try:
            if self.use_playwright and PLAYWRIGHT_AVAILABLE:
                await self._explore_with_playwright(urls, max_depth, max_pages_per_depth, all_pages, visited_urls)
            elif SELENIUM_AVAILABLE:
                await self._explore_with_selenium(urls, max_depth, max_pages_per_depth, all_pages, visited_urls)
            else:
                raise RuntimeError("No browser automation tools available (Playwright or Selenium)")
                
        except Exception as e:
            logger.error(f"Error during web exploration: {e}")
            raise
        
        logger.info(f"Exploration completed. Collected {len(all_pages)} pages across {max_depth} depth levels")
        return all_pages
    
    async def _collect_with_playwright(self, urls: List[str]) -> List[WebPageData]:
        """Collect web data using Playwright"""
        
        web_pages = []
        
        try:
            async with async_playwright() as p:
                # Launch browser
                self.browser = await p.chromium.launch(headless=True)
                self.page = await self.browser.new_page()
                
                # Set viewport
                await self.page.set_viewport_size({"width": 1280, "height": 720})
                
                for i, url in enumerate(urls[:self.max_pages]):
                    logger.info(f"Collecting data with Playwright for {url}")
                    
                    try:
                        # Navigate to page with more flexible waiting strategy
                        try:
                            await self.page.goto(url, wait_until='domcontentloaded', timeout=self.timeout * 1000)
                            # Wait for network to be idle, but with shorter timeout
                            await self.page.wait_for_load_state("networkidle", timeout=15000)
                        except Exception as e:
                            logger.warning(f"Network idle timeout for {url}, continuing with current state: {e}")
                            # Continue even if network doesn't become idle
                        
                        # Wait for page to load
                        await asyncio.sleep(2)
                        
                        # Get page title
                        title = await self.page.title()
                        
                        # Take screenshot
                        self.page_counter += 1
                        screenshot_path = self.output_dir / f"page_{i+1}_{self.page_counter}.png"
                        await self.page.screenshot(path=str(screenshot_path), full_page=True)
                        
                        # Extract DOM elements
                        elements = await self._extract_dom_elements_playwright()
                        
                        # Get page metrics
                        metrics = await self.page.evaluate("""
                            () => {
                                return {
                                    loadTime: performance.timing.loadEventEnd - performance.timing.navigationStart,
                                    pageSize: document.documentElement.outerHTML.length
                                }
                            }
                        """)
                        
                        # Create page data
                        page_data = WebPageData(
                            url=url,
                            title=title,
                            elements=elements,
                            screenshots=[str(screenshot_path)],
                            load_time=metrics.get('loadTime', 0) / 1000.0,
                            page_size=metrics.get('pageSize', 0)
                        )
                        
                        # Save DOM data
                        dom_file = self.output_dir / f"page_{i+1}_dom.json"
                        self._save_dom_data(page_data, dom_file)
                        
                        web_pages.append(page_data)
                        
                        # Delay between requests
                        await asyncio.sleep(self.delay)
                        
                    except Exception as e:
                        logger.error(f"Error collecting data from {url}: {e}")
                        continue
                
                # Close browser
                await self.browser.close()
                
        except Exception as e:
            logger.error(f"Error with Playwright: {e}")
            if self.browser:
                await self.browser.close()
        
        logger.info(f"Collected data for {len(web_pages)} pages using Playwright")
        return web_pages
    
    async def _collect_with_selenium(self, urls: List[str]) -> List[WebPageData]:
        """Collect web data using Selenium"""
        
        web_pages = []
        
        try:
            # Setup Chrome options for faster performance
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")  # Don't load images for faster loading
            # Keep JavaScript enabled for DOM extraction
            chrome_options.add_argument("--window-size=1280,720")
            chrome_options.add_argument("--remote-debugging-port=9222")
            
            # Initialize driver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            self.driver.implicitly_wait(5)  # Reduce implicit wait
            
            for i, url in enumerate(urls[:self.max_pages]):
                logger.info(f"Collecting data with Selenium for {url}")
                
                try:
                    # Navigate to page
                    self.driver.get(url)
                    
                    # Wait for page to load
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                    
                    # Get page title
                    title = self.driver.title
                    
                    # Take screenshot
                    self.page_counter += 1
                    screenshot_path = self.output_dir / f"page_{i+1}_{self.page_counter}.png"
                    self.driver.save_screenshot(str(screenshot_path))
                    
                    # Extract DOM elements
                    elements = self._extract_dom_elements_selenium()
                    
                    # Get page metrics
                    page_size = len(self.driver.page_source)
                    
                    # Create page data
                    page_data = WebPageData(
                        url=url,
                        title=title,
                        elements=elements,
                        screenshots=[str(screenshot_path)],
                        load_time=0.0,  # Selenium doesn't provide easy access to load time
                        page_size=page_size
                    )
                    
                    # Save DOM data
                    dom_file = self.output_dir / f"page_{i+1}_{self.page_counter}_dom.json"
                    self._save_dom_data(page_data, dom_file)
                    
                    web_pages.append(page_data)
                    
                    # Delay between requests
                    await asyncio.sleep(self.delay)
                    
                except Exception as e:
                    logger.error(f"Error collecting data from {url}: {e}")
                    continue
            
            # Close driver
            self.driver.quit()
            
        except Exception as e:
            logger.error(f"Error with Selenium: {e}")
            if self.driver:
                self.driver.quit()
        
        logger.info(f"Collected data for {len(web_pages)} pages using Selenium")
        return web_pages
    
    async def _explore_with_selenium(self, start_urls: List[str], max_depth: int, max_pages_per_depth: int, 
                                   all_pages: List[WebPageData], visited_urls: Set[str]):
        """Explore website using Selenium with intelligent priority-based collection"""
        
        # Initialize browser
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        options.add_argument('--disable-images')
        # options.add_argument('--disable-javascript')  # Keep JS enabled for exploration
        
        driver = None
        try:
            driver = webdriver.Chrome(options=options)
            driver.implicitly_wait(5)
            
            # Use priority queue for intelligent collection
            from queue import PriorityQueue
            page_queue = PriorityQueue()
            
            # Initialize queue, sort starting URLs by priority
            for url in start_urls:
                if url not in visited_urls:
                    # First collect page information to calculate priority
                    try:
                        driver.get(url)
                        await asyncio.sleep(2)
                        
                        title = driver.title
                        content = driver.page_source
                        
                        # Temporarily collect elements to calculate priority
                        elements = self._extract_dom_elements_selenium(driver)
                        
                        priority = self._calculate_page_priority(url, title, content, elements)
                        
                        # Priority queue uses negative numbers because Python PriorityQueue is min-heap
                        page_queue.put((-priority, url, 0))  # (negative priority, URL, depth)
                        
                        logger.info(f"🎯 Page {url} priority: {priority}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to calculate priority for {url}: {e}")
                        # If unable to calculate priority, use default priority
                        page_queue.put((-100, url, 0))
            
            # Start priority collection
            while not page_queue.empty() and len(all_pages) < self.max_pages:
                priority, url, depth = page_queue.get()
                priority = -priority  # Restore positive priority
                
                if url in visited_urls or depth > max_depth:
                    continue
                
                logger.info(f"🔍 Exploring page (priority: {priority}, depth: {depth}): {url}")
                
                # Collect current page
                page_data = await self._collect_single_page_selenium(driver, url, depth)
                if page_data:
                    all_pages.append(page_data)
                    visited_urls.add(url)
                    
                    # If there is still space, explore links
                    if len(all_pages) < self.max_pages and depth < max_depth:
                        # Get explorable links
                        links_to_explore = await self._find_explorable_links_selenium(
                            driver, url, max_pages_per_depth
                        )
                        
                        # Calculate priority for each link and add to queue
                        for link_url in links_to_explore:
                            if link_url not in visited_urls and len(all_pages) < self.max_pages:
                                try:
                                    # Navigate to link page to calculate priority
                                    driver.get(link_url)
                                    await asyncio.sleep(2)
                                    
                                    title = driver.title
                                    content = driver.page_source
                                    
                                    # Temporarily collect elements
                                    elements = self._extract_dom_elements_selenium(driver)
                                    
                                    link_priority = self._calculate_page_priority(link_url, title, content, elements)
                                    
                                    # Depth penalty
                                    link_priority -= depth * 50
                                    
                                    page_queue.put((-link_priority, link_url, depth + 1))
                                    
                                    logger.debug(f"  📎 Link {link_url} priority: {link_priority}")
                                    
                                except Exception as e:
                                    logger.warning(f"Failed to calculate priority for link {link_url}: {e}")
                                    # Use default priority
                                    page_queue.put((-50, link_url, depth + 1))
                
        finally:
            if driver:
                driver.quit()
    
    async def _explore_from_url_selenium(self, driver, url: str, max_depth: int, max_pages_per_depth: int,
                                       all_pages: List[WebPageData], visited_urls: Set[str], current_depth: int):
        """Recursively explore from a URL using Selenium"""
        
        if current_depth >= max_depth or url in visited_urls or len(all_pages) >= self.max_pages:
            return
        
        visited_urls.add(url)
        logger.info(f"Exploring depth {current_depth}: {url}")
        
        try:
            # Navigate to page
            driver.get(url)
            await asyncio.sleep(2)  # Wait for page load
            
            # Collect current page data
            page_data = await self._collect_single_page_selenium(driver, url, current_depth)
            all_pages.append(page_data)
            
            # If we haven't reached max depth, explore links
            if current_depth < max_depth - 1 and len(all_pages) < self.max_pages:
                links_to_explore = await self._find_explorable_links_selenium(driver, url, self.max_links_per_page)
                
                for link_url in links_to_explore:
                    if len(all_pages) >= self.max_pages:
                        break
                    await self._explore_from_url_selenium(driver, link_url, max_depth, max_pages_per_depth,
                                                        all_pages, visited_urls, current_depth + 1)
                        
        except Exception as e:
            # Check if it's a timeout error
            if "Timeout" in str(e) or "timeout" in str(e).lower():
                logger.warning(f"Timeout exploring {url}: {e}. Skipping this page.")
            else:
                logger.error(f"Error exploring {url}: {e}")
    
    async def _collect_single_page_selenium(self, driver, url: str, depth: int) -> WebPageData:
        """Collect data from a single page using Selenium"""
        
        try:
            # Get page information
            title = driver.title
            page_source = driver.page_source
            
            # Take screenshot
            self.page_counter += 1
            screenshot_path = self.output_dir / f"page_{depth}_{self.page_counter}.png"
            driver.save_screenshot(str(screenshot_path))
            
            # Extract DOM elements
            elements = self._extract_dom_elements_selenium(driver)
            
            # Get page metrics
            load_time = 0.0  # Could be enhanced with timing
            page_size = len(page_source)
            
            # Determine page type
            page_type = self._determine_page_type(title, page_source, elements)
            
            # Extract links for further exploration
            links = self._extract_links_from_page(driver, url)
            
            return WebPageData(
                url=url,
                title=title,
                elements=elements,
                screenshots=[screenshot_path],
                load_time=load_time,
                page_size=page_size,
                links=links,
                clickable_elements=[e for e in elements if e.is_clickable],
                form_elements=[e for e in elements if e.is_input],
                table_elements=[e for e in elements if e.element_type == "table"],
                page_type=page_type,
                exploration_depth=depth
            )
            
        except Exception as e:
            # Check if it's a timeout error
            if "Timeout" in str(e) or "timeout" in str(e).lower():
                logger.warning(f"Timeout collecting page data for {url}: {e}. Skipping this page.")
            else:
                logger.error(f"Error collecting page data for {url}: {e}")
            return self._create_empty_page_data(url, depth)
    
    async def _find_explorable_links_selenium(self, driver, current_url: str, max_links: int) -> List[str]:
        """Find links that can be explored from current page"""
        
        try:
            # Find all links
            links_elements = driver.find_elements(By.TAG_NAME, "a")
            
            # Convert to link objects for LLM evaluation
            links = []
            for link in links_elements[:max_links * 2]:  # Get more links to filter
                try:
                    href = link.get_attribute("href")
                    text = link.text.strip()
                    visible = link.is_displayed()
                    
                    if href:
                        links.append({
                            'href': href,
                            'text': text,
                            'visible': visible
                        })
                except:
                    continue
            
            # Use configured method to evaluate and select links
            if self.use_llm_crawling and self.llm_executor:
                # Get current page information for context
                title = driver.title
                content = driver.page_source
                
                # Use LLM to evaluate and select links
                selected_links = await self._evaluate_links(
                    links, current_url, title, content, max_links
                )
                
                return selected_links
            else:
                # Use rule-based selection
                selected_links = []
                for link in links:
                    if len(selected_links) >= max_links:
                        break
                    
                    href = link.get('href', '')
                    if self._is_explorable_link(href, current_url):
                        selected_links.append(href)
            
                logger.info(f"📋 Rule-based evaluation: selected {len(selected_links)} links from {len(links)} available")
                return selected_links
            
        except Exception as e:
            logger.error(f"Error finding explorable links: {e}")
            return []


    
    def _is_explorable_link(self, href: str, current_url: str) -> bool:
        """Determine if a link should be explored using configuration-based rules"""
        
        if not href:
            return False
        
        # Parse URLs
        try:
            from urllib.parse import urlparse
            href_parsed = urlparse(href)
            current_parsed = urlparse(current_url)
            
            # Domain checking - allow same domain and subdomains if configured
            if href_parsed.netloc and self.allow_subdomains:
                current_domain = current_parsed.netloc
                href_domain = href_parsed.netloc
                
                # Check if it's the same domain or a subdomain
                if not (href_domain == current_domain or 
                       href_domain.endswith('.' + current_domain) or
                       current_domain.endswith('.' + href_domain)):
                    return False
            
            # Scheme checking - use configured allowed schemes
            if href_parsed.scheme not in self.allowed_schemes:
                return False
            
            # Pattern-based filtering - use configured skip patterns
            for pattern in self.skip_patterns:
                if pattern in href.lower():
                    return False
            
            # Anchor link handling - use configured setting
            if self.skip_anchor_only:
                if href.startswith('#') or (href_parsed.path == '' and href_parsed.fragment):
                    return False
            
            return True
            
        except Exception:
            return False
    
    async def _evaluate_links(self, links: List[Dict[str, Any]], current_url: str, 
                                     current_title: str, current_content: str, max_links: int) -> List[str]:
        """Evaluate and select links using configured method (LLM or rule-based)"""
        
        if not links:
            return []
        
        # Use LLM evaluation if enabled
        if self.use_llm_crawling and self.llm_executor:
            try:
                # Prepare context for LLM
                context = self._prepare_link_evaluation_context(links, current_url, current_title, current_content)
                
                # Create LLM prompt for link evaluation
                prompt = self._create_link_evaluation_prompt(context, max_links)
                
                # Execute LLM evaluation
                response, tokens_used = self.llm_executor._execute_with_retries(prompt)
                
                # Debug: Log LLM response
                logger.debug(f"🤖 LLM link evaluation response: {response}")
                
                # Parse LLM response
                selected_links = self._parse_link_evaluation_response(response, links, current_url)
                
                # Debug: Log parsing result
                logger.debug(f"🤖 Parsed {len(selected_links)} links from LLM response")
                
                logger.info(f"🤖 LLM evaluated {len(links)} links, selected {len(selected_links)} for diverse web tasks")
                return selected_links[:max_links]
                
            except Exception as e:
                logger.warning(f"LLM link evaluation failed: {e}, falling back to rule-based selection")
                return self._evaluate_links_with_rules(links, current_url, max_links)
        else:
            # Use rule-based evaluation
            return self._evaluate_links_with_rules(links, current_url, max_links)
    
    def _evaluate_links_with_rules(self, links: List[Dict[str, Any]], current_url: str, max_links: int) -> List[str]:
        """Evaluate links using rule-based filtering"""
        
        selected_links = []
        for link in links:
            if len(selected_links) >= max_links:
                break
            
            href = link.get('href', '')
            if self._is_explorable_link(href, current_url):
                selected_links.append(href)
        
        logger.info(f"📋 Rule-based evaluation: selected {len(selected_links)} links from {len(links)} available")
        return selected_links
    
    def _prepare_link_evaluation_context(self, links: List[Dict[str, Any]], current_url: str, 
                                       current_title: str, current_content: str) -> str:
        """Prepare context information for LLM link evaluation"""
        
        # Extract link information
        link_info = []
        for i, link in enumerate(links[:20]):  # Limit to first 20 links for context
            link_info.append({
                'index': i + 1,
                'href': link.get('href', ''),
                'text': link.get('text', '')[:100],  # Truncate text
                'visible': link.get('visible', True)
            })
        
        # Prepare current page context
        current_context = {
            'url': current_url,
            'title': current_title,
            'content_preview': current_content[:500] + "..." if len(current_content) > 500 else current_content
        }
        
        return {
            'current_page': current_context,
            'available_links': link_info
        }
    
    def _create_link_evaluation_prompt(self, context: Dict[str, Any], max_links: int) -> str:
        """Create LLM prompt for intelligent link evaluation focused on task value"""
        
        current_page = context['current_page']
        links = context['available_links']
        
        prompt = f"""You are an expert web task generation specialist. Your goal is to select links that lead to pages with HIGH TASK VALUE for generating diverse web automation tasks. Focus on QUALITY over QUANTITY.

Current Page Context:
- URL: {current_page['url']}
- Title: {current_page['title']}
- Content Preview: {current_page['content_preview']}

Available Links:
"""
        
        for link in links:
            prompt += f"- Link {link['index']}: {link['text']} -> {link['href']}\n"
        
        prompt += f"""
TASK: Select up to {max_links} links that lead to pages with the HIGHEST TASK GENERATION VALUE.

CORE SELECTION CRITERIA:

1. **Information Density** - Pages should contain:
   - Structured information (tables, lists, forms, charts)
   - Rich textual content (not just images/ads)
   - Multiple data points for comparison/extraction
   - Clear, readable content (avoid code snippets, gibberish)

2. **Task Relevance** - Pages should enable realistic tasks:
   - EXTRACTION tasks: Extract specific data from tables, forms, or structured content
   - COMPARISON tasks: Compare prices, features, options across multiple items
   - FACT_VERIFICATION tasks: Verify information against multiple sources
   - FORM_FILLING tasks: Complete registration, contact, or data entry forms
   - NAVIGATION tasks: Multi-step workflows, search and filter operations
   - DATA_ANALYSIS tasks: Work with charts, graphs, or complex data structures

3. **Multi-hop Potential** - Pages should have:
   - Multiple logically connected content sections
   - Cross-references between different data elements
   - Hierarchical information structure
   - Related content that can be combined for complex tasks

4. **Content Quality** - Pages should be:
   - Well-structured with clear DOM hierarchy
   - Free from excessive ads or irrelevant content
   - Accessible without login requirements
   - Complete (not partial content or broken pages)

5. **Safety & Compliance** - Avoid pages with:
   - Personal/private information (emails, phone numbers, addresses)
   - Copyrighted/proprietary content requiring subscriptions
   - Potentially harmful or inappropriate content
   - Robots.txt violations

PAGE TYPES TO PRIORITIZE:
- E-commerce product pages with specifications, reviews, pricing
- Service booking/registration pages with multi-step forms
- Data-rich pages (statistics, reports, dashboards)
- Content management pages with structured data
- Search results pages with filtering/sorting options
- User account/dashboard pages with multiple functions

PAGE TYPES TO AVOID:
- Static information pages with no interactive elements
- Image galleries or video pages with minimal text
- API documentation or technical reference pages
- Simple landing pages with basic navigation only
- Social media feeds or comment sections
- Download pages or file repositories

INSTRUCTIONS:
- Return only the link indices (e.g., "1, 3, 5, 7") of the selected links
- Prioritize pages with HIGH information density and task potential
- Choose diverse page types to enable various task categories
- Focus on pages that would generate MEANINGFUL automation tasks
- Quality over quantity - better to select fewer high-value pages

Selected link indices:"""
        
        return prompt
    
    def _parse_link_evaluation_response(self, response: str, links: List[Dict[str, Any]], current_url: str = '') -> List[str]:
        """Parse LLM response to extract selected link URLs"""
        
        try:
            import re
            
            # If no links available, return empty list
            if not links:
                return []
            
            # Try multiple parsing strategies
            selected_urls = []
            
            # Strategy 1: Look for explicit "Selected link indices:" format
            pattern1 = r'Selected link indices:\s*([\d,\s]+)'
            match1 = re.search(pattern1, response, re.IGNORECASE)
            if match1:
                indices_text = match1.group(1)
                indices = re.findall(r'\d+', indices_text)
                logger.debug(f"🤖 Strategy 1 - Found explicit indices: {indices}")
                
                for num in indices:
                    try:
                        index = int(num) - 1
                        if 0 <= index < len(links):
                            href = links[index].get('href', '')
                            if href and self._is_explorable_link(href, current_url):
                                selected_urls.append(href)
                                logger.debug(f"🤖 Selected link {index + 1}: {href}")
                    except ValueError:
                        continue
            
            # Strategy 2: If no explicit indices, look for numbers that could be valid indices
            if not selected_urls:
                # Find all numbers in response
                all_numbers = re.findall(r'\b(\d+)\b', response)
                logger.debug(f"🤖 Strategy 2 - All numbers found: {all_numbers}")
                
                # Filter numbers that could be valid link indices (1 to len(links))
                valid_indices = []
                for num in all_numbers:
                    try:
                        index = int(num)
                        if 1 <= index <= len(links):
                            valid_indices.append(index)
                    except ValueError:
                        continue
                
                logger.debug(f"🤖 Strategy 2 - Valid indices: {valid_indices}")
                
                # Use the first few valid indices
                for index in valid_indices[:5]:  # Limit to first 5
                    href = links[index - 1].get('href', '')
                    logger.debug(f"🤖 Checking link {index}: {href}")
                    if href and self._is_explorable_link(href, current_url):
                        selected_urls.append(href)
                        logger.debug(f"🤖 Selected link {index}: {href}")
                    else:
                        logger.debug(f"🤖 Link {index} not explorable: {href}")
            
            # Strategy 3: If still no links, fallback to first few valid links
            if not selected_urls:
                logger.debug(f"🤖 Strategy 3 - Fallback to first few valid links")
                for i, link in enumerate(links[:5]):
                    href = link.get('href', '')
                    if href and self._is_explorable_link(href, current_url):
                        selected_urls.append(href)
                        logger.debug(f"🤖 Fallback selected link {i + 1}: {href}")
            
            logger.debug(f"🤖 Total selected URLs: {len(selected_urls)}")
            return selected_urls
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM link evaluation response: {e}")
            # Fallback to first few valid links
            fallback_urls = []
            for link in links[:3]:
                href = link.get('href', '')
                if href and self._is_explorable_link(href, current_url):
                    fallback_urls.append(href)
            logger.debug(f"🤖 Fallback URLs: {fallback_urls}")
            return fallback_urls
    
    async def _assess_page_difficulty_with_llm(self, title: str, content: str, elements: List[WebElement], screenshot_path: Optional[Path] = None) -> Dict[str, Any]:
        """Use LLM to assess page quality and task generation potential with optional screenshot analysis"""
        
        if not self.llm_executor:
            return {
                'information_density_score': 0.5,
                'task_relevance_categories': [],
                'multi_hop_potential': 'medium',
                'content_quality': [],
                'safety_compliance': [],
                'task_generation_potential': 'medium',
                'suggested_task_types': [],
                'reasoning': 'LLM assessment not available'
            }
        
        try:
            # Prepare page information for LLM
            page_info = {
                'title': title,
                'content_preview': content[:1000] + "..." if len(content) > 1000 else content,
                'elements': [elem.to_dict() for elem in elements[:50]],  # Limit elements
                'screenshot_available': screenshot_path is not None and screenshot_path.exists()
            }
            
            # Create LLM prompt for comprehensive assessment
            prompt = self._create_difficulty_assessment_prompt(page_info)
            
            # Execute LLM assessment with optional screenshot
            if screenshot_path and screenshot_path.exists():
                # Use vision-capable LLM for screenshot analysis
                response, tokens_used = await self._assess_page_with_vision(prompt, screenshot_path)
            else:
                # Use text-only LLM assessment
                response, tokens_used = self.llm_executor._execute_with_retries(prompt)
            
            # Parse LLM response
            assessment = self._parse_difficulty_assessment_response(response)
            
            return assessment
            
        except Exception as e:
            logger.warning(f"LLM page assessment failed: {e}")
            return {
                'information_density_score': 0.5,
                'task_relevance_categories': [],
                'multi_hop_potential': 'medium',
                'content_quality': [],
                'safety_compliance': [],
                'task_generation_potential': 'medium',
                'suggested_task_types': [],
                'reasoning': f'Assessment failed: {str(e)}'
            }
    
    async def _assess_page_with_vision(self, prompt: str, screenshot_path: Path) -> Tuple[str, int]:
        """Assess page using vision-capable LLM with screenshot"""
        
        try:
            # Check if LLM executor supports vision
            if hasattr(self.llm_executor, '_execute_with_retries_with_images'):
                # Use vision-capable execution
                response, tokens_used = self.llm_executor._execute_with_retries_with_images(prompt, [str(screenshot_path)])
                logger.info(f"🤖 Using vision-capable LLM for screenshot analysis")
            else:
                # Fallback to text-only analysis
                response, tokens_used = self.llm_executor._execute_with_retries(prompt)
                logger.info(f"🤖 Using text-only LLM (vision not available)")
            
            return response, tokens_used
            
        except Exception as e:
            logger.warning(f"Vision-based assessment failed: {e}, falling back to text-only")
            # Fallback to text-only analysis
            response, tokens_used = self.llm_executor._execute_with_retries(prompt)
            return response, tokens_used
    
    def _should_include_page(self, assessment: Dict[str, Any]) -> bool:
        """Determine if a page should be included based on LLM assessment"""
        
        # Check if assessment is available
        if not assessment or 'task_generation_potential' not in assessment:
            return True  # Default to include if no assessment
        
        # Quality filters
        task_potential = assessment.get('task_generation_potential', 'low')
        info_density = assessment.get('information_density_score', 0.0)
        content_quality = assessment.get('content_quality', [])
        safety_compliance = assessment.get('safety_compliance', [])
        
        # Filter criteria
        if task_potential == 'low':
            logger.info("📋 Page filtered out: Low task generation potential")
            return False
        
        if info_density < 0.3:
            logger.info("📋 Page filtered out: Low information density")
            return False
        
        # Check content quality
        if 'readability' not in content_quality:
            logger.info("📋 Page filtered out: Poor readability")
            return False
        
        if 'completeness' not in content_quality:
            logger.info("📋 Page filtered out: Incomplete content")
            return False
        
        # Check safety compliance
        if 'privacy_safe' not in safety_compliance:
            logger.info("📋 Page filtered out: Privacy concerns")
            return False
        
        if 'content_safe' not in safety_compliance:
            logger.info("📋 Page filtered out: Content safety concerns")
            return False
        
        return True
    
    def _create_difficulty_assessment_prompt(self, page_info: Dict[str, Any]) -> str:
        """Create LLM prompt for comprehensive page assessment focused on task generation value"""
        
        screenshot_note = ""
        if page_info.get('screenshot_available', False):
            screenshot_note = """
IMPORTANT: You have access to a screenshot of this page. Use visual analysis to enhance your assessment:
- Analyze visual layout and information organization
- Identify visual elements like charts, tables, forms, buttons
- Assess visual complexity and user interface design
- Look for visual indicators of interactive elements
- Consider visual hierarchy and content structure
"""
        
        prompt = f"""You are an expert web task generation specialist. Assess this page's value for generating high-quality web automation tasks.

Page Information:
- Title: {page_info['title']}
- Content Preview: {page_info['content_preview']}
- Interactive Elements: {len(page_info['elements'])} elements
- Screenshot Available: {page_info.get('screenshot_available', False)}
{screenshot_note}

TASK: Provide a comprehensive assessment of this page's task generation potential.

ASSESSMENT CRITERIA:

1. **Information Density Score** (0.0-1.0):
   - 0.0-0.3: Low density (mostly images, minimal text, poor structure)
   - 0.3-0.7: Medium density (some structured content, moderate text)
   - 0.7-1.0: High density (rich structured data, tables, forms, comprehensive content)

2. **Visual Complexity Assessment** (if screenshot available):
   - "visual_richness": Rich visual elements, charts, graphs, complex layouts
   - "ui_complexity": Complex user interface with multiple interaction areas
   - "layout_quality": Well-organized, professional layout design
   - "visual_hierarchy": Clear visual hierarchy and information structure
   - "interactive_indicators": Visual cues for interactive elements

3. **Task Relevance Categories** (select all that apply):
   - "extraction": Extract data from tables, forms, or structured content
   - "comparison": Compare prices, features, options across multiple items
   - "fact_verification": Verify information against multiple sources
   - "form_filling": Complete registration, contact, or data entry forms
   - "navigation": Multi-step workflows, search and filter operations
   - "data_analysis": Work with charts, graphs, or complex data structures
   - "multi_hop": Tasks requiring information from multiple page sections
   - "visual_interaction": Tasks involving visual elements and layout

4. **Multi-hop Potential** (high/medium/low):
   - **High**: Multiple logically connected sections, cross-references, hierarchical structure
   - **Medium**: Some related content sections, moderate connectivity
   - **Low**: Single topic, minimal cross-references, flat structure

5. **Content Quality Assessment**:
   - "readability": Clear, well-structured text (avoid code, gibberish)
   - "completeness": Full content, not partial or broken
   - "accessibility": No login required, public content
   - "structure": Well-organized DOM hierarchy
   - "relevance": Content relevant to the page's purpose

6. **Safety & Compliance Check**:
   - "privacy_safe": No personal/private information exposed
   - "copyright_safe": No proprietary/subscription content
   - "content_safe": No harmful or inappropriate content
   - "robots_compliant": Respects robots.txt guidelines

7. **Task Generation Potential** (high/medium/low):
   - **High**: Rich in structured data, multiple interaction types, enables complex tasks
   - **Medium**: Moderate data richness, some interaction opportunities
   - **Low**: Limited data, minimal interactions, mainly informational

8. **Suggested Task Types** (specific task examples this page could generate):
   - List 3-5 specific, realistic tasks this page could support

Return your assessment in JSON format:
{{
    "information_density_score": 0.0-1.0,
    "visual_complexity": ["visual_richness", "ui_complexity", "layout_quality"],
    "task_relevance_categories": ["extraction", "comparison", "form_filling"],
    "multi_hop_potential": "high|medium|low",
    "content_quality": ["readability", "completeness", "accessibility"],
    "safety_compliance": ["privacy_safe", "copyright_safe", "content_safe"],
    "task_generation_potential": "high|medium|low",
    "suggested_task_types": [
        "Extract product specifications from the comparison table",
        "Fill out the contact form with specific information",
        "Compare prices across different product options"
    ],
    "reasoning": "brief explanation of assessment and task potential"
}}

Assessment:"""
        
        return prompt
    
    def _parse_difficulty_assessment_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for difficulty assessment"""
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                assessment = json.loads(json_match.group())
                return assessment
            else:
                logger.warning("No JSON found in LLM difficulty assessment response")
                return {'difficulty': 'medium', 'complexity_score': 0.5, 'interaction_types': []}
                
        except Exception as e:
            logger.warning(f"Failed to parse LLM difficulty assessment: {e}")
            return {'difficulty': 'medium', 'complexity_score': 0.5, 'interaction_types': []}
    
    def _extract_links_from_page(self, driver, current_url: str) -> List[str]:
        """Extract all links from the current page"""
        
        try:
            links = driver.find_elements(By.TAG_NAME, "a")
            extracted_links = []
            
            for link in links:
                try:
                    href = link.get_attribute("href")
                    if href and self._is_explorable_link(href, current_url):
                        extracted_links.append(href)
                except:
                    continue
            
            return extracted_links
            
        except Exception as e:
            logger.error(f"Error extracting links: {e}")
            return []
    
    def _determine_page_type(self, title: str, page_source: str, elements: List[WebElement]) -> str:
        """Determine the type of page based on content and elements"""
        
        title_lower = title.lower()
        source_lower = page_source.lower()
        
        # Check for common page types
        if any(word in title_lower for word in ['login', 'signin', 'auth']):
            return 'login'
        elif any(word in title_lower for word in ['signup', 'register', 'create account']):
            return 'signup'
        elif any(word in title_lower for word in ['product', 'item', 'detail']):
            return 'product'
        elif any(word in title_lower for word in ['cart', 'basket', 'checkout']):
            return 'cart'
        elif any(word in title_lower for word in ['search', 'results']):
            return 'search'
        elif any(word in title_lower for word in ['about', 'contact', 'help']):
            return 'info'
        elif any(word in title_lower for word in ['privacy', 'terms', 'policy']):
            return 'legal'
        else:
            return 'content'
    
    async def _determine_website_type_with_llm(self, title: str, page_source: str, elements: List[WebElement], url: str) -> tuple[str, str]:
        """Use LLM to determine the type of website and provide a description"""
        
        # Check if LLM website type detection is enabled
        if not self.use_llm_website_type_detection:
            logger.info("LLM website type detection is disabled, using rule-based detection")
            return self._determine_website_type_rule_based(title, page_source, elements, url)
        
        if not self.llm_executor:
            # Fallback to rule-based detection if LLM is not available
            logger.info("LLM executor not available, using rule-based website type detection")
            return self._determine_website_type_rule_based(title, page_source, elements, url)
        
        try:
            # Prepare page content for LLM analysis
            page_content = self._prepare_page_content_for_llm(title, page_source, elements)
            
            prompt = f"""Analyze this web page and determine the type of website and its primary function.

Page Information:
- URL: {url}
- Title: {title}
- Content: {page_content[:2000]}...  # Truncated for brevity

Please classify this website into one of these categories and provide a brief description:

1. **crm** - Customer Relationship Management systems (SuiteCRM, Salesforce, HubSpot, etc.)
2. **ecommerce** - Online shopping and retail websites
3. **blog** - Content publishing and blogging platforms
4. **news** - News and media websites
5. **social** - Social media platforms
6. **education** - Educational and learning platforms
7. **corporate** - Business and corporate websites
8. **government** - Government and official websites
9. **portfolio** - Personal portfolio and resume websites
10. **content** - General content and information websites

Respond in this exact JSON format:
{{
    "website_type": "category_name",
    "website_description": "Brief description of what this website does and its primary function"
}}

Focus on the main purpose and functionality of the website based on the content and elements present."""
            
            # Get LLM response with website type detection specific settings
            response = self.llm_executor.execute_simple(
                prompt, 
                task_id="website_type_detection"
            )
            
            # Parse response
            import json
            import re
            
            # Extract JSON from response (response is ExecutionResult object)
            json_match = re.search(r'\{.*\}', response.answer, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                website_type = result.get('website_type', 'content')
                website_description = result.get('website_description', 'General content website')
                logger.info(f"🤖 LLM classified website as: {website_type}")
                return website_type, website_description
            else:
                logger.warning("No JSON found in LLM website type response, using fallback")
                return self._determine_website_type_rule_based(title, page_source, elements, url)
                
        except Exception as e:
            logger.warning(f"LLM website type detection failed: {e}, using fallback")
            return self._determine_website_type_rule_based(title, page_source, elements, url)
    
    def _determine_website_type_rule_based(self, title: str, page_source: str, elements: List[WebElement], url: str) -> tuple[str, str]:
        """Rule-based fallback for website type detection"""
        
        title_lower = title.lower()
        source_lower = page_source.lower()
        url_lower = url.lower()
        
        logger.debug(f"🔍 Rule-based website type detection - Title: '{title}', URL: {url}")
        
        # Check for CRM systems
        if any(word in title_lower for word in ['suitecrm', 'crm', 'customer relationship', 'salesforce', 'hubspot']):
            logger.info(f"📋 Rule-based detection: Identified as CRM system (title contains: {[word for word in ['suitecrm', 'crm', 'customer relationship', 'salesforce', 'hubspot'] if word in title_lower]})")
            return 'crm', 'Customer Relationship Management system for managing customers, contacts, leads, and sales opportunities'
        
        # Check for e-commerce sites
        if any(word in title_lower for word in ['shop', 'store', 'buy', 'cart', 'checkout', 'product']) or \
           any(word in source_lower for word in ['add to cart', 'buy now', 'shopping cart', 'product catalog']):
            return 'ecommerce', 'E-commerce website for online shopping and product sales'
        
        # Check for blogs
        if any(word in title_lower for word in ['blog', 'article', 'post']) or \
           any(word in source_lower for word in ['blog', 'article', 'post', 'comment']):
            return 'blog', 'Blog or content publishing website'
        
        # Check for news sites
        if any(word in title_lower for word in ['news', 'breaking', 'latest']) or \
           any(word in source_lower for word in ['news', 'breaking', 'latest', 'headlines']):
            return 'news', 'News and media website'
        
        # Check for social media
        if any(word in title_lower for word in ['social', 'profile', 'feed']) or \
           any(word in source_lower for word in ['social', 'profile', 'feed', 'timeline']):
            return 'social', 'Social media platform'
        
        # Check for educational sites
        if any(word in title_lower for word in ['learn', 'course', 'tutorial', 'education']) or \
           any(word in source_lower for word in ['learn', 'course', 'tutorial', 'education']):
            return 'education', 'Educational or learning platform'
        
        # Check for corporate/business sites
        if any(word in title_lower for word in ['about us', 'company', 'corporate', 'business']) or \
           any(word in source_lower for word in ['about us', 'company', 'corporate', 'business']):
            return 'corporate', 'Corporate or business website'
        
        # Check for government sites
        if '.gov' in url_lower or any(word in title_lower for word in ['government', 'official']):
            return 'government', 'Government or official website'
        
        # Check for portfolio/personal sites
        if any(word in title_lower for word in ['portfolio', 'personal', 'resume']) or \
           any(word in source_lower for word in ['portfolio', 'personal', 'resume']):
            return 'portfolio', 'Personal portfolio or resume website'
        
        # Default to content site
        logger.info(f"📋 Rule-based detection: Defaulting to content website (no specific patterns matched)")
        return 'content', 'General content website'
    
    def _prepare_page_content_for_llm(self, title: str, page_source: str, elements: List[WebElement]) -> str:
        """Prepare page content for LLM analysis"""
        
        # Extract key information from page source
        import re
        
        # Get text content (remove HTML tags)
        text_content = re.sub(r'<[^>]+>', ' ', page_source)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        
        # Extract key elements information
        element_info = []
        for elem in elements[:20]:  # Limit to first 20 elements
            if elem.text_content.strip():
                element_info.append(f"{elem.element_type}: {elem.text_content[:100]}")
        
        # Combine information
        content = f"Title: {title}\n"
        content += f"Text Content: {text_content[:1000]}\n"
        content += f"Interactive Elements: {', '.join(element_info[:10])}"
        
        return content
    
    def _create_empty_page_data(self, url: str, depth: int) -> WebPageData:
        """Create empty page data for failed collection"""
        
        return WebPageData(
            url=url,
            title="Error loading page",
            elements=[],
            screenshots=[],
            load_time=0.0,
            page_size=0,
            links=[],
            clickable_elements=[],
            form_elements=[],
            table_elements=[],
            page_type="error",
            website_type="unknown",
            website_description="Error loading page",
            exploration_depth=depth
        )
    
    

    
    async def _extract_dom_elements_playwright(self, page=None, som_mapping=None) -> List[WebElement]:
        """Extract DOM elements using Playwright - only SoM marked elements if mapping provided"""
        
        elements = []
        
        # Use provided page or fall back to self.page
        current_page = page or self.page
        if not current_page:
            logger.error("No page available for DOM extraction")
            return elements
        
        try:
            logger.debug("🔍 Starting DOM element extraction...")
            if som_mapping:
                # Extract only SoM marked elements
                logger.info(f"🔍 Extracting only SoM marked elements ({len(som_mapping)} elements)")
                element_data = await current_page.evaluate("""
                    (somMapping) => {
                        try {
                            console.log('🔍 JavaScript execution started');
                            console.log('🔍 somMapping parameter:', somMapping);
                            
                        // Enhanced element type identification function - IMPROVED VERSION
                        function identifyElementType(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 1. 基于语义和功能的智能识别（优先级最高）
                            
                            // 业务数据元素识别（新增）
                            if (isBusinessDataElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'business_data';
                            }
                            
                            // 搜索框识别（新增）
                            if (isSearchElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'search_box';
                            }
                            
                            // 表单控件识别（改进）
                            if (isFormControlElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'form_control';
                            }
                            
                            // 菜单元素识别
                            if (isMenuElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'menu';
                            }
                            
                            // 标签页元素识别
                            if (isTabElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'tab';
                            }
                            
                            // 过滤器元素识别
                            if (isFilterElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'filter';
                            }
                            
                            // 模态框元素识别
                            if (isModalElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'modal';
                            }
                            
                            // 表单元素识别
                            if (isFormElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'form';
                            }
                            
                            // 下拉菜单识别
                            if (isDropdownElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'dropdown';
                            }
                            
                            // 面包屑识别
                            if (isBreadcrumbElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'breadcrumb';
                            }
                            
                            // 分页器识别
                            if (isPaginatorElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'paginator';
                            }
                            
                            // 通知元素识别
                            if (isNotificationElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'toast';
                            }
                            
                            // 2. 基于标签名的基本识别（保持原有逻辑作为后备）
                            if (tagName === 'button') {
                                // 智能按钮类型识别
                                if (classList.some(cls => cls.toLowerCase().includes('submit'))) {
                                    return 'submit';
                                } else if (classList.some(cls => cls.toLowerCase().includes('search'))) {
                                    return 'search_button';
                                } else if (classList.some(cls => cls.toLowerCase().includes('filter'))) {
                                    return 'filter_button';
                                } else {
                                    return 'button';
                                }
                            } else if (tagName === 'input') {
                                // 智能输入框类型识别
                                const inputType = element.getAttribute('type') || 'text';
                                if (inputType === 'search' || classList.some(cls => cls.toLowerCase().includes('search'))) {
                                    return 'search_box';
                                } else if (inputType === 'email') {
                                    return 'email_input';
                                } else if (inputType === 'password') {
                                    return 'password_input';
                                } else if (inputType === 'submit') {
                                    return 'submit';
                                } else {
                                    return 'input';
                                }
                            } else if (tagName === 'a') {
                                // 智能链接类型识别
                                if (classList.some(cls => cls.toLowerCase().includes('nav'))) {
                                    return 'navigation';
                                } else if (classList.some(cls => cls.toLowerCase().includes('menu'))) {
                                    return 'menu_link';
                                } else if (classList.some(cls => cls.toLowerCase().includes('detail'))) {
                                    return 'detail_link';
                                } else {
                                    return 'link';
                                }
                            } else if (tagName === 'select') {
                                return 'select';
                            } else if (tagName === 'textarea') {
                                return 'textarea';
                            } else if (tagName === 'table') {
                                return 'table';
                            } else if (tagName === 'img') {
                                return 'image';
                            } else if (tagName === 'form') {
                                return 'form';
                            } else if (tagName === 'div' || tagName === 'section') {
                                // 智能容器类型识别
                                if (classList.some(cls => cls.toLowerCase().includes('card'))) {
                                    return 'card';
                                } else if (classList.some(cls => cls.toLowerCase().includes('list'))) {
                                    return 'list';
                                } else if (classList.some(cls => cls.toLowerCase().includes('detail'))) {
                                    return 'detail';
                                } else if (classList.some(cls => cls.toLowerCase().includes('dashboard'))) {
                                    return 'dashboard';
                                }
                            }
                            
                            // 3. 默认类型
                            return 'clickable';
                        }
                        // Helper functions for element type identification
                        function isMenuElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查role属性
                            if (role === 'menu' || role === 'menuitem' || role === 'menubar') {
                                return true;
                            }
                            
                            // 检查CSS类名
                            const menuKeywords = ['menu', 'nav-menu', 'main-menu', 'primary-menu', 'sidebar-menu', 'navbar-nav'];
                            if (classList.some(cls => menuKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查aria标签
                            if (ariaLabel && ariaLabel.toLowerCase().includes('menu')) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (parentLower.includes('nav') || parentLower.includes('sidebar') || 
                                    parentLower.includes('menu') || parentLower.includes('navbar')) {
                                    return true;
                                }
                            }
                            
                            // 检查是否在导航容器中
                            const parent = element.parentElement;
                            if (parent) {
                                const parentClass = parent.className || '';
                                const parentTag = parent.tagName.toLowerCase();
                                if (parentTag === 'nav' || parentClass.toLowerCase().includes('nav') || 
                                    parentClass.toLowerCase().includes('menu')) {
                                    return true;
                                }
                            }
                            
                            return false;
                        }
                        
                        function isTabElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 首先排除table元素
                            if (tagName === 'table') {
                                return false;
                            }
                            
                            // 检查role属性
                            if (role === 'tab' || role === 'tablist') {
                                return true;
                            }
                            
                            // 检查CSS类名
                            const tabKeywords = ['tab', 'tab-item', 'nav-tab', 'tab-nav'];
                            if (classList.some(cls => tabKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查aria标签
                            if (ariaLabel && ariaLabel.toLowerCase().includes('tab')) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (parentLower.includes('tab') || parentLower.includes('nav-tabs')) {
                                    return true;
                                }
                            }
                            
                            // 检查是否在标签页容器中
                            const parent = element.parentElement;
                            if (parent) {
                                const parentClass = parent.className || '';
                                if (parentClass.toLowerCase().includes('tab') || parentClass.toLowerCase().includes('nav-tabs')) {
                                    return true;
                                }
                            }
                            
                            return false;
                        }
                        
                        function isFilterElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查CSS类名
                            const filterKeywords = ['filter', 'filters', 'filter-panel', 'filter-controls', 'filter-options', 'search-form'];
                            if (classList.some(cls => filterKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查aria标签
                            if (ariaLabel && (ariaLabel.toLowerCase().includes('filter') || ariaLabel.toLowerCase().includes('search'))) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (parentLower.includes('filter') || parentLower.includes('search') || 
                                    parentLower.includes('advanced-search')) {
                                    return true;
                                }
                            }
                            
                            // 检查是否在过滤器容器中
                            const parent = element.parentElement;
                            if (parent) {
                                const parentClass = parent.className || '';
                                if (parentClass.toLowerCase().includes('filter') || parentClass.toLowerCase().includes('search')) {
                                    return true;
                                }
                            }
                            
                            // 检查data属性
                            if (element.hasAttribute('data-filter') || element.hasAttribute('data-search')) {
                                return true;
                            }
                            
                            return false;
                        }
                        
                        function isModalElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查role属性
                            if (role === 'dialog' || role === 'modal') {
                                return true;
                            }
                            
                            // 检查CSS类名
                            const modalKeywords = ['modal', 'dialog', 'popup', 'overlay', 'lightbox'];
                            if (classList.some(cls => modalKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查aria标签
                            if (ariaLabel && (ariaLabel.toLowerCase().includes('modal') || ariaLabel.toLowerCase().includes('dialog'))) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (parentLower.includes('modal') || parentLower.includes('dialog')) {
                                    return true;
                                }
                            }
                            
                            return false;
                        }
                        
                        function isFormElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查标签名
                            if (tagName === 'form') {
                                return true;
                            }
                            
                            // 检查CSS类名
                            const formKeywords = ['form', 'form-group', 'form-control', 'form-container'];
                            if (classList.some(cls => formKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (parentLower.includes('form') || parentLower.includes('form-group')) {
                                    return true;
                                }
                            }
                            
                            // 检查是否在表单容器中
                            const parent = element.parentElement;
                            if (parent) {
                                const parentClass = parent.className || '';
                                if (parentClass.toLowerCase().includes('form')) {
                                    return true;
                                }
                            }
                            
                            return false;
                        }
                        
                        function isDropdownElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // IMPORTANT: Check if this is actually a navigation element first
                            // Navigation elements with dropdown-toggle should be treated as navigation, not dropdown
                            const hasNavigationClass = classList.some(cls => cls.toLowerCase().includes('nav'));
                            if (hasNavigationClass) {
                                return false; // Let navigation logic handle it
                            }
                            
                            // 检查CSS类名
                            const dropdownKeywords = ['dropdown', 'drop-down', 'select-dropdown', 'menu-dropdown'];
                            if (classList.some(cls => dropdownKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查aria标签
                            if (ariaLabel && ariaLabel.toLowerCase().includes('dropdown')) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (parentLower.includes('dropdown')) {
                                    return true;
                                }
                            }
                            
                            return false;
                        }
                        
                        function isBreadcrumbElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查role属性
                            if (role === 'navigation' && ariaLabel && ariaLabel.toLowerCase().includes('breadcrumb')) {
                                return true;
                            }
                            
                            // 检查CSS类名
                            const breadcrumbKeywords = ['breadcrumb', 'breadcrumbs', 'crumb'];
                            if (classList.some(cls => breadcrumbKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            return false;
                        }
                        
                        function isPaginatorElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查CSS类名
                            const paginatorKeywords = ['pagination', 'paginator', 'page-link', 'page-item'];
                            if (classList.some(cls => paginatorKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (parentLower.includes('pagination') || parentLower.includes('paginator')) {
                                    return true;
                                }
                            }
                            
                            return false;
                        }
                        
                        function isNotificationElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查role属性
                            if (role === 'alert' || role === 'status') {
                                return true;
                            }
                            
                            // 检查CSS类名
                            const notificationKeywords = ['toast', 'notification', 'alert', 'message', 'snackbar'];
                            if (classList.some(cls => notificationKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            return false;
                        }
                        
                        // NEW: Business data element detection - IMPROVED VERSION
                        function isBusinessDataElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 0. 首先检查元素是否有有意义的内容（关键修复：避免空内容元素被标记为业务数据）
                            const textContent = element.textContent || '';
                            const placeholder = element.placeholder || '';
                            const value = element.value || '';
                            const ariaLabelValue = element.getAttribute('aria-label') || '';
                            
                            // 检查是否有有意义的文本内容
                            const hasValidContent = (
                                (textContent && textContent.trim().length > 0) ||
                                (placeholder && placeholder.trim().length > 0 && placeholder !== 'Search...') ||
                                (value && value.trim().length > 0 && value !== 'Search...')
                            );
                            
                            // 如果没有有意义的内容，则不应该被标记为业务数据
                            // 但允许一些特殊情况：如果元素有明确的业务数据属性或在业务数据表格中
                            if (!hasValidContent) {
                                // 特殊情况1：在数据表格中的元素（即使内容为空也可能是业务数据占位符）
                                if (tagName === 'td' || tagName === 'th' || tagName === 'tr') {
                                    let parent = element.parentElement;
                                    while (parent) {
                                        if (parent.tagName === 'TABLE') {
                                            const tableClasses = parent.className.toLowerCase();
                                            if (!tableClasses.includes('nav') && !tableClasses.includes('menu')) {
                                                return true; // 数据表格中的空单元格仍然可以是业务数据
                                            }
                                        }
                                        parent = parent.parentElement;
                                    }
                                }
                                
                                // 特殊情况2：有明确业务数据相关的data-*属性
                                const dataAttributes = Array.from(element.attributes).filter(attr => attr.name.startsWith('data-'));
                                const hasBusinessDataAttributes = dataAttributes.some(attr => {
                                    const attrName = attr.name.toLowerCase();
                                    const attrValue = attr.value.toLowerCase();
                                    return ['record', 'entity', 'model', 'id', 'customer', 'user', 'client', 'lead', 'product', 'order'].some(keyword => 
                                        attrName.includes(keyword) || attrValue.includes(keyword)
                                    );
                                });
                                
                                if (!hasBusinessDataAttributes) {
                                    return false; // 没有有意义的内容且没有业务数据属性，不应标记为业务数据
                                }
                            }
                            
                            // 1. 首先排除明显的导航元素（但保留可能是业务数据的链接和列表项）
                            if (tagName === 'nav' || tagName === 'ul') {
                                return false;
                            }

                            // 对于列表项，需要更仔细的判断 - 可能包含业务数据
                            if (tagName === 'li') {
                                // 如果列表项包含业务数据关键词，则认为是业务数据
                                const businessKeywords = ['customer', 'account', 'contact', 'lead', 'opportunity', 'product', 'order', 'company', 'corporation', 'inc', 'ltd', 'llc'];
                                const textLower = textContent.toLowerCase();

                                if (businessKeywords.some(keyword => textLower.includes(keyword))) {
                                    return true;
                                }

                                // 如果列表项很长且包含多个实体，可能是业务数据列表
                                const words = textContent.trim().split(/\s+/);
                                if (words.length > 3 && textContent.length > 50) {
                                    return true;
                                }

                                // 默认排除导航相关的列表项
                                return false;
                            }
                            
                            // 2. 对于链接元素，需要更仔细的判断
                            if (tagName === 'a') {
                                // 检查链接的上下文和内容，判断是否为业务数据链接
                                const href = element.href || '';
                                const parentContext = element.parentElement ? element.parentElement.className.toLowerCase() : '';
                                
                                // 排除明显的导航链接
                                const navigationLinkKeywords = ['nav', 'menu', 'breadcrumb', 'sidebar', 'header', 'footer'];
                                if (classList.some(cls => navigationLinkKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                    return false;
                                }
                                
                                // 排除导航相关的父级上下文
                                if (parentContext && navigationLinkKeywords.some(keyword => parentContext.includes(keyword))) {
                                    return false;
                                }
                                
                                // 检查是否为业务数据相关的链接
                                const businessLinkKeywords = ['contact', 'customer', 'user', 'client', 'lead', 'product', 'item', 'order', 'invoice', 'deal', 'ticket', 'case'];
                                const linkTextLower = textContent.toLowerCase();
                                const hrefLower = href.toLowerCase();
                                
                                // 如果链接文本或href包含业务数据关键词，且不在导航上下文中，则认为是业务数据
                                if (businessLinkKeywords.some(keyword => 
                                    linkTextLower.includes(keyword) || hrefLower.includes(keyword))) {
                                    return true;
                                }
                                
                                // 如果链接指向具体的业务数据页面（如详情页、记录页等）
                                if (hrefLower.includes('detail') || hrefLower.includes('record') || 
                                    hrefLower.includes('view') || hrefLower.includes('edit') ||
                                    hrefLower.includes('id=') || hrefLower.includes('record=')) {
                                    return true;
                                }
                                
                                // 默认情况下，不在导航上下文中的链接可能是业务数据
                                return !parentContext.includes('nav') && !parentContext.includes('menu');
                            }
                            
                            // 3. 排除明显的导航相关CSS类（但不包括link）
                            const navigationKeywords = ['nav', 'menu', 'breadcrumb', 'sidebar', 'header', 'footer'];
                            if (classList.some(cls => navigationKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return false;
                            }
                            
                            // 4. 检查是否在数据表格中（真正的业务数据）
                            if (tagName === 'td' || tagName === 'th' || tagName === 'tr') {
                                // 进一步检查是否在真正的数据表格中，而不是导航表格
                                let parent = element.parentElement;
                                while (parent) {
                                    if (parent.tagName === 'TABLE') {
                                        const tableClasses = parent.className.toLowerCase();
                                        // 排除导航表格
                                        if (tableClasses.includes('nav') || tableClasses.includes('menu')) {
                                            return false;
                                        }
                                        return true;
                                    }
                                    parent = parent.parentElement;
                                }
                                return true;
                            }
                            
                            // 5. 检查CSS类名中的业务数据关键词（更精确的匹配）
                            const businessKeywords = [
                                'contact', 'customer', 'user', 'client', 'lead',
                                'product', 'item', 'goods', 'service',
                                'order', 'invoice', 'quote', 'deal',
                                'ticket', 'case', 'issue', 'bug',
                                'report', 'analytics', 'metrics', 'statistics',
                                'data', 'record', 'entry', 'row'
                            ];
                            
                            // 更精确的关键词匹配，避免误匹配导航元素
                            for (const cls of classList) {
                                const clsLower = cls.toLowerCase();
                                for (const keyword of businessKeywords) {
                                    // 确保关键词不是导航的一部分
                                    if (clsLower.includes(keyword) && 
                                        !clsLower.includes('nav') && 
                                        !clsLower.includes('menu')) {
                                        return true;
                                    }
                                }
                            }
                            
                            // 6. 检查父级上下文（更严格的检查）
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                // 排除导航相关的父级上下文
                                if (parentLower.includes('nav') || parentLower.includes('menu') || parentLower.includes('sidebar')) {
                                    return false;
                                }
                                
                                // 只匹配明确的业务数据上下文
                                const strictBusinessKeywords = [
                                    'contact', 'customer', 'user', 'client', 'lead',
                                    'product', 'item', 'goods', 'service',
                                    'order', 'invoice', 'quote', 'deal',
                                    'ticket', 'case', 'issue', 'bug',
                                    'report', 'analytics', 'metrics', 'statistics'
                                ];
                                
                                for (const keyword of strictBusinessKeywords) {
                                    if (parentLower.includes(keyword) && 
                                        !parentLower.includes('nav') && 
                                        !parentLower.includes('menu')) {
                                        return true;
                                    }
                                }
                            }
                            
                            return false;
                        }
                        
                        // NEW: Search element detection
                        function isSearchElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查标签名
                            if (tagName === 'input' && element.getAttribute('type') === 'search') {
                                return true;
                            }
                            
                            // 检查CSS类名
                            const searchKeywords = ['search', 'search-box', 'search-input', 'search-field'];
                            if (classList.some(cls => searchKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查placeholder
                            const placeholder = element.getAttribute('placeholder') || '';
                            if (placeholder.toLowerCase().includes('search') || placeholder.toLowerCase().includes('查找')) {
                                return true;
                            }
                            
                            // 检查aria标签
                            if (ariaLabel && (ariaLabel.toLowerCase().includes('search') || ariaLabel.toLowerCase().includes('查找'))) {
                                return true;
                            }
                            
                            return false;
                        }
                        
                        // NEW: Form control element detection
                        function isFormControlElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 首先排除form元素本身
                            if (tagName === 'form') {
                                return false;
                            }
                            
                            // 检查是否在表单中
                            let parent = element.parentElement;
                            while (parent) {
                                if (parent.tagName === 'FORM' || 
                                    parent.getAttribute('role') === 'form' ||
                                    (parent.className && parent.className.toLowerCase().includes('form'))) {
                                    return true;
                                }
                                parent = parent.parentElement;
                            }
                            
                            // 检查CSS类名
                            const formControlKeywords = ['form-control', 'form-group', 'input-group'];
                            if (classList.some(cls => formControlKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            return false;
                        }
                        
                        // NEW: Card element detection
                        function isCardElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查CSS类名
                            const cardKeywords = ['card', 'card-body', 'card-header', 'card-footer', 'panel', 'widget'];
                            if (classList.some(cls => cardKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (cardKeywords.some(keyword => parentLower.includes(keyword))) {
                                    return true;
                                }
                            }
                            
                            return false;
                        }
                        
                        // NEW: List element detection
                        function isListElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查标签名
                            if (tagName === 'ul' || tagName === 'li' || tagName === 'ol') {
                                return true;
                            }
                            
                            // 检查CSS类名
                            const listKeywords = ['list', 'list-group', 'list-item', 'item-list'];
                            if (classList.some(cls => listKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            return false;
                        }
                        
                        // NEW: Detail element detection
                        function isDetailElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查CSS类名
                            const detailKeywords = ['detail', 'details', 'detail-view', 'detail-panel'];
                            if (classList.some(cls => detailKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (detailKeywords.some(keyword => parentLower.includes(keyword))) {
                                    return true;
                                }
                            }
                            
                            return false;
                        }
                        // Unified description generation function
                        function generateElementDescription(element, tagName, textContent, placeholder, value, href, type, classList, ariaLabel, alt) {
                            // Get parent context (heading, form, section)
                            let parentContext = '';
                            let parent = element.parentElement;
                            for (let i = 0; i < 3 && parent; i++) {
                                if (parent.tagName && ['H1', 'H2', 'H3', 'H4', 'H5', 'H6'].includes(parent.tagName)) {
                                    parentContext = parent.textContent?.trim() || '';
                                    break;
                                } else if (parent.tagName === 'FORM') {
                                    parentContext = 'form';
                                    break;
                                } else if (parent.getAttribute('role') === 'form') {
                                    parentContext = 'form';
                                    break;
                                } else if (parent.className && typeof parent.className === 'string' && parent.className.includes('form')) {
                                    parentContext = 'form';
                                    break;
                                }
                                parent = parent.parentElement;
                            }
                            
                            // Rule-based description generation
                            let description = '';
                            if (tagName === 'button') {
                                if (textContent) {
                                    description = 'Button element, text ' + JSON.stringify(textContent);
                                } else if (classList.some(cls => cls.includes('search'))) {
                                    description = 'Button element, search button';
                                } else if (classList.some(cls => cls.includes('filter'))) {
                                    description = 'Button element, filter button';
                                } else if (classList.some(cls => cls.includes('submit'))) {
                                    description = 'Button element, submit button';
                                } else {
                                    description = 'Button element, action button';
                                }
                            } else if (tagName === 'input') {
                                if (type === 'text' || type === 'search') {
                                    if (placeholder) {
                                        description = 'Input element, type ' + type + ', placeholder ' + JSON.stringify(placeholder);
                                    } else if (classList.some(cls => cls.includes('search'))) {
                                        description = 'Input element, type search, search field';
                                    } else {
                                        description = 'Input element, type ' + type;
                                    }
                                } else if (type === 'email') {
                                    description = 'Input element, type email, email field';
                                } else if (type === 'password') {
                                    description = 'Input element, type password, password field';
                                } else if (type === 'submit') {
                                    description = 'Input element, type submit, submit button';
                                } else {
                                    description = 'Input element, type ' + type;
                                }
                            } else if (tagName === 'a') {
                                if (textContent) {
                                    description = 'Link element, text ' + JSON.stringify(textContent);
                                } else if (href) {
                                    description = 'Link element, navigation link';
                                } else {
                                    description = 'Link element, clickable link';
                                }
                            } else if (tagName === 'select') {
                                description = 'Select element, dropdown selection field';
                            } else if (tagName === 'textarea') {
                                description = 'Textarea element, multi-line text input field';
                            } else if (tagName === 'table') {
                                description = 'Table element, data table';
                            } else if (tagName === 'img') {
                                if (alt) {
                                    description = 'Image element, alt text ' + JSON.stringify(alt);
                                } else {
                                    description = 'Image element';
                                }
                            } else if (tagName === 'form') {
                                description = 'Form element, form container';
                            } else {
                                description = tagName + ' element';
                            }
                            
                            // Add parent context if available
                            if (parentContext) {
                                description += ', inside ' + JSON.stringify(parentContext);
                            }
                            
                            return description;
                        }
                        
                        const elements = [];
                        const processedElements = new Set(); // Track processed elements to avoid duplicates
                        
                        // Extract elements based on SoM mapping
                        console.log('🔍 Starting element extraction, somMapping:', somMapping);
                        console.log('🔍 somMapping entries count:', Object.keys(somMapping).length);
                        
                        Object.entries(somMapping).forEach(([markId, elementInfo]) => {
                            console.log(`🔍 Processing mark ${markId}:`, elementInfo);
                            
                            // Find element by matching position and properties
                            const targetRect = elementInfo.rect_viewport;
                            const targetType = elementInfo.type;
                            const targetTag = elementInfo.tag;
                            const targetText = elementInfo.text;
                            
                            let element = null;
                            
                            // Validate required properties
                            if (!targetRect || !targetTag) {
                                console.warn(`Missing required properties for mark ${markId}:`, elementInfo);
                                return; // Skip this element
                            }
                            
                            // Try to find element by exact position match
                            const candidates = document.querySelectorAll(targetTag);
                            console.log(`🔍 Found ${candidates.length} candidates for tag ${targetTag}`);
                            
                            for (const candidate of candidates) {
                                const rect = candidate.getBoundingClientRect();
                                const candidateText = candidate.textContent?.trim() || candidate.value || candidate.placeholder || '';
                                
                                console.log(`🔍 Checking candidate:`, {
                                    tag: candidate.tagName,
                                    text: candidateText,
                                    rect: { left: rect.left, top: rect.top, width: rect.width, height: rect.height },
                                    target: { x: targetRect.x, y: targetRect.y, width: targetRect.width, height: targetRect.height }
                                });
                                
                                // Check for exact position match
                                if (Math.abs(rect.left - targetRect.x) < 1 &&
                                    Math.abs(rect.top - targetRect.y) < 1 &&
                                    Math.abs(rect.width - targetRect.width) < 1 &&
                                    Math.abs(rect.height - targetRect.height) < 1) {
                                    element = candidate;
                                    console.log(`🔍 Found exact match for mark ${markId}`);
                                    break;
                                }
                            }
                            
                            // Fallback: find by text content match (for elements with text)
                            if (!element && targetText) {
                                for (const candidate of candidates) {
                                    const candidateText = candidate.textContent?.trim() || candidate.value || candidate.placeholder || '';
                                    if (candidateText === targetText) {
                                        element = candidate;
                                        break;
                                    }
                                }
                            }
                            
                            // Final fallback: find by tag and type
                            if (!element) {
                                const selectors = {
                                    'clickable': 'button, a, [role="button"]',
                                    'input': 'input, textarea',
                                    'select': 'select, input[type="radio"], input[type="checkbox"]',
                                    'navigation': 'a, nav a, .pagination a',
                                    'result': 'h1, h2, h3, .title, .product-title'
                                };
                                
                                const selector = selectors[targetType] || targetTag;
                                const matchingElements = document.querySelectorAll(selector);
                                if (matchingElements.length > 0) {
                                    element = matchingElements[0]; // Take first matching element
                                }
                            }
                            
                            if (element) {
                                try {
                                    const rect = element.getBoundingClientRect();
                                    const computedStyle = window.getComputedStyle(element);
                                    
                                    // Generate description using unified function
                                    const textContent = elementInfo.text || element.textContent?.trim() || '';
                                    const placeholder = element.placeholder || '';
                                    const value = element.value || '';
                                    const href = element.href || '';
                                    const type = element.type || '';
                                    const tagName = elementInfo.tag.toLowerCase();
                                    const classList = Array.from(element.classList);
                                    const ariaLabel = element.getAttribute('aria-label') || '';
                                    const alt = element.getAttribute('alt') || '';
                                    
                                    // Get parent context for better element understanding
                                    let parentContext = '';
                                    let parent = element.parentElement;
                                    for (let i = 0; i < 3 && parent; i++) {
                                        if (parent.tagName && ['H1', 'H2', 'H3', 'H4', 'H5', 'H6'].includes(parent.tagName)) {
                                            parentContext = parent.textContent?.trim() || '';
                                            break;
                                        } else if (parent.tagName === 'FORM') {
                                            parentContext = 'form';
                                            break;
                                        } else if (parent.getAttribute('role') === 'form') {
                                            parentContext = 'form';
                                            break;
                                        } else if (parent.className && typeof parent.className === 'string' && parent.className.includes('form')) {
                                            parentContext = 'form';
                                            break;
                                        }
                                        parent = parent.parentElement;
                                    }
                                
                                // 使用增强的元素类型识别
                                const role = element.getAttribute('role') || '';
                                const identifiedType = identifyElementType(element, tagName, classList, role, ariaLabel, parentContext);
                                
                                const description = generateElementDescription(element, tagName, textContent, placeholder, value, href, type, classList, ariaLabel, alt);
                                
                                elements.push({
                                    element_id: element.id || markId, // Use pure SoM mark ID to match WebAgent
                                    element_type: identifiedType, // 使用识别出的类型而不是elementInfo.type
                                    tag_name: elementInfo.tag,
                                    text_content: elementInfo.text || '',
                                    placeholder: element.placeholder || '',
                                    value: element.value || '',
                                    href: element.href || '',
                                    src: element.src || '',
                                    x: rect.left,
                                    y: rect.top,
                                    width: rect.width,
                                    height: rect.height,
                                    css_classes: Array.from(element.classList),
                                    css_selector: element.tagName.toLowerCase(),
                                    is_clickable: identifiedType === 'clickable' || identifiedType === 'navigation' || identifiedType === 'menu' || identifiedType === 'tab',
                                    is_input: identifiedType === 'input',
                                    is_visible: computedStyle.display !== 'none' && computedStyle.visibility !== 'hidden',
                                    is_enabled: !element.disabled,
                                    input_type: element.type || '',
                                    required: element.required || false,
                                    options: element.tagName === 'SELECT' ? Array.from(element.options).map(opt => opt.text) : [],
                                    attributes: Object.fromEntries(Array.from(element.attributes).map(attr => [attr.name, attr.value])),
                                    som_mark: markId,
                                    som_type: identifiedType, // 使用识别出的类型
                                    description: description
                                });
                                } catch (elemError) {
                                    console.warn(`Error processing element ${markId}:`, elemError);
                                    // Continue with next element
                                }

                            }
                        });
                        
                        console.log(`🔍 Final result: found ${elements.length} elements`);
                        console.log('🔍 Elements:', elements);
                        console.log('🔍 JavaScript execution completed successfully');
                        
                        return elements;
                        } catch (error) {
                            console.error('Error in DOM element extraction:', error);
                            return [];
                        }
                    }
                """, som_mapping)
                
                # Process the returned element data
                if element_data and isinstance(element_data, list):
                    logger.debug(f"🔍 JavaScript execution completed, found {len(element_data)} elements")
                    
                    for i, data in enumerate(element_data):
                        try:
                            logger.debug(f"🔍 Processing element {i+1}/{len(element_data)}: {data.get('element_type', 'unknown')}")
                            
                            # Refine description if needed
                            raw_description = data.get('description', '')
                            refined_description = raw_description
                            
                            # Enhanced type mapping for better Graph Builder alignment
                            raw_element_type = data.get('element_type', 'unknown')
                            enhanced_element_type = self._enhance_element_type_mapping(raw_element_type, data)
                            
                            # Enhanced clickable detection
                            enhanced_is_clickable = self._enhance_clickable_detection(enhanced_element_type, data)
                            
                            # Enhanced input detection
                            enhanced_is_input = self._enhance_input_detection(enhanced_element_type, data)
                            
                            # Create WebElement object with enhanced type mapping
                            element = WebElement(
                                element_id=data.get('element_id', f'element_{i}'),
                                element_type=enhanced_element_type,
                                tag_name=data.get('tag_name', ''),
                                text_content=data.get('text_content', ''),
                                placeholder=data.get('placeholder', ''),
                                value=data.get('value', ''),
                                href=data.get('href', ''),
                                src=data.get('src', ''),
                                x=data.get('x', 0),
                                y=data.get('y', 0),
                                width=data.get('width', 0),
                                height=data.get('height', 0),
                                css_classes=data.get('css_classes', []),
                                css_selector=data.get('css_selector', ''),
                                is_clickable=enhanced_is_clickable,
                                is_input=enhanced_is_input,
                                is_visible=data.get('is_visible', True),
                                is_enabled=data.get('is_enabled', True),
                                input_type=data.get('input_type', ''),
                                required=data.get('required', False),
                                options=data.get('options', []),
                                attributes=data.get('attributes', {}),
                                som_mark=data.get('som_mark', ''),
                                som_type=enhanced_element_type,  # 使用增强的类型
                                description=refined_description
                            )
                            elements.append(element)
                        except Exception as elem_error:
                            logger.warning(f"⚠️ Failed to process element {i+1}: {elem_error}")
                            continue
                    
                    logger.debug(f"🔍 DOM extraction completed, created {len(elements)} WebElement objects")
                elif element_data:
                    logger.warning(f"⚠️ JavaScript execution returned unexpected data type: {type(element_data)}")
                    logger.debug(f"🔍 Data content: {element_data}")
                else:
                    logger.warning("⚠️ JavaScript execution returned no data")
                    
            else:
                # Extract all interactive elements (fallback)
                logger.info("🔍 Extracting all interactive elements (no SoM mapping provided)")
                logger.debug("🔍 About to execute JavaScript for DOM extraction...")
                
                # First, let's test the JavaScript code in smaller chunks to identify the exact error
                try:
                    logger.debug("🔍 Testing basic JavaScript structure...")
                    test_result = await current_page.evaluate("""
                    () => {
                        console.log('🔍 Basic structure test...');
                        return { success: true, message: 'Basic structure works' };
                    }
                    """)
                    logger.debug(f"✅ Basic structure test passed: {test_result}")
                except Exception as e:
                    logger.error(f"❌ Basic structure test failed: {e}")
                    raise
                
                try:
                    logger.debug("🔍 Testing selectors definition...")
                    test_result = await current_page.evaluate("""
                    () => {
                        const interactiveSelectors = [
                            'input', 'button', 'a', 'select', 'textarea',
                            '[onclick]', '[role="button"]', '[tabindex]',
                            'form', 'table', 'img', 'video', 'audio',
                            '[data-action]', '[data-toggle]', '[data-target]',
                            '.btn', '.button', '.form-control', '.form-group',
                            '.search-box', '.search-input', '.filter',
                            '.dropdown', '.menu', '.nav-item',
                            '.card', '.item', '.product', '.article',
                            '.comment', '.reply', '.like', '.share',
                            '.rating', '.star', '.review',
                            '.pagination', '.page-link', '.page-item'
                        ];
                        return { success: true, count: interactiveSelectors.length };
                    }
                    """)
                    logger.debug(f"✅ Selectors test passed: {test_result}")
                except Exception as e:
                    logger.error(f"❌ Selectors test failed: {e}")
                    raise
                
                try:
                    logger.debug("🔍 Testing business data selectors...")
                    test_result = await current_page.evaluate("""
                    () => {
                        const businessDataSelectors = [
                            'table tbody tr', 'table td', 'table th',
                            '.table tbody tr', '.table td', '.table th',
                            '.list-view tbody tr', '.list-view td', '.list-view th',
                            '.data-table tbody tr', '.data-table td', '.data-table th',
                            '.contact-name', '.contact-email', '.contact-phone',
                            '.opportunity-name', '.opportunity-value', '.opportunity-stage',
                            '.account-name', '.account-type', '.account-industry',
                            '.lead-name', '.lead-company', '.lead-status',
                            '.quote-name', '.quote-amount', '.quote-status',
                            '.task-name', '.task-status', '.task-priority',
                            '.meeting-title', '.meeting-date', '.meeting-time',
                            '.document-name', '.document-type', '.document-size',
                            '.email-subject', '.email-sender', '.email-date',
                            '.item-name', '.item-title', '.item-description',
                            '.product-name', '.product-price', '.product-category',
                            '.customer-name', '.customer-email', '.customer-phone',
                            '.project-name', '.project-status', '.project-manager',
                            'input[name*="name"]', 'input[name*="email"]', 'input[name*="phone"]',
                            'input[name*="company"]', 'input[name*="title"]', 'input[name*="subject"]',
                            'textarea[name*="description"]', 'textarea[name*="notes"]',
                            '.data-field', '.field-value', '.info-item',
                            '.detail-item', '.property-value', '.attribute-value'
                        ];
                        return { success: true, count: businessDataSelectors.length };
                    }
                    """)
                    logger.debug(f"✅ Business data selectors test passed: {test_result}")
                except Exception as e:
                    logger.error(f"❌ Business data selectors test failed: {e}")
                    raise
                
                try:
                    logger.debug("🔍 Testing regex patterns...")
                    test_result = await current_page.evaluate("""
                    () => {
                        const testText = 'John Smith';
                        const isBusinessData = (
                            /^[A-Z][a-z]+ [A-Z][a-z]+/.test(testText) ||
                            /^[A-Z][a-z]+$/.test(testText) ||
                            /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(testText) ||
                            testText.includes('@') ||
                            testText.includes('$') ||
                            testText.includes('Active') ||
                            testText.includes('Inactive')
                        );
                        return { success: true, isBusinessData: isBusinessData };
                    }
                    """)
                    logger.debug(f"✅ Regex test passed: {test_result}")
                except Exception as e:
                    logger.error(f"❌ Regex test failed: {e}")
                    raise
                
                try:
                    logger.debug("🔍 Testing element processing logic...")
                    test_result = await current_page.evaluate("""
                    () => {
                        const elements = [];
                        const processedElements = new Set();
                        
                        // Test with a simple element
                        const testElement = document.createElement('button');
                        testElement.textContent = 'Test Button';
                        testElement.id = 'test-button';
                        
                        const textContent = testElement.textContent?.trim() || '';
                        const tagName = testElement.tagName.toLowerCase();
                        
                        let description = '';
                        if (tagName === 'button') {
                            if (textContent) {
                                description = 'Button element, text ' + JSON.stringify(textContent);
                            } else {
                                description = 'Button element, action button';
                            }
                        }
                        
                        elements.push({
                            element_id: testElement.id || 'test_element',
                            element_type: tagName,
                            tag_name: tagName,
                            text_content: textContent,
                            description: description
                        });
                        
                        return { success: true, elements: elements.length, description: description };
                    }
                    """)
                    logger.debug(f"✅ Element processing test passed: {test_result}")
                except Exception as e:
                    logger.error(f"❌ Element processing test failed: {e}")
                    raise
                
                try:
                    logger.debug("🔍 Testing business data processing...")
                    test_result = await current_page.evaluate("""
                    () => {
                        const elements = [];
                        const processedElements = new Set();
                        
                        // Test business data detection
                        const testText = 'John Smith';
                        const isBusinessData = (
                            /^[A-Z][a-z]+ [A-Z][a-z]+/.test(testText) ||
                            /^[A-Z][a-z]+$/.test(testText) ||
                            /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(testText) ||
                            testText.includes('@') ||
                            testText.includes('$') ||
                            testText.includes('Active') ||
                            testText.includes('Inactive')
                        );
                        
                        if (isBusinessData) {
                            let description = '';
                            if (/^[A-Z][a-z]+ [A-Z][a-z]+/.test(testText)) {
                                description = 'Business data element, person name';
                            } else if (/^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(testText)) {
                                description = 'Business data element, email address';
                            } else if (testText.includes('@')) {
                                description = 'Business data element, email address';
                            } else if (testText.includes('$')) {
                                description = 'Business data element, currency amount';
                            } else if (testText.includes('Active') || testText.includes('Inactive')) {
                                description = 'Business data element, status value';
                            } else {
                                description = 'Business data element';
                            }
                            
                            elements.push({
                                element_id: 'test_business',
                                element_type: 'business_data',
                                tag_name: 'div',
                                text_content: testText,
                                description: description
                            });
                        }
                        
                        return { success: true, elements: elements.length, isBusinessData: isBusinessData };
                    }
                    """)
                    logger.debug(f"✅ Business data processing test passed: {test_result}")
                except Exception as e:
                    logger.error(f"❌ Business data processing test failed: {e}")
                    raise
                
                # Now try the full JavaScript code
                logger.debug("🔍 About to execute the complete JavaScript code...")
                element_data = await current_page.evaluate("""
                () => {
                    console.log('🔍 Starting DOM element extraction...');
                    
                    // Unified description generation function
                    function generateElementDescription(element, tagName, textContent, placeholder, value, href, type, classList, ariaLabel, alt) {
                        // Get parent context (heading, form, section)
                        let parentContext = '';
                        let parent = element.parentElement;
                        for (let i = 0; i < 3 && parent; i++) {
                            if (parent.tagName && ['H1', 'H2', 'H3', 'H4', 'H5', 'H6'].includes(parent.tagName)) {
                                parentContext = parent.textContent?.trim() || '';
                                break;
                            } else if (parent.tagName === 'FORM') {
                                parentContext = 'form';
                                break;
                            } else if (parent.getAttribute('role') === 'form') {
                                parentContext = 'form';
                                break;
                            } else if (parent.className && typeof parent.className === 'string' && parent.className.includes('form')) {
                                parentContext = 'form';
                                break;
                            }
                            parent = parent.parentElement;
                        }
                        
                        // Enhanced element type identification function
                        function identifyElementType(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 1. 基于语义和功能的智能识别
                            
                            // 菜单元素识别
                            if (isMenuElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'menu';
                            }
                            
                            // 标签页元素识别
                            if (isTabElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'tab';
                            }
                            
                            // 过滤器元素识别
                            if (isFilterElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'filter';
                            }
                            
                            // 模态框元素识别
                            if (isModalElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'modal';
                            }
                            
                            // 表单元素识别
                            if (isFormElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'form';
                            }
                            
                            // 下拉菜单识别
                            if (isDropdownElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'dropdown';
                            }
                            
                            // 面包屑识别
                            if (isBreadcrumbElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'breadcrumb';
                            }
                            
                            // 分页器识别
                            if (isPaginatorElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'paginator';
                            }
                            
                            // 通知元素识别
                            if (isNotificationElement(element, tagName, classList, role, ariaLabel, parentContext)) {
                                return 'toast';
                            }
                            
                            // 2. 基于标签名的基本识别（保持原有逻辑作为后备）
                            if (tagName === 'button') {
                                return 'clickable';
                            } else if (tagName === 'input') {
                                return 'input';
                            } else if (tagName === 'a') {
                                // 智能链接类型识别
                                if (classList.some(cls => cls.toLowerCase().includes('nav'))) {
                                    return 'navigation';
                                } else if (classList.some(cls => cls.toLowerCase().includes('menu'))) {
                                    return 'menu_link';
                                } else if (classList.some(cls => cls.toLowerCase().includes('detail'))) {
                                    return 'detail_link';
                                } else {
                                    return 'link';
                                }
                            } else if (tagName === 'select') {
                                return 'select';
                            } else if (tagName === 'textarea') {
                                return 'input';
                            } else if (tagName === 'table') {
                                return 'table';
                            } else if (tagName === 'img') {
                                return 'image';
                            } else if (tagName === 'form') {
                                return 'form';
                            }
                            
                            // 3. 默认类型
                            return 'clickable';
                        }
                        
                        // Helper functions for element type identification
                        function isMenuElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查role属性
                            if (role === 'menu' || role === 'menuitem' || role === 'menubar') {
                                return true;
                            }
                            
                            // 检查CSS类名
                            const menuKeywords = ['menu', 'nav-menu', 'main-menu', 'primary-menu', 'sidebar-menu', 'navbar-nav'];
                            if (classList.some(cls => menuKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查aria标签
                            if (ariaLabel && ariaLabel.toLowerCase().includes('menu')) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (parentLower.includes('nav') || parentLower.includes('sidebar') || 
                                    parentLower.includes('menu') || parentLower.includes('navbar')) {
                                    return true;
                                }
                            }
                            
                            // 检查是否在导航容器中
                            const parent = element.parentElement;
                            if (parent) {
                                const parentClass = parent.className || '';
                                const parentTag = parent.tagName.toLowerCase();
                                if (parentTag === 'nav' || parentClass.toLowerCase().includes('nav') || 
                                    parentClass.toLowerCase().includes('menu')) {
                                    return true;
                                }
                            }
                            
                            return false;
                        }
                        
                        function isTabElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 首先排除table元素
                            if (tagName === 'table') {
                                return false;
                            }
                            
                            // 检查role属性
                            if (role === 'tab' || role === 'tablist') {
                                return true;
                            }
                            
                            // 检查CSS类名
                            const tabKeywords = ['tab', 'tab-item', 'nav-tab', 'tab-nav'];
                            if (classList.some(cls => tabKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查aria标签
                            if (ariaLabel && ariaLabel.toLowerCase().includes('tab')) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (parentLower.includes('tab') || parentLower.includes('nav-tabs')) {
                                    return true;
                                }
                            }
                            
                            // 检查是否在标签页容器中
                            const parent = element.parentElement;
                            if (parent) {
                                const parentClass = parent.className || '';
                                if (parentClass.toLowerCase().includes('tab') || parentClass.toLowerCase().includes('nav-tabs')) {
                                    return true;
                                }
                            }
                            
                            return false;
                        }
                        
                        function isFilterElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查CSS类名
                            const filterKeywords = ['filter', 'filters', 'filter-panel', 'filter-controls', 'filter-options', 'search-form'];
                            if (classList.some(cls => filterKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查aria标签
                            if (ariaLabel && (ariaLabel.toLowerCase().includes('filter') || ariaLabel.toLowerCase().includes('search'))) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (parentLower.includes('filter') || parentLower.includes('search') || 
                                    parentLower.includes('advanced-search')) {
                                    return true;
                                }
                            }
                            
                            // 检查是否在过滤器容器中
                            const parent = element.parentElement;
                            if (parent) {
                                const parentClass = parent.className || '';
                                if (parentClass.toLowerCase().includes('filter') || parentClass.toLowerCase().includes('search')) {
                                    return true;
                                }
                            }
                            
                            // 检查data属性
                            if (element.hasAttribute('data-filter') || element.hasAttribute('data-search')) {
                                return true;
                            }
                            
                            return false;
                        }
                        
                        function isModalElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查role属性
                            if (role === 'dialog' || role === 'modal') {
                                return true;
                            }
                            
                            // 检查CSS类名
                            const modalKeywords = ['modal', 'dialog', 'popup', 'overlay', 'lightbox'];
                            if (classList.some(cls => modalKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查aria标签
                            if (ariaLabel && (ariaLabel.toLowerCase().includes('modal') || ariaLabel.toLowerCase().includes('dialog'))) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (parentLower.includes('modal') || parentLower.includes('dialog')) {
                                    return true;
                                }
                            }
                            
                            return false;
                        }
                        
                        function isFormElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查标签名
                            if (tagName === 'form') {
                                return true;
                            }
                            
                            // 检查CSS类名
                            const formKeywords = ['form', 'form-group', 'form-control', 'form-container'];
                            if (classList.some(cls => formKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (parentLower.includes('form') || parentLower.includes('form-group')) {
                                    return true;
                                }
                            }
                            
                            // 检查是否在表单容器中
                            const parent = element.parentElement;
                            if (parent) {
                                const parentClass = parent.className || '';
                                if (parentClass.toLowerCase().includes('form')) {
                                    return true;
                                }
                            }
                            
                            return false;
                        }
                        
                        function isDropdownElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // IMPORTANT: Check if this is actually a navigation element first
                            // Navigation elements with dropdown-toggle should be treated as navigation, not dropdown
                            const hasNavigationClass = classList.some(cls => cls.toLowerCase().includes('nav'));
                            if (hasNavigationClass) {
                                return false; // Let navigation logic handle it
                            }
                            
                            // 检查CSS类名
                            const dropdownKeywords = ['dropdown', 'drop-down', 'select-dropdown', 'menu-dropdown'];
                            if (classList.some(cls => dropdownKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查aria标签
                            if (ariaLabel && ariaLabel.toLowerCase().includes('dropdown')) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (parentLower.includes('dropdown')) {
                                    return true;
                                }
                            }
                            
                            return false;
                        }
                        
                        function isBreadcrumbElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查role属性
                            if (role === 'navigation' && ariaLabel && ariaLabel.toLowerCase().includes('breadcrumb')) {
                                return true;
                            }
                            
                            // 检查CSS类名
                            const breadcrumbKeywords = ['breadcrumb', 'breadcrumbs', 'crumb'];
                            if (classList.some(cls => breadcrumbKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            return false;
                        }
                        
                        function isPaginatorElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查CSS类名
                            const paginatorKeywords = ['pagination', 'paginator', 'page-link', 'page-item'];
                            if (classList.some(cls => paginatorKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            // 检查父级上下文
                            if (parentContext) {
                                const parentLower = parentContext.toLowerCase();
                                if (parentLower.includes('pagination') || parentLower.includes('paginator')) {
                                    return true;
                                }
                            }
                            
                            return false;
                        }
                        
                        function isNotificationElement(element, tagName, classList, role, ariaLabel, parentContext) {
                            // 检查role属性
                            if (role === 'alert' || role === 'status') {
                                return true;
                            }
                            
                            // 检查CSS类名
                            const notificationKeywords = ['toast', 'notification', 'alert', 'message', 'snackbar'];
                            if (classList.some(cls => notificationKeywords.some(keyword => cls.toLowerCase().includes(keyword)))) {
                                return true;
                            }
                            
                            return false;
                        }
                        
                        // Enhanced element description generation function
                        function generateElementDescription(element, tagName, textContent, placeholder, value, href, type, classList, ariaLabel, alt, parentContext) {
                            // Rule-based description generation
                            let description = '';
                            if (tagName === 'button') {
                                if (textContent) {
                                    description = 'Button element, text ' + JSON.stringify(textContent);
                                } else if (classList.some(cls => cls.includes('search'))) {
                                    description = 'Button element, search button';
                                } else if (classList.some(cls => cls.includes('filter'))) {
                                    description = 'Button element, filter button';
                                } else if (classList.some(cls => cls.includes('submit'))) {
                                    description = 'Button element, submit button';
                                } else {
                                    description = 'Button element, action button';
                                }
                            } else if (tagName === 'input') {
                            if (type === 'text' || type === 'search') {
                                if (placeholder) {
                                    description = 'Input element, type ' + type + ', placeholder ' + JSON.stringify(placeholder);
                                } else if (classList.some(cls => cls.includes('search'))) {
                                    description = 'Input element, type search, search field';
                                } else {
                                    description = 'Input element, type ' + type;
                                }
                            } else if (type === 'email') {
                                description = 'Input element, type email, email field';
                            } else if (type === 'password') {
                                description = 'Input element, type password, password field';
                            } else if (type === 'submit') {
                                description = 'Input element, type submit, submit button';
                            } else {
                                description = 'Input element, type ' + type;
                            }
                        } else if (tagName === 'a') {
                            if (textContent) {
                                description = 'Link element, text ' + JSON.stringify(textContent);
                            } else if (href) {
                                description = 'Link element, navigation link';
                            } else {
                                description = 'Link element, clickable link';
                            }
                        } else if (tagName === 'select') {
                            description = 'Select element, dropdown selection field';
                        } else if (tagName === 'textarea') {
                            description = 'Textarea element, multi-line text input field';
                        } else if (tagName === 'table') {
                            description = 'Table element, data table';
                        } else if (tagName === 'img') {
                            if (alt) {
                                description = 'Image element, alt text ' + JSON.stringify(alt);
                            } else {
                                description = 'Image element';
                            }
                        } else if (tagName === 'form') {
                            description = 'Form element, form container';
                        } else {
                            description = tagName + ' element';
                        }
                        
                        // Add parent context if available
                        if (parentContext) {
                            description += ', inside ' + JSON.stringify(parentContext);
                        }
                        
                        return description;
                    }
                    
                    const elements = [];
                    const processedElements = new Set(); // Track processed elements to avoid duplicates
                    
                    // Interactive elements selectors
                    const interactiveSelectors = [
                        'input', 'button', 'a', 'select', 'textarea',
                        '[onclick]', '[role="button"]', '[tabindex]',
                        'form', 'table', 'img', 'video', 'audio',
                        '[data-action]', '[data-toggle]', '[data-target]',
                        '.btn', '.button', '.form-control', '.form-group',
                        '.search-box', '.search-input', '.filter',
                        '.dropdown', '.menu', '.nav-item',
                        '.card', '.item', '.product', '.article',
                        '.comment', '.reply', '.like', '.share',
                        '.rating', '.star', '.review',
                        '.pagination', '.page-link', '.page-item'
                    ];
                    
                    // Business data selectors - specifically for CRM data
                    const businessDataSelectors = [
                        // Table rows and cells containing business data
                        'table tbody tr', 'table td', 'table th',
                        '.table tbody tr', '.table td', '.table th',
                        '.list-view tbody tr', '.list-view td', '.list-view th',
                        '.data-table tbody tr', '.data-table td', '.data-table th',
                        
                        // CRM specific elements
                        '.contact-name', '.contact-email', '.contact-phone',
                        '.opportunity-name', '.opportunity-value', '.opportunity-stage',
                        '.account-name', '.account-type', '.account-industry',
                        '.lead-name', '.lead-company', '.lead-status',
                        '.quote-name', '.quote-amount', '.quote-status',
                        '.task-name', '.task-status', '.task-priority',
                        '.meeting-title', '.meeting-date', '.meeting-time',
                        '.document-name', '.document-type', '.document-size',
                        '.email-subject', '.email-sender', '.email-date',
                        
                        // Generic business data containers
                        '.item-name', '.item-title', '.item-description',
                        '.product-name', '.product-price', '.product-category',
                        '.customer-name', '.customer-email', '.customer-phone',
                        '.project-name', '.project-status', '.project-manager',
                        
                        // Form fields with business data
                        'input[name*="name"]', 'input[name*="email"]', 'input[name*="phone"]',
                        'input[name*="company"]', 'input[name*="title"]', 'input[name*="subject"]',
                        'textarea[name*="description"]', 'textarea[name*="notes"]',
                        
                        // Data display elements
                        '.data-field', '.field-value', '.info-item',
                        '.detail-item', '.property-value', '.attribute-value'
                    ];
                    
                    // Process interactive elements
                    console.log('🔍 Processing interactive elements...');
                    interactiveSelectors.forEach(selector => {
                        const foundElements = document.querySelectorAll(selector);
                        console.log(`🔍 Found ${foundElements.length} elements for selector: ${selector}`);
                        foundElements.forEach((el, index) => {
                            const rect = el.getBoundingClientRect();
                            const computedStyle = window.getComputedStyle(el);
                            const elementSignature = `${el.tagName}-${el.textContent?.trim()}-${rect.left}-${rect.top}`;
                            
                            if (processedElements.has(elementSignature)) {
                                return; // Skip duplicate elements
                            }
                            processedElements.add(elementSignature);
                            
                            // Extract element information for description generation
                            const textContent = el.textContent?.trim() || '';
                            const placeholder = el.placeholder || '';
                            const value = el.value || '';
                            const href = el.href || '';
                            const type = el.type || '';
                            const tagName = el.tagName.toLowerCase();
                            const classList = Array.from(el.classList);
                            const ariaLabel = el.getAttribute('aria-label') || '';
                            const alt = el.getAttribute('alt') || '';
                            
                            // Generate description using unified function
                            console.log(`🔍 Generating description for ${tagName} element with text: ${textContent}`);
                            const description = generateElementDescription(el, tagName, textContent, placeholder, value, href, type, classList, ariaLabel, alt);
                            console.log(`🔍 Final description: ${description}`);
                            
                            elements.push({
                                element_id: el.id || `${selector}_${index}`,
                                element_type: el.tagName.toLowerCase(),
                                tag_name: el.tagName.toLowerCase(),
                                text_content: textContent,
                                placeholder: placeholder,
                                value: value,
                                href: href,
                                src: el.src || '',
                                x: rect.left,
                                y: rect.top,
                                width: rect.width,
                                height: rect.height,
                                css_classes: classList,
                                css_selector: selector,
                                is_clickable: el.onclick !== null || el.tagName === 'A' || el.tagName === 'BUTTON' || el.classList.contains('btn') || el.classList.contains('button'),
                                is_input: el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.classList.contains('form-control') || el.classList.contains('search-input'),
                                is_visible: computedStyle.display !== 'none' && computedStyle.visibility !== 'hidden',
                                is_enabled: !el.disabled,
                                input_type: type,
                                required: el.required || false,
                                options: el.tagName === 'SELECT' ? Array.from(el.options).map(opt => opt.text) : [],
                                attributes: Object.fromEntries(Array.from(el.attributes).map(attr => [attr.name, attr.value])),
                                description: description
                            });
                        });
                    });
                    
                    // Process business data elements
                    console.log('🔍 Processing business data elements...');
                    businessDataSelectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach((el, index) => {
                            const rect = el.getBoundingClientRect();
                            const computedStyle = window.getComputedStyle(el);
                            const textContent = el.textContent?.trim() || '';
                            
                            // Skip empty or very short content
                            if (!textContent || textContent.length < 2) {
                                return;
                            }
                            
                            // Skip if it's just a single character or common UI text
                            if (textContent.length === 1 || 
                                ['x', '-', '+', '=', '>', '<', '|', '/'].includes(textContent)) {
                                return;
                            }
                            
                            const elementSignature = `${el.tagName}-${textContent}-${rect.left}-${rect.top}`;
                            
                            if (processedElements.has(elementSignature)) {
                                return; // Skip duplicate elements
                            }
                            processedElements.add(elementSignature);
                            
                            // Determine if this is business data based on content patterns
                            const isBusinessData = (
                                // Names (2+ words or single word with capital letters)
                                /^[A-Z][a-z]+ [A-Z][a-z]+/.test(textContent) ||
                                /^[A-Z][a-z]+$/.test(textContent) ||
                                // Email addresses
                                /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(textContent) ||
                                // Phone numbers
                                /^[+]?[0-9\\s\\-\\(\\)]+$/.test(textContent) ||
                                // Currency amounts
                                /^\\$?[0-9,]+(\\.?[0-9]{2})?$/.test(textContent) ||
                                // Dates
                                /^[0-9]{1,2}\\/[0-9]{1,2}\\/[0-9]{4}$/.test(textContent) ||
                                /^[A-Za-z]+ [0-9]{1,2},? [0-9]{4}$/.test(textContent) ||
                                // Status values
                                /^(Active|Inactive|Pending|Completed|Open|Closed|New|Old)$/i.test(textContent) ||
                                // Company names (multiple words with capitals)
                                /^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+/.test(textContent) ||
                                // CRM-specific business terms (case-insensitive)
                                /^(Accounts?|Contacts?|Opportunities?|Leads?|Quotes?|Calendar|Documents?|Emails?|Meetings?|Tasks?|Projects?|Companies?|Customers?|Vendors?|Partners?)$/i.test(textContent) ||
                                // Simple patterns for now
                                textContent.includes('@') ||
                                textContent.includes('$') ||
                                textContent.includes('Active') ||
                                textContent.includes('Inactive') ||
                                textContent.includes('Contact') ||
                                textContent.includes('Lead') ||
                                textContent.includes('Opportunity') ||
                                textContent.includes('Account') ||
                                textContent.includes('Quote') ||
                                textContent.includes('Meeting') ||
                                textContent.includes('Task') ||
                                textContent.includes('Document') ||
                                textContent.includes('Email') ||
                                textContent.includes('Project') ||
                                textContent.includes('Company') ||
                                textContent.includes('Customer') ||
                                textContent.includes('Vendor') ||
                                textContent.includes('Partner')
                            );
                            
                            if (isBusinessData) {
                                // Generate description for business data
                                let description = '';
                                if (/^[A-Z][a-z]+ [A-Z][a-z]+/.test(textContent)) {
                                    description = 'Business data element, person name';
                                } else if (/^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(textContent)) {
                                    description = 'Business data element, email address';
                                } else if (textContent.includes('@')) {
                                    description = 'Business data element, email address';
                                } else if (textContent.includes('$')) {
                                    description = 'Business data element, currency amount';
                                } else if (textContent.includes('Active') || textContent.includes('Inactive')) {
                                    description = 'Business data element, status value';
                                } else {
                                    description = 'Business data element';
                                }
                                
                                elements.push({
                                    element_id: el.id || `business_${selector}_${index}`,
                                    element_type: 'business_data',
                                    tag_name: el.tagName.toLowerCase(),
                                    text_content: textContent,
                                    placeholder: el.placeholder || '',
                                    value: el.value || '',
                                    href: el.href || '',
                                    src: el.src || '',
                                    x: rect.left,
                                    y: rect.top,
                                    width: rect.width,
                                    height: rect.height,
                                    css_classes: Array.from(el.classList),
                                    css_selector: selector,
                                    is_clickable: el.onclick !== null || el.tagName === 'A' || el.tagName === 'BUTTON' || el.classList.contains('btn') || el.classList.contains('button') || el.classList.contains('clickable') || el.classList.contains('selectable') || el.classList.contains('expandable') || el.getAttribute('data-clickable') === 'true' || el.getAttribute('role') === 'button' || el.getAttribute('tabindex') !== null,
                                    is_input: false,
                                    is_visible: computedStyle.display !== 'none' && computedStyle.visibility !== 'hidden',
                                    is_enabled: true,
                                    input_type: '',
                                    required: false,
                                    options: [],
                                    attributes: Object.fromEntries(Array.from(el.attributes).map(attr => [attr.name, attr.value])),
                                    description: description
                                });
                            }
                        });
                    });
                    
                    console.log(`🔍 DOM extraction completed, found ${elements.length} total elements`);
                    return elements;
                }
            """)
            
            logger.debug(f"🔍 JavaScript execution completed, found {len(element_data)} elements")
            
            # Convert to WebElement objects
            for i, data in enumerate(element_data):
                logger.debug(f"🔍 Processing element {i+1}/{len(element_data)}: {data.get('element_type', 'unknown')}")
                
                # Get raw description from data
                raw_description = data.get('description', '')
                logger.debug(f"🔍 Raw description: {raw_description}")
                
                # Refine description if refiner is available
                refined_description = raw_description
                if self.description_refiner and raw_description:
                    logger.debug("🔍 Refining description with LLM...")
                    refined_description = self.description_refiner.refine_description(raw_description)
                    logger.debug(f"🔍 Refined description: {refined_description}")
                
                element = WebElement(
                    element_id=data['element_id'],
                    element_type=data['element_type'],
                    tag_name=data['tag_name'],
                    text_content=data['text_content'],
                    placeholder=data['placeholder'],
                    value=data['value'],
                    href=data['href'],
                    src=data['src'],
                    x=data['x'],
                    y=data['y'],
                    width=data['width'],
                    height=data['height'],
                    css_classes=data['css_classes'],
                    css_selector=data['css_selector'],
                    is_clickable=data['is_clickable'],
                    is_input=data['is_input'],
                    is_visible=data['is_visible'],
                    is_enabled=data['is_enabled'],
                    input_type=data['input_type'],
                    required=data['required'],
                    options=data['options'],
                    attributes=data['attributes'],
                    som_mark=data.get('som_mark', ''),
                    som_type=data.get('som_type', ''),
                    description=refined_description
                )
                elements.append(element)
            
            logger.debug(f"🔍 DOM extraction completed, created {len(elements)} WebElement objects")
                
        except Exception as e:
            logger.error(f"Error extracting DOM elements with Playwright: {e}")
            logger.debug(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.debug(f"🔍 Full traceback: {traceback.format_exc()}")
        
        return elements
    
    def _extract_dom_elements_selenium(self, driver=None) -> List[WebElement]:
        """Extract DOM elements using Selenium"""
        
        elements = []
        
        try:
            # Get all interactive elements
            selectors = [
                'input', 'button', 'a', 'select', 'textarea',
                '[onclick]', '[role="button"]', '[tabindex]',
                'form', 'table', 'img', 'video', 'audio',
                # 增加更多交互元素选择器
                '[data-action]', '[data-toggle]', '[data-target]',
                '.btn', '.button', '.form-control', '.form-group',
                '.search-box', '.search-input', '.filter',
                '.dropdown', '.menu', '.nav-item',
                '.card', '.item', '.product', '.article',
                '.comment', '.reply', '.like', '.share',
                '.rating', '.star', '.review',
                '.pagination', '.page-link', '.page-item'
            ]
            
            # Use provided driver or fall back to self.driver
            current_driver = driver or self.driver
            if not current_driver:
                logger.error("No driver available for DOM extraction")
                return elements
                
            for selector in selectors:
                try:
                    found_elements = current_driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    for index, el in enumerate(found_elements):
                        try:
                            # Get element properties
                            rect = el.rect
                            computed_style = current_driver.execute_script(
                                "return window.getComputedStyle(arguments[0]);", el
                            )
                            
                            # Get element attributes
                            attributes = {}
                            for attr_name in ['id', 'class', 'type', 'href', 'src', 'placeholder', 'value']:
                                attr_value = el.get_attribute(attr_name)
                                if attr_value:
                                    attributes[attr_name] = attr_value
                            
                            # Get options for select elements
                            options = []
                            if el.tag_name == 'select':
                                option_elements = el.find_elements(By.TAG_NAME, 'option')
                                options = [opt.text for opt in option_elements if opt.text]
                            
                            element = WebElement(
                                element_id=el.get_attribute('id') or f"{selector}_{index}",
                                element_type=el.tag_name,
                                tag_name=el.tag_name,
                                text_content=el.text.strip(),
                                placeholder=el.get_attribute('placeholder') or '',
                                value=el.get_attribute('value') or '',
                                href=el.get_attribute('href') or '',
                                src=el.get_attribute('src') or '',
                                x=rect['x'],
                                y=rect['y'],
                                width=rect['width'],
                                height=rect['height'],
                                css_classes=el.get_attribute('class').split() if el.get_attribute('class') else [],
                                css_selector=selector,
                                is_clickable=el.tag_name in ['a', 'button'] or el.get_attribute('onclick') is not None or 'btn' in (el.get_attribute('class') or '') or 'button' in (el.get_attribute('class') or ''),
                                is_input=el.tag_name in ['input', 'textarea', 'select'] or 'form-control' in (el.get_attribute('class') or '') or 'search-input' in (el.get_attribute('class') or ''),
                                is_visible=el.is_displayed(),
                                is_enabled=el.is_enabled(),
                                input_type=el.get_attribute('type') or '',
                                required=el.get_attribute('required') is not None,
                                options=options,
                                attributes=attributes
                            )
                            elements.append(element)
                            
                        except Exception as e:
                            logger.warning(f"Error processing element {selector}: {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error finding elements with selector {selector}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting DOM elements with Selenium: {e}")
        
        return elements
    
    
    
    def _save_dom_data(self, page_data: WebPageData, dom_file: Path):
        """Save DOM data to JSON file"""
        
        dom_data = {
            "url": page_data.url,
            "title": page_data.title,
            "page_type": page_data.page_type,
            "load_time": page_data.load_time,
            "page_size": page_data.page_size,
            "elements": [element.to_dict() for element in page_data.elements],
            "links": page_data.links,
            "clickable_elements": page_data.clickable_elements,
            "form_elements": page_data.form_elements,
            "table_elements": page_data.table_elements,
            "collected_at": time.time()
        }
        
        with open(dom_file, 'w', encoding='utf-8') as f:
            json.dump(dom_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved DOM data to {dom_file}")
    

    
    async def collect_single_page(self, url: str) -> WebPageData:
        """Collect data from a single web page"""
        
        logger.info(f"Collecting data from single page: {url}")
        
        # Use real browser automation
        if self.use_playwright and PLAYWRIGHT_AVAILABLE:
            return await self._collect_single_page_playwright(self.page, url, 0)
        elif SELENIUM_AVAILABLE:
            # TODO: Implement Selenium single page collection
            raise NotImplementedError("Selenium single page collection not implemented yet")
        else:
            raise RuntimeError("No browser automation tools available (Playwright or Selenium)")
    
    def save_web_data(self, web_pages: List[WebPageData], output_file: str = "web_data.json"):
        """Save collected web data to file"""
        
        output_path = self.output_dir / output_file
        
        data = {
            "collection_time": time.time(),
            "total_pages": len(web_pages),
            "pages": [page.to_dict() for page in web_pages]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved web data to {output_path}")
    
    def load_web_data(self, input_file: str = "web_data.json") -> List[WebPageData]:
        """Load web data from file"""
        
        input_path = self.output_dir / input_file
        
        if not input_path.exists():
            logger.warning(f"Web data file not found: {input_path}")
            return []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        web_pages = []
        for page_data in data.get('pages', []):
            elements = [WebElement(**elem_data) for elem_data in page_data.get('elements', [])]
            page = WebPageData(
                url=page_data['url'],
                title=page_data['title'],
                elements=elements,
                screenshots=page_data.get('screenshots', []),
                load_time=page_data.get('load_time', 0.0),
                page_size=page_data.get('page_size', 0),
                links=page_data.get('links', []),
                clickable_elements=page_data.get('clickable_elements', []),
                form_elements=page_data.get('form_elements', []),
                table_elements=page_data.get('table_elements', []),
                page_type=page_data.get('page_type', 'content')
            )
            web_pages.append(page)
        
        logger.info(f"Loaded {len(web_pages)} web pages from {input_path}")
        return web_pages


    async def _explore_with_playwright(self, start_urls: List[str], max_depth: int, max_pages_per_depth: int, 
                                     all_pages: List[WebPageData], visited_urls: Set[str]):
        """Explore website using Playwright with intelligent priority-based collection"""
        
        try:
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(
                    headless=self.config.get('headless', True),
                    args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
                )
                
                # Create new page
                page = await browser.new_page()
                await page.set_viewport_size({"width": 1280, "height": 720})
                
                # Use priority queue for intelligent collection
                from queue import PriorityQueue
                page_queue = PriorityQueue()
                
                # Initialize queue, sort starting URLs by priority
                for url in start_urls:
                    if url not in visited_urls:
                        # First collect page information to calculate priority
                        try:
                            await page.goto(url, wait_until='networkidle', timeout=self.timeout * 1000)
                            await asyncio.sleep(3)  # Increased from 2 to 3 seconds
                            
                            title = await page.title()
                            content = await page.content()
                            
                            # Temporarily collect elements to calculate priority
                            som_mapping = await self._mark_elements_with_som(page)
                            elements = await self._extract_dom_elements_playwright(page, som_mapping)
                            
                            priority = self._calculate_page_priority(url, title, content, elements)
                            
                            # Priority queue uses negative numbers because Python PriorityQueue is min-heap
                            page_queue.put((-priority, url, 0))  # (negative priority, URL, depth)
                            
                            logger.info(f"🎯 Page {url} priority: {priority}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to calculate priority for {url}: {e}")
                            # If unable to calculate priority, use default priority
                            page_queue.put((-100, url, 0))
                
                # Start priority collection
                while not page_queue.empty() and len(all_pages) < self.max_pages:
                    priority, url, depth = page_queue.get()
                    priority = -priority  # Restore positive priority
                    
                    if url in visited_urls or depth > max_depth:
                        continue
                    
                    logger.info(f"🔍 Exploring page (priority: {priority}, depth: {depth}): {url}")
                    
                    # Collect current page
                    page_data = await self._collect_single_page_playwright(page, url, depth)
                    if page_data:
                        all_pages.append(page_data)
                        visited_urls.add(url)
                        
                        # If there is still space, explore links
                        if len(all_pages) < self.max_pages and depth < max_depth:
                            # Get explorable links
                            links_to_explore = await self._find_explorable_links_playwright(
                                page, url, max_pages_per_depth
                            )
                            
                            # Calculate priority for each link and add to queue
                            for link_url in links_to_explore:
                                if link_url not in visited_urls and len(all_pages) < self.max_pages:
                                    try:
                                        # Navigate to link page to calculate priority
                                        await page.goto(link_url, wait_until='networkidle', timeout=self.timeout * 1000)
                                        await asyncio.sleep(3)  # Increased from 2 to 3 seconds
                                        
                                        title = await page.title()
                                        content = await page.content()
                                        
                                        # Temporarily collect elements
                                        som_mapping = await self._mark_elements_with_som(page)
                                        elements = await self._extract_dom_elements_playwright(page, som_mapping)
                                        
                                        link_priority = self._calculate_page_priority(link_url, title, content, elements)
                                        
                                        # Depth penalty
                                        link_priority -= depth * 50
                                        
                                        page_queue.put((-link_priority, link_url, depth + 1))
                                        
                                        logger.debug(f"  📎 Link {link_url} priority: {link_priority}")
                                        
                                    except Exception as e:
                                        logger.warning(f"Failed to calculate priority for link {link_url}: {e}")
                                        # Use default priority
                                        page_queue.put((-50, link_url, depth + 1))
                
                # Close browser
                await browser.close()
                
        except Exception as e:
            logger.error(f"Error with Playwright exploration: {e}")
    
    async def _find_explorable_interactive_elements_playwright(self, page, current_url: str, max_elements: int = 3) -> List[Dict[str, Any]]:
        """Find explorable interactive elements (buttons, forms, etc.) that can lead to new pages"""
        
        try:
            # Extract interactive elements that might lead to new pages
            interactive_elements = await page.evaluate("""
                () => {
                    const elements = [];
                    
                    // Find buttons that might lead to new pages
                    const buttons = Array.from(document.querySelectorAll('button, input[type="submit"], input[type="button"]'));
                    buttons.forEach((btn, index) => {
                        if (btn.offsetWidth > 0 && btn.offsetHeight > 0 && btn.textContent.trim()) {
                            // Create a more specific selector based on element type
                            let selector;
                            if (btn.tagName === 'BUTTON') {
                                selector = `button:nth-of-type(${index + 1})`;
                            } else if (btn.type === 'submit') {
                                selector = `input[type="submit"]:nth-of-type(${index + 1})`;
                            } else {
                                selector = `input[type="button"]:nth-of-type(${index + 1})`;
                            }
                            
                            elements.push({
                                type: 'button',
                                text: btn.textContent.trim(),
                                selector: selector,
                                onclick: btn.onclick ? 'has_onclick' : 'no_onclick',
                                form: btn.form ? 'has_form' : 'no_form'
                            });
                        }
                    });
                    
                    // Find form submissions that might lead to new pages
                    const forms = Array.from(document.querySelectorAll('form'));
                    forms.forEach((form, index) => {
                        if (form.action && form.action !== window.location.href) {
                            elements.push({
                                type: 'form',
                                action: form.action,
                                method: form.method || 'GET',
                                selector: `form:nth-of-type(${index + 1})`,
                                inputs: Array.from(form.querySelectorAll('input')).length
                            });
                        }
                    });
                    
                    // Find clickable divs or spans that might be navigation elements
                    const clickableDivs = Array.from(document.querySelectorAll('div[onclick], span[onclick], div[role="button"], span[role="button"]'));
                    clickableDivs.forEach((div, index) => {
                        if (div.offsetWidth > 0 && div.offsetHeight > 0 && div.textContent.trim()) {
                            elements.push({
                                type: 'clickable_div',
                                text: div.textContent.trim(),
                                selector: `${div.tagName.toLowerCase()}[onclick]:nth-of-type(${index + 1})`,
                                onclick: div.onclick ? 'has_onclick' : 'no_onclick'
                            });
                        }
                    });
                    
                    return elements.slice(0, 20); // Limit to 20 elements
                }
            """)
            
            # Filter and prioritize elements
            explorable_elements = []
            for element in interactive_elements:
                # Skip elements that are likely not navigation-related
                if element['type'] == 'button':
                    text = element['text'].lower()
                    # Look for navigation-related button text
                    if any(keyword in text for keyword in ['submit', 'next', 'continue', 'proceed', 'search', 'go', 'enter', 'login', 'sign']):
                        explorable_elements.append(element)
                elif element['type'] == 'form':
                    # Include forms that might lead to new pages
                    explorable_elements.append(element)
                elif element['type'] == 'clickable_div':
                    text = element['text'].lower()
                    # Look for navigation-related text
                    if any(keyword in text for keyword in ['next', 'continue', 'proceed', 'search', 'go', 'enter', 'login', 'sign']):
                        explorable_elements.append(element)
                
                if len(explorable_elements) >= max_elements:
                    break
            
            logger.info(f"🔍 Found {len(explorable_elements)} explorable interactive elements")
            return explorable_elements
            
        except Exception as e:
            logger.error(f"Error finding explorable interactive elements: {e}")
            return []

    async def _explore_interactive_element_playwright(self, page, element: Dict[str, Any], current_url: str) -> Optional[str]:
        """Explore an interactive element and return the new URL if navigation occurs"""
        
        try:
            original_url = page.url
            
            # Click the element
            if element['type'] == 'button':
                await page.click(element['selector'])
            elif element['type'] == 'form':
                # For forms, try to fill and submit
                form_selector = element['selector']
                inputs = await page.query_selector_all(f"{form_selector} input")
                
                # Fill form inputs with dummy data
                for i, input_elem in enumerate(inputs):
                    input_type = await input_elem.get_attribute('type')
                    if input_type in ['text', 'email', 'search']:
                        await input_elem.fill(f"test_data_{i}")
                    elif input_type == 'password':
                        await input_elem.fill("test_password")
                
                # Submit the form
                submit_button = await page.query_selector(f"{form_selector} input[type='submit'], {form_selector} button[type='submit']")
                if submit_button:
                    await submit_button.click()
                else:
                    # Try pressing Enter
                    await page.keyboard.press('Enter')
            elif element['type'] == 'clickable_div':
                await page.click(element['selector'])
            
            # Wait for navigation
            await asyncio.sleep(3)
            
            # Check if URL changed
            new_url = page.url
            if new_url != original_url:
                logger.info(f"✅ Interactive element navigation successful: {original_url} -> {new_url}")
                return new_url
            else:
                logger.info(f"⚠️ Interactive element click did not change URL: {element['type']} - {element.get('text', element.get('action', 'unknown'))}")
                return None
                
        except Exception as e:
            logger.warning(f"Error exploring interactive element: {e}")
            return None
    
    async def _explore_from_url_playwright(self, page, url: str, max_depth: int, max_pages_per_depth: int,
                                         all_pages: List[WebPageData], visited_urls: Set[str], current_depth: int):
        """Recursively explore from a URL using Playwright with enhanced interactive element exploration"""
        
        if current_depth >= max_depth or url in visited_urls or len(all_pages) >= self.max_pages:
            return
        
        visited_urls.add(url)
        logger.info(f"Exploring depth {current_depth}: {url}")
        
        try:
            # Navigate to page
            await page.goto(url, wait_until='networkidle', timeout=self.timeout * 1000)
            await asyncio.sleep(2)  # Wait for page load
            
            # Collect current page data
            page_data = await self._collect_single_page_playwright(page, url, current_depth)
            
            # Only add page if it passed quality filters
            if page_data is not None:
                all_pages.append(page_data)
                logger.info(f"✅ Page added to collection: {url}")
            else:
                logger.info(f"📋 Page filtered out: {url}")
            
            # If we haven't reached max depth, explore both links and interactive elements
            if current_depth < max_depth - 1 and len(all_pages) < self.max_pages:
                # Explore links
                links_to_explore = await self._find_explorable_links_playwright(page, url, self.max_links_per_page)
                
                for link_url in links_to_explore:
                    if len(all_pages) >= self.max_pages:
                        break
                    await self._explore_from_url_playwright(page, link_url, max_depth, max_pages_per_depth,
                                                          all_pages, visited_urls, current_depth + 1)
                
                # Explore interactive elements (buttons, forms, etc.)
                if len(all_pages) < self.max_pages:
                    interactive_elements = await self._find_explorable_interactive_elements_playwright(page, url, max_pages_per_depth)
                    
                    for element in interactive_elements:
                        if len(all_pages) >= self.max_pages:
                            break
                        
                        # Try to explore the interactive element
                        new_url = await self._explore_interactive_element_playwright(page, element, url)
                        
                        if new_url and new_url not in visited_urls:
                            # Navigate back to original page for next exploration
                            await page.goto(url, wait_until='networkidle', timeout=self.timeout * 1000)
                            await asyncio.sleep(3)  # Increased from 2 to 3 seconds
                            
                            # Explore the new URL
                            await self._explore_from_url_playwright(page, new_url, max_depth, max_pages_per_depth,
                                                          all_pages, visited_urls, current_depth + 1)
                        
        except Exception as e:
            # Check if it's a timeout error
            if "Timeout" in str(e) or "timeout" in str(e).lower():
                logger.warning(f"Timeout exploring {url}: {e}. Skipping this page.")
            else:
                logger.error(f"Error exploring {url}: {e}")
    
    async def _collect_single_page_playwright(self, page, url: str, depth: int) -> WebPageData:
        """Collect data from a single page using Playwright"""
        
        try:
            # Check if this is a login page and attempt to login
            if depth == 0 and 'localhost:8080' in url:
                try:
                    # First check if this actually looks like a login page
                    page_content = await page.content()
                    title = await page.title()
                    
                    # Check for login indicators
                    login_indicators = [
                        'login', 'signin', 'sign in', 'log in', 'username', 'password',
                        'suitecrm', 'crm', 'admin', 'dashboard'
                    ]
                    
                    is_login_page = any(indicator in title.lower() or indicator in page_content.lower() 
                                      for indicator in login_indicators)
                    
                    if is_login_page:
                        logger.info("🔍 Detected potential login page, attempting SuiteCRM login")
                        from .tool import WebTools
                        login_success = await WebTools.login_suitecrm(page)
                        if login_success:
                            logger.info("🔐 Successfully logged in to SuiteCRM, continuing with data collection")
                            # Wait a moment for the page to fully load after login
                            await asyncio.sleep(2)
                        else:
                            logger.warning("⚠️ Login failed, continuing with original page")
                    else:
                        logger.info("🔍 Not a login page, skipping login attempt")
                except Exception as e:
                    logger.warning(f"⚠️ Login attempt failed: {e}, continuing with original page")
            
            # Enhanced waiting for dynamic content to load
            logger.info(f"⏳ Waiting for dynamic content to load on {url}")
            
            # Wait for network to be idle
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
                logger.info("✅ Network is idle")
            except Exception as e:
                logger.warning(f"⚠️ Network idle wait timeout: {e}")
            
            # Additional wait for dynamic content (especially for SPAs)
            await asyncio.sleep(4)  # Wait for 4 seconds for dynamic content
            
            # Wait for specific elements that indicate content is loaded
            try:
                # Wait for common content indicators
                await page.wait_for_selector('table, .table, .cdk-table, .list-view-table, tbody, tr, td', 
                                           timeout=5000, state='visible')
                logger.info("✅ Table content is visible")
            except Exception as e:
                logger.warning(f"⚠️ Table content wait timeout: {e}")
            
            # Additional wait for any remaining dynamic content
            await asyncio.sleep(2)
            
            # Get page information
            title = await page.title()
            page_content = await page.content()
            
            # Mark interactive elements with SoM before extraction
            som_mapping = await self._mark_elements_with_som(page)
            
            # Take screenshot AFTER SoM marking to include the markers
            self.page_counter += 1
            screenshot_path = self.output_dir / f"page_{depth}_{self.page_counter}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            logger.info(f"🔍 SoM mapping created with {len(som_mapping)} elements")
            
            # Extract DOM elements (only SoM-marked elements will be included)
            elements = await self._extract_dom_elements_playwright(page, som_mapping)
            logger.info(f"🔍 Extracted {len(elements)} elements from SoM mapping")
            
            # Log SoM marked elements
            som_marked_elements = [e for e in elements if e.som_mark]
            logger.info(f"🔍 SoM marked elements: {len(som_marked_elements)}/{len(elements)}")
            for elem in som_marked_elements[:5]:  # Show first 5
                logger.info(f"  - {elem.som_mark}: {elem.som_type} - {elem.text_content[:50]}")
            
            # Get page metrics
            metrics = await page.evaluate("""
                () => {
                    return {
                        loadTime: performance.timing.loadEventEnd - performance.timing.navigationStart,
                        pageSize: document.documentElement.outerHTML.length
                    }
                }
            """)
            
            load_time = metrics.get('loadTime', 0) / 1000.0
            page_size = metrics.get('pageSize', len(page_content))
            
            # Determine page type and website type
            page_type = self._determine_page_type(title, page_content, elements)
            logger.info(f"🔍 Determining website type for: {url}")
            website_type, website_description = await self._determine_website_type_with_llm(title, page_content, elements, url)
            logger.info(f"✅ Website type determined: {website_type} - {website_description}")
            
            # Extract links for further exploration
            links = await self._extract_links_from_page_playwright(page, url)
            
            # Use LLM to assess page quality and task generation potential if enabled
            llm_assessment = None
            if self.use_llm_crawling and self.llm_executor:
                try:
                    # Pass screenshot path for visual analysis
                    llm_assessment = await self._assess_page_difficulty_with_llm(
                        title, page_content, elements, screenshot_path
                    )
                    
                    # Log assessment results
                    task_potential = llm_assessment.get('task_generation_potential', 'unknown')
                    info_density = llm_assessment.get('information_density_score', 0.0)
                    multi_hop = llm_assessment.get('multi_hop_potential', 'unknown')
                    visual_analysis = "with visual analysis" if screenshot_path and screenshot_path.exists() else "text-only"
                    
                    logger.info(f"🤖 LLM Assessment ({visual_analysis}): {task_potential} potential, "
                              f"{info_density:.2f} density, {multi_hop} multi-hop")
                    
                    # Check if page should be included based on quality filters
                    if not self._should_include_page(llm_assessment):
                        logger.info(f"📋 Page filtered out by quality criteria: {url}")
                        return None  # Skip this page
                        
                except Exception as e:
                    logger.warning(f"LLM page assessment failed: {e}")
            
            # Create page data
            page_data = WebPageData(
                url=url,
                title=title,
                elements=elements,
                screenshots=[screenshot_path],
                load_time=load_time,
                page_size=page_size,
                links=links,
                clickable_elements=[e.element_id for e in elements if e.is_clickable],
                form_elements=[e.element_id for e in elements if e.is_input],
                table_elements=[e.element_id for e in elements if e.element_type == "table"],
                page_type=page_type,
                website_type=website_type,
                website_description=website_description,
                exploration_depth=depth
            )
            
            # Clean up SoM markers to prevent accumulation and reduce complexity
            await self._cleanup_som_markers(page)
            
            # Add LLM assessment to page data if available
            if llm_assessment:
                page_data.llm_assessment = llm_assessment
            
            # Save DOM data
            dom_file = self.output_dir / f"page_{depth}_{self.page_counter}_dom.json"
            self._save_dom_data(page_data, dom_file)
            
            return page_data
            
        except Exception as e:
            # Check if it's a timeout error
            if "Timeout" in str(e) or "timeout" in str(e).lower():
                logger.warning(f"Timeout collecting page data for {url}: {e}. Skipping this page.")
            else:
                logger.error(f"Error collecting page data for {url}: {e}")
            
            # Ensure SoM markers are cleaned up even in case of error
            try:
                await self._cleanup_som_markers(page)
                logger.debug(f"🧹 Cleaned up SoM markers after error for {url}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Failed to cleanup SoM markers after error: {cleanup_error}")
            
            return self._create_empty_page_data(url, depth)
    
    async def _find_explorable_links_playwright(self, page, current_url: str, max_links: int) -> List[str]:
        """Find links that can be explored from current page using Playwright - 改进版本确保收集所有导航链接"""
        
        try:
            # 获取页面上的所有链接，包括导航链接
            links = await page.evaluate("""
                () => {
                    const links = [];
                    const baseUrl = window.location.origin;
                    const visitedUrls = new Set();
                    
                    // 1. 获取所有锚点标签（包括SPA导航元素）
                    const anchors = Array.from(document.querySelectorAll('a'));
                    anchors.forEach(anchor => {
                        const href = anchor.href;
                        const text = anchor.textContent.trim();
                        
                        // 检查是否是SPA导航元素（没有href但有导航相关的类）
                        const isSPANavigation = !href && (
                            anchor.classList.contains('nav-link') || 
                            anchor.classList.contains('top-nav-link') ||
                            anchor.classList.contains('navigation') ||
                            anchor.classList.contains('dropdown-toggle') ||
                            text.toLowerCase().includes('nav') ||
                            anchor.getAttribute('role') === 'navigation'
                        );
                        
                        // 包含传统链接和SPA导航元素
                        if ((href && 
                            href !== window.location.href && 
                            href.startsWith(baseUrl) &&
                            text.length > 0 &&
                            !href.includes('#') &&
                            !href.includes('javascript:') &&
                            !href.includes('mailto:') &&
                            !href.includes('tel:') &&
                            !visitedUrls.has(href)) ||
                            (isSPANavigation && text.length > 0)) {
                            
                            if (href) {
                                visitedUrls.add(href);
                            }
                            links.push({
                                href: href || `#${text.toLowerCase().replace(/\s+/g, '-')}`, // 为SPA元素创建虚拟URL
                                text: text,
                                visible: anchor.offsetParent !== null,
                                isNavigation: true,
                                isSPA: isSPANavigation
                            });
                        }
                    });
                    
                    // 2. 获取所有导航菜单项
                    const navItems = Array.from(document.querySelectorAll('nav a, .nav a, .menu a, .navigation a'));
                    navItems.forEach(item => {
                        const href = item.href;
                        const text = item.textContent.trim();
                        
                        if (href && 
                            href !== window.location.href && 
                            href.startsWith(baseUrl) &&
                            text.length > 0 &&
                            !visitedUrls.has(href)) {
                            
                            visitedUrls.add(href);
                            links.push({
                                href: href,
                                text: text,
                                visible: item.offsetParent !== null,
                                isNavigation: true
                            });
                        }
                    });
                    
                    // 3. 获取所有下拉菜单项
                    const dropdownItems = Array.from(document.querySelectorAll('.dropdown-menu a, .dropdown a, [role="menuitem"]'));
                    dropdownItems.forEach(item => {
                        const href = item.href;
                        const text = item.textContent.trim();
                        
                        if (href && 
                            href !== window.location.href && 
                            href.startsWith(baseUrl) &&
                            text.length > 0 &&
                            !visitedUrls.has(href)) {
                            
                            visitedUrls.add(href);
                            links.push({
                                href: href,
                                text: text,
                                visible: item.offsetParent !== null,
                                isNavigation: true
                            });
                        }
                    });
                    
                    return links;
                }
            """)
            
            # 使用配置的方法评估和选择链接
            if self.use_llm_crawling and self.llm_executor:
                # 获取当前页面信息作为上下文
                title = await page.title()
                content = await page.content()
                
                # 使用LLM评估和选择链接
                selected_links = await self._evaluate_links(
                    links, current_url, title, content, max_links
                )
                
                return selected_links
            else:
                # 使用基于规则的选择，优先选择导航链接
                navigation_links = []
                other_links = []
                        
                for link in links:
                    href = link.get('href', '')
                    if self._is_explorable_link(href, current_url):
                        if link.get('isNavigation', False):
                            navigation_links.append(href)
                        else:
                            other_links.append(href)
                
                # 优先返回导航链接，然后返回其他链接
                selected_links = navigation_links + other_links
                
                logger.info(f"🔗 Found {len(navigation_links)} navigation links and {len(other_links)} other links on {current_url}")
                logger.info(f"📋 Rule-based evaluation: selected {len(selected_links)} links from {len(links)} available")
                
                return selected_links[:max_links]
            
        except Exception as e:
            logger.error(f"Error finding explorable links: {e}")
            return []
    
    async def _extract_links_from_page_playwright(self, page, current_url: str) -> List[str]:
        """Extract all links from the current page using Playwright - 改进版本确保收集所有导航链接"""
        
        try:
            links = await page.evaluate("""
                () => {
                    const links = [];
                    const baseUrl = window.location.origin;
                    const visitedUrls = new Set();
                    
                    // 1. 获取所有锚点标签
                    const anchors = Array.from(document.querySelectorAll('a[href]'));
                    anchors.forEach(anchor => {
                        const href = anchor.href;
                        const text = anchor.textContent.trim();
                        
                        if (href && 
                            href !== window.location.href && 
                            href.startsWith(baseUrl) &&
                            text.length > 0 &&
                            !href.includes('#') &&
                            !href.includes('javascript:') &&
                            !href.includes('mailto:') &&
                            !href.includes('tel:') &&
                            !visitedUrls.has(href)) {
                            
                            visitedUrls.add(href);
                            links.push({
                                href: href,
                                text: text,
                                isNavigation: anchor.classList.contains('nav-link') || 
                                             anchor.classList.contains('top-nav-link') ||
                                             anchor.classList.contains('navigation') ||
                                             text.toLowerCase().includes('nav') ||
                                             anchor.getAttribute('role') === 'navigation'
                            });
                        }
                    });
                    
                    // 2. 获取所有导航菜单项
                    const navItems = Array.from(document.querySelectorAll('nav a, .nav a, .menu a, .navigation a'));
                    navItems.forEach(item => {
                        const href = item.href;
                        const text = item.textContent.trim();
                        
                        if (href && 
                            href !== window.location.href && 
                            href.startsWith(baseUrl) &&
                            text.length > 0 &&
                            !visitedUrls.has(href)) {
                            
                            visitedUrls.add(href);
                            links.push({
                                href: href,
                                text: text,
                                isNavigation: true
                            });
                        }
                    });
                    
                    // 3. 获取所有下拉菜单项
                    const dropdownItems = Array.from(document.querySelectorAll('.dropdown-menu a, .dropdown a, [role="menuitem"]'));
                    dropdownItems.forEach(item => {
                        const href = item.href;
                        const text = item.textContent.trim();
                        
                        if (href && 
                            href !== window.location.href && 
                            href.startsWith(baseUrl) &&
                            text.length > 0 &&
                            !visitedUrls.has(href)) {
                            
                            visitedUrls.add(href);
                            links.push({
                                href: href,
                                text: text,
                                isNavigation: true
                            });
                        }
                    });
                    
                    return links;
                }
            """)
            
            # 过滤和优先化链接
            navigation_links = []
            other_links = []
            
            for link in links:
                href = link.get('href', '')
                if self._is_explorable_link(href, current_url):
                    if link.get('isNavigation', False):
                        navigation_links.append(href)
                    else:
                        other_links.append(href)
            
            # 优先返回导航链接，然后返回其他链接
            extracted_links = navigation_links + other_links
            
            logger.info(f"🔗 Extracted {len(navigation_links)} navigation links and {len(other_links)} other links from {current_url}")
            
            return extracted_links
            
        except Exception as e:
            logger.error(f"Error extracting links: {e}")
            return []
        
    async def _evaluate_page_quality_with_llm(self, page_data: WebPageData) -> Dict[str, Any]:
        """Use LLM to evaluate page quality, information density, and link validity"""
        
        if not hasattr(self, 'llm_available') or not self.llm_available:
            logger.warning("LLM not available for page quality evaluation")
            return {
                "quality_score": 0.5,
                "information_density": "medium",
                "link_validity": "unknown",
                "is_suitable_for_tasks": True,
                "reasoning": "LLM not available, defaulting to accept"
            }
        
        try:
            from agent_framework.executors import LLMExecutor
            
            # Prepare page content for LLM analysis
            page_content = {
                "url": page_data.url,
                "title": page_data.title,
                "text_content": page_data.text_content[:2000],  # Limit content length
                "interactive_elements": len(page_data.interactive_elements),
                "links": len(page_data.links),
                "forms": len([e for e in page_data.elements if e.element_type == "form"]),
                "buttons": len([e for e in page_data.elements if e.element_type == "button"]),
                "inputs": len([e for e in page_data.elements if e.element_type == "input"])
            }
            
            # Create LLM prompt for page quality evaluation
            prompt = f"""
You are a web page quality evaluator. Analyze the following page data and provide a comprehensive quality assessment.

Page Data:
{json.dumps(page_content, indent=2)}

Evaluation Criteria:
1. Information Density: How much useful, structured information does the page contain?
2. Link Validity: Are the links likely to lead to relevant, accessible content?
3. Interactive Potential: Does the page have sufficient interactive elements for task generation?
4. Task Suitability: Is this page suitable for generating meaningful web tasks?

Please provide your assessment in the following JSON format:
{{
    "quality_score": 0.0-1.0,
    "information_density": "low|medium|high",
    "link_validity": "low|medium|high",
    "interactive_potential": "low|medium|high",
    "is_suitable_for_tasks": true/false,
    "reasoning": "Detailed explanation of your assessment",
    "recommended_filters": ["list", "of", "specific", "issues", "to", "address"]
}}

Focus on:
- Content relevance and structure
- Navigation complexity
- Form completeness
- Search functionality
- Data presentation quality
- Mobile responsiveness indicators
"""
            
            # Get LLM evaluation
            from agent_framework.executors import ExecutionConfig
            execution_config = ExecutionConfig(
                model_name="gpt-4o-mini",
                model_provider="openai",
                max_tokens=500,
                temperature=0.1
            )
            executor = LLMExecutor(execution_config)
            result = await executor.execute_async(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse LLM response
            try:
                evaluation = json.loads(result.answer)
                logger.info(f"Page quality evaluation for {page_data.url}: {evaluation.get('quality_score', 0.0)}")
                return evaluation
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM evaluation response: {result.answer}")
                return {
                    "quality_score": 0.5,
                    "information_density": "medium",
                    "link_validity": "medium",
                    "is_suitable_for_tasks": True,
                    "reasoning": "LLM response parsing failed, defaulting to accept"
                }
                
        except Exception as e:
            logger.error(f"Error in LLM page quality evaluation: {e}")
            return {
                "quality_score": 0.5,
                "information_density": "medium",
                "link_validity": "medium",
                "is_suitable_for_tasks": True,
                "reasoning": f"Evaluation failed: {str(e)}, defaulting to accept"
            }

    async def _filter_pages_by_quality(self, pages: List[WebPageData]) -> List[WebPageData]:
        """Keep all pages - no quality filtering to maintain navigation continuity"""
        
        logger.info(f"📊 Keeping all {len(pages)} collected pages to maintain navigation continuity")
        
        # Store basic metadata for all pages
        for page in pages:
            page.metadata = getattr(page, 'metadata', {})
            page.metadata['quality_evaluation'] = {
                "quality_score": 0.5,
                "information_density": "medium",
                "link_validity": "medium",
                "is_suitable_for_tasks": True,
                "reasoning": "All pages kept to maintain navigation continuity"
            }
        
        return pages
    
    async def _mark_elements_with_som(self, page) -> Dict[str, Any]:
        """Mark interactive elements with SoM (Set-of-Mark) - consistent with web_agent.py"""
        try:
            logger.info("🔍 Starting SoM marker injection for web collection...")
            
            # Additional wait to ensure page is fully loaded before marking
            logger.info("⏳ Additional wait for page stability before SoM marking...")
            await asyncio.sleep(1)
            
            # Wait for any remaining dynamic content
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
                logger.info("✅ DOM content loaded")
            except Exception as e:
                logger.warning(f"⚠️ DOM content loaded wait timeout: {e}")
            
            # Final wait for any JavaScript execution
            await asyncio.sleep(1)
            
            # First, remove any existing markers
            await page.evaluate("""
                () => {
                    const old = document.getElementById('som-overlay');
                    if (old && old.__som_cleanup__) {
                        old.__som_cleanup__();
                    }
                }
            """)
            
            # Inject CSS for optimized marker styling with color coding (same as web_agent.py)
            await page.evaluate("""
                () => {
                    if (!document.getElementById('som-styles')) {
                        const style = document.createElement('style');
                        style.id = 'som-styles';
                        style.textContent = `
                            .som-wrapper {
                                position: absolute;
                                border: 3px solid; /* 增加边框宽度 */
                                background: rgba(255, 255, 255, 0.1); /* 添加半透明背景 */
                                pointer-events: none;
                                z-index: 10000;
                                box-shadow: 0 0 5px rgba(0, 0, 0, 0.3); /* 添加阴影效果 */
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
                                border-color: #ff9500; /* 橙色 - 导航类 */
                            }
                            .som-wrapper.paginator {
                                border-color: #ff9500; /* 橙色 - 分页类 */
                            }
                            .som-wrapper.result {
                                border-color: #af52de; /* 紫色 - 结果信息类 */
                            }
                            .som-wrapper.tab {
                                border-color: #00bcd4; /* 青色 - 标签页类 */
                            }
                            .som-wrapper.modal {
                                border-color: #9c27b0; /* 紫色 - 模态框类 */
                            }
                            .som-wrapper.toast {
                                border-color: #ffc107; /* 黄色 - 通知类 */
                            }
                            .som-wrapper.breadcrumb {
                                border-color: #795548; /* 棕色 - 面包屑类 */
                            }
                            .som-wrapper.link {
                                border-color: #1976d2; /* 深蓝色 - 链接类 */
                            }
                            .som-wrapper.detail_link {
                                border-color: #388e3c; /* 深绿色 - 详情链接类 */
                            }
                            .som-wrapper.dropdown {
                                border-color: #7b1fa2; /* 深紫色 - 下拉菜单类 */
                            }
                            .som-wrapper.submenu {
                                border-color: #ba68c8; /* 浅紫色 - 子菜单类 */
                            }
                            .som-wrapper.menu {
                                border-color: #1565c0; /* 深蓝色 - 菜单类 */
                            }
                            .som-wrapper.content {
                                border-color: #9e9e9e; /* 灰色 - 内容类 */
                            }
                            .som-wrapper.list {
                                border-color: #ff9800; /* 橙色 - 列表类 */
                            }
                            .som-wrapper.table {
                                border-color: #424242; /* 深灰色 - 表格类 */
                            }
                            .som-wrapper.business_data {
                                border-color: #e91e63; /* 粉色 - 业务数据类 */
                            }
                            .som-wrapper.card {
                                border-color: #42a5f5; /* 浅蓝色 - 卡片类 */
                            }
                            .som-wrapper.detail {
                                border-color: #2e7d32; /* 深绿色 - 详情类 */
                            }
                            .som-wrapper.item {
                                border-color: #66bb6a; /* 浅绿色 - 项目类 */
                            }
                            .som-wrapper.filter {
                                border-color: #f57c00; /* 深橙色 - 过滤器类 */
                            }
                            .som-wrapper.filter_panel {
                                border-color: #fbc02d; /* 深黄色 - 过滤器面板类 */
                            }
                            .som-wrapper.notification_area {
                                border-color: #fff59d; /* 浅黄色 - 通知区域类 */
                            }
                            .som-wrapper.tab_container {
                                border-color: #006064; /* 深青色 - 标签页容器类 */
                            }
                            .som-label {
                                position: absolute;
                                top: -10px; left: -10px;
                                font-weight: 700;
                                font-size: 12px; /* 增加字体大小 */
                                line-height: 14px;
                                padding: 2px 6px; /* 增加内边距 */
                                border-radius: 999px;
                                border: 2px solid rgba(255,255,255,0.9); /* 增加边框宽度 */
                                background: rgba(0, 0, 0, 0.8); /* 添加深色背景 */
                                color: white; /* 确保文字是白色 */
                                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8); /* 添加文字阴影 */
                            }
                                text-shadow: 0 1px 1px rgba(0,0,0,0.3);
                                pointer-events: none;
                                user-select: none;
                                min-width: 12px;
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
                            .som-label.paginator {
                                background: #ff9500;
                                color: #fff;
                            }
                            .som-label.result {
                                background: #af52de;
                                color: #fff;
                            }
                            .som-label.tab {
                                background: #00bcd4;
                                color: #fff;
                            }
                            .som-label.modal {
                                background: #9c27b0;
                                color: #fff;
                            }
                            .som-label.toast {
                                background: #ffc107;
                                color: #000;
                            }
                            .som-label.breadcrumb {
                                background: #795548;
                                color: #fff;
                            }
                            .som-label.link {
                                background: #1976d2;
                                color: #fff;
                            }
                            .som-label.detail_link {
                                background: #388e3c;
                                color: #fff;
                            }
                            .som-label.dropdown {
                                background: #7b1fa2;
                                color: #fff;
                            }
                            .som-label.submenu {
                                background: #ba68c8;
                                color: #fff;
                            }
                            .som-label.menu {
                                background: #1565c0;
                                color: #fff;
                            }
                            .som-label.content {
                                background: #9e9e9e;
                                color: #fff;
                            }
                            .som-label.list {
                                background: #ff9800;
                                color: #fff;
                            }
                            .som-label.table {
                                background: #424242;
                                color: #fff;
                            }
                            .som-label.business_data {
                                background: #e91e63;
                                color: #fff;
                            }
                            .som-label.card {
                                background: #42a5f5;
                                color: #fff;
                            }
                            .som-label.detail {
                                background: #2e7d32;
                                color: #fff;
                            }
                            .som-label.item {
                                background: #66bb6a;
                                color: #fff;
                            }
                            .som-label.filter {
                                background: #f57c00;
                                color: #fff;
                            }
                            .som-label.filter_panel {
                                background: #fbc02d;
                                color: #000;
                            }
                            .som-label.notification_area {
                                background: #fff59d;
                                color: #000;
                            }
                            .som-label.tab_container {
                                background: #006064;
                                color: #fff;
                            }
                        `;
                        document.head.appendChild(style);
                    }
                }
            """)
            
                        # Get optimized elements and inject markers (simplified and robust)
            mark_mapping = await page.evaluate("""
                () => {
                    const markMapping = {};
                    let markCounter = 1;
                    
                    // Global tracking to prevent duplicate marking
                    const processedElements = new Set();
                    const elementSignature = (el) => {
                        const rect = el.getBoundingClientRect();
                        const text = el.textContent?.trim() || el.value || el.placeholder || '';
                        const tag = el.tagName.toLowerCase();
                        const className = el.className || '';
                        // More specific signature to avoid duplicates
                        return `${tag}-${text.substring(0, 20)}-${Math.round(rect.left)}-${Math.round(rect.top)}-${Math.round(rect.width)}-${Math.round(rect.height)}`;
                    };
                    
                    // Enhanced navigation detection function
                    const isNavigationElement = (el) => {
                        const tagName = el.tagName.toLowerCase();
                        const className = el.className || '';
                        const textContent = el.textContent?.trim() || '';
                        const href = el.href || '';
                        const role = el.getAttribute('role') || '';
                        const ariaLabel = el.getAttribute('aria-label') || '';
                        const dataToggle = el.getAttribute('data-toggle') || '';
                        const onclick = el.getAttribute('onclick') || '';
                        
                        // Check if element is in navigation context
                        const isInNavContext = () => {
                            let parent = el.parentElement;
                            while (parent && parent !== document.body) {
                                const parentTag = parent.tagName.toLowerCase();
                                const parentClass = (parent.className && typeof parent.className === 'string') ? parent.className : (parent.className ? parent.className.toString() : '');
                                const parentId = parent.id || '';
                                
                                if (parentTag === 'nav' || 
                                    parentClass.includes('nav') || 
                                    parentClass.includes('menu') || 
                                    parentClass.includes('navbar') ||
                                    parentId.includes('nav') || 
                                    parentId.includes('menu') ||
                                    parent.getAttribute('role') === 'navigation' ||
                                    parent.getAttribute('role') === 'menubar') {
                                    return true;
                                }
                                parent = parent.parentElement;
                            }
                            return false;
                        };
                        
                        // Navigation indicators
                        const safeClassName = (className && typeof className === 'string') ? className : (className ? className.toString() : '');
                        const hasNavigationClass = safeClassName.includes('nav') || 
                                                 safeClassName.includes('menu') || 
                                                 safeClassName.includes('dropdown');
                        
                        const hasNavigationText = ['accounts', 'contacts', 'opportunities', 'leads', 'quotes', 
                                                 'calendar', 'documents', 'home', 'about', 'services', 'products',
                                                 'dashboard', 'profile', 'settings', 'admin', 'user', 'login', 'logout'].some(
                            keyword => textContent.toLowerCase().includes(keyword));
                        
                        const hasNavigationRole = role === 'menuitem' || role === 'navigation' || role === 'menubar';
                        
                        const hasNavigationAria = ariaLabel.toLowerCase().includes('navigation') || 
                                                ariaLabel.toLowerCase().includes('menu');
                        
                        const hasNavigationData = dataToggle === 'dropdown' || 
                                                el.hasAttribute('data-nav') ||
                                                el.hasAttribute('data-menu');
                        
                        const hasNavigationBehavior = onclick || 
                                                    href === '#' || 
                                                    href === 'javascript:void(0)' ||
                                                    href === '';
                        
                        // Check if it's a link that looks like navigation
                        const isNavigationLink = tagName === 'a' && (
                            hasNavigationClass || 
                            hasNavigationText || 
                            hasNavigationRole ||
                            hasNavigationAria ||
                            hasNavigationData ||
                            hasNavigationBehavior ||
                            isInNavContext()
                        );
                        
                        // Check if it's a button that acts like navigation
                        const isNavigationButton = (tagName === 'button' || tagName === 'a') && (
                            hasNavigationClass || 
                            hasNavigationText || 
                            hasNavigationRole ||
                            hasNavigationAria ||
                            hasNavigationData ||
                            hasNavigationBehavior ||
                            isInNavContext()
                        );
                        
                        return isNavigationLink || isNavigationButton;
                    };
                    
                    // Enhanced selectors including business data elements - aligned with web_agent.py
                    const clickableSelectors = [
                        // Priority 1: Explicit business actions
                        'button[class*="create"]',   // Create buttons
                        'button[class*="edit"]',     // Edit buttons
                        'button[class*="delete"]',   // Delete buttons
                        'button[class*="save"]',     // Save buttons
                        'button[class*="submit"]',   // Submit buttons
                        'a[href*="create"]',         // Create links
                        'a[href*="edit"]',           // Edit links
                        'a[href*="delete"]',         // Delete links

                        // Priority 2: Search and filter (removed buttons - they have no content)
                        // 'button[class*="search"]',   // Search button - removed, has no text content
                        // 'button[class*="filter"]',   // Filter button - removed, has no text content
                        'button[class*="bulk"]',     // Bulk action button
                        'input[type="search"]',      // Search inputs
                        '[role="search"]',           // Search role

                        // Priority 3: Navigation
                        'a[href*="home"]',           // Home link
                        'a[href*="dashboard"]',      // Dashboard links
                        'a[href*="profile"]',        // Profile links
                        'a[href*="settings"]',       // Settings links

                        // Priority 4: General interactive elements
                        'button',                    // General buttons
                        'a[href]',                   // Links
                        '[role="button"]',           // Role buttons
                        'input[type="submit"]',      // Submit buttons
                        'input[type="button"]',      // Button inputs
                        'div[onclick]',             // Divs with click handlers
                        'span[onclick]',            // Spans with click handlers
                        '[data-clickable="true"]'   // Explicitly marked clickable
                    ];
                    
                    const inputSelectors = [
                        'input[type="search"]',     // Search input
                        'input[placeholder*="Search"]',  // Search placeholder
                        'input[type="text"]',       // Text inputs
                        'input[type="email"]',      // Email inputs
                        'input[type="password"]',   // Password inputs
                        'input[type="url"]',        // URL inputs
                        'input[type="tel"]',        // Telephone inputs
                        'input[type="number"]',     // Number inputs
                        'input:not([type])',        // Default text inputs
                        'textarea',                 // Text areas
                        '[contenteditable="true"]'  // Content editable elements
                    ];
                    
                    const selectSelectors = [
                        'select',                   // Dropdown selects
                        'input[type="radio"]',      // Radio buttons
                        'input[type="checkbox"]'    // Checkboxes
                    ];
                    
                    const navigationSelectors = [
                        // Primary navigation selectors
                        '.top-nav-link',            // Main navigation
                        '.nav-link-nongrouped',     // Non-grouped nav links
                        '.dropdown-toggle',         // Dropdown toggles (often navigation)
                        '.nav-link',                // Nav links
                        'nav a',                    // Navigation links
                        '[role="menuitem"]',        // Menu items
                        '[role="navigation"]',      // Navigation roles
                        
                        // Context-based navigation detection
                        'nav a[href*="accounts"]',  // Accounts navigation
                        'nav a[href*="contacts"]',  // Contacts navigation
                        'nav a[href*="opportunities"]',  // Opportunities navigation
                        'nav a[href*="leads"]',     // Leads navigation
                        'nav a[href*="quotes"]',    // Quotes navigation
                        'nav a[href*="calendar"]',  // Calendar navigation
                        'nav a[href*="documents"]', // Documents navigation
                        
                        // CSS class-based navigation detection
                        'a[class*="nav"]',          // Any link with nav in class name
                        'a[class*="menu"]',         // Any link with menu in class name
                        'a[class*="dropdown"]',     // Any link with dropdown in class name
                        
                        // Text content-based navigation detection (removed :contains as it's not valid CSS)
                        // These will be handled by the isNavigationElement function instead
                        
                        // Structural navigation detection
                        'header a',                 // Links in header
                        '.header a',                // Links in header class
                        '.navbar a',                // Links in navbar
                        '.menu a',                  // Links in menu
                        '.navigation a',            // Links in navigation
                        '.nav-bar a',               // Links in nav-bar
                        '.main-nav a',              // Links in main-nav
                        '.primary-nav a',           // Links in primary-nav
                        '.secondary-nav a',         // Links in secondary-nav
                        
                        // ARIA-based navigation detection
                        '[aria-label*="navigation"]', // Elements with navigation in aria-label
                        '[aria-label*="menu"]',     // Elements with menu in aria-label
                        '[aria-expanded="true"]',   // Expanded navigation elements
                        '[aria-haspopup="true"]',   // Elements with popup (dropdowns)
                        
                        // Data attribute-based navigation detection
                        '[data-toggle="dropdown"]', // Bootstrap dropdown toggles
                        '[data-target*="nav"]',     // Elements targeting navigation
                        '[data-nav]',               // Elements with nav data attribute
                        
                        // Generic navigation patterns
                        'a[href="#"]',              // Hash links (often navigation)
                        'a[href="javascript:void(0)"]', // JavaScript void links (often navigation)
                        'a[onclick]',               // Links with onclick (often navigation)
                        
                        // Fallback: any link in navigation context
                        'nav a',                    // General nav links
                        '.nav a',                   // General nav links with class
                        '#nav a',                   // General nav links with id
                        '#navigation a',            // General navigation links with id
                        '#menu a',                  // General menu links with id
                        '#navbar a'                 // General navbar links with id
                    ];
                    
                    const paginationSelectors = [
                        'a[href*="page"]',          // Page links
                        'a[href*="next"]',          // Next page
                        'a[href*="prev"]',          // Previous page
                        'a[href*="previous"]',      // Previous page
                        'button[class*="next"]',    // Next buttons
                        'button[class*="prev"]',    // Previous buttons
                        '.pagination a',            // Pagination links
                        '.pagination button',       // Pagination buttons
                        '[data-page]',              // Page data attributes
                        '[aria-label*="page"]'      // Page aria labels
                    ];
                    
                    const resultSelectors = [
                        'h1', 'h2', 'h3', 'h4',    // Headings
                        '.product-title', '.item-title', '.result-title',
                        '.search-result h3', '.search-result h4',
                        'table th', 'table td:first-child', // Table headers
                        '.price', '.cost', '.amount',       // Price information
                        '.description', '.summary'          // Descriptions
                    ];
                    
                    const tabSelectors = [
                        '[role="tab"]',             // Tab roles
                        '.tab', '.tab-item',        // Tab classes
                        '[data-tab]',               // Tab data attributes
                        'button[aria-selected]',    // Selected tabs
                        '.nav-tabs a', '.tab-nav a' // Tab navigation
                    ];
                    
                    const modalSelectors = [
                        '[role="dialog"]',          // Dialog roles
                        '.modal', '.dialog',        // Modal classes
                        '[data-modal]',             // Modal data attributes
                        '.popup', '.overlay'        // Popup elements
                    ];
                    
                    const toastSelectors = [
                        '.toast', '.notification',  // Toast notifications
                        '.alert', '.message',       // Alert messages
                        '[role="alert"]',           // Alert roles
                        '.snackbar', '.popup-message' // Snackbar messages
                    ];
                    
                    const breadcrumbSelectors = [
                        '.breadcrumb', '.breadcrumbs', // Breadcrumb navigation
                        '[role="navigation"][aria-label*="breadcrumb"]',
                        '.nav-breadcrumb', '.crumb'    // Breadcrumb classes
                    ];
                    
                    const linkSelectors = [
                        'a[href]:not([href="#"]):not([href="javascript:void(0)"]):not([href=""]):not([href=" "])', // Valid links with actual href
                        '.link'        // Link classes (excluding .nav-link to avoid conflicts with navigation)
                    ];
                    
                    const detailLinkSelectors = [
                        'a[href*="detail"]',        // Detail page links
                        'a[href*="view"]',          // View page links
                        'a[href*="show"]',          // Show page links
                        'a[href*="info"]',          // Info page links
                        '.detail-link', '.view-link', '.show-link' // Link classes
                    ];
                    
                    const dropdownSelectors = [
                        '.dropdown', '.drop-down',  // Dropdown classes
                        '.select-dropdown', '[data-dropdown]', // Dropdown data
                        '.menu-dropdown', '.dropdown-menu'    // Dropdown menus
                    ];
                    
                    const submenuSelectors = [
                        '.submenu', '.sub-menu',    // Submenu classes
                        '.nested-menu', '.child-menu', // Nested menus
                        '.sub-nav'                  // Sub navigation
                    ];
                    
                    const menuSelectors = [
                        '.menu', '.nav-menu',       // Menu classes
                        '[role="menu"]',            // Menu roles
                        '.main-menu', '.primary-menu' // Main menus
                    ];
                    
                    const contentSelectors = [
                        '.content', '.main-content', // Content areas
                        '.article', '.post',         // Article content
                        '.text-content', '.body-content' // Text content
                    ];
                    
                    const listSelectors = [
                        '.list', '.item-list',      // List classes
                        '.product-list', '.user-list', // Specific lists
                        'ul.list', 'ol.list', '.grid-list' // List types
                    ];
                    
                    const tableSelectors = [
                        'table[class*="list-view"]', 'table', // Tables
                        '.table', '.data-table', '.grid-table', // Table classes
                        '.results-table', '.list-view-table'   // Specific tables
                    ];
                    
                    const cardSelectors = [
                        '.card', '.item-card',      // Card classes
                        '.product-card', '.user-card', // Specific cards
                        '.info-card', '.content-card'  // Content cards
                    ];
                    
                    const detailSelectors = [
                        '.detail', '.details',      // Detail classes
                        '.item-detail', '.product-detail', // Specific details
                        '.user-detail', '.info-detail'     // Info details
                    ];
                    
                    const itemSelectors = [
                        '.item', '.list-item',      // Item classes
                        '.product-item', '.user-item', // Specific items
                        '.entry', '.element'        // Generic items
                    ];
                    
                    const filterSelectors = [
                        '.filter', '.filters',      // Filter classes
                        '.filter-panel', '.filter-controls', // Filter controls
                        '.filter-options', '[data-filter]'   // Filter data
                    ];
                    
                    const filterPanelSelectors = [
                        '.filter-panel', '.filter-sidebar', // Filter panels
                        '.filter-container', '.filter-box',  // Filter containers
                        '.filter-section'                    // Filter sections
                    ];
                    
                    const notificationAreaSelectors = [
                        '.notification-area', '.notifications', // Notification areas
                        '.alert-area', '.message-area',         // Alert areas
                        '.status-area'                          // Status areas
                    ];
                    
                    const tabContainerSelectors = [
                        '.tab-container', '.tabs',   // Tab containers
                        '.tab-group', '.tab-panel',  // Tab groups
                        '[role="tablist"]'           // Tab list roles
                    ];
                    
                    // Business data selectors - CRM specific content (preserved for business logic)
                    const businessDataSelectors = [
                        // Table rows and cells containing business data
                        'table tbody tr', 'table td', 'table th',
                        '.table tbody tr', '.table td', '.table th',
                        '.list-view tbody tr', '.list-view td', '.list-view th',
                        '.data-table tbody tr', '.data-table td', '.data-table th',
                        
                        // SuiteCRM specific table selectors
                        '.list-view-data tbody tr', '.list-view-data td', '.list-view-data th',
                        '.table-responsive tbody tr', '.table-responsive td', '.table-responsive th',
                        '.ng-star-inserted tbody tr', '.ng-star-inserted td', '.ng-star-inserted th',
                        '[class*="list-view"] tbody tr', '[class*="list-view"] td', '[class*="list-view"] th',
                        '[class*="table"] tbody tr', '[class*="table"] td', '[class*="table"] th',
                        
                        // Additional SuiteCRM table selectors for better coverage
                        'tbody tr', 'tbody td', 'tbody th',
                        '.cdk-row', '.cdk-cell', '.cdk-header-cell',
                        '.mat-row', '.mat-cell', '.mat-header-cell',
                        '[role="row"]', '[role="cell"]', '[role="columnheader"]',
                        '.list-view-row', '.list-view-cell', '.list-view-header',
                        '.data-row', '.data-cell', '.data-header',
                        '.item-row', '.item-cell', '.item-header',
                        '.content-row', '.content-cell', '.content-header',
                        '.main-content tbody tr', '.main-content tbody td', '.main-content tbody th',
                        '.page-content tbody tr', '.page-content tbody td', '.page-content tbody th',
                        '.container tbody tr', '.container tbody td', '.container tbody th',
                        
                        // Comprehensive table element selectors for SuiteCRM
                        'tr', 'td', 'th',
                        '.cdk-table tr', '.cdk-table td', '.cdk-table th',
                        '.list-view-table tr', '.list-view-table td', '.list-view-table th',
                        '.striped-table tr', '.striped-table td', '.striped-table th',
                        'table tr', 'table td', 'table th',
                        '[class*="cdk"] tr', '[class*="cdk"] td', '[class*="cdk"] th',
                        '[class*="table"] tr', '[class*="table"] td', '[class*="table"] th',
                        '[class*="list"] tr', '[class*="list"] td', '[class*="list"] th',
                        
                        // CRM specific elements
                        '.contact-name', '.contact-email', '.contact-phone',
                        '.opportunity-name', '.opportunity-value', '.opportunity-stage',
                        '.account-name', '.account-type', '.account-industry',
                        '.lead-name', '.lead-company', '.lead-status',
                        '.quote-name', '.quote-amount', '.quote-status',
                        '.task-name', '.task-status', '.task-priority',
                        '.meeting-title', '.meeting-date', '.meeting-time',
                        '.document-name', '.document-type', '.document-size',
                        '.email-subject', '.email-sender', '.email-date',
                        
                        // Generic business data containers
                        '.item-name', '.item-title', '.item-description',
                        '.product-name', '.product-price', '.product-category',
                        '.customer-name', '.customer-email', '.customer-phone',
                        '.project-name', '.project-status', '.project-manager',
                        
                        // Form fields with business data
                        'input[name*="name"]', 'input[name*="email"]', 'input[name*="phone"]',
                        'input[name*="company"]', 'input[name*="title"]', 'input[name*="subject"]',
                        'textarea[name*="description"]', 'textarea[name*="notes"]',
                        
                        // Data display elements
                        '.data-field', '.field-value', '.info-item',
                        '.detail-item', '.property-value', '.attribute-value',
                        
                        // Additional SuiteCRM specific selectors
                        '[data-field-name]', '[data-field-type]', '[data-field-value]',
                        '.field-value', '.field-name', '.field-type',
                        '.list-view-field', '.list-view-cell', '.list-view-row'
                    ];
                    
                    // Element types with business data support - aligned with web_agent.py
                    // Navigation should be processed BEFORE link to avoid conflicts
                    const elementTypes = [
                        { type: 'navigation', selectors: navigationSelectors, maxElements: 15 }, // Increased maxElements and moved to front
                        { type: 'clickable', selectors: clickableSelectors, maxElements: 15 },
                        { type: 'input', selectors: inputSelectors, maxElements: 10 },
                        { type: 'select', selectors: selectSelectors, maxElements: 8 },
                        { type: 'paginator', selectors: paginationSelectors, maxElements: 8 },
                        { type: 'result', selectors: resultSelectors, maxElements: 5 },
                        { type: 'tab', selectors: tabSelectors, maxElements: 8 },
                        { type: 'modal', selectors: modalSelectors, maxElements: 5 },
                        { type: 'toast', selectors: toastSelectors, maxElements: 5 },
                        { type: 'breadcrumb', selectors: breadcrumbSelectors, maxElements: 5 },
                        { type: 'link', selectors: linkSelectors, maxElements: 10 },
                        { type: 'detail_link', selectors: detailLinkSelectors, maxElements: 8 },
                        { type: 'dropdown', selectors: dropdownSelectors, maxElements: 8 },
                        { type: 'submenu', selectors: submenuSelectors, maxElements: 8 },
                        { type: 'menu', selectors: menuSelectors, maxElements: 8 },
                        { type: 'content', selectors: contentSelectors, maxElements: 10 },
                        { type: 'list', selectors: listSelectors, maxElements: 8 },
                        { type: 'table', selectors: tableSelectors, maxElements: 5 },
                        { type: 'card', selectors: cardSelectors, maxElements: 8 },
                        { type: 'detail', selectors: detailSelectors, maxElements: 8 },
                        { type: 'item', selectors: itemSelectors, maxElements: 10 },
                        { type: 'filter', selectors: filterSelectors, maxElements: 8 },
                        { type: 'filter_panel', selectors: filterPanelSelectors, maxElements: 5 },
                        { type: 'notification_area', selectors: notificationAreaSelectors, maxElements: 5 },
                        { type: 'tab_container', selectors: tabContainerSelectors, maxElements: 5 },
                        // Preserve business data type for CRM-specific logic
                        { type: 'business_data', selectors: businessDataSelectors, maxElements: 50 }
                    ];
                    
                    // Process elements with strict filtering
                    elementTypes.forEach(({ type, selectors, maxElements }) => {
                        let elementCount = 0;
                        
                        selectors.forEach(selector => {
                            if (elementCount >= maxElements) return;
                            
                            document.querySelectorAll(selector).forEach(el => {
                                if (elementCount >= maxElements) return;
                                
                                // Check if this element has already been processed globally
                                const signature = elementSignature(el);
                                if (processedElements.has(signature)) {
                                    return; // Skip already processed elements
                                }
                                
                                const rect = el.getBoundingClientRect();
                                const style = window.getComputedStyle(el);
                                const className = el.className || '';
                                const tagName = el.tagName.toLowerCase();
                                
                                // ENHANCED: For link type, check if it's actually navigation first
                                if (type === 'link' && isNavigationElement(el)) {
                                    // Skip this element for link processing - it will be handled by navigation
                                    return;
                                }
                                
                                // CRITICAL: Skip any SoM-related elements
                                const safeClassName = (className && typeof className === 'string') ? className : (className ? className.toString() : '');
                                if (safeClassName.includes('som-wrapper') || safeClassName.includes('som-label') || 
                                    safeClassName.includes('som-') || tagName === 'som-wrapper' || tagName === 'som-label') {
                                    return;
                                }
                                
                                // Enhanced visibility and size checks - only mark significant elements
                                if (rect.width < 20 || rect.height < 20 || 
                                    style.display === 'none' || style.visibility === 'hidden' ||
                                    rect.top < 0 || rect.left < 0 ||
                                    rect.bottom > window.innerHeight + 100 ||
                                    rect.right > window.innerWidth + 100) {
                                    return;
                                }
                                
                                // Additional position validation - ensure element is actually visible
                                if (rect.top < 0 || rect.left < 0 || 
                                    rect.bottom > window.innerHeight || 
                                    rect.right > window.innerWidth) {
                                    return;
                                }
                                
                                // Skip small pagination buttons and minor UI elements
                                if (rect.width < 30 && rect.height < 30 && 
                                    (safeClassName.includes('pagination') || safeClassName.includes('button-group'))) {
                                    return;
                                }
                                
                                // Skip elements with very short text (likely icons or minor buttons)
                                const text = el.textContent?.trim() || el.value || el.placeholder || '';
                                if (text.length < 2 && !el.href && !safeClassName.includes('search')) {
                                    return;
                                }
                                
                                // Special handling for business data elements
                                if (type === 'business_data') {
                                    // Simplified business data detection - more lenient
                                    const isBusinessData = (
                                        // Basic patterns
                                        text.includes('@') ||  // Email addresses
                                        text.includes('$') ||  // Currency
                                        text.includes('Active') || text.includes('Inactive') ||
                                        text.includes('New') || text.includes('Open') || text.includes('Closed') ||
                                        text.includes('Pending') || text.includes('Completed') ||
                                        
                                        // CRM-specific terms
                                        text.includes('Contact') || text.includes('Lead') ||
                                        text.includes('Opportunity') || text.includes('Account') ||
                                        text.includes('Quote') || text.includes('Meeting') ||
                                        text.includes('Task') || text.includes('Document') ||
                                        text.includes('Email') || text.includes('Project') ||
                                        text.includes('Company') || text.includes('Customer') ||
                                        text.includes('Vendor') || text.includes('Partner') ||
                                        
                                        // SuiteCRM specific patterns
                                        text.includes('Website') || text.includes('Mobile') ||
                                        text.includes('Cloud') || text.includes('Cybersecurity') ||
                                        text.includes('Data') || text.includes('E-commerce') ||
                                        text.includes('Digital') || text.includes('IT') ||
                                        text.includes('Software') || text.includes('System') ||
                                        
                                        // Company names
                                        text.includes('Acme') || text.includes('Globex') ||
                                        text.includes('Soylent') || text.includes('Initech') ||
                                        text.includes('Umbrella') || text.includes('Massive') ||
                                        text.includes('Stark') || text.includes('Wayne') ||
                                        text.includes('Wonka') ||
                                        
                                        // Person names (simplified)
                                        text.includes('Bruce Wayne') || text.includes('Clark Kent') ||
                                        text.includes('Diana Prince') || text.includes('Barry Allen') ||
                                        text.includes('Hal Jordan') || text.includes('Arthur Curry') ||
                                        text.includes('Victor Stone') || text.includes('Peter Parker') ||
                                        text.includes('Tony Stark') || text.includes('Natasha Romanoff') ||
                                        
                                        // Email templates
                                        text.includes('Email template') || text.includes('template') ||
                                        
                                        // Status and field names
                                        text.includes('Name') || text.includes('Status') ||
                                        text.includes('Account Name') || text.includes('Office Phone') ||
                                        text.includes('Email') || text.includes('User') ||
                                        
                                        // Date patterns
                                        text.includes('08/25/2025') || text.includes('05/24/2013') ||
                                        text.includes('13:26') || text.includes('14:31') ||
                                        
                                        // For table cells, be more lenient
                                        (tagName === 'td' || tagName === 'th' || tagName === 'tr') ||
                                        
                                        // Any meaningful text in business context
                                        (text.length > 2 && text.length < 100 && 
                                         !text.includes('TEMPLATES') && !text.includes('ACCOUNTS') &&
                                         !text.includes('CONTACTS') && !text.includes('OPPORTUNITIES') &&
                                         !text.includes('LEADS') && !text.includes('QUOTES') &&
                                         !text.includes('CALENDAR') && !text.includes('DOCUMENTS') &&
                                         !text.includes('EMAILS') && !text.includes('MORE'))
                                    );
                                    
                                    // Simplified business data validation
                                    if (!isBusinessData) {
                                        return; // Skip if not business data
                                    }
                                    
                                    // Skip page titles but NOT navigation headers (they should be processed as navigation elements)
                                    if (tagName === 'h1' || tagName === 'h2' || tagName === 'h3' || 
                                        safeClassName.includes('page-title') || safeClassName.includes('nav-title')) {
                                        return; // Skip page titles only
                                    }
                                    
                                    // For business_data type, skip navigation elements as they should be processed as navigation type
                                    if (text.includes('TEMPLATES') || text.includes('ACCOUNTS') || 
                                        text.includes('CONTACTS') || text.includes('OPPORTUNITIES') ||
                                        text.includes('LEADS') || text.includes('QUOTES') || 
                                        text.includes('CALENDAR') || text.includes('DOCUMENTS') ||
                                        text.includes('EMAILS') || text.includes('MORE')) {
                                        return; // Skip navigation elements from business_data processing
                                    }
                                }
                                
                                // Enhanced duplicate detection - check text content and position
                                let isDuplicate = false;
                                const currentText = el.textContent?.trim() || el.value || el.placeholder || '';
                                const currentTag = el.tagName.toLowerCase();
                                
                                for (const existing of Object.values(markMapping)) {
                                    const existingRect = existing.rect_viewport;
                                    const existingText = existing.text;
                                    const existingTag = existing.tag;
                                    
                                    // Check for exact position match
                                    if (Math.abs(rect.left - existingRect.x) < 1 &&
                                        Math.abs(rect.top - existingRect.y) < 1 &&
                                        Math.abs(rect.width - existingRect.width) < 1 &&
                                        Math.abs(rect.height - existingRect.height) < 1) {
                                        isDuplicate = true;
                                        break;
                                    }
                                    
                                    // Check for same text and tag (likely duplicate elements)
                                    if (currentText && existingText && 
                                        currentText === existingText && 
                                        currentTag === existingTag) {
                                        isDuplicate = true;
                                        break;
                                    }
                                    
                                    // Check for overlapping elements
                                    if (rect.left < existingRect.x + existingRect.width &&
                                        rect.left + rect.width > existingRect.x &&
                                        rect.top < existingRect.y + existingRect.height &&
                                        rect.top + rect.height > existingRect.y) {
                                        // If overlapping significantly, prefer the one with more text
                                        if (currentText.length <= existingText.length) {
                                            isDuplicate = true;
                                            break;
                                        }
                                    }
                                }
                                
                                if (!isDuplicate) {
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
                                    
                                    // Store mapping
                                    markMapping[`M${markCounter}`] = {
                                        type: type,
                                        tag: el.tagName.toLowerCase(),
                                        text: el.textContent?.trim() || el.value || el.placeholder || '',
                                        className: el.className || '',
                                        rect_viewport: {
                                            x: rect.left,
                                            y: rect.top,
                                            width: rect.width,
                                            height: rect.height,
                                            centerX: Math.round(rect.left + rect.width / 2),
                                            centerY: Math.round(rect.top + rect.height / 2)
                                        }
                                    };
                                    
                                    // Mark this element as processed
                                    processedElements.add(signature);
                                    markCounter++;
                                    elementCount++;
                                    
                                    // Debug logging
                                    console.log(`Marked element: M${markCounter-1} - ${type} - ${el.tagName} - "${text}" - pos:(${rect.left},${rect.top})`);
                                }
                            });
                        });
                    });
                    
                    return markMapping;
                }
            """)
            
            logger.info(f"✅ SoM marker injection completed for web collection, found {len(mark_mapping)} elements")
            
            # Log element type distribution
            type_counts = {}
            for mapping in mark_mapping.values():
                element_type = mapping.get('type', 'unknown')
                type_counts[element_type] = type_counts.get(element_type, 0) + 1
            logger.info(f"📊 Element type distribution: {type_counts}")
            
            return mark_mapping
            
        except Exception as e:
            logger.error(f"Failed to inject SoM markers for web collection: {e}")
            return {}
    
    async def _cleanup_som_markers(self, page):
        """Remove SoM markers from DOM - 彻底清理所有SoM相关元素，与web_agent.py保持一致"""
        try:
            logger.info("🧹 Starting comprehensive SoM cleanup in web collector...")
            
            cleanup_result = await page.evaluate("""
                () => {
                    let cleanupStats = {
                        wrappers_removed: 0,
                        styles_removed: 0,
                        overlays_removed: 0,
                        total_cleaned: 0
                    };
                    
                    // 1. 移除所有SoM wrapper元素
                    const wrappers = document.querySelectorAll('.som-wrapper');
                    wrappers.forEach(wrapper => {
                        wrapper.remove();
                        cleanupStats.wrappers_removed++;
                    });
                    
                    // 2. 移除SoM样式表
                    const somStyles = document.getElementById('som-styles');
                    if (somStyles) {
                        somStyles.remove();
                        cleanupStats.styles_removed++;
                    }
                    
                    // 3. 移除SoM overlay
                    const overlay = document.getElementById('som-overlay');
                    if (overlay) {
                        if (overlay.__som_cleanup__) {
                            overlay.__som_cleanup__();
                        }
                        overlay.remove();
                        cleanupStats.overlays_removed++;
                    }
                    
                    // 4. 移除任何其他可能的SoM相关元素
                    const somElements = document.querySelectorAll('[class*="som-"], [id*="som-"]');
                    somElements.forEach(element => {
                        element.remove();
                        cleanupStats.total_cleaned++;
                    });
                    
                    // 5. 清理可能残留的CSS规则
                    const styleSheets = Array.from(document.styleSheets);
                    styleSheets.forEach(sheet => {
                        try {
                            if (sheet.href === null) { // 内联样式表
                                const rules = Array.from(sheet.cssRules || []);
                                rules.forEach((rule, index) => {
                                    if (rule.selectorText && rule.selectorText.includes('som-')) {
                                        sheet.deleteRule(index);
                                        cleanupStats.total_cleaned++;
                                    }
                                });
                            }
                        } catch (e) {
                            // 跨域样式表可能无法访问
                        }
                    });
                    
                    // 6. 强制垃圾回收（如果可用）
                    if (window.gc) {
                        window.gc();
                    }
                    
                    return cleanupStats;
                }
            """)
            
            logger.info(f"✅ SoM cleanup completed in web collector: {cleanup_result}")
            
            # 验证清理是否成功
            verification_result = await page.evaluate("""
                () => {
                    const remainingWrappers = document.querySelectorAll('.som-wrapper').length;
                    const remainingStyles = document.getElementById('som-styles');
                    const remainingOverlays = document.getElementById('som-overlay');
                    
                    return {
                        remaining_wrappers: remainingWrappers,
                        remaining_styles: !!remainingStyles,
                        remaining_overlays: !!remainingOverlays,
                        is_clean: remainingWrappers === 0 && !remainingStyles && !remainingOverlays
                    };
                }
            """)
            
            if verification_result['is_clean']:
                logger.info("✅ SoM cleanup verification passed in web collector - all markers removed")
            else:
                logger.warning(f"⚠️ SoM cleanup verification failed in web collector: {verification_result}")
                # 尝试二次清理
                await self._force_cleanup_som_markers_collector(page)
                
            # 最终验证清理结果
            await self._verify_som_cleanup_collector(page)
                
        except Exception as e:
            logger.error(f"❌ Failed to cleanup SoM markers in web collector: {e}")
            # 尝试强制清理
            await self._force_cleanup_som_markers_collector(page)
    
    async def _force_cleanup_som_markers_collector(self, page):
        """强制清理SoM标记 - web collector备用清理方法"""
        try:
            logger.info("🔄 Attempting force cleanup of SoM markers in web collector...")
            
            await page.evaluate("""
                () => {
                    // 强制移除所有可能的SoM元素
                    const allElements = document.querySelectorAll('*');
                    allElements.forEach(element => {
                        if (element.className && typeof element.className === 'string' && 
                            (element.className.includes('som-') || element.className.includes('som-wrapper') || element.className.includes('som-label'))) {
                            element.remove();
                        }
                        if (element.id && element.id.includes('som-')) {
                            element.remove();
                        }
                    });
                    
                    // 移除所有样式表
                    const styles = document.querySelectorAll('style');
                    styles.forEach(style => {
                        if (style.textContent && style.textContent.includes('som-')) {
                            style.remove();
                        }
                    });
                    
                    // 清理可能残留的CSS规则
                    const styleSheets = Array.from(document.styleSheets);
                    styleSheets.forEach(sheet => {
                        try {
                            if (sheet.href === null) { // 内联样式表
                                const rules = Array.from(sheet.cssRules || []);
                                rules.forEach((rule, index) => {
                                    if (rule.selectorText && rule.selectorText.includes('som-')) {
                                        sheet.deleteRule(index);
                                    }
                                });
                            }
                        } catch (e) {
                            // 跨域样式表可能无法访问
                        }
                    });
                }
            """)
            
            logger.info("✅ Force cleanup completed in web collector")
            
        except Exception as e:
            logger.error(f"❌ Force cleanup failed in web collector: {e}")
    
    async def _verify_som_cleanup_collector(self, page):
        """验证SoM清理结果，确保完全清理 - web collector版本"""
        try:
            verification_result = await page.evaluate("""
                () => {
                    const remainingWrappers = document.querySelectorAll('.som-wrapper').length;
                    const remainingLabels = document.querySelectorAll('.som-label').length;
                    const remainingStyles = document.getElementById('som-styles');
                    const remainingOverlays = document.getElementById('som-overlay');
                    const remainingSomElements = document.querySelectorAll('[class*="som-"], [id*="som-"]').length;
                    
                    return {
                        remaining_wrappers: remainingWrappers,
                        remaining_labels: remainingLabels,
                        remaining_styles: !!remainingStyles,
                        remaining_overlays: !!remainingOverlays,
                        remaining_som_elements: remainingSomElements,
                        is_clean: remainingWrappers === 0 && remainingLabels === 0 && !remainingStyles && !remainingOverlays && remainingSomElements === 0
                    };
                }
            """)
            
            if verification_result['is_clean']:
                logger.info("✅ SoM cleanup verification passed in web collector - page is completely clean")
            else:
                logger.warning(f"⚠️ SoM cleanup verification failed in web collector: {verification_result}")
                # 如果验证失败，再次尝试强制清理
                await self._force_cleanup_som_markers_collector(page)
                
                # 再次验证
                final_verification = await page.evaluate("""
                    () => {
                        const remainingWrappers = document.querySelectorAll('.som-wrapper').length;
                        const remainingLabels = document.querySelectorAll('.som-label').length;
                        const remainingStyles = document.getElementById('som-styles');
                        const remainingOverlays = document.getElementById('som-overlay');
                        const remainingSomElements = document.querySelectorAll('[class*="som-"], [id*="som-"]').length;
                        
                        return {
                            remaining_wrappers: remainingWrappers,
                            remaining_labels: remainingLabels,
                            remaining_styles: !!remainingStyles,
                            remaining_overlays: !!remainingOverlays,
                            remaining_som_elements: remainingSomElements,
                            is_clean: remainingWrappers === 0 && remainingLabels === 0 && !remainingStyles && !remainingOverlays && remainingSomElements === 0
                        };
                    }
                """)
                
                if final_verification['is_clean']:
                    logger.info("✅ SoM cleanup verification passed in web collector after force cleanup")
                else:
                    logger.error(f"❌ SoM cleanup verification failed in web collector even after force cleanup: {final_verification}")
                    
        except Exception as e:
            logger.error(f"❌ SoM cleanup verification failed in web collector: {e}")
    
    def _calculate_page_priority(self, url: str, title: str, content: str, elements: List[WebElement]) -> int:
        """
        计算页面优先级，用于智能收集排序
        优先级从高到低：全局骨架 → 功能入口 → 深层交互 → 辅助页面
        """
        priority = 0
        
        # 1. 全局骨架页面 (最高优先级)
        if self._is_global_skeleton_page(url, title, content):
            priority += 1000
        
        # 2. 功能入口页面 (高优先级)
        if self._is_functional_entry_page(url, title, content, elements):
            priority += 800
        
        # 3. 搜索和列表页面 (高优先级)
        if self._is_search_or_list_page(url, title, content, elements):
            priority += 600
        
        # 4. 表单页面 (中高优先级)
        if self._is_form_page(url, title, content, elements):
            priority += 500
        
        # 5. 详情页面 (中等优先级)
        if self._is_detail_page(url, title, content, elements):
            priority += 300
        
        # 6. 过滤和分页页面 (中等优先级)
        if self._is_filter_or_pagination_page(url, title, content, elements):
            priority += 200
        
        # 7. 辅助页面 (低优先级)
        if self._is_auxiliary_page(url, title, content, elements):
            priority += 100
        
        # 深度惩罚：深度越深，优先级越低
        depth_penalty = self._get_url_depth(url) * 50
        priority -= depth_penalty
        
        return priority
    
    def _is_global_skeleton_page(self, url: str, title: str, content: str) -> bool:
        """判断是否为全局骨架页面（首页、导航页等）"""
        url_lower = url.lower()
        title_lower = title.lower()
        
        # 首页特征
        if (url_lower.endswith('/') or 
            url_lower.endswith('/index') or 
            url_lower.endswith('/home') or
            'home' in title_lower or
            'welcome' in title_lower or
            'main' in title_lower):
            return True
        
        # 导航页面特征
        if any(keyword in url_lower for keyword in ['nav', 'menu', 'sitemap', 'breadcrumb']):
            return True
        
        return False
    
    def _is_functional_entry_page(self, url: str, title: str, content: str, elements: List[WebElement]) -> bool:
        """判断是否为功能入口页面"""
        url_lower = url.lower()
        title_lower = title.lower()
        
        # 检查是否有重要的功能元素
        has_search = any(e.som_type == 'input' and 'search' in e.text_content.lower() for e in elements)
        has_forms = any(e.som_type == 'input' for e in elements)
        has_tables = any(e.element_type == 'table' for e in elements)
        has_cards = any('card' in e.css_classes for e in elements)
        
        # 功能入口页面特征
        if (has_search or has_forms or has_tables or has_cards or
            any(keyword in url_lower for keyword in ['dashboard', 'overview', 'main', 'portal'])):
            return True
        
        return False
    
    def _is_search_or_list_page(self, url: str, title: str, content: str, elements: List[WebElement]) -> bool:
        """判断是否为搜索或列表页面"""
        url_lower = url.lower()
        title_lower = title.lower()
        
        # 搜索页面特征
        if any(keyword in url_lower for keyword in ['search', 'query', 'find', 'lookup']):
            return True
        
        # 列表页面特征
        if any(keyword in url_lower for keyword in ['list', 'table', 'grid', 'items', 'products']):
            return True
        
        # 检查页面元素
        has_search_box = any(e.som_type == 'input' and 'search' in e.text_content.lower() for e in elements)
        has_list_items = any(e.element_type in ['table', 'ul', 'ol'] for e in elements)
        
        return has_search_box or has_list_items
    
    def _is_form_page(self, url: str, title: str, content: str, elements: List[WebElement]) -> bool:
        """判断是否为表单页面"""
        url_lower = url.lower()
        
        # 表单页面特征
        if any(keyword in url_lower for keyword in ['form', 'input', 'submit', 'register', 'login', 'contact']):
            return True
        
        # 检查表单元素
        form_elements = [e for e in elements if e.som_type in ['input', 'select']]
        return len(form_elements) >= 2  # 至少2个表单元素
    
    def _is_detail_page(self, url: str, title: str, content: str, elements: List[WebElement]) -> bool:
        """判断是否为详情页面"""
        url_lower = url.lower()
        
        # 详情页面特征
        if any(keyword in url_lower for keyword in ['detail', 'view', 'show', 'item', 'product', 'article']):
            return True
        
        # 检查详情页面元素
        has_detail_content = any(e.element_type in ['h1', 'h2', 'h3'] for e in elements)
        has_actions = any(e.som_type == 'clickable' for e in elements)
        
        return has_detail_content and has_actions
    
    def _is_filter_or_pagination_page(self, url: str, title: str, content: str, elements: List[WebElement]) -> bool:
        """判断是否为过滤或分页页面"""
        url_lower = url.lower()
        
        # 过滤页面特征
        if any(keyword in url_lower for keyword in ['filter', 'sort', 'order']):
            return True
        
        # 分页页面特征
        if any(keyword in url_lower for keyword in ['page', 'pagination']):
            return True
        
        # 检查分页元素
        has_pagination = any(e.som_type == 'navigation' for e in elements)
        has_filters = any('filter' in e.text_content.lower() for e in elements)
        
        return has_pagination or has_filters
    
    def _is_auxiliary_page(self, url: str, title: str, content: str, elements: List[WebElement]) -> bool:
        """判断是否为辅助页面（帮助、关于、设置等）"""
        url_lower = url.lower()
        title_lower = title.lower()
        
        # 辅助页面特征
        auxiliary_keywords = ['help', 'about', 'contact', 'settings', 'profile', 'account', 'privacy', 'terms']
        if any(keyword in url_lower or keyword in title_lower for keyword in auxiliary_keywords):
            return True
        
        return False
    
    def _get_url_depth(self, url: str) -> int:
        """获取URL深度"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split('/') if p]
            return len(path_parts)
        except:
            return 0
    
    # Enhanced click simulation methods
    async def _find_clickable_elements_playwright(self, page) -> List[Dict[str, Any]]:
        """Find all clickable elements that could lead to navigation using Playwright"""
        
        clickable_elements = await page.evaluate("""
            () => {
                const clickableSelectors = [
                    // Standard navigation elements
                    'a[href]',
                    'button',
                    '[role="button"]',
                    '[role="menuitem"]',
                    '[role="navigation"]',
                    
                    // JavaScript clickable elements
                    'div[onclick]',
                    'span[onclick]',
                    'li[onclick]',
                    'td[onclick]',
                    
                    // Common navigation patterns
                    '.nav-link',
                    '.nav-item',
                    '.menu-item',
                    '.tab',
                    '.breadcrumb a',
                    
                    // Framework-specific selectors
                    '[data-toggle]',
                    '[data-target]',
                    '[ng-click]'
                ];
                
                const elements = [];
                const seen = new Set();
                
                // Process standard selectors
                clickableSelectors.forEach(selector => {
                    try {
                        document.querySelectorAll(selector).forEach(el => {
                            if (elements.length >= 50) return; // Limit total elements
                            
                            const rect = el.getBoundingClientRect();
                            const isVisible = rect.width > 0 && rect.height > 0 && 
                                            window.getComputedStyle(el).display !== 'none';
                            
                            if (!isVisible) return;
                            
                            const text = el.textContent?.trim() || '';
                            // 改进去重逻辑：使用更精确的键值，避免误判重复
                            const key = `${el.tagName}-${text}-${el.className}-${rect.left}-${rect.top}`;
                            
                            if (seen.has(key)) return;
                            seen.add(key);
                            
                            // Generate a simple selector
                            let selector = el.tagName.toLowerCase();
                            if (el.id) {
                                selector = `#${el.id}`;
                            } else if (el.className) {
                                const classes = el.className.split(' ');
                                if (classes.length > 0) {
                                    selector = `.${classes[0]}`;
                                }
                            }
                            
                            elements.push({
                                selector: selector,
                                textContent: text,
                                elementType: el.tagName.toLowerCase(),
                                onclickContent: el.getAttribute('onclick') || '',
                                href: el.href || '',
                                isVisible: true,
                                isEnabled: !el.disabled,
                                position: { x: rect.left, y: rect.top }
                            });
                        });
                    } catch (e) {
                        // Skip invalid selectors
                        console.warn('Invalid selector:', selector, e);
                    }
                });
                
                // Handle framework-specific attributes that can't be used in CSS selectors
                const allElements = document.querySelectorAll('*');
                allElements.forEach(el => {
                    if (elements.length >= 50) return;
                    
                    // Check for Vue.js and Angular attributes
                    const hasVueClick = el.hasAttribute('v-on:click') || el.hasAttribute('@click');
                    const hasAngularClick = el.hasAttribute('(click)') || el.hasAttribute('ng-click');
                    
                    if (hasVueClick || hasAngularClick) {
                        const rect = el.getBoundingClientRect();
                        const isVisible = rect.width > 0 && rect.height > 0 && 
                                        window.getComputedStyle(el).display !== 'none';
                        
                        if (!isVisible) return;
                        
                        const text = el.textContent?.trim() || '';
                        // 改进去重逻辑：使用更精确的键值，避免误判重复
                        const key = `${el.tagName}-${text}-${el.className}-${rect.left}-${rect.top}`;
                        
                        if (seen.has(key)) return;
                        seen.add(key);
                        
                        // Generate a simple selector
                        let selector = el.tagName.toLowerCase();
                        if (el.id) {
                            selector = `#${el.id}`;
                        } else if (el.className) {
                            const classes = el.className.split(' ');
                            if (classes.length > 0) {
                                selector = `.${classes[0]}`;
                            }
                        }
                        
                        elements.push({
                            selector: selector,
                            textContent: text,
                            elementType: el.tagName.toLowerCase(),
                            onclickContent: el.getAttribute('onclick') || '',
                            href: el.href || '',
                            isVisible: true,
                            isEnabled: !el.disabled,
                            position: { x: rect.left, y: rect.top }
                        });
                    }
                });
                
                return elements;
            }
        """)
        
        return clickable_elements[:self.max_clickable_elements]
    
    def _calculate_dom_signature(self, elements: List[WebElement]) -> str:
        """计算DOM内容的特征签名，用于去重 - 改进版本更精确识别页面差异"""
        if not elements:
            return "empty"
        
        # 提取关键特征
        element_types = []
        text_contents = []
        form_elements = []
        business_data = []
        navigation_elements = []
        
        for element in elements:
            # 元素类型
            element_types.append(element.element_type)
            
            # 文本内容（前100个字符，增加长度以捕获更多差异）
            if element.text_content:
                clean_text = element.text_content[:100].lower().strip()
                if len(clean_text) > 3:  # 只保留有意义的文本
                    text_contents.append(clean_text)
            
            # 表单元素
            if element.is_input:
                form_elements.append(f"{element.input_type}:{element.placeholder}")
            
            # 业务数据（CRM特定内容）
            if element.text_content and len(element.text_content) > 5:
                # 检查是否是业务数据（姓名、邮箱、公司等）
                if any(keyword in element.text_content.lower() for keyword in [
                    'account', 'contact', 'lead', 'opportunity', 'meeting', 'task',
                    'email', 'phone', 'company', 'title', 'status', 'amount', 'date'
                ]):
                    business_data.append(element.text_content[:50].lower().strip())
            
            # 导航元素（用于识别页面类型）
            if element.element_type in ['navigation', 'link'] and element.text_content:
                navigation_elements.append(element.text_content[:30].lower().strip())
        
        # 排序以确保一致性
        element_types.sort()
        text_contents.sort()
        form_elements.sort()
        business_data.sort()
        navigation_elements.sort()
        
        # 构建更精确的签名
        signature_parts = [
            f"types:{','.join(element_types[:20])}",  # 限制类型数量
            f"texts:{','.join(text_contents[:15])}",  # 增加文本数量
            f"forms:{','.join(form_elements)}",
            f"business:{','.join(business_data[:10])}",  # 业务数据
            f"nav:{','.join(navigation_elements[:10])}"  # 导航元素
        ]
        
        return "|".join(signature_parts)
    
    async def _simulate_clicks_with_new_page_strategy(self, page, browser, current_url: str, depth: int, nav_manager) -> List[str]:
        """
        使用开新标签页策略的点击模拟 - 天然BFS，父节点不需要回退
        每访问一个子节点，直接创建新页面，访问完成后关闭，父页面环境保持不变
        """
        """Simulate clicks on elements and discover new pages - 改进版本确保点击所有导航链接"""
        
        if not self.enable_click_simulation:
            logger.warning(f"⚠️ Click simulation is disabled: enable_click_simulation={self.enable_click_simulation}")
            return []
        
        if depth >= self.click_simulation_depth:
            logger.info(f"🛑 Reached max click simulation depth: {depth} >= {self.click_simulation_depth}")
            return []
        
        logger.info(f"🖱️ Starting click simulation for {current_url} (depth: {depth})")
        
        # 查找可点击元素，特别关注导航元素
        clickable_elements = await self._find_clickable_elements_playwright(page)
        logger.info(f"🔍 Found {len(clickable_elements)} clickable elements")
        
        discovered_urls = []
        
        # 只处理真正可点击的导航元素
        navigation_elements = []
        
        for element in clickable_elements:
            element_text = element['textContent'].lower().strip()
            
            # 跳过空文本或无效元素
            if not element_text or len(element_text) < 2:
                continue
                
                            # 跳过明显的非导航元素和会打开模态框的元素
                if any(keyword in element_text for keyword in [
                    'copyright', 'powered by', 'about', 'help', 'terms', 'privacy', 'footer', 'close', 'cancel',
                    'license', 'version', 'info', 'information', 'legal', 'disclaimer', 'support',
                    'skip to content', 'skip to main', 'accessibility', 'screen reader', 'sr-only',
                    'jump to', 'go to main', 'bypass', 'skip navigation',
                    # 跳过可能导致点击问题的元素
                    'acme corporation', 'recently viewed', 'field-link', 'last viewed',
                    'globex industries', 'soylent corp', 'initech', 'umbrella corporation',
                    'field-link', 'record', 'detail'
                ]):
                    continue
            
            # 收集所有有意义的导航元素，包括SuiteCRM的主要模块
            # 优先收集主要模块导航链接
            is_main_module = any(keyword in element_text for keyword in [
                'accounts', 'contacts', 'leads', 'opportunities', 'calendar', 'documents', 'reports', 
                'emails', 'tasks', 'meetings', 'calls', 'quotes', 'invoices', 'campaigns', 'cases'
            ])
            
            # 收集其他有意义的导航元素
            is_navigation = any(keyword in element_text for keyword in [
                # 通用导航关键词
                'nav', 'menu', 'home', 'dashboard', 'search', 'browse', 'view', 'list',
                # 操作关键词
                'create', 'new', 'add', 'import', 'export', 'filter', 'insights',
                # 管理相关
                'admin', 'settings', 'users', 'roles', 'products', 'targets'
            ])
            
            # 如果是主要模块或导航元素，且文本长度合适，则添加到导航元素列表
            if (is_main_module or is_navigation) and len(element_text) > 1 and len(element_text) < 100:
                # 主要模块优先
                if is_main_module:
                    navigation_elements.insert(0, element)  # 插入到列表开头
                else:
                    navigation_elements.append(element)
        
        # 点击更多导航元素，确保探索所有主要模块
        elements_to_click = navigation_elements[:20]  # 增加点击数量到20个，确保覆盖所有主要模块
        
        logger.info(f"🎯 Will click {len(elements_to_click)} navigation elements")
        
        for i, element in enumerate(elements_to_click):
            try:
                # 清理文本内容，移除多余的换行符和空格
                clean_text = ' '.join(element['textContent'].strip().split())
                logger.info(f"🖱️ Clicking element {i+1}/{len(elements_to_click)}: {clean_text} ({element['elementType']})")
                
                # Store current URL and hash
                original_url = page.url
                original_hash = await page.evaluate("() => window.location.hash")
                
                # 只点击真正有意义的导航元素
                element_text = element['textContent'].lower().strip()
                
                # 跳过空文本或无效元素
                if not element_text or len(element_text) < 2:
                    logger.debug(f"⏭️ Skipping empty/invalid element: '{element['textContent']}'")
                    continue
                
                # 跳过明显的非导航元素
                if any(keyword in element_text for keyword in [
                    'copyright', 'powered by', 'about', 'help', 'terms', 'privacy', 'footer', 'close', 'cancel',
                    'skip to content', 'skip to main', 'accessibility', 'screen reader', 'sr-only',
                    'jump to', 'go to main', 'bypass', 'skip navigation'
                ]):
                    logger.debug(f"⏭️ Skipping non-navigation element: {element['textContent']}")
                    continue
                
                # 记录导航元素（清理空白字符）
                clean_text = ' '.join(element['textContent'].strip().split())
                if len(clean_text) > 100:  # 如果文本太长，截断
                    clean_text = clean_text[:100] + "..."
                logger.info(f"🎯 Clicking navigation element: {clean_text}")
                
                # Try to close any existing modals first
                try:
                    await page.evaluate("""
                        () => {
                            // Close any open modals
                            const modals = document.querySelectorAll('.modal, .modal-dialog, [role="dialog"]');
                            modals.forEach(modal => {
                                if (modal.classList.contains('show') || modal.style.display !== 'none') {
                                    const closeBtn = modal.querySelector('.close, .btn-close, [data-dismiss="modal"]');
                                    if (closeBtn) {
                                        closeBtn.click();
                                    }
                                }
                            });
                        }
                    """)
                    await asyncio.sleep(0.5)
                except:
                    pass
                
                # 在点击之前设置通用页面状态捕获器（避免重复设置）
                # 只在第一次点击时设置监听器
                if not hasattr(page, '_page_state_collector_set'):
                    await page.evaluate("""
                        () => {
                            // 避免重复设置监听器
                            if (window._page_state_collector_set) return;
                            window._page_state_collector_set = true;
                            
                            // 初始化状态记录
                            window._lastStateUrl = null;
                            window._virtualStateCounter = 0;
                            window._lastContentHash = null;
                            
                            // 通用页面状态捕获回调
                            function notifyPageStateChange(url, type = 'navigation') {
                                // 避免重复的相同URL事件
                                if (window._lastStateUrl === url) return;
                                window._lastStateUrl = url;
                                
                                window.dispatchEvent(new CustomEvent('pageStateChange', {
                                    detail: { 
                                        type: type, 
                                        url: url,
                                        timestamp: Date.now()
                                    }
                                }));
                            }
                            
                            // =============== 1. 标准导航捕获（传统页面刷新） ===============
                            window.addEventListener("load", () => notifyPageStateChange(location.href, 'load'));
                            window.addEventListener("beforeunload", () => notifyPageStateChange(location.href, 'beforeunload'));
                            
                            // =============== 2. History API 捕获（SPA 路由） ===============
                            const originalPushState = history.pushState;
                            history.pushState = function() {
                                originalPushState.apply(history, arguments);
                                const newUrl = arguments[2] || location.href;
                                notifyPageStateChange(newUrl, 'pushState');
                            };
                            
                            const originalReplaceState = history.replaceState;
                            history.replaceState = function() {
                                originalReplaceState.apply(history, arguments);
                                const newUrl = arguments[2] || location.href;
                                notifyPageStateChange(newUrl, 'replaceState');
                            };
                            
                            window.addEventListener('popstate', () => notifyPageStateChange(location.href, 'popstate'));
                            
                            // =============== 3. Hash 路由捕获 ===============
                            window.addEventListener('hashchange', () => notifyPageStateChange(location.href, 'hashchange'));
                            
                            // =============== 4. 无 URL 变化的 UI 状态捕获 ===============
                            // 计算页面内容哈希值
                            function calculateContentHash() {
                                const mainContent = document.querySelector('main') || 
                                                   document.querySelector('#content') || 
                                                   document.querySelector('.content') || 
                                                   document.body;
                                return mainContent ? mainContent.innerHTML.length + '_' + mainContent.children.length : '0';
                            }
                            
                            // 检测DOM变化
                            const observer = new MutationObserver((mutations) => {
                                // 检查是否有显著的内容变化
                                const significantChange = mutations.some(mutation => {
                                    // 新增或删除节点
                                    if (mutation.addedNodes.length > 0 || mutation.removedNodes.length > 0) {
                                        return true;
                                    }
                                    
                                    // 属性变化（可能是动态内容加载）
                                    if (mutation.type === 'attributes' && 
                                        (mutation.attributeName === 'class' || 
                                         mutation.attributeName === 'style' ||
                                         mutation.attributeName === 'data-loaded')) {
                                        return true;
                                    }
                                    
                                    return false;
                                });
                                
                                if (significantChange) {
                                    const currentHash = calculateContentHash();
                                    if (window._lastContentHash !== currentHash) {
                                        window._lastContentHash = currentHash;
                                        window._virtualStateCounter++;
                                        
                                        // 生成虚拟URL来标记状态变化
                                        const virtualUrl = location.href + `#virtual-${window._virtualStateCounter}-${Date.now()}`;
                                        notifyPageStateChange(virtualUrl, 'virtualState');
                                    }
                                }
                            });
                            
                            // 观察整个文档的变化
                            observer.observe(document.body, { 
                                childList: true, 
                                subtree: true,
                                attributes: true,
                                attributeFilter: ['class', 'style', 'data-loaded', 'data-state']
                            });
                            
                            // =============== 5. 框架特定路由捕获 ===============
                            // Angular 路由变化
                            if (window.angular && window.angular.element) {
                                try {
                                    const $rootScope = angular.element(document.body).scope();
                                    if ($rootScope) {
                                        $rootScope.$on('$stateChangeStart', function(event, toState, toParams) {
                                            const stateKey = toState.name + JSON.stringify(toParams);
                                            if (window._lastAngularState !== stateKey) {
                                                window._lastAngularState = stateKey;
                                                const angularUrl = location.href + `#angular-${toState.name}`;
                                                notifyPageStateChange(angularUrl, 'angularStateChange');
                                            }
                                        });
                                    }
                                } catch (e) {
                                    // Angular 可能未完全加载，忽略错误
                                }
                            }
                            
                            // React Router 变化
                            if (window.ReactRouterDOM) {
                                try {
                                    const originalNavigate = window.ReactRouterDOM.navigate;
                                    if (originalNavigate) {
                                        window.ReactRouterDOM.navigate = function(to, options) {
                                            originalNavigate(to, options);
                                            if (window._lastReactRoute !== to) {
                                                window._lastReactRoute = to;
                                                const reactUrl = location.href + `#react-${to}`;
                                                notifyPageStateChange(reactUrl, 'reactRouterNavigate');
                                            }
                                        };
                                    }
                                } catch (e) {
                                    // React Router 可能未完全加载，忽略错误
                                }
                            }
                            
                            // Vue Router 变化
                            if (window.Vue && window.Vue.prototype && window.Vue.prototype.$router) {
                                try {
                                    const originalPush = window.Vue.prototype.$router.push;
                                    window.Vue.prototype.$router.push = function(to) {
                                        originalPush.call(this, to);
                                        const vueUrl = location.href + `#vue-${to.path || to}`;
                                        notifyPageStateChange(vueUrl, 'vueRouterPush');
                                    };
                                } catch (e) {
                                    // Vue Router 可能未完全加载，忽略错误
                                }
                            }
                            
                            // 初始状态记录
                            notifyPageStateChange(location.href, 'initial');
                        }
                    """)
                    
                    # 设置事件监听器
                    await page.evaluate("""
                        () => {
                            if (!window.pageStateChanges) {
                                window.pageStateChanges = [];
                                window.addEventListener('pageStateChange', function(event) {
                                    window.pageStateChanges.push(event.detail);
                                });
                            }
                        }
                    """)
                    
                    page._page_state_collector_set = True
                
                # 原来的点击模拟策略
                try:
                    # 直接点击元素
                    await page.click(element['selector'], timeout=8000)  # 增加超时时间到8秒
                    await asyncio.sleep(self.wait_after_click)
                    
                    # 检查URL是否改变
                    new_url = page.url
                    if new_url != current_url and new_url not in discovered_urls:
                        discovered_urls.append(new_url)
                        logger.info(f"🔄 URL changed: {current_url} -> {new_url}")
                        
                        # 立即收集新页面，而不是返回到原始URL
                        logger.info(f"📄 Immediately collecting new page: {new_url}")
                        try:
                            # 等待页面完全加载
                            await page.wait_for_load_state("networkidle", timeout=5000)
                            await asyncio.sleep(1)
                            
                            # 注入SoM标记
                            som_mapping = await self._mark_elements_with_som(page)
                            
                            # 截图
                            import os
                            existing_pages = len([f for f in os.listdir(self.output_dir) if f.startswith('page_') and f.endswith('.png')])
                            page_number = existing_pages + 1
                            screenshot_path = self.output_dir / f"page_{page_number}.png"
                            await page.screenshot(path=str(screenshot_path), full_page=True)
                            
                            # 提取DOM元素
                            elements = await self._extract_dom_elements_playwright(page=page, som_mapping=som_mapping)
                            
                            # 清理SoM标记
                            await self._cleanup_som_markers(page)
                            
                            # 保存DOM数据
                            dom_file = self.output_dir / f"page_{page_number}_dom.json"
                            title = await page.title()
                            
                            page_data = WebPageData(
                                url=new_url,
                                title=title,
                                elements=elements,
                                screenshots=[str(screenshot_path)],
                                load_time=0.0,
                                page_size=0,
                                links=[],
                                clickable_elements=[e.element_id for e in elements if e.is_clickable],
                                form_elements=[e.element_id for e in elements if e.is_input],
                                table_elements=[e.element_id for e in elements if e.element_type == "table"],
                                page_type="content",
                                website_type="crm",
                                website_description="Customer Relationship Management system",
                                exploration_depth=depth + 1
                            )
                            
                            self._save_dom_data(page_data, dom_file)
                            logger.info(f"✅ Immediately collected page: {new_url}")
                            
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to immediately collect page {new_url}: {e}")
                    
                    # 返回原始URL继续点击其他元素
                    if new_url != current_url:
                        await page.goto(current_url, wait_until="networkidle")
                        await asyncio.sleep(1.0)
                        
                except Exception as e:
                    clean_text = ' '.join(element['textContent'].strip().split())
                    error_msg = str(e)
                    
                    # 提供更具体的错误信息
                    if "Timeout" in error_msg and "exceeded" in error_msg:
                        logger.warning(f"⏰ Click timeout - element may not be accessible: {clean_text}")
                    elif "not visible" in error_msg:
                        logger.warning(f"👁️ Element not visible - skipping: {clean_text}")
                    elif "not enabled" in error_msg:
                        logger.warning(f"🔒 Element not enabled - skipping: {clean_text}")
                    elif "not stable" in error_msg:
                        logger.warning(f"📱 Element not stable - skipping: {clean_text}")
                    else:
                        logger.warning(f"❌ Failed to click element {clean_text}: {e}")
                    
                    # 如果是超时错误，直接跳过，不尝试fallback
                    if "timeout" in error_msg.lower() or "Timeout" in error_msg:
                        logger.info(f"⏭️ Skipping element due to timeout: {clean_text}")
                        continue
                    
                    # 尝试fallback点击
                    try:
                        await page.click(element['selector'], timeout=8000)  # 增加超时时间到8秒
                        logger.info(f"🎯 Fallback: clicked element in parent page: {clean_text}")
                    except Exception as click_error:
                        error_msg = str(click_error)
                        # If click fails, try to close any modals that might have opened
                        try:
                            await page.evaluate("""
                                () => {
                                    const modals = document.querySelectorAll('.modal.show, .modal-dialog.show');
                                    modals.forEach(modal => {
                                        const closeBtn = modal.querySelector('.close, .btn-close');
                                        if (closeBtn) closeBtn.click();
                                    });
                                }
                            """)
                            await asyncio.sleep(0.5)
                        except:
                            pass
                        
                        # Provide more specific error messages
                        if "Timeout" in error_msg and "exceeded" in error_msg:
                            logger.warning(f"⏰ Fallback click timeout - element may not be accessible: {clean_text}")
                        elif "not visible" in error_msg:
                            logger.warning(f"👁️ Fallback click failed - element not visible: {clean_text}")
                        elif "not enabled" in error_msg:
                            logger.warning(f"🔒 Fallback click failed - element not enabled: {clean_text}")
                        else:
                            logger.debug(f"⚠️ Fallback click failed for {clean_text}: {click_error}")
                        continue
                
                # Wait for navigation or content change
                await asyncio.sleep(self.wait_after_click)
                
                # 对于SPA导航，需要等待更长时间让hash更新
                await asyncio.sleep(1.0)  # 额外等待1秒
                
                # 检查页面状态变化事件
                try:
                    page_state_changes = await page.evaluate("() => window.pageStateChanges || []")
                    if page_state_changes:
                        logger.info(f"🎯 Detected {len(page_state_changes)} page state changes:")
                        for change in page_state_changes:
                            logger.info(f"   - Type: {change.get('type', 'unknown')}")
                            if 'url' in change:
                                logger.info(f"     URL: {change['url']}")
                            if 'state' in change:
                                logger.info(f"     State: {change['state']}")
                            if 'to' in change:
                                logger.info(f"     To: {change['to']}")
                            if 'timestamp' in change:
                                logger.info(f"     Time: {change['timestamp']}")
                        
                        # 如果检测到页面状态变化，立即保存当前页面的DOM和截图
                        current_url = page.url
                        if current_url not in discovered_urls:
                            discovered_urls.append(current_url)
                            logger.info(f"📄 Saving DOM and screenshot for page state change: {current_url}")
                            
                            # 等待页面完全加载
                            await page.wait_for_load_state("networkidle", timeout=5000)
                            await asyncio.sleep(1)
                            
                            # 保存截图
                            try:
                                screenshot_path = f"output/debug_collection/page_state_change_{len(discovered_urls)}.png"
                                await page.screenshot(path=screenshot_path, full_page=True)
                                logger.info(f"📸 Screenshot saved: {screenshot_path}")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to save screenshot: {e}")
                            
                            # 保存DOM
                            try:
                                dom_data = await page.evaluate("""
                                    () => {
                                        return {
                                            url: window.location.href,
                                            title: document.title,
                                            html: document.documentElement.outerHTML,
                                            timestamp: new Date().toISOString()
                                        }
                                    }
                                """)
                                
                                import json
                                import os
                                os.makedirs("output/debug_collection", exist_ok=True)
                                dom_path = f"output/debug_collection/page_state_change_{len(discovered_urls)}_dom.json"
                                with open(dom_path, 'w', encoding='utf-8') as f:
                                    json.dump(dom_data, f, ensure_ascii=False, indent=2)
                                logger.info(f"📄 DOM saved: {dom_path}")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to save DOM: {e}")
                        
                        # 清除页面状态变化记录
                        await page.evaluate("() => { window.pageStateChanges = []; }")
                except Exception as e:
                    logger.debug(f"⚠️ Error checking page state changes: {e}")
                
                # 新页面策略已经处理了URL发现，这里只检查父页面的状态
                # 父页面环境保持不变，不需要检查URL变化
                
                # 检测DOM内容变化（SPA导航的核心逻辑）
                try:
                    # 等待动态内容加载
                    await page.wait_for_load_state("networkidle", timeout=5000)
                    await asyncio.sleep(1)  # 额外等待确保内容完全加载
                    
                    # 更准确的DOM内容变化检测
                    content_analysis = await page.evaluate("""
                        () => {
                            // 获取页面主要内容的快照
                            const getContentSnapshot = () => {
                                const selectors = [
                                    'main', '.content', '.container', '#content', 
                                    '.main-content', '.page-content', '.article-content',
                                    '.post-content', '.entry-content', '.body-content',
                                    '.ng-star-inserted', '[ng-version]', '.router-outlet'
                                ];
                                
                                let mainContent = null;
                                for (const selector of selectors) {
                                    mainContent = document.querySelector(selector);
                                    if (mainContent) break;
                                }
                                
                                if (!mainContent) {
                                    mainContent = document.body;
                                }
                                
                                // 获取内容特征
                                const textContent = mainContent.textContent.trim();
                                const visibleElements = mainContent.querySelectorAll('*:not([style*="display: none"]):not([style*="visibility: hidden"])');
                                const formElements = mainContent.querySelectorAll('input, select, textarea, button');
                                const links = mainContent.querySelectorAll('a[href]');
                                
                                return {
                                    textLength: textContent.length,
                                    visibleElements: visibleElements.length,
                                    formElements: formElements.length,
                                    links: links.length,
                                    hasForms: formElements.length > 0,
                                    hasNavigation: links.length > 5,
                                    mainText: textContent.substring(0, 200) // 前200个字符作为特征
                                };
                            };
                            
                            return getContentSnapshot();
                        }
                    """)
                    
                    # 检查内容是否发生了实质性变化
                    content_changed = False
                    clean_text = ' '.join(element['textContent'].strip().split())
                    
                    # 如果内容有足够的变化，认为页面发生了变化
                    if (content_analysis['textLength'] > 100 and 
                        content_analysis['visibleElements'] > 10):
                        content_changed = True
                        logger.info(f"📄 Content changed after click on: {clean_text}")
                        logger.info(f"   - Text length: {content_analysis['textLength']}")
                        logger.info(f"   - Visible elements: {content_analysis['visibleElements']}")
                        logger.info(f"   - Form elements: {content_analysis['formElements']}")
                    
                    # 检查URL hash变化
                    current_hash = await page.evaluate("() => window.location.hash")
                    hash_changed = current_hash != original_hash
                    
                    if hash_changed:
                        logger.debug(f"🔍 Hash changed: {original_hash} -> {current_hash}")
                    
                    # 如果内容变化或hash变化，都认为是新页面
                    if content_changed or hash_changed:
                        # 构建新URL（优先使用hash变化后的URL）
                        if hash_changed:
                            new_page_url = f"{original_url.split('#')[0]}{current_hash}"
                        else:
                            # 如果只是内容变化，创建一个虚拟URL
                            element_text = element['textContent'].lower().strip()
                            if any(keyword in element_text for keyword in ['accounts', 'contacts', 'leads', 'opportunities', 'calendar', 'documents']):
                                # 为主要模块创建虚拟URL
                                module_name = next((keyword for keyword in ['accounts', 'contacts', 'leads', 'opportunities', 'calendar', 'documents'] if keyword in element_text), 'page')
                                new_page_url = f"{original_url}#/{module_name}"
                            else:
                                # 为其他内容创建通用虚拟URL
                                new_page_url = f"{original_url}#/content-{len(discovered_urls) + 1}"
                        
                        # 添加到发现的URL列表
                        if new_page_url not in discovered_urls:
                            discovered_urls.append(new_page_url)
                            logger.info(f"🎯 New page discovered: {clean_text} -> {new_page_url}")
                            
                            # 立即收集新页面
                            logger.info(f"📄 Immediately collecting new page: {new_page_url}")
                            try:
                                # 等待页面完全加载
                                await page.wait_for_load_state("networkidle", timeout=5000)
                                await asyncio.sleep(1)
                                
                                # 注入SoM标记
                                som_mapping = await self._mark_elements_with_som(page)
                                
                                # 截图
                                import os
                                existing_pages = len([f for f in os.listdir(self.output_dir) if f.startswith('page_') and f.endswith('.png')])
                                page_number = existing_pages + 1
                                screenshot_path = self.output_dir / f"page_{page_number}.png"
                                await page.screenshot(path=str(screenshot_path), full_page=True)
                                
                                # 提取DOM元素
                                elements = await self._extract_dom_elements_playwright(page=page, som_mapping=som_mapping)
                                
                                # 清理SoM标记
                                await self._cleanup_som_markers(page)
                                
                                # DOM内容去重检查（在点击模拟中）
                                is_duplicate_in_click = False
                                # 这里需要检查是否与已收集的页面重复
                                # 由于这里没有直接访问web_pages，我们通过文件系统检查
                                import os
                                existing_dom_files = [f for f in os.listdir(self.output_dir) if f.endswith('_dom.json')]
                                if len(existing_dom_files) > 1:  # 不是第一个页面
                                    current_signature = self._calculate_dom_signature(elements)
                                    # 检查与现有DOM文件的相似性
                                    for dom_file in existing_dom_files:
                                        try:
                                            with open(self.output_dir / dom_file, 'r') as f:
                                                import json
                                                existing_data = json.load(f)
                                                existing_elements = [WebElement(**elem) for elem in existing_data.get('elements', [])]
                                                existing_signature = self._calculate_dom_signature(existing_elements)
                                                if current_signature == existing_signature:
                                                    is_duplicate_in_click = True
                                                    logger.info(f"🔄 Skipping duplicate content in click simulation: {new_page_url}")
                                                    break
                                        except Exception as e:
                                            logger.debug(f"⚠️ Error checking existing DOM file {dom_file}: {e}")
                                
                                if not is_duplicate_in_click:
                                    # 保存DOM数据
                                    dom_file = self.output_dir / f"page_{page_number}_dom.json"
                                    title = await page.title()
                                    
                                    page_data = WebPageData(
                                        url=new_page_url,
                                        title=title,
                                        elements=elements,
                                        screenshots=[str(screenshot_path)],
                                        load_time=0.0,
                                        page_size=0,
                                        links=[],
                                        clickable_elements=[e.element_id for e in elements if e.is_clickable],
                                        form_elements=[e.element_id for e in elements if e.is_input],
                                        table_elements=[e.element_id for e in elements if e.element_type == "table"],
                                        page_type="content",
                                        website_type="crm",
                                        website_description="Customer Relationship Management system",
                                        exploration_depth=depth + 1
                                    )
                                    
                                    self._save_dom_data(page_data, dom_file)
                                    logger.info(f"✅ Immediately collected page: {new_page_url}")
                                else:
                                    logger.info(f"⏭️ Skipped duplicate page in click simulation: {new_page_url}")
                                
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to immediately collect page {new_page_url}: {e}")
                    
                    # 如果内容没有变化，记录调试信息
                    else:
                        logger.debug(f"🔍 No significant content change detected for: {clean_text}")
                
                except Exception as e:
                    logger.debug(f"⚠️ Error checking content changes: {e}")
                
                # 新页面策略：父页面环境保持不变，不需要回退
                # 每个子节点都在新页面中处理，完成后关闭新页面
            
            except Exception as e:
                clean_text = ' '.join(element['textContent'].strip().split())
                logger.warning(f"Failed to click element {clean_text}: {e}")
                continue
        
        logger.info(f"✅ Click simulation completed. Discovered {len(discovered_urls)} new URLs")
        return discovered_urls
    

    
    async def collect_with_click_simulation(self, urls: List[str]) -> List[WebPageData]:
        """Hybrid collection strategy: URL traversal + click simulation using two-level BFS"""
        
        if not self.enable_click_simulation:
            logger.warning("Click simulation is disabled. Use enable_click_simulation=True to enable.")
            return await self.collect_web_data(urls)
        
        logger.info(f"🚀 Starting hybrid web collection strategy for {len(urls)} URLs")
        
        web_pages = []
        visited_states = set()  # (URL, DOM hash) for deduplication
        
        # Initialize queues
        url_queue = deque([(url, 0) for url in urls])  # (URL, depth)
        interaction_queue = deque()  # (URL, element, depth)
        
        # State tracking
        max_urls = self.config.get('max_pages', 50)
        max_click_depth = self.config.get('click_simulation_depth', 3)
        max_states = self.config.get('max_states', 1000)
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.set_viewport_size({"width": 1280, "height": 720})
                

                
                # Outer BFS: URL-based traversal
                while url_queue and len(web_pages) < max_urls and len(visited_states) < max_states:
                    current_url, url_depth = url_queue.popleft()
                    
                    try:
                        logger.info(f"🌐 Loading URL: {current_url} (depth: {url_depth})")
                        
                        # 使用更宽松的加载策略
                        await page.goto(current_url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                        # 给页面一些时间加载，不强制等待networkidle
                        await asyncio.sleep(2)
                        
                        # Auto-login for SuiteCRM (only on first page)
                        if "localhost:8080" in current_url and len(web_pages) == 0:
                            logger.info("🔐 Detected SuiteCRM (localhost:8080), attempting auto-login...")
                            try:
                                from .tool import WebTools
                                login_success = await WebTools.login_suitecrm(page)  # SuiteCRM 特定的登录方法
                                if login_success:
                                    logger.info("✅ Auto-login successful")
                                    # Wait a bit after login for page to stabilize
                                    await asyncio.sleep(3)
                                else:
                                    logger.warning("⚠️ Auto-login failed, continuing without login")
                            except Exception as e:
                                logger.error(f"❌ Auto-login error: {e}")
                                logger.warning("⚠️ Continuing without login")
                        
                        # Mark elements with SoM and extract DOM elements
                        som_mapping = await self._mark_elements_with_som(page)
                        elements = await self._extract_dom_elements_playwright(page=page, som_mapping=som_mapping)
                        
                        # Calculate DOM signature for deduplication
                        dom_signature = self._calculate_dom_signature(elements)
                        state_key = (current_url, dom_signature)
                        
                        # Check if state already visited
                        if state_key in visited_states:
                            logger.info(f"⏭️ Skipping duplicate state: {current_url}")
                            continue
                        
                        visited_states.add(state_key)
                        
                        # Wait for SPA content to load completely before taking screenshot
                        try:
                            # Wait for network to be idle
                            await page.wait_for_load_state("networkidle", timeout=5000)
                            
                            # Wait for Angular/AngularJS to finish rendering
                            await page.evaluate("""
                                () => {
                                    return new Promise((resolve) => {
                                        // Wait for Angular to finish
                                        if (window.angular) {
                                            const $rootScope = angular.element(document.body).scope();
                                            if ($rootScope) {
                                                $rootScope.$apply();
                                            }
                                        }
                                        
                                        // Wait for any pending animations
                                        setTimeout(resolve, 1000);
                                    });
                                }
                            """)
                            
                            # Additional wait for dynamic content
                            await asyncio.sleep(2)
                            
                            # Check if page content has meaningful differences
                            content_analysis = await page.evaluate("""
                                () => {
                                    // Get main content area
                                    const mainContent = document.querySelector('main') || 
                                                       document.querySelector('.content') || 
                                                       document.querySelector('#content') || 
                                                       document.querySelector('.main-content') ||
                                                       document.body;
                                    
                                    if (!mainContent) return { hasContent: false, contentLength: 0 };
                                    
                                    const textContent = mainContent.textContent.trim();
                                    const visibleElements = mainContent.querySelectorAll('*:not([style*="display: none"]):not([style*="visibility: hidden"])');
                                    
                                    // Check for meaningful content
                                    const hasContent = textContent.length > 100 && visibleElements.length > 10;
                                    
                                    return {
                                        hasContent: hasContent,
                                        contentLength: textContent.length,
                                        visibleElements: visibleElements.length,
                                        mainText: textContent.substring(0, 200)
                                    };
                                }
                            """)
                            
                            if not content_analysis.get('hasContent', False):
                                logger.warning(f"⚠️ Page {current_url} has insufficient content, may be duplicate")
                            
                        except Exception as e:
                            logger.warning(f"⚠️ Wait for SPA content failed: {e}")
                        
                        # Collect page data
                        title = await page.title()
                        
                        # Create more descriptive screenshot filename
                        url_parts = current_url.replace('http://localhost:8080', '').replace('/', '_').replace('?', '_').replace('&', '_')
                        if len(url_parts) > 50:
                            url_parts = url_parts[:50]
                        screenshot_filename = f"page_{len(web_pages) + 1}_{url_parts}.png"
                        screenshot_path = self.output_dir / screenshot_filename
                        
                        # Take screenshot with SoM markers
                        await page.screenshot(path=str(screenshot_path), full_page=True)
                        
                        # Clean up SoM markers after screenshot
                        await self._cleanup_som_markers(page)
                        
                        # Save DOM data
                        dom_file = self.output_dir / f"page_{len(web_pages) + 1}_dom.json"
                        self._save_dom_data(WebPageData(
                            url=current_url,
                            title=title,
                            elements=elements,
                            screenshots=[str(screenshot_path)],
                            load_time=0.0
                        ), dom_file)
                        
                        page_data = WebPageData(
                            url=current_url,
                            title=title,
                            elements=elements,
                            screenshots=[str(screenshot_path)],
                            load_time=0.0
                        )
                        web_pages.append(page_data)
                        
                        logger.info(f"✅ Collected page {len(web_pages)}: {current_url}")
                        
                        # Phase 1: URL extraction (add to URL queue)
                        new_urls = await self._extract_urls_from_page(page, current_url)
                        for new_url in new_urls:
                            if new_url not in [url for url, _ in url_queue] and new_url not in [page.url for page in web_pages]:
                                url_queue.append((new_url, url_depth + 1))
                                logger.info(f"🔗 Added URL to queue: {new_url}")
                        
                        # Phase 2: Interaction extraction (add to interaction queue)
                        if url_depth < max_click_depth:
                            interactive_elements = self._find_interactive_elements(elements)
                            for element in interactive_elements:
                                interaction_queue.append((current_url, element, url_depth))
                                logger.info(f"🖱️ Added interaction to queue: {element.text_content[:30]}...")
                        
                        # Inner BFS: Click simulation on current page
                        await self._simulate_clicks_inner_bfs(page, current_url, url_depth, interaction_queue, visited_states, web_pages, max_click_depth)
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing URL {current_url}: {e}")
                        continue
                
                await browser.close()
            
        except Exception as e:
            logger.error(f"Error with hybrid collection: {e}")
        
        logger.info(f"🎉 Hybrid collection completed. Collected {len(web_pages)} pages, {len(visited_states)} unique states")
        
        # Collection summary
        logger.info(f"📊 Collection summary:")
        logger.info(f"  - Total pages collected: {len(web_pages)}")
        logger.info(f"  - Unique states visited: {len(visited_states)}")
        logger.info(f"  - URLs in queue: {len(url_queue)}")
        logger.info(f"  - Interactions in queue: {len(interaction_queue)}")
        
        return web_pages

    async def _extract_urls_from_page(self, page, current_url: str) -> List[str]:
        """Extract all URLs from current page"""
        try:
            urls = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => {
                        const href = link.href;
                        // Only include same-domain URLs
                        if (href.startsWith(window.location.origin) || href.startsWith('/')) {
                            return href;
                        }
                        return null;
                    }).filter(url => url !== null);
                }
            """)
            return list(set(urls))  # Remove duplicates
        except Exception as e:
            logger.error(f"Error extracting URLs: {e}")
            return []

    def _find_interactive_elements(self, elements: List[WebElement]) -> List[WebElement]:
        """Find interactive elements for click simulation"""
        interactive_elements = []
        
        for element in elements:
            # Navigation elements
            if element.element_type == "navigation":
                interactive_elements.append(element)
            # Buttons
            elif element.element_type == "button":
                interactive_elements.append(element)
            # Form elements
            elif element.is_input and element.input_type in ["submit", "button"]:
                interactive_elements.append(element)
            # Links that might be interactive
            elif element.element_type == "link" and element.text_content:
                # Filter out navigation links (already handled)
                if not any(keyword in element.text_content.lower() for keyword in [
                    'accounts', 'contacts', 'leads', 'opportunities', 'calendar', 'documents', 'reports'
                ]):
                    interactive_elements.append(element)
        
        return interactive_elements[:10]  # Limit to 10 elements per page

    async def _simulate_clicks_inner_bfs(self, page, current_url: str, url_depth: int, 
                                       interaction_queue: deque, visited_states: set, 
                                       web_pages: List[WebPageData], max_click_depth: int):
        """Inner BFS: Click simulation on current page"""
        if url_depth >= max_click_depth:
            return
        
        # Get interactions for current page
        current_interactions = [item for item in interaction_queue if item[0] == current_url]
        
        for _, element, depth in current_interactions[:5]:  # Limit to 5 clicks per page
            try:
                logger.info(f"🖱️ Clicking element: {element.text_content[:30]}...")
                
                # Store original state
                original_url = page.url
                original_dom = await page.evaluate("() => document.body.innerHTML")
                
                # Perform click
                await page.click(element.css_selector, timeout=5000)
                await page.wait_for_load_state("networkidle", timeout=3000)
                
                # Check for state change
                new_url = page.url
                new_dom = await page.evaluate("() => document.body.innerHTML")
                
                # URL changed - add to URL queue
                if new_url != original_url:
                    logger.info(f"🔗 URL changed: {original_url} -> {new_url}")
                    # This will be handled by outer BFS
                    continue
                
                # DOM changed - check if new state
                if new_dom != original_dom:
                    # Mark elements with SoM and extract new elements
                    som_mapping = await self._mark_elements_with_som(page)
                    new_elements = await self._extract_dom_elements_playwright(page=page, som_mapping=som_mapping)
                    new_dom_signature = self._calculate_dom_signature(new_elements)
                    new_state_key = (new_url, new_dom_signature)
                    
                    if new_state_key not in visited_states:
                        visited_states.add(new_state_key)
                        
                        # Collect new page data
                        title = await page.title()
                        screenshot_path = self.output_dir / f"page_{len(web_pages) + 1}.png"
                        await page.screenshot(path=str(screenshot_path), full_page=True)
                        
                        # Clean up SoM markers after screenshot
                        await self._cleanup_som_markers(page)
                        
                        # Save DOM data
                        dom_file = self.output_dir / f"page_{len(web_pages) + 1}_dom.json"
                        self._save_dom_data(WebPageData(
                            url=new_url,
                            title=title,
                            elements=new_elements,
                            screenshots=[str(screenshot_path)],
                            load_time=0.0
                        ), dom_file)
                        
                        page_data = WebPageData(
                            url=new_url,
                            title=title,
                            elements=new_elements,
                            screenshots=[str(screenshot_path)],
                            load_time=0.0
                        )
                        web_pages.append(page_data)
                        
                        logger.info(f"✅ Collected new state: {new_url}")
                        
                        # Add new interactive elements to queue
                        new_interactive_elements = self._find_interactive_elements(new_elements)
                        for new_element in new_interactive_elements:
                            interaction_queue.append((new_url, new_element, depth + 1))
                
                else:
                    logger.info(f"⏭️ No state change detected")
                
            except Exception as e:
                error_msg = str(e)
                if "Timeout" in error_msg and "exceeded" in error_msg:
                    logger.warning(f"⏰ Element click timeout - element may not be visible or clickable: {element.text_content[:30]}...")
                elif "not visible" in error_msg:
                    logger.warning(f"👁️ Element not visible - skipping: {element.text_content[:30]}...")
                elif "not enabled" in error_msg:
                    logger.warning(f"🔒 Element not enabled - skipping: {element.text_content[:30]}...")
                elif "not stable" in error_msg:
                    logger.warning(f"📱 Element not stable - skipping: {element.text_content[:30]}...")
                else:
                    logger.error(f"❌ Error clicking element: {e}")
                continue

    async def collect_with_hybrid_strategy(self, urls: List[str]) -> List[WebPageData]:
        """
        Hybrid collection strategy: URL traversal + click simulation
        Uses two-level BFS: outer BFS for URL queue, inner BFS for interaction queue
        Enhanced with business data filtering to maintain page coherence
        """
        logger.info("🔄 Starting hybrid web collection strategy with business data filtering")
        
        web_pages = []
        visited_states = set()  # (URL, DOM hash) for deduplication
        
        # Initialize queues
        url_queue = deque([(url, 0) for url in urls])  # (URL, depth)
        interaction_queue = deque()  # (URL, element, depth)
        
        # State tracking
        max_urls = self.config.get('max_pages', 50)
        max_click_depth = self.config.get('click_simulation_depth', 3)
        max_states = self.config.get('max_states', 1000)
        
        # Business data filtering settings
        min_business_data_elements = self.config.get('min_business_data_elements', 3)  # Minimum business data elements required
        business_data_coherence_threshold = self.config.get('business_data_coherence_threshold', 0.3)  # Ratio of pages with business data
        consecutive_empty_pages_limit = self.config.get('consecutive_empty_pages_limit', 5)  # Increased limit for navigation coherence
        
        # Track business data statistics and navigation paths
        total_pages_processed = 0
        pages_with_business_data = 0
        consecutive_empty_pages = 0
        business_data_pages = []  # Pages with business data for coherence
        navigation_paths = {}  # Track navigation paths to business data pages
        essential_pages = set()  # Pages that are essential for navigation
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Login if needed
            if self.config.get('auto_login', False):
                await self._login_to_suitecrm(page)
            
            # Outer BFS: URL-based traversal
            while url_queue and len(web_pages) < max_urls and len(visited_states) < max_states:
                current_url, url_depth = url_queue.popleft()
                
                try:
                    logger.info(f"🌐 Loading URL: {current_url} (depth: {url_depth})")
                    
                    # 使用更宽松的加载策略
                    await page.goto(current_url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                    # 给页面一些时间加载，不强制等待networkidle
                    await asyncio.sleep(2)
                    
                    # Mark elements with SoM and extract DOM elements
                    som_mapping = await self._mark_elements_with_som(page)
                    elements = await self._extract_dom_elements_playwright(page=page, som_mapping=som_mapping)
                    
                    # Calculate DOM signature for deduplication
                    dom_signature = self._calculate_dom_signature(elements)
                    state_key = (current_url, dom_signature)
                    
                    # Check if state already visited
                    if state_key in visited_states:
                        logger.info(f"⏭️ Skipping duplicate state: {current_url}")
                        continue
                    
                    visited_states.add(state_key)
                    total_pages_processed += 1
                    
                    # Check if page contains business data
                    business_data_elements = self._count_business_data_elements(elements)
                    has_business_data = business_data_elements >= min_business_data_elements
                    
                    # Check if this is a navigation page
                    is_navigation_page = self._is_navigation_page(elements, current_url)
                    
                    if has_business_data:
                        pages_with_business_data += 1
                        consecutive_empty_pages = 0
                        business_data_pages.append(current_url)
                        logger.info(f"✅ Page contains {business_data_elements} business data elements: {current_url}")
                        
                        # Mark this page and its navigation path as essential
                        self._mark_navigation_path_as_essential(current_url, url_depth, navigation_paths, essential_pages)
                    elif is_navigation_page:
                        consecutive_empty_pages = 0  # Reset counter for navigation pages
                        essential_pages.add(current_url)
                        logger.info(f"🔗 Page is navigation page (no business data): {current_url}")
                    else:
                        consecutive_empty_pages += 1
                        logger.info(f"⚠️ Page has only {business_data_elements} business data elements: {current_url}")
                    
                    # Check if we should skip this page based on navigation coherence
                    should_skip = False
                    skip_reason = ""
                    
                    # Never skip essential navigation pages or navigation pages
                    if current_url in essential_pages or is_navigation_page:
                        logger.info(f"🔗 Keeping essential/navigation page: {current_url}")
                        should_skip = False
                    else:
                        # Skip if too many consecutive empty pages (but not essential ones)
                        if consecutive_empty_pages > consecutive_empty_pages_limit:
                            should_skip = True
                            skip_reason = f"Too many consecutive non-essential pages without business data ({consecutive_empty_pages})"
                        
                        # Skip if business data ratio is too low and we have enough pages
                        if total_pages_processed > 10:  # Increased threshold
                            business_data_ratio = pages_with_business_data / total_pages_processed
                            if business_data_ratio < business_data_coherence_threshold:
                                should_skip = True
                                skip_reason = f"Business data ratio too low ({business_data_ratio:.2f} < {business_data_coherence_threshold})"
                    
                    if should_skip:
                        logger.info(f"⏭️ Skipping page due to business data coherence: {skip_reason}")
                        continue
                    
                    # Collect page data
                    title = await page.title()
                    screenshot_path = self.output_dir / f"page_{len(web_pages) + 1}.png"
                    await page.screenshot(path=str(screenshot_path), full_page=True)
                    
                    # Clean up SoM markers after screenshot
                    await self._cleanup_som_markers(page)
                    
                    # Save DOM data
                    dom_file = self.output_dir / f"page_{len(web_pages) + 1}_dom.json"
                    self._save_dom_data(WebPageData(
                        url=current_url,
                        title=title,
                        elements=elements,
                        screenshots=[str(screenshot_path)],
                        load_time=0.0
                    ), dom_file)
                    
                    page_data = WebPageData(
                        url=current_url,
                        title=title,
                        elements=elements,
                        screenshots=[str(screenshot_path)],
                        load_time=0.0
                    )
                    web_pages.append(page_data)
                    
                    logger.info(f"✅ Collected page {len(web_pages)}: {current_url}")
                    
                    # Log business data statistics
                    if total_pages_processed > 0:
                        business_data_ratio = pages_with_business_data / total_pages_processed
                        logger.info(f"📊 Business data statistics: {pages_with_business_data}/{total_pages_processed} pages ({business_data_ratio:.2f})")
                    
                    # Phase 1: URL extraction (add to URL queue)
                    new_urls = await self._extract_urls_from_page(page, current_url)
                    for new_url in new_urls:
                        if new_url not in [url for url, _ in url_queue] and new_url not in [page.url for page in web_pages]:
                            url_queue.append((new_url, url_depth + 1))
                            # Track navigation path from current page to new URL
                            if current_url not in navigation_paths:
                                navigation_paths[current_url] = []
                            navigation_paths[current_url].append(new_url)
                            logger.info(f"🔗 Added URL to queue: {new_url}")
                    
                    # Phase 2: Interaction extraction (add to interaction queue)
                    if url_depth < max_click_depth:
                        interactive_elements = self._find_interactive_elements(elements)
                        for element in interactive_elements:
                            interaction_queue.append((current_url, element, url_depth))
                            logger.info(f"🖱️ Added interaction to queue: {element.text_content[:30]}...")
                    
                    # Inner BFS: Click simulation on current page
                    await self._simulate_clicks_inner_bfs(page, current_url, url_depth, interaction_queue, visited_states, web_pages, max_click_depth)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing URL {current_url}: {e}")
                    continue
            
            await browser.close()
        
        # Final business data statistics
        if total_pages_processed > 0:
            final_business_data_ratio = pages_with_business_data / total_pages_processed
            logger.info(f"📊 Final business data statistics:")
            logger.info(f"   - Total pages processed: {total_pages_processed}")
            logger.info(f"   - Pages with business data: {pages_with_business_data}")
            logger.info(f"   - Business data ratio: {final_business_data_ratio:.2f}")
            logger.info(f"   - Pages collected: {len(web_pages)}")
            logger.info(f"   - Business data pages: {len(business_data_pages)}")
            logger.info(f"   - Essential navigation pages: {len(essential_pages)}")
            logger.info(f"   - Navigation paths tracked: {len(navigation_paths)}")
        
        logger.info(f"🎉 Hybrid collection completed. Collected {len(web_pages)} pages, {len(visited_states)} unique states")
        return web_pages

    async def _login_to_suitecrm(self, page):
        """Login to SuiteCRM using Playwright"""
        try:
            # Open SuiteCRM login page
            await page.goto("https://localhost:8080/index.php?entryPoint=login")
            
            # Fill in login form
            await page.fill('input[name="username"]', 'admin')
            await page.fill('input[name="password"]', 'password')
            
            # Submit form
            await page.click('input[type="submit"]')
            
            # Wait for login to complete
            await page.waitForNavigation(waitUntil="networkidle")
            
            logger.info("🔐 SuiteCRM login successful")
        except Exception as e:
            logger.error(f"❌ Failed to login to SuiteCRM: {e}")

    async def _extract_urls_from_page(self, page, current_url: str) -> List[str]:
        """Extract all URLs from current page"""
        try:
            urls = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => {
                        const href = link.href;
                        // Only include same-domain URLs
                        if (href.startsWith(window.location.origin) || href.startsWith('/')) {
                            return href;
                        }
                        return null;
                    }).filter(url => url !== null);
                }
            """)
            return list(set(urls))  # Remove duplicates
        except Exception as e:
            logger.error(f"Error extracting URLs: {e}")
            return []

    def _find_interactive_elements(self, elements: List[WebElement]) -> List[WebElement]:
        """Find interactive elements for click simulation"""
        interactive_elements = []
        
        for element in elements:
            # Navigation elements
            if element.element_type == "navigation":
                interactive_elements.append(element)
            # Buttons
            elif element.element_type == "button":
                interactive_elements.append(element)
            # Form elements
            elif element.is_input and element.input_type in ["submit", "button"]:
                interactive_elements.append(element)
            # Links that might be interactive
            elif element.element_type == "link" and element.text_content:
                # Filter out navigation links (already handled)
                if not any(keyword in element.text_content.lower() for keyword in [
                    'accounts', 'contacts', 'leads', 'opportunities', 'calendar', 'documents', 'reports'
                ]):
                    interactive_elements.append(element)
        
        return interactive_elements[:10]  # Limit to 10 elements per page

    def _count_business_data_elements(self, elements: List[WebElement]) -> int:
        """Count business data elements in a page"""
        business_data_count = 0
        
        for element in elements:
            # Check if element is marked as business data
            if getattr(element, 'som_type', '') == 'business_data':
                business_data_count += 1
                continue
            
            # Check element text content for business data patterns
            text_content = element.text_content.lower().strip()
            if not text_content:
                continue
            
            # Business data patterns
            business_patterns = [
                # Names (2+ words or single word with capital letters)
                r'^[a-z]+ [a-z]+$',  # Two word names
                r'^[a-z]+$',  # Single word names
                # Email addresses
                r'^[^\s@]+@[^\s@]+\.[^\s@]+$',
                # Phone numbers
                r'^[+]?[0-9\s\-\(\)]+$',
                # Currency amounts
                r'^\$?[0-9,]+(\.?[0-9]{2})?$',
                # Dates
                r'^[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}$',
                r'^[a-z]+ [0-9]{1,2},? [0-9]{4}$',
                # Status values
                r'^(active|inactive|pending|completed|open|closed|new|old|prospecting|proposal|negotiation|closed won|closed lost)$',
                # Company names (multiple words with capitals)
                r'^[a-z]+ [a-z]+ [a-z]+',
                # CRM-specific business terms
                r'^(accounts?|contacts?|opportunities?|leads?|quotes?|calendar|documents?|emails?|meetings?|tasks?|projects?|companies?|customers?|vendors?|partners?)$',
                # SuiteCRM specific business data patterns
                r'^(website redesign|mobile app development|cloud migration|cybersecurity upgrade|data analytics implementation|e-commerce platform|digital transformation|it consulting|software development|system integration)$',
                r'^(acme corporation|globex industries|soylent corp|initech|umbrella corporation|massive dynamic|stark industries|wayne enterprises|wonka industries)$'
            ]
            
            # Check if text matches any business pattern
            for pattern in business_patterns:
                if re.match(pattern, text_content):
                    business_data_count += 1
                    break
            
            # Additional simple checks
            if any(keyword in text_content for keyword in [
                '@', '$', 'active', 'inactive', 'contact', 'lead', 'opportunity', 
                'account', 'quote', 'meeting', 'task', 'document', 'email', 
                'project', 'company', 'customer', 'vendor', 'partner',
                'website', 'mobile', 'cloud', 'cybersecurity', 'data', 
                'e-commerce', 'digital', 'it', 'software', 'system',
                'acme', 'globex', 'soylent', 'initech', 'umbrella', 
                'massive', 'stark', 'wayne', 'wonka'
            ]):
                business_data_count += 1
        
        return business_data_count

    def _mark_navigation_path_as_essential(self, business_data_url: str, depth: int, 
                                          navigation_paths: Dict[str, List[str]], 
                                          essential_pages: Set[str]) -> None:
        """Mark navigation path to business data page as essential"""
        essential_pages.add(business_data_url)
        
        # Mark parent pages in the navigation path as essential
        for parent_url, child_urls in navigation_paths.items():
            if business_data_url in child_urls:
                essential_pages.add(parent_url)
                logger.debug(f"🔗 Marked parent page as essential: {parent_url} -> {business_data_url}")
                
                # Recursively mark grandparent pages
                for grandparent_url, parent_child_urls in navigation_paths.items():
                    if parent_url in parent_child_urls:
                        essential_pages.add(grandparent_url)
                        logger.debug(f"🔗 Marked grandparent page as essential: {grandparent_url} -> {parent_url} -> {business_data_url}")
        
        logger.info(f"🔗 Marked navigation path to business data page: {business_data_url} (depth: {depth})")

    def _is_navigation_page(self, elements: List[WebElement], url: str) -> bool:
        """Check if page is a navigation page that should be preserved"""
        # Check for navigation indicators in URL
        navigation_url_patterns = [
            'index', 'home', 'dashboard', 'main', 'menu', 'navigation',
            'accounts', 'contacts', 'leads', 'opportunities', 'calendar',
            'documents', 'reports', 'settings', 'admin', 'user'
        ]
        
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in navigation_url_patterns):
            return True
        
        # Check for navigation elements in DOM
        navigation_elements = 0
        for element in elements:
            # Navigation menus, breadcrumbs, etc.
            if element.element_type == "navigation":
                navigation_elements += 1
            # Main navigation links
            elif element.element_type == "link" and any(keyword in element.text_content.lower() for keyword in [
                'accounts', 'contacts', 'leads', 'opportunities', 'calendar', 
                'documents', 'reports', 'settings', 'admin', 'user', 'home',
                'dashboard', 'main', 'menu'
            ]):
                navigation_elements += 1
        
        # If page has significant navigation elements, consider it a navigation page
                return navigation_elements >= 3  # At least 3 navigation elements

    def _enhance_element_type_mapping(self, raw_element_type: str, element_data: Dict[str, Any]) -> str:
        """Enhanced element type mapping for better Graph Builder alignment"""
        # Enhanced type mapping based on JavaScript identification
        type_mapping = {
            # Business data types
            'business_data': 'business_data',
            'search_box': 'search_box',
            'form_control': 'form_control',
            
            # Button types
            'button': 'button',
            'submit': 'submit',
            'search_button': 'search_button',
            'filter_button': 'filter_button',
            
            # Input types
            'input': 'input',
            'email_input': 'email_input',
            'password_input': 'password_input',
            
            # Link types
            'link': 'link',
            'navigation': 'navigation',
            'menu_link': 'menu_link',
            'detail_link': 'detail_link',
            
            # Container types
            'card': 'card',
            'list': 'list',
            'detail': 'detail',
            'dashboard': 'dashboard',
            
            # Form types
            'form': 'form',
            'select': 'select',
            'textarea': 'textarea',
            
            # UI component types
            'menu': 'menu',
            'tab': 'tab',
            'filter': 'filter',
            'modal': 'modal',
            'dropdown': 'dropdown',
            'breadcrumb': 'breadcrumb',
            'paginator': 'paginator',
            'toast': 'toast',
            
            # Content types
            'table': 'table',
            'image': 'image',
            'content': 'content',
            
            # Legacy types (fallback)
            'clickable': 'button',
            'unknown': 'content'
        }
        
        # Get enhanced type
        enhanced_type = type_mapping.get(raw_element_type, raw_element_type)
        
        # Additional context-based enhancement
        if enhanced_type == 'content':
            # Try to determine more specific type from element data
            css_classes = element_data.get('css_classes', [])
            css_classes_str = ' '.join(css_classes).lower()
            
            if any(keyword in css_classes_str for keyword in ['card', 'panel', 'widget']):
                enhanced_type = 'card'
            elif any(keyword in css_classes_str for keyword in ['list', 'item']):
                enhanced_type = 'list'
            elif any(keyword in css_classes_str for keyword in ['detail', 'info']):
                enhanced_type = 'detail'
            elif any(keyword in css_classes_str for keyword in ['dashboard', 'overview']):
                enhanced_type = 'dashboard'
        
        return enhanced_type

    def _enhance_clickable_detection(self, element_type: str, element_data: Dict[str, Any]) -> bool:
        """Enhanced clickable detection based on element type and properties"""
        # Direct clickable types
        clickable_types = {
            'button', 'submit', 'search_button', 'filter_button',
            'link', 'navigation', 'menu_link', 'detail_link',
            'menu', 'tab', 'dropdown', 'breadcrumb'
        }
        
        if element_type in clickable_types:
            return True
        
        # Check for clickable attributes
        attributes = element_data.get('attributes', {})
        if any(attr in attributes for attr in ['onclick', 'onmousedown', 'onmouseup', 'onmouseover']):
            return True
        
        # Check for role attribute
        if attributes.get('role') in ['button', 'link', 'menuitem', 'tab']:
            return True
        
        # Check for tabindex (indicates keyboard navigation)
        if 'tabindex' in attributes:
            return True
        
        # Special handling for business data elements
        if element_type == 'business_data':
            # Check if business data element is actually clickable
            tag_name = element_data.get('tag_name', '').lower()
            href = element_data.get('href', '')
            
            # If it's a link tag or has href, it's clickable
            if tag_name == 'a' or href:
                return True
            
            # Check if it has clickable styling or cursor
            css_classes = element_data.get('css_classes', [])
            css_classes_str = ' '.join(css_classes).lower()
            
            # Check for clickable indicators in CSS classes
            clickable_indicators = ['clickable', 'link', 'button', 'cursor-pointer', 'hover']
            if any(indicator in css_classes_str for indicator in clickable_indicators):
                return True
            
            # Check for interactive attributes specific to business data
            if any(attr in attributes for attr in ['data-href', 'data-link', 'data-url', 'data-action']):
                return True
        
        return False

    def _enhance_input_detection(self, element_type: str, element_data: Dict[str, Any]) -> bool:
        """Enhanced input detection based on element type and properties"""
        # Direct input types
        input_types = {
            'input', 'email_input', 'password_input', 'search_box',
            'select', 'textarea', 'form_control'
        }
        
        if element_type in input_types:
            return True
        
        # Check for input attributes
        attributes = element_data.get('attributes', {})
        if 'type' in attributes or 'value' in attributes or 'placeholder' in attributes:
            return True
        
        # Check for form-related attributes
        if any(attr in attributes for attr in ['name', 'required', 'disabled', 'readonly']):
            return True
        
        return False