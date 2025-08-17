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
from urllib.parse import urljoin, urlparse
import re

from loguru import logger

# LLM executor will be imported when needed
LLM_AVAILABLE = None  # Will be checked when actually needed

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
            "attributes": self.attributes
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
            "exploration_depth": self.exploration_depth,
            "llm_assessment": self.llm_assessment
        }


class WebCollector:
    """Web page collector for crawling and extracting web data"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Collection settings
        self.max_pages = self.config.get('max_pages', 10)
        self.timeout = self.config.get('timeout', 30)
        self.delay = self.config.get('delay', 1.0)
        
        # Output settings
        self.base_output_dir = Path(self.config.get('output_dir', 'data/web_screenshots'))
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use base output directory directly (no timestamp subdirectory)
        self.output_dir = self.base_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Counter for unique page identification
        self.page_counter = 0
        
        # Browser automation settings
        self.use_playwright = self.config.get('use_playwright', True)  # Prefer Playwright over Selenium
        
        # LLM-based intelligent crawling settings
        self.use_llm_crawling = self.config.get('use_llm_crawling', False)
        self.llm_model_name = self.config.get('llm_model_name', 'gpt-4o-mini')
        self.llm_temperature = self.config.get('llm_temperature', 0.3)
        self.llm_max_tokens = self.config.get('llm_max_tokens', 1000)
        self.llm_quality_threshold = self.config.get('llm_quality_threshold', 0.6)
        
        # Link filtering configuration
        exploration_config = self.config.get('exploration', {})
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
        
        # Initialize LLM executor if available
        self.llm_executor = None
        if self.use_llm_crawling:
            try:
                # Lazy import to avoid circular dependencies
                from agent_framework.executors import LLMExecutor
                self.llm_executor = LLMExecutor.get_instance()
                logger.info(f"✅ LLM-based crawling enabled with model: {self.llm_model_name}")
            except ImportError:
                logger.warning("⚠️ LLM executor not available. Install with: pip install openai anthropic")
                self.use_llm_crawling = False
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize LLM executor: {e}")
                self.use_llm_crawling = False
        
        # Browser automation setup
        self.browser = None
        self.page = None
        self.driver = None
        
        # Check available automation tools
        if self.use_playwright and not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available, falling back to Selenium")
            self.use_playwright = False
        
        if not self.use_playwright and not SELENIUM_AVAILABLE:
            logger.warning("No browser automation available")
            raise RuntimeError("No browser automation tools available (Playwright or Selenium)")
        
        logger.info(f"WebCollector initialized with max_pages={self.max_pages}, timeout={self.timeout}s")
        if self.use_llm_crawling:
            logger.info(f"🤖 LLM-based intelligent crawling: Enabled")
        else:
            logger.info(f"📋 Rule-based crawling: Enabled")
    
    async def collect_web_data(self, urls: List[str]) -> List[WebPageData]:
        """Collect web data from multiple URLs"""
        
        logger.info(f"Starting web data collection for {len(urls)} URLs")
        
        # Use real browser automation
        if self.use_playwright and PLAYWRIGHT_AVAILABLE:
            return await self._collect_with_playwright(urls)
        elif SELENIUM_AVAILABLE:
            return await self._collect_with_selenium(urls)
        else:
            raise RuntimeError("No browser automation tools available (Playwright or Selenium)")
    

    
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
                        # Navigate to page
                        await self.page.goto(url, wait_until='networkidle', timeout=self.timeout * 1000)
                        
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
        """Explore website using Selenium with multi-step navigation"""
        
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
            
            # Start exploration from each URL
            for start_url in start_urls:
                await self._explore_from_url_selenium(driver, start_url, max_depth, max_pages_per_depth, 
                                                    all_pages, visited_urls, current_depth=0)
                
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
            
            # Use LLM-based intelligent selection if enabled
            if self.use_llm_crawling and self.llm_executor:
                try:
                    # Get current page information for context
                    title = driver.title
                    content = driver.page_source
                    
                    # Use LLM to evaluate and select links
                    selected_links = await self._evaluate_links_with_llm(
                        links, current_url, title, content, max_links
                    )
                    
                    if selected_links:
                        logger.info(f"🤖 LLM selected {len(selected_links)} links for exploration")
                        return selected_links
                    else:
                        logger.warning("LLM link evaluation returned no links, falling back to rule-based selection")
                        
                except Exception as e:
                    logger.warning(f"LLM link evaluation failed: {e}, falling back to rule-based selection")
            
            # Fallback to rule-based selection
            explorable_links = []
            for link in links:
                href = link['href']
                if self._is_explorable_link(href, current_url):
                    explorable_links.append(href)
                    if len(explorable_links) >= max_links:
                        break
            
            return explorable_links[:max_links]
            
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
    
    async def _evaluate_links_with_llm(self, links: List[Dict[str, Any]], current_url: str, 
                                     current_title: str, current_content: str, max_links: int) -> List[str]:
        """Use LLM to intelligently evaluate and rank links for web task generation"""
        
        if not self.llm_executor or not links:
            return []
        
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
            return [link['href'] for link in links[:max_links] if self._is_explorable_link(link['href'], current_url)]
    
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
            exploration_depth=depth
        )
    

    
    async def _extract_dom_elements_playwright(self, page=None) -> List[WebElement]:
        """Extract DOM elements using Playwright"""
        
        elements = []
        
        # Use provided page or fall back to self.page
        current_page = page or self.page
        if not current_page:
            logger.error("No page available for DOM extraction")
            return elements
        
        try:
            # Get all interactive elements
            element_data = await current_page.evaluate("""
                () => {
                    const elements = [];
                    const selectors = [
                        'input', 'button', 'a', 'select', 'textarea',
                        '[onclick]', '[role="button"]', '[tabindex]',
                        'form', 'table', 'img', 'video', 'audio'
                    ];
                    
                    selectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach((el, index) => {
                            const rect = el.getBoundingClientRect();
                            const computedStyle = window.getComputedStyle(el);
                            
                            elements.push({
                                element_id: el.id || `${selector}_${index}`,
                                element_type: el.tagName.toLowerCase(),
                                tag_name: el.tagName.toLowerCase(),
                                text_content: el.textContent?.trim() || '',
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
                                is_clickable: el.onclick !== null || el.tagName === 'A' || el.tagName === 'BUTTON',
                                is_input: el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT',
                                is_visible: computedStyle.display !== 'none' && computedStyle.visibility !== 'hidden',
                                is_enabled: !el.disabled,
                                input_type: el.type || '',
                                required: el.required || false,
                                options: el.tagName === 'SELECT' ? Array.from(el.options).map(opt => opt.text) : [],
                                attributes: Object.fromEntries(Array.from(el.attributes).map(attr => [attr.name, attr.value]))
                            });
                        });
                    });
                    
                    return elements;
                }
            """)
            
            # Convert to WebElement objects
            for data in element_data:
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
                    attributes=data['attributes']
                )
                elements.append(element)
                
        except Exception as e:
            logger.error(f"Error extracting DOM elements with Playwright: {e}")
        
        return elements
    
    def _extract_dom_elements_selenium(self, driver=None) -> List[WebElement]:
        """Extract DOM elements using Selenium"""
        
        elements = []
        
        try:
            # Get all interactive elements
            selectors = [
                'input', 'button', 'a', 'select', 'textarea',
                '[onclick]', '[role="button"]', '[tabindex]',
                'form', 'table', 'img', 'video', 'audio'
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
                                is_clickable=el.tag_name in ['a', 'button'] or el.get_attribute('onclick') is not None,
                                is_input=el.tag_name in ['input', 'textarea', 'select'],
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
    
    def _classify_page_type(self, url: str) -> str:
        """Classify page type based on URL"""
        
        url_lower = url.lower()
        
        if any(keyword in url_lower for keyword in ['login', 'signup', 'register', 'form']):
            return "form"
        elif any(keyword in url_lower for keyword in ['search', 'query', 'find']):
            return "search"
        elif any(keyword in url_lower for keyword in ['product', 'item', 'buy', 'shop']):
            return "product"
        elif any(keyword in url_lower for keyword in ['news', 'article', 'blog', 'content']):
            return "content"
        else:
            return "content"
    

            # Content elements
            elements.extend([
                WebElement(
                    element_id="article_title",
                    element_type="heading",
                    tag_name="h1",
                    text_content="Sample Article",
                    x=100, y=100, width=500, height=60
                ),
                WebElement(
                    element_id="article_content",
                    element_type="text",
                    tag_name="p",
                    text_content="This is sample article content...",
                    x=100, y=200, width=500, height=200
                ),
                WebElement(
                    element_id="read_more",
                    element_type="link",
                    tag_name="a",
                    text_content="Read More",
                    is_clickable=True,
                    x=100, y=450, width=100, height=30
                )
            ])
        
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
        """Explore website using Playwright with multi-step navigation"""
        
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
                
                # Start exploration from each URL
                for start_url in start_urls:
                    await self._explore_from_url_playwright(page, start_url, max_depth, max_pages_per_depth, 
                                                          all_pages, visited_urls, current_depth=0)
                
                # Close browser
                await browser.close()
                
        except Exception as e:
            logger.error(f"Error with Playwright exploration: {e}")
    
    async def _explore_from_url_playwright(self, page, url: str, max_depth: int, max_pages_per_depth: int,
                                         all_pages: List[WebPageData], visited_urls: Set[str], current_depth: int):
        """Recursively explore from a URL using Playwright"""
        
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
            
            # If we haven't reached max depth, explore links
            if current_depth < max_depth - 1 and len(all_pages) < self.max_pages:
                links_to_explore = await self._find_explorable_links_playwright(page, url, self.max_links_per_page)
                
                for link_url in links_to_explore:
                    if len(all_pages) >= self.max_pages:
                        break
                    await self._explore_from_url_playwright(page, link_url, max_depth, max_pages_per_depth,
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
            # Get page information
            title = await page.title()
            page_content = await page.content()
            
            # Take screenshot
            self.page_counter += 1
            screenshot_path = self.output_dir / f"page_{depth}_{self.page_counter}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            
            # Extract DOM elements
            elements = await self._extract_dom_elements_playwright(page)
            
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
            
            # Determine page type
            page_type = self._determine_page_type(title, page_content, elements)
            
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
                exploration_depth=depth
            )
            
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
            return self._create_empty_page_data(url, depth)
    
    async def _find_explorable_links_playwright(self, page, current_url: str, max_links: int) -> List[str]:
        """Find links that can be explored from current page using Playwright"""
        
        try:
            # Get all links from the page
            links = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => ({
                        href: link.href,
                        text: link.textContent.trim(),
                        visible: link.offsetParent !== null
                    })).filter(link => link.visible && link.href);
                }
            """)
            
            # Use LLM-based intelligent selection if enabled
            if self.use_llm_crawling and self.llm_executor:
                try:
                    # Get current page information for context
                    title = await page.title()
                    content = await page.content()
                    
                    # Use LLM to evaluate and select links
                    selected_links = await self._evaluate_links_with_llm(
                        links, current_url, title, content, max_links
                    )
                    
                    if selected_links:
                        logger.info(f"🤖 LLM selected {len(selected_links)} links for exploration")
                        return selected_links
                    else:
                        logger.warning("LLM link evaluation returned no links, falling back to rule-based selection")
                        
                except Exception as e:
                    logger.warning(f"LLM link evaluation failed: {e}, falling back to rule-based selection")
            
            # Fallback to rule-based selection
            explorable_links = []
            for link in links:
                href = link['href']
                if self._is_explorable_link(href, current_url):
                    explorable_links.append(href)
                    if len(explorable_links) >= max_links:
                        break
            
            return explorable_links[:max_links]
            
        except Exception as e:
            logger.error(f"Error finding explorable links: {e}")
            return []
    
    async def _extract_links_from_page_playwright(self, page, current_url: str) -> List[str]:
        """Extract all links from the current page using Playwright"""
        
        try:
            links = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => link.href).filter(href => href && href.trim());
                }
            """)
            
            extracted_links = []
            for href in links:
                if self._is_explorable_link(href, current_url):
                    extracted_links.append(href)
            
            return extracted_links
            
        except Exception as e:
            logger.error(f"Error extracting links: {e}")
            return []
        