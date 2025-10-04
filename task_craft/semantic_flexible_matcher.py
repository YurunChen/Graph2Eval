"""
Semantic Flexible Matcher - Semantic Flexible Matching
Let LLM participate in semantic completion and fill optional slots, implementing "pattern softening + dynamic combination + LLM completion"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import re
import json
from graph_rag.graph_builder import TaskGraph
from graph_rag.edge_types import GraphEdge, EdgeType
from graph_rag.node_types import GraphNode, NodeType
from .task_seeds import TaskSeedPattern, TaskSeedType
from .web_subgraph_sampler import SubgraphSample
from .graph_regex_engine import GraphRegexMatch
from loguru import logger

@dataclass
class SemanticCompletion:
    """Semantic completion result"""
    completion_type: str  # "node_inference", "edge_inference", "slot_filling"
    inferred_element: str
    confidence_score: float
    reasoning: str
    fallback_options: List[str] = field(default_factory=list)

@dataclass
class FlexibleTaskPattern:
    """Flexible task pattern - pattern that supports semantic completion"""
    pattern_id: str
    name: str
    description: str
    
    # Core components (required)
    core_components: Dict[str, str] = field(default_factory=dict)
    
    # Optional components (can be completed)
    optional_components: Dict[str, str] = field(default_factory=dict)
    
    # Semantic completion hints
    completion_hints: List[str] = field(default_factory=list)
    
    # Fallback versions
    fallback_versions: List[str] = field(default_factory=list)
    
    # Corresponding task seed type
    seed_type: TaskSeedType = TaskSeedType.BASIC_NAVIGATION

class SemanticFlexibleMatcher:
    """Semantic flexible matcher"""
    
    def __init__(self, llm_executor=None):
        self.llm_executor = llm_executor
        self.flexible_patterns = self._initialize_flexible_patterns()
    
    def _initialize_flexible_patterns(self) -> Dict[str, FlexibleTaskPattern]:
        """Initialize flexible task patterns"""
        patterns = {}
        
        # 1. Basic navigation pattern - supports semantic completion
        patterns["flexible_navigation"] = FlexibleTaskPattern(
            pattern_id="flexible_navigation",
            name="Flexible Navigation Pattern",
            description="Flexible navigation pattern that supports semantic completion",
            core_components={
                "navigation_element": "Navigation|Button",
                "target_page": "Page"
            },
            optional_components={
                "intermediate_nav": "Navigation",
                "breadcrumb": "Breadcrumb",
                "menu_item": "Button"
            },
            completion_hints=[
                "If page title contains 'blog', 'article', 'content', can infer content page",
                "If page has navigation menu, can infer sub-pages",
                "If page has pagination, can infer list page"
            ],
            fallback_versions=[
                "simple_navigation",  # Only navigation elements
                "breadcrumb_navigation",  # Breadcrumb navigation
                "menu_navigation"  # Menu navigation
            ],
            seed_type=TaskSeedType.BASIC_NAVIGATION
        )
        
        # 2. Content browsing pattern - supports semantic completion
        patterns["flexible_browsing"] = FlexibleTaskPattern(
            pattern_id="flexible_browsing",
            name="Flexible Content Browsing Pattern",
            description="Flexible content browsing that supports semantic completion",
            core_components={
                "interactive_element": "Button|Navigation",
                "content_area": "Page"
            },
            optional_components={
                "content_filter": "Filter",
                "content_sort": "Button",
                "content_pagination": "Paginator",
                "content_search": "SearchBox"
            },
            completion_hints=[
                "If page has article list, can infer content browsing functionality",
                "If page has category tags, can infer filtering functionality",
                "If page has search box, can infer search functionality"
            ],
            fallback_versions=[
                "simple_browsing",  # Simple browsing
                "filtered_browsing",  # Browsing with filters
                "search_browsing"  # Browsing with search
            ],
            seed_type=TaskSeedType.CONTENT_BROWSING
        )
        
        # 3. Button interaction pattern - supports semantic completion
        patterns["flexible_button"] = FlexibleTaskPattern(
            pattern_id="flexible_button",
            name="Flexible Button Interaction Pattern",
            description="Flexible button interaction that supports semantic completion",
            core_components={
                "button_element": "Button",
                "action_response": "Content"
            },
            optional_components={
                "modal_dialog": "Modal",
                "page_change": "Page",
                "notification": "Toast",
                "form_trigger": "Form"
            },
            completion_hints=[
                "If page has buttons, can infer button functionality testing",
                "If page has modals, can infer modal interaction",
                "If page has navigation, can infer page changes"
            ],
            fallback_versions=[
                "simple_button",  # Simple button click
                "modal_button",  # Button with modal
                "navigation_button"  # Button with navigation
            ],
            seed_type=TaskSeedType.BUTTON_INTERACTION
        )
        
        # 4. Search pattern - supports semantic completion
        patterns["flexible_search"] = FlexibleTaskPattern(
            pattern_id="flexible_search",
            name="Flexible Search Pattern",
            description="Flexible search pattern that supports semantic completion",
            core_components={
                "search_box": "SearchBox|Input",
                "search_button": "Button"
            },
            optional_components={
                "search_filters": "Filter",
                "search_results": "List",
                "search_suggestions": "List",
                "search_history": "List"
            },
            completion_hints=[
                "If page has search box, can infer search results page",
                "If page has filter options, can infer advanced search",
                "If page has suggestion list, can infer search suggestions"
            ],
            fallback_versions=[
                "simple_search",  # Simple search
                "filtered_search",  # Search with filters
                "advanced_search"  # Advanced search
            ],
            seed_type=TaskSeedType.BUSINESS_SEARCH_FILTER
        )
        
        return patterns
    
    def match_with_semantic_completion(self, subgraph: SubgraphSample, seed_type: TaskSeedType) -> List[Dict[str, Any]]:
        """Use semantic completion for flexible matching"""
        matches = []
        
        # Find corresponding flexible pattern
        pattern = self._get_pattern_for_seed_type(seed_type)
        if not pattern:
            return matches
        
        # 1. Analyze subgraph structure
        subgraph_analysis = self._analyze_subgraph_for_completion(subgraph)
        
        # 2. Check core component matching
        core_matches = self._match_core_components(pattern, subgraph_analysis)
        
        # 3. Perform semantic completion for each core match
        for core_match in core_matches:
            completions = self._perform_semantic_completion(pattern, core_match, subgraph_analysis)
            
            if completions:
                # 4. Generate complete task pattern
                complete_pattern = self._build_complete_pattern(pattern, core_match, completions)
                matches.append(complete_pattern)
        
        # 5. If no complete match found, try fallback versions
        if not matches:
            fallback_matches = self._try_fallback_versions(pattern, subgraph_analysis)
            matches.extend(fallback_matches)
        
        return matches
    
    def _get_pattern_for_seed_type(self, seed_type: TaskSeedType) -> Optional[FlexibleTaskPattern]:
        """Get corresponding flexible pattern based on seed type"""
        for pattern in self.flexible_patterns.values():
            if pattern.seed_type == seed_type:
                return pattern
        return None
    
    def _analyze_subgraph_for_completion(self, subgraph: SubgraphSample) -> Dict[str, Any]:
        """Analyze subgraph to support semantic completion"""
        analysis = {
            "node_types": defaultdict(list),
            "edge_types": defaultdict(list),
            "page_content": {},
            "semantic_clues": [],
            "interactive_elements": [],
            "content_elements": [],
            "navigation_elements": []
        }
        
        # Analyze node types and content
        for node_id, node in subgraph.nodes.items():
            analysis["node_types"][node.node_type.value].append(node_id)
            
            # Collect semantic clues
            if hasattr(node, 'metadata') and node.metadata:
                if node.metadata.text_content:
                    analysis["semantic_clues"].append({
                        "node_id": node_id,
                        "text": node.metadata.text_content,
                        "type": node.node_type.value
                    })
                
                # Categorize elements
                if node.node_type in [NodeType.BUTTON, NodeType.NAVIGATION]:
                    analysis["interactive_elements"].append(node_id)
                elif node.node_type in [NodeType.INPUT, NodeType.SEARCH_BOX]:
                    analysis["content_elements"].append(node_id)
                elif node.node_type == NodeType.NAVIGATION:
                    analysis["navigation_elements"].append(node_id)
        
        # Analyze edge types
        for edge_id, edge in subgraph.edges.items():
            analysis["edge_types"][edge.edge_type.value].append(edge_id)
        
        # Analyze page content (if available)
        if hasattr(subgraph, 'page_info') and subgraph.page_info:
            analysis["page_content"] = {
                "title": getattr(subgraph.page_info, 'title', ''),
                "url": getattr(subgraph.page_info, 'url', ''),
                "description": getattr(subgraph.page_info, 'description', '')
            }
        
        return analysis
    
    def _match_core_components(self, pattern: FlexibleTaskPattern, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match core components"""
        core_matches = []
        
        for component_name, component_type in pattern.core_components.items():
            # Support type aliases (separated by |)
            allowed_types = component_type.split('|')
            
            # Find matching nodes
            matching_nodes = []
            for node_type in allowed_types:
                if node_type in analysis["node_types"]:
                    matching_nodes.extend(analysis["node_types"][node_type])
            
            if matching_nodes:
                core_matches.append({
                    "component_name": component_name,
                    "component_type": component_type,
                    "matching_nodes": matching_nodes,
                    "confidence": 1.0
                })
        
        return core_matches
    
    def _perform_semantic_completion(self, pattern: FlexibleTaskPattern, core_match: Dict[str, Any], analysis: Dict[str, Any]) -> List[SemanticCompletion]:
        """Perform semantic completion"""
        completions = []
        
        # If no LLM executor, return empty list
        if not self.llm_executor:
            logger.warning("No LLM executor available for semantic completion")
            return completions
        
        try:
            # Build completion prompt
            prompt = self._create_completion_prompt(pattern, core_match, analysis)
            
            # Call LLM for completion
            response, tokens_used = self.llm_executor._execute_with_retries(prompt)
            
            # Parse completion results
            completions = self._parse_llm_completion_response(response)
            
            logger.info(f"Generated {len(completions)} semantic completions for pattern {pattern.pattern_id}")
            
        except Exception as e:
            logger.error(f"Error in semantic completion: {e}")
        
        return completions
    
    def _create_completion_prompt(self, pattern: FlexibleTaskPattern, core_match: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Create semantic completion prompt"""
        prompt = f"""You are a web task pattern completion expert. Based on the given pattern and available elements, suggest semantic completions.

Pattern: {pattern.name}
Description: {pattern.description}

Core Components (already matched):
{chr(10).join([f"- {name}: {comp_type}" for name, comp_type in pattern.core_components.items()])}

Optional Components (need completion):
{chr(10).join([f"- {name}: {comp_type}" for name, comp_type in pattern.optional_components.items()])}

Available Elements:
- Node Types: {list(analysis['node_types'].keys())}
- Interactive Elements: {analysis['interactive_elements']}
- Content Elements: {analysis['content_elements']}
- Navigation Elements: {analysis['navigation_elements']}

Semantic Clues:
{chr(10).join([f"- {clue['text']} ({clue['type']})" for clue in analysis['semantic_clues'][:5]])}

Please suggest completions for the optional components. Return your response in JSON format:

```json
{{
    "completions": [
        {{
            "completion_type": "node_inference",
            "inferred_element": "element_id",
            "confidence_score": 0.8,
            "reasoning": "Explanation for this completion",
            "fallback_options": ["alternative1", "alternative2"]
        }}
    ]
}}
```"""
        return prompt
    
    def _build_complete_pattern(self, pattern: FlexibleTaskPattern, core_match: Dict[str, Any], completions: List[SemanticCompletion]) -> Dict[str, Any]:
        """Build complete task pattern"""
        complete_pattern = {
            "pattern_id": pattern.pattern_id,
            "name": pattern.name,
            "description": pattern.description,
            "core_components": core_match,
            "completions": completions,
            "confidence": self._calculate_pattern_confidence(core_match, completions),
            "seed_type": pattern.seed_type
        }
        
        return complete_pattern
    
    def _calculate_pattern_confidence(self, core_match: Dict[str, Any], completions: List[SemanticCompletion]) -> float:
        """Calculate pattern confidence"""
        # Base confidence from core matching
        base_confidence = core_match.get("confidence", 0.5)
        
        # Adjust confidence based on completion quality
        if completions:
            avg_completion_confidence = sum(c.confidence_score for c in completions) / len(completions)
            return (base_confidence + avg_completion_confidence) / 2
        else:
            return base_confidence
        
        # 1. Semantic inference based on page content
        page_completions = self._infer_from_page_content(pattern, analysis)
        completions.extend(page_completions)
        
        # 2. Semantic inference based on element text
        text_completions = self._infer_from_element_text(pattern, analysis)
        completions.extend(text_completions)
        
        # 3. Semantic inference based on graph structure
        structure_completions = self._infer_from_graph_structure(pattern, analysis)
        completions.extend(structure_completions)
        
        # 4. Use LLM for advanced semantic inference
        if self.llm_executor:
            llm_completions = self._infer_with_llm(pattern, core_match, analysis)
            completions.extend(llm_completions)
        
        return completions
    
    def _infer_from_page_content(self, pattern: FlexibleTaskPattern, analysis: Dict[str, Any]) -> List[SemanticCompletion]:
        """Infer from page content"""
        completions = []
        
        page_content = analysis.get("page_content", {})
        title = page_content.get("title", "").lower()
        url = page_content.get("url", "").lower()
        
        # Infer based on title
        if "blog" in title or "article" in title or "post" in title:
            completions.append(SemanticCompletion(
                completion_type="node_inference",
                inferred_element="content_page",
                confidence_score=0.8,
                reasoning="Page title suggests content publishing functionality",
                fallback_options=["list_page", "detail_page"]
            ))
        
        if "search" in title or "find" in title:
            completions.append(SemanticCompletion(
                completion_type="node_inference",
                inferred_element="search_page",
                confidence_score=0.7,
                reasoning="Page title suggests search functionality",
                fallback_options=["browse_page", "filter_page"]
            ))
        
        # Infer based on URL
        if "/search" in url or "?s=" in url:
            completions.append(SemanticCompletion(
                completion_type="node_inference",
                inferred_element="search_results",
                confidence_score=0.9,
                reasoning="URL pattern indicates search results page",
                fallback_options=["filtered_results", "browse_results"]
            ))
        
        return completions
    
    def _infer_from_element_text(self, pattern: FlexibleTaskPattern, analysis: Dict[str, Any]) -> List[SemanticCompletion]:
        """Infer from element text"""
        completions = []
        
        for clue in analysis.get("semantic_clues", []):
            text = clue["text"].lower()
            node_type = clue["type"]
            
            # Infer based on button text
            if node_type == "Button":
                if "search" in text or "find" in text:
                    completions.append(SemanticCompletion(
                        completion_type="node_inference",
                        inferred_element="search_button",
                        confidence_score=0.8,
                        reasoning=f"Button text '{text}' suggests search functionality",
                        fallback_options=["filter_button", "browse_button"]
                    ))
                
                if "submit" in text or "send" in text or "post" in text:
                    completions.append(SemanticCompletion(
                        completion_type="node_inference",
                        inferred_element="submit_button",
                        confidence_score=0.9,
                        reasoning=f"Button text '{text}' suggests form submission",
                        fallback_options=["action_button", "confirm_button"]
                    ))
            
            # Infer based on input placeholder
            elif node_type == "Input":
                if "search" in text or "find" in text:
                    completions.append(SemanticCompletion(
                        completion_type="node_inference",
                        inferred_element="search_input",
                        confidence_score=0.8,
                        reasoning=f"Input placeholder '{text}' suggests search functionality",
                        fallback_options=["filter_input", "browse_input"]
                    ))
        
        return completions
    
    def _infer_from_graph_structure(self, pattern: FlexibleTaskPattern, analysis: Dict[str, Any]) -> List[SemanticCompletion]:
        """Infer from graph structure"""
        completions = []
        
        # Check if there are navigation edges
        if "NavTo" in analysis["edge_types"]:
            completions.append(SemanticCompletion(
                completion_type="edge_inference",
                inferred_element="navigation_flow",
                confidence_score=0.7,
                reasoning="Navigation edges suggest multi-page workflow",
                fallback_options=["single_page_flow", "modal_flow"]
            ))
        
        # Check if there are form element combinations
        if len(analysis["content_elements"]) >= 2:
            completions.append(SemanticCompletion(
                completion_type="node_inference",
                inferred_element="form_section",
                confidence_score=0.6,
                reasoning="Multiple input elements suggest form functionality",
                fallback_options=["search_section", "filter_section"]
            ))
        
        # Check if there are interactive elements
        if len(analysis["interactive_elements"]) >= 3:
            completions.append(SemanticCompletion(
                completion_type="node_inference",
                inferred_element="interactive_interface",
                confidence_score=0.7,
                reasoning="Multiple interactive elements suggest rich interface",
                fallback_options=["simple_interface", "basic_interface"]
            ))
        
        return completions
    
    def _infer_with_llm(self, pattern: FlexibleTaskPattern, core_match: Dict[str, Any], analysis: Dict[str, Any]) -> List[SemanticCompletion]:
        """Use LLM for advanced semantic inference"""
        if not self.llm_executor:
            return []
        
        try:
            # Build LLM prompt
            prompt = self._build_llm_completion_prompt(pattern, core_match, analysis)
            
            # Call LLM
            response = self.llm_executor.execute_simple(prompt)
            
            # Parse LLM response
            completions = self._parse_llm_completion_response(response)
            
            return completions
            
        except Exception as e:
            # LLM call failed, return empty list
            return []
    
    def _build_llm_completion_prompt(self, pattern: FlexibleTaskPattern, core_match: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Build LLM completion prompt"""
        prompt = f"""
You are a professional web interface analysis expert. Please perform semantic completion based on the following information to infer missing UI components and interaction patterns.

## Task Pattern Information
- Pattern Name: {pattern.name}
- Pattern Description: {pattern.description}
- Core Components: {pattern.core_components}
- Optional Components: {pattern.optional_components}

## Already Matched Core Components
{core_match}

## Subgraph Analysis Information
- Node Types: {dict(analysis['node_types'])}
- Edge Types: {dict(analysis['edge_types'])}
- Semantic Clues: {analysis['semantic_clues'][:5]}  # Limited quantity
- Page Content: {analysis.get('page_content', {})}

## Semantic Completion Hints
{pattern.completion_hints}

## Task
Please analyze the above information to infer potentially missing UI components and interaction patterns. Consider the following aspects:
1. Infer functionality based on page content and element text
2. Infer interaction flow based on graph structure
3. Infer optional components based on pattern hints

## Output Format
Please output inference results in the following JSON format strictly, without adding any additional text or explanations:

```json
{
    "completions": [
        {
            "completion_type": "node_inference",
            "inferred_element": "element_name",
            "confidence_score": 0.8,
            "reasoning": "Inference reasoning",
            "fallback_options": ["option1", "option2"]
        }
    ]
}
```

**Important Requirements:**
1. Must output strictly according to the above JSON format
2. Do not add any text before or after the JSON
3. Ensure all fields have correct comma separators
4. String values must be enclosed in double quotes
5. Arrays and objects must be properly closed
"""
        
        return prompt
    
    def _parse_llm_completion_response(self, response: Any) -> List[SemanticCompletion]:
        """Parse LLM completion response"""
        completions = []
        
        try:
            # Handle ExecutionResult object
            if hasattr(response, 'result'):
                response = response.result
            elif hasattr(response, 'content'):
                response = response.content
            elif not isinstance(response, str):
                response = str(response)
                logger.warning(f"Response converted to string: {type(response)}")
            
            # Try multiple JSON extraction methods
            json_patterns = [
                r'```json\s*(.*?)\s*```',  # JSON in code blocks
                r'\{[^{}]*"completions"[^{}]*\}',  # JSON containing completions
                r'\{.*?\}',  # Any JSON object
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, response, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group()
                        # Clean JSON string
                        json_str = re.sub(r'[^\x20-\x7E]', '', json_str)  # Remove non-ASCII characters
                        data = json.loads(json_str)
                        
                        # Parse completion results
                        for completion_data in data.get("completions", []):
                            completion = SemanticCompletion(
                                completion_type=completion_data.get("completion_type", "node_inference"),
                                inferred_element=completion_data.get("inferred_element", ""),
                                confidence_score=completion_data.get("confidence_score", 0.5),
                                reasoning=completion_data.get("reasoning", ""),
                                fallback_options=completion_data.get("fallback_options", [])
                            )
                            completions.append(completion)
                        
                        if completions:
                            return completions
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.debug(f"Failed to parse JSON with pattern {pattern}: {e}")
                        # 打印具体的错误位置信息
                        if isinstance(e, json.JSONDecodeError):
                            logger.debug(f"JSON decode error at line {e.lineno}, column {e.colno}: {e.msg}")
                            # 尝试显示错误位置附近的字符
                            try:
                                lines = json_str.split('\n')
                                if e.lineno <= len(lines):
                                    error_line = lines[e.lineno - 1]
                                    logger.debug(f"Error line: {error_line}")
                                    if e.colno <= len(error_line):
                                        logger.debug(f"Error position: {error_line[:e.colno]}^")
                            except:
                                pass
                        continue
            
            # If no valid JSON found, try to extract from ExecutionResult
            if hasattr(response, 'result') and response.result:
                try:
                    # Try to extract JSON from result
                    result_str = str(response.result)
                    for pattern in json_patterns:
                        json_match = re.search(pattern, result_str, re.DOTALL)
                        if json_match:
                            try:
                                json_str = json_match.group()
                                json_str = re.sub(r'[^\x20-\x7E]', '', json_str)
                                data = json.loads(json_str)
                                
                                for completion_data in data.get("completions", []):
                                    completion = SemanticCompletion(
                                        completion_type=completion_data.get("completion_type", "node_inference"),
                                        inferred_element=completion_data.get("inferred_element", ""),
                                        confidence_score=completion_data.get("confidence_score", 0.5),
                                        reasoning=completion_data.get("reasoning", ""),
                                        fallback_options=completion_data.get("fallback_options", [])
                                    )
                                    completions.append(completion)
                                
                                if completions:
                                    return completions
                            except (json.JSONDecodeError, ValueError, KeyError) as e:
                                logger.debug(f"Failed to parse JSON from result with pattern {pattern}: {e}")
                                continue
                except Exception as e:
                    logger.debug(f"Failed to extract from ExecutionResult: {e}")
            
            # If no valid JSON found, try to fix common JSON format issues
            try:
                response_str = str(response)
                
                # Method 1: Fix truncated JSON
                if '"reasoning": "' in response_str and not response_str.strip().endswith('}'):
                    # Try to find the last complete JSON object
                    brace_count = 0
                    last_complete_pos = -1
                    for i, char in enumerate(response_str):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                last_complete_pos = i
                    
                    if last_complete_pos > 0:
                        fixed_json = response_str[:last_complete_pos + 1]
                        try:
                            data = json.loads(fixed_json)
                            for completion_data in data.get("completions", []):
                                completion = SemanticCompletion(
                                    completion_type=completion_data.get("completion_type", "node_inference"),
                                    inferred_element=completion_data.get("inferred_element", ""),
                                    confidence_score=completion_data.get("confidence_score", 0.5),
                                    reasoning=completion_data.get("reasoning", ""),
                                    fallback_options=completion_data.get("fallback_options", [])
                                )
                                completions.append(completion)
                            
                            if completions:
                                logger.info(f"Successfully parsed truncated JSON with {len(completions)} completions")
                                return completions
                        except json.JSONDecodeError:
                            pass
                
                # Method 2: Fix JSON with missing commas
                if '"reasoning": ' in response_str:
                    # Try to fix missing comma issues
                    # Find "reasoning": "..." without comma after it
                    # Match "reasoning": "..." without comma after it
                    pattern = r'"reasoning":\s*"([^"]*)"\s*(?!,)'
                    matches = re.finditer(pattern, response_str)
                    
                    for match in matches:
                        # Try to add comma after reasoning field
                        start_pos = match.start()
                        end_pos = match.end()
                        
                        # Check if there are other fields after
                        after_reasoning = response_str[end_pos:].strip()
                        if after_reasoning.startswith('"') or after_reasoning.startswith('}'):
                            # Add comma after reasoning
                            fixed_json = response_str[:end_pos] + ',' + response_str[end_pos:]
                            try:
                                data = json.loads(fixed_json)
                                for completion_data in data.get("completions", []):
                                    completion = SemanticCompletion(
                                        completion_type=completion_data.get("completion_type", "node_inference"),
                                        inferred_element=completion_data.get("inferred_element", ""),
                                        confidence_score=completion_data.get("confidence_score", 0.5),
                                        reasoning=completion_data.get("reasoning", ""),
                                        fallback_options=completion_data.get("fallback_options", [])
                                    )
                                    completions.append(completion)
                                
                                if completions:
                                    logger.info(f"Successfully parsed JSON with missing comma fix, {len(completions)} completions")
                                    return completions
                            except json.JSONDecodeError:
                                continue
                
                # Method 3: Try to extract partially valid JSON
                if '"completions"' in response_str:
                    # Try to extract completions array
                    start_match = re.search(r'"completions":\s*\[', response_str)
                    if start_match:
                        start_pos = start_match.start()
                        # Find corresponding closing bracket
                        brace_count = 0
                        in_array = False
                        end_pos = -1
                        
                        for i in range(start_pos, len(response_str)):
                            char = response_str[i]
                            if char == '[':
                                if not in_array:
                                    in_array = True
                                brace_count += 1
                            elif char == ']':
                                brace_count -= 1
                                if in_array and brace_count == 0:
                                    end_pos = i
                                    break
                        
                        if end_pos > start_pos:
                            # Construct a simple JSON structure
                            json_part = response_str[start_pos-1:end_pos+1]  # Include "completions": [...]
                            simple_json = "{" + json_part + "}"
                            try:
                                data = json.loads(simple_json)
                                for completion_data in data.get("completions", []):
                                    completion = SemanticCompletion(
                                        completion_type=completion_data.get("completion_type", "node_inference"),
                                        inferred_element=completion_data.get("inferred_element", ""),
                                        confidence_score=completion_data.get("confidence_score", 0.5),
                                        reasoning=completion_data.get("reasoning", ""),
                                        fallback_options=completion_data.get("fallback_options", [])
                                    )
                                    completions.append(completion)
                                
                                if completions:
                                    logger.info(f"Successfully parsed partial JSON with {len(completions)} completions")
                                    return completions
                            except json.JSONDecodeError:
                                pass
                                
            except Exception as e:
                logger.debug(f"Failed to fix JSON: {e}")
            
            # If all methods fail, print complete response for debugging
            logger.warning(f"No valid JSON found in response. Full response:")
            logger.warning(f"Response type: {type(response)}")
            logger.warning(f"Full response content:")
            logger.warning(f"{str(response)}")
            return completions
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM completion response: {e}")
            return []
    
    def _apply_completions(self, pattern: FlexibleTaskPattern, core_match: Dict[str, Any], completions: List[SemanticCompletion]) -> List[Dict[str, Any]]:
        """Apply semantic completion results"""
        enhanced_matches = []
        
        for completion in completions:
            # Create enhanced match result
            enhanced_match = core_match.copy()
            
            # Add inferred elements
            if completion.completion_type == "node_inference":
                enhanced_match["inferred_nodes"] = enhanced_match.get("inferred_nodes", [])
                enhanced_match["inferred_nodes"].append({
                    "element": completion.inferred_element,
                    "confidence": completion.confidence_score,
                    "reasoning": completion.reasoning
                })
            
            elif completion.completion_type == "edge_inference":
                enhanced_match["inferred_edges"] = enhanced_match.get("inferred_edges", [])
                enhanced_match["inferred_edges"].append({
                    "edge_type": completion.inferred_element,
                    "confidence": completion.confidence_score,
                    "reasoning": completion.reasoning
                })
            
            elif completion.completion_type == "slot_filling":
                enhanced_match["filled_slots"] = enhanced_match.get("filled_slots", {})
                enhanced_match["filled_slots"][completion.inferred_element] = {
                    "value": completion.fallback_options[0] if completion.fallback_options else "",
                    "confidence": completion.confidence_score,
                    "reasoning": completion.reasoning
                }
            
            # Calculate overall confidence
            enhanced_match["confidence_score"] = min(1.0, enhanced_match.get("confidence_score", 0.5) + completion.confidence_score * 0.3)
            
            enhanced_matches.append(enhanced_match)
        
        return enhanced_matches
    
    def _try_fallback_versions(self, subgraph: SubgraphSample, seed_type: TaskSeedType) -> List[Dict[str, Any]]:
        """Try fallback version matching"""
        matches = []
        
        # Define fallback mapping
        fallback_mapping = {
            TaskSeedType.BUSINESS_SEARCH_FILTER: [
                TaskSeedType.BASIC_NAVIGATION,
                TaskSeedType.CONTENT_BROWSING,
                TaskSeedType.BUTTON_INTERACTION
            ],
            TaskSeedType.BUTTON_INTERACTION: [
                TaskSeedType.BUTTON_INTERACTION,
                TaskSeedType.BASIC_NAVIGATION
            ],
            TaskSeedType.MULTI_HOP_NAVIGATION: [
                TaskSeedType.BASIC_NAVIGATION,
                TaskSeedType.CONTENT_BROWSING,
                TaskSeedType.MENU_EXPLORATION
            ],
        }
        
        # Get fallback seed types
        fallback_types = fallback_mapping.get(seed_type, [TaskSeedType.BASIC_NAVIGATION])
        
        # Try each fallback type
        for fallback_type in fallback_types:
            try:
                fallback_matches = self.match_with_semantic_completion(subgraph, fallback_type)
                if fallback_matches:
                    # Adjust confidence scores (fallback matches have lower confidence)
                    for match in fallback_matches:
                        match["confidence_score"] = match.get("confidence_score", 0.5) * 0.7
                        match["fallback_type"] = fallback_type.value
                    matches.extend(fallback_matches)
            except Exception as e:
                logger.warning(f"Failed to try fallback version {fallback_type.value}: {e}")
                continue
        
        return matches