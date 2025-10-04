"""
Graph Regex Engine - Graph Regular Expression Engine
Implements regex-like graph pattern syntax with flexible matching and dynamic composition
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import re
from collections import defaultdict
from loguru import logger
from graph_rag.graph_builder import TaskGraph
from graph_rag.edge_types import GraphEdge, EdgeType
from graph_rag.node_types import GraphNode, NodeType
from .task_seeds import TaskSeedPattern, TaskSeedType
from .web_subgraph_sampler import SubgraphSample

@dataclass
class GraphRegexPattern:
    """Graph regular expression pattern"""
    pattern_id: str
    name: str
    description: str
    
    # Graph regular expression syntax
    regex_syntax: str
    
    # Parsed pattern components
    pattern_components: List[Dict[str, Any]] = field(default_factory=list)
    
    # Pattern variables and constraints
    variables: Dict[str, str] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    
    # Corresponding task seed type
    seed_type: TaskSeedType = TaskSeedType.BASIC_NAVIGATION

@dataclass 
class GraphRegexMatch:
    """Graph regular expression match result"""
    pattern: GraphRegexPattern
    matched_nodes: List[str] = field(default_factory=list)
    matched_edges: List[str] = field(default_factory=list)
    variable_bindings: Dict[str, str] = field(default_factory=dict)
    match_quality: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern.pattern_id,
            "matched_nodes": self.matched_nodes,
            "matched_edges": self.matched_edges,
            "variable_bindings": self.variable_bindings,
            "match_quality": self.match_quality
        }

class GraphRegexEngine:
    """Graph regular expression engine"""
    
    def __init__(self):
        self.patterns = self._initialize_regex_patterns()
    
    def _initialize_regex_patterns(self) -> Dict[str, GraphRegexPattern]:
        """Initialize graph regular expression patterns - reorganized into hierarchical structure"""
        patterns = {}
        
        # ===== 1. Business Data Patterns (High Priority) =====
        
        # 1.1. Business data search pattern
        patterns["business_data_search"] = GraphRegexPattern(
            pattern_id="business_data_search",
            name="Business Data Search Pattern",
            description="Search pattern based on business data, using real business data as search terms",
            regex_syntax="SearchBox($search) -[Fills]-> BusinessData($query) -[Controls]-> Button($submit) -[NavTo]-> List($results) -[Contains]-> BusinessData($data)",
            variables={
                "search": "SearchBox",
                "query": "BusinessData",
                "submit": "Button", 
                "results": "List",
                "data": "BusinessData"
            },
            seed_type=TaskSeedType.BUSINESS_SEARCH_FILTER
        )
        
        
        # 1.3. Business context navigation pattern
        patterns["business_context_navigation"] = GraphRegexPattern(
            pattern_id="business_context_navigation",
            name="Business Context Navigation Pattern",
            description="Navigation pattern based on business context",
            regex_syntax="Page($current) -[Contains]-> BusinessContext($context) -[NavTo]-> Navigation($nav) -[Contains]-> BusinessData($data)",
            variables={
                "current": "Page",
                "context": "BusinessContext",
                "nav": "Navigation",
                "data": "BusinessData"
            },
            seed_type=TaskSeedType.BUSINESS_NAVIGATION
        )
        
        
        # ===== 2. General Interaction Patterns (Medium Priority) =====
        
        # 2.1. General search filter pattern (non-business data)
        patterns["general_search_filter"] = GraphRegexPattern(
            pattern_id="general_search_filter",
            name="General Search and Filter Pattern",
            description="General search pattern with optional filters and paginators (non-business data)",
            regex_syntax="SearchBox($search) -[Fills]-> Content($query) -[Controls]-> Button($submit) -[NavTo]-> List($results) -[Filters]?-> (Filter($filter))* -[NavTo]?-> (Paginator($page))?",
            variables={
                "search": "SearchBox",
                "query": "Content",
                "submit": "Button", 
                "results": "List",
                "filter": "Filter",
                "page": "Paginator"
            },
            seed_type=TaskSeedType.BUSINESS_SEARCH_FILTER
        )
        
        # 2.2. General button interaction pattern
        patterns["general_button_interaction"] = GraphRegexPattern(
            pattern_id="general_button_interaction",
            name="General Button Interaction Pattern",
            description="General button interaction pattern with optional responses and feedback",
            regex_syntax="Page($page) -[Contains]-> Button($button) -[Opens]?-> (Toast($feedback) | Modal($feedback) | Page($new_page))?",
            variables={
                "page": "Page",
                "button": "Button",
                "feedback": "Toast",
                "new_page": "Page"
            },
            seed_type=TaskSeedType.BUTTON_INTERACTION
        )
        
        # 2.3. General navigation pattern
        patterns["general_navigation"] = GraphRegexPattern(
            pattern_id="general_navigation",
            name="General Navigation Pattern",
            description="General navigation pattern with optional intermediate steps",
            regex_syntax="Navigation($nav1) -[NavTo]-> (Navigation($nav2) -[NavTo]->)? Navigation($target)",
            variables={
                "nav1": "Navigation",
                "nav2": "Navigation", 
                "target": "Navigation"
            },
            seed_type=TaskSeedType.BASIC_NAVIGATION
        )
        
        # 2.4. General content browsing pattern
        patterns["general_content_browsing"] = GraphRegexPattern(
            pattern_id="general_content_browsing",
            name="General Content Browsing Pattern", 
            description="General content browsing with multiple optional interactive elements",
            regex_syntax="Page($page) -[Contains]-> (Button($btn1))+ -[NavTo|Contains]?-> (Button($btn2))*",
            variables={
                "page": "Page",
                "btn1": "Button",
                "btn2": "Button"
            },
            seed_type=TaskSeedType.CONTENT_BROWSING
        )
        
        # 2.5. Multi-hop navigation pattern
        patterns["multi_hop_navigation"] = GraphRegexPattern(
            pattern_id="multi_hop_navigation",
            name="Multi-hop Navigation Pattern",
            description="Multi-hop navigation with variable-length navigation paths",
            regex_syntax="Navigation($start) (-[NavTo]-> Navigation($hop)){1,3} -[NavTo]-> Navigation($end)",
            variables={
                "start": "Navigation",
                "hop": "Navigation",
                "end": "Navigation"
            },
            seed_type=TaskSeedType.MULTI_HOP_NAVIGATION
        )
        
        # ===== 3. Basic Interaction Patterns (Fallback / Simple, Low Priority) =====
        
        # 3.1. Simple button interaction
        patterns["simple_button_interaction"] = GraphRegexPattern(
            pattern_id="simple_button_interaction",
            name="Simple Button Interaction",
            description="Simple button click interaction (fallback mode)",
            regex_syntax="Page($page) -[Contains]-> Button($btn)",
            variables={
                "page": "Page",
                "btn": "Button"
            },
            seed_type=TaskSeedType.BUTTON_INTERACTION
        )
        
        # 3.2. Simple input interaction
        patterns["simple_input_interaction"] = GraphRegexPattern(
            pattern_id="simple_input_interaction", 
            name="Simple Input Interaction",
            description="Simple input field filling (fallback mode)",
            regex_syntax="Page($page) -[Contains]-> Input($input)",
            variables={
                "page": "Page",
                "input": "Input"
            },
            seed_type=TaskSeedType.BUTTON_INTERACTION
        )
        
        # 3.3. Simple content browsing
        patterns["simple_content_browsing"] = GraphRegexPattern(
            pattern_id="simple_content_browsing",
            name="Simple Content Browsing Pattern",
            description="Simple page content browsing (fallback mode)",
            regex_syntax="Page($page) -[Contains]-> Content($content)",
            variables={
                "page": "Page",
                "content": "Content"
            },
            seed_type=TaskSeedType.CONTENT_BROWSING
        )
        
        # 3.4. Simple link navigation
        patterns["simple_link_navigation"] = GraphRegexPattern(
            pattern_id="simple_link_navigation",
            name="Simple Link Navigation Pattern",
            description="Simple link navigation pattern (fallback mode)",
            regex_syntax="Page($page) -[Contains]-> Link($link)",
            variables={
                "page": "Page",
                "link": "Link"
            },
            seed_type=TaskSeedType.BASIC_NAVIGATION
        )
        
        # 3.5. Multi-link browsing
        patterns["multi_link_browsing"] = GraphRegexPattern(
            pattern_id="multi_link_browsing",
            name="Multi-Link Browsing Pattern",
            description="Multi-link browsing pattern (fallback mode)",
            regex_syntax="Page($page) -[Contains]-> Link($link1) -[Contains]-> Link($link2)",
            variables={
                "page": "Page",
                "link1": "Link",
                "link2": "Link"
            },
            seed_type=TaskSeedType.CONTENT_BROWSING
        )
        
        # 3.6. Link to content
        patterns["link_to_content"] = GraphRegexPattern(
            pattern_id="link_to_content",
            name="Link to Content Pattern",
            description="Navigate from link to content (fallback mode)",
            regex_syntax="Page($page) -[Contains]-> Link($link) -[Contains]-> ResultItem($content)",
            variables={
                "page": "Page",
                "link": "Link",
                "content": "ResultItem"
            },
            seed_type=TaskSeedType.CONTENT_BROWSING
        )
        
        # 3.7. Multi-button interaction (fallback)
        patterns["multi_button_interaction"] = GraphRegexPattern(
            pattern_id="multi_button_interaction",
            name="Multi-Button Interaction Pattern",
            description="Combined interaction of multiple buttons (fallback mode)",
            regex_syntax="Page($page) -[Contains]-> Button($btn1) -[NavTo]-> Button($btn2)",
            variables={
                "page": "Page",
                "btn1": "Button",
                "btn2": "Button"
            },
            seed_type=TaskSeedType.BUTTON_INTERACTION
        )
        
        # 3.8. Multi-step interaction (fallback)
        patterns["multi_step_interaction"] = GraphRegexPattern(
            pattern_id="multi_step_interaction",
            name="Multi-Step Interaction Pattern",
            description="Multi-step interaction including navigation, browsing, verification, etc. (fallback mode)",
            regex_syntax="Page($page) -[Contains]-> Button($btn1) -[NavTo]-> Button($btn2) -[Contains]-> ResultItem($content)",
            variables={
                "page": "Page",
                "btn1": "Button",
                "btn2": "Button",
                "content": "ResultItem"
            },
            seed_type=TaskSeedType.CONTENT_BROWSING
        )
        
        # 3.9. Content exploration (fallback)
        patterns["content_exploration"] = GraphRegexPattern(
            pattern_id="content_exploration",
            name="Content Exploration Pattern",
            description="Deep content exploration and browsing (fallback mode)",
            regex_syntax="Page($page) -[Contains]-> ResultItem($content1) -[Contains]-> ResultItem($content2)",
            variables={
                "page": "Page",
                "content1": "ResultItem",
                "content2": "ResultItem"
            },
            seed_type=TaskSeedType.CONTENT_BROWSING
        )
        
        
        # 3.11. Information aggregation (fallback)
        patterns["information_aggregation"] = GraphRegexPattern(
            pattern_id="information_aggregation",
            name="Information Aggregation Pattern",
            description="Aggregate information from multiple sources (fallback mode)",
            regex_syntax="Page($page) -[Contains]-> Content($content1) -[RefersTo]-> Content($content2)",
            variables={
                "page": "Page",
                "content1": "Content",
                "content2": "Content"
            },
            seed_type=TaskSeedType.CONTENT_BROWSING
        )
        
        return patterns
    
    def parse_regex_pattern(self, pattern: GraphRegexPattern) -> List[Dict[str, Any]]:
        """Parse graph regular expression syntax"""
        components = []
        syntax = pattern.regex_syntax.strip()
        
        # Decompose pattern syntax into tokens
        tokens = self._tokenize_regex(syntax)
        
        # Parse token sequence
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if self._is_node_token(token):
                # Parse node
                node_component = self._parse_node_token(token)
                components.append(node_component)
            
            elif self._is_edge_token(token):
                # Parse edge
                edge_component = self._parse_edge_token(token)
                components.append(edge_component)
            
            elif self._is_quantifier_token(token):
                # Parse quantifier (?, *, +, {n,m})
                if components:
                    # Apply quantifier to the last component
                    quantifier = self._parse_quantifier(token)
                    components[-1]["quantifier"] = quantifier
            
            elif self._is_group_token(token):
                # Parse grouping (...)
                group_component = self._parse_group_token(token)
                components.append(group_component)
            
            elif self._is_alternative_token(token):
                # Parse alternative syntax (A|B)
                alternative_component = self._parse_alternative_token(token)
                components.append(alternative_component)
            
            i += 1
        
        pattern.pattern_components = components
        return components
    
    def _tokenize_regex(self, syntax: str) -> List[str]:
        """Decompose graph regular expression into tokens"""
        # Improved token pattern for better handling of complex syntax
        token_pattern = r'(\w+\(\$?\w+\)|\-\[[^\]]+\]\-\>|[?*+]|\{\d+,?\d*\}|\([^)]+\)|\w+)'
        tokens = re.findall(token_pattern, syntax)
        tokens = [token.strip() for token in tokens if token.strip()]
        
        # Handle special cases: alternative syntax like Toast|Modal
        processed_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Check if it's alternative syntax (A|B)
            if '|' in token and not token.startswith('-['):
                # This is alternative syntax, needs special handling
                processed_tokens.append(token)
            else:
                processed_tokens.append(token)
            
            i += 1
        
        return processed_tokens
    
    def _is_node_token(self, token: str) -> bool:
        """Check if token is a node token"""
        return re.match(r'\w+\(\$?\w+\)', token) is not None
    
    def _is_edge_token(self, token: str) -> bool:
        """Check if token is an edge token"""
        return re.match(r'-\[[^\]]+\]->', token) is not None
    
    def _is_quantifier_token(self, token: str) -> bool:
        """Check if token is a quantifier token"""
        return token in ['?', '*', '+'] or re.match(r'\{\d+,?\d*\}', token) is not None
    
    def _is_group_token(self, token: str) -> bool:
        """Check if token is a group token"""
        return token.startswith('(') and token.endswith(')')
    
    def _is_alternative_token(self, token: str) -> bool:
        """Check if token is an alternative syntax token"""
        return '|' in token and not token.startswith('-[')
    
    def _parse_node_token(self, token: str) -> Dict[str, Any]:
        """Parse node token"""
        # Format: NodeType($variable)
        match = re.match(r'(\w+)\(\$(\w+)\)', token)
        if match:
            node_type = match.group(1)
            variable = match.group(2)
            return {
                "type": "node",
                "node_type": node_type,
                "variable": variable,
                "quantifier": {"min": 1, "max": 1}  # Default quantifier
            }
        return {}
    
    def _parse_edge_token(self, token: str) -> Dict[str, Any]:
        """Parse edge token"""
        # Format: -[EdgeType|EdgeType2]->
        match = re.match(r'-\[([^\]]+)\]->', token)
        if match:
            edge_types = match.group(1).split('|')
            return {
                "type": "edge",
                "edge_types": edge_types,
                "quantifier": {"min": 1, "max": 1}  # Default quantifier
            }
        return {}
    
    def _parse_quantifier(self, token: str) -> Dict[str, int]:
        """Parse quantifier"""
        if token == '?':
            return {"min": 0, "max": 1}
        elif token == '*':
            return {"min": 0, "max": float('inf')}
        elif token == '+':
            return {"min": 1, "max": float('inf')}
        elif re.match(r'\{\d+,?\d*\}', token):
            # Format: {n} or {n,m}
            numbers = re.findall(r'\d+', token)
            if len(numbers) == 1:
                n = int(numbers[0])
                return {"min": n, "max": n}
            elif len(numbers) == 2:
                n, m = int(numbers[0]), int(numbers[1])
                return {"min": n, "max": m}
        
        return {"min": 1, "max": 1}
    
    def _parse_group_token(self, token: str) -> Dict[str, Any]:
        """Parse group token"""
        # Remove brackets and recursively parse content
        inner_content = token[1:-1]  # Remove ( and )
        return {
            "type": "group",
            "content": inner_content,
            "quantifier": {"min": 1, "max": 1}
        }
    
    def _parse_alternative_token(self, token: str) -> Dict[str, Any]:
        """Parse alternative syntax token"""
        # Format: (A|B) or A|B
        if token.startswith('(') and token.endswith(')'):
            inner_content = token[1:-1]
        else:
            inner_content = token
        
        alternatives = [alt.strip() for alt in inner_content.split('|')]
        return {
            "type": "alternative",
            "alternatives": alternatives,
            "quantifier": {"min": 1, "max": 1}
        }
    
    def match_pattern(self, pattern: GraphRegexPattern, subgraph: SubgraphSample) -> List[GraphRegexMatch]:
        """Match graph regular expression pattern in subgraph"""
        matches = []
        
        # Parse pattern (if not parsed yet)
        if not pattern.pattern_components:
            self.parse_regex_pattern(pattern)
        
        # Try to find matches in subgraph
        potential_matches = self._find_pattern_matches(pattern, subgraph)
        
        for match_candidate in potential_matches:
            if self._validate_match(match_candidate, pattern, subgraph):
                matches.append(match_candidate)
        
        return matches
    
    def _find_pattern_matches(self, pattern: GraphRegexPattern, subgraph: SubgraphSample) -> List[GraphRegexMatch]:
        """Find potential pattern matches in subgraph"""
        matches = []
        
        # Try matching from each possible starting node
        for start_node_id, start_node in subgraph.nodes.items():
            match = self._try_match_from_node(pattern, subgraph, start_node_id)
            if match:
                matches.append(match)
        
        return matches
    
    def _try_match_from_node(self, pattern: GraphRegexPattern, subgraph: SubgraphSample, start_node_id: str) -> Optional[GraphRegexMatch]:
        """Try to match pattern starting from specified node"""
        match = GraphRegexMatch(pattern=pattern)
        current_nodes = [start_node_id]
        component_index = 0
        
        # Traverse pattern components
        while component_index < len(pattern.pattern_components) and current_nodes:
            component = pattern.pattern_components[component_index]
            
            if component["type"] == "node":
                # Match node component
                matched_nodes = self._match_node_component(component, subgraph, current_nodes)
                if matched_nodes:
                    match.matched_nodes.extend(matched_nodes)
                    # Update variable bindings
                    if component["variable"] not in match.variable_bindings:
                        match.variable_bindings[component["variable"]] = matched_nodes[0]
                    current_nodes = matched_nodes
                else:
                    # Check if it's an optional component
                    if component["quantifier"]["min"] == 0:
                        # Optional component, skip
                        pass
                    else:
                        # Required component match failed
                        return None
            
            elif component["type"] == "edge":
                # Match edge component
                next_nodes, matched_edges = self._match_edge_component(component, subgraph, current_nodes)
                if next_nodes:
                    match.matched_edges.extend(matched_edges)
                    current_nodes = next_nodes
                else:
                    # Check if it's an optional component
                    if component["quantifier"]["min"] == 0:
                        # Optional edge, continue using current nodes
                        pass
                    else:
                        # Required edge match failed
                        return None
            
            elif component["type"] == "alternative":
                # Match alternative syntax component
                matched_alternative = self._match_alternative_component(component, subgraph, current_nodes)
                if matched_alternative:
                    # Process matched alternative
                    if "matched_nodes" in matched_alternative:
                        match.matched_nodes.extend(matched_alternative["matched_nodes"])
                    if "matched_edges" in matched_alternative:
                        match.matched_edges.extend(matched_alternative["matched_edges"])
                    current_nodes = matched_alternative.get("next_nodes", current_nodes)
                else:
                    # Check if it's an optional component
                    if component["quantifier"]["min"] == 0:
                        # Optional alternative, skip
                        pass
                    else:
                        # Required alternative match failed
                        return None
            
            component_index += 1
        
        # Calculate match quality
        match.match_quality = self._calculate_match_quality(match, pattern, subgraph)
        
        return match if match.matched_nodes else None
    
    def _match_node_component(self, component: Dict[str, Any], subgraph: SubgraphSample, candidate_nodes: List[str]) -> List[str]:
        """Match node component"""
        matched_nodes = []
        required_type = component["node_type"]
        quantifier = component["quantifier"]
        
        for node_id in candidate_nodes:
            if node_id not in subgraph.nodes:
                continue
            
            node = subgraph.nodes[node_id]
            
            # Check node type match
            if self._node_type_matches(node, required_type):
                matched_nodes.append(node_id)
                
                # Check quantifier constraints
                if len(matched_nodes) >= quantifier["max"]:
                    break
        
        # Check minimum match count
        if len(matched_nodes) >= quantifier["min"]:
            return matched_nodes
        else:
            return []
    
    def _match_edge_component(self, component: Dict[str, Any], subgraph: SubgraphSample, source_nodes: List[str]) -> Tuple[List[str], List[str]]:
        """Match edge component"""
        target_nodes = []
        matched_edges = []
        allowed_edge_types = component["edge_types"]
        quantifier = component["quantifier"]
        
        for source_node_id in source_nodes:
            # Find matching edges from this node
            for edge_id, edge in subgraph.edges.items():
                if edge.source_node_id == source_node_id:
                    # Check edge type match
                    if self._edge_type_matches(edge, allowed_edge_types):
                        target_nodes.append(edge.target_node_id)
                        matched_edges.append(edge_id)
                        
                        # Check quantifier constraints
                        if len(matched_edges) >= quantifier["max"]:
                            break
        
        # Remove duplicates
        target_nodes = list(set(target_nodes))
        
        # Check minimum match count
        if len(matched_edges) >= quantifier["min"]:
            return target_nodes, matched_edges
        else:
            return [], []
    
    def _match_alternative_component(self, component: Dict[str, Any], subgraph: SubgraphSample, source_nodes: List[str]) -> Optional[Dict[str, Any]]:
        """Match alternative syntax component"""
        alternatives = component["alternatives"]
        
        # Try to match each alternative
        for alternative in alternatives:
            # Check if it's a node type
            if re.match(r'\w+\(\$?\w+\)', alternative):
                # This is a node alternative
                node_component = {
                    "type": "node",
                    "node_type": alternative.split('(')[0],
                    "variable": alternative.split('(')[1].rstrip(')'),
                    "quantifier": {"min": 1, "max": 1}
                }
                matched_nodes = self._match_node_component(node_component, subgraph, source_nodes)
                if matched_nodes:
                    return {
                        "matched_nodes": matched_nodes,
                        "next_nodes": matched_nodes
                    }
            
            # Check if it's an edge type
            elif alternative.startswith('-[') and alternative.endswith('->'):
                # This is an edge alternative
                edge_component = {
                    "type": "edge",
                    "edge_types": [alternative[2:-3]],  # Remove -[ and ]->
                    "quantifier": {"min": 1, "max": 1}
                }
                next_nodes, matched_edges = self._match_edge_component(edge_component, subgraph, source_nodes)
                if next_nodes:
                    return {
                        "matched_edges": matched_edges,
                        "next_nodes": next_nodes
                    }
        
        return None
    
    def _node_type_matches(self, node: GraphNode, required_type: str) -> bool:
        """Check if node type matches (supports flexible matching)"""
        actual_type = node.node_type.value
        
        # Direct match
        if actual_type == required_type:
            return True
        
        # Flexible matching mapping - aligned with actual NodeType enum values
        flexible_mapping = {
            "Navigation": ["Navigation", "Button", "Link", "Menu"],
            "Button": ["Button", "Navigation", "Link", "Submit"],
            "SearchBox": ["SearchBox", "Input", "Button"],
            "Input": ["Input", "SearchBox", "Textarea", "Select"],
            "Page": ["Page", "Section"],
            "Form": ["Form", "Input", "Button"],
            "List": ["List", "Table", "Card", "Item"],
            "Detail": ["Detail", "Card", "Content"],
            "Filter": ["Filter", "Button", "Select"],
            "Paginator": ["Paginator", "Button", "Navigation"],
            "Toast": ["Toast", "Modal", "NotificationArea"],
            "Modal": ["Modal", "Toast"],
            "Content": ["Content", "Text", "Card", "Item"],
            "ResultItem": ["ResultItem", "Item", "Card", "Content"],
            "Link": ["Link", "Button", "Navigation"],
            "Menu": ["Menu", "Navigation", "Dropdown"],
            "Card": ["Card", "Item", "Content"],
            "Item": ["Item", "Card", "Content"],
            "Text": ["Text", "Content"],
            "Image": ["Image", "Content"],
            "Icon": ["Icon", "Button"],
            "Tab": ["Tab", "Button"],
            "TabContainer": ["TabContainer", "Menu"],
            "Dropdown": ["Dropdown", "Menu", "Select"],
            "Submenu": ["Submenu", "Menu"],
            "Breadcrumb": ["Breadcrumb", "Navigation"],
            "Dashboard": ["Dashboard", "Page"],
            "DetailLink": ["DetailLink", "Link", "Button"],
            "FilterPanel": ["FilterPanel", "Filter"],
            "Collapsible": ["Collapsible", "Button"],
            "ScrollArea": ["ScrollArea", "Content"],
            "NotificationArea": ["NotificationArea", "Toast", "Modal"]
        }
        
        allowed_types = flexible_mapping.get(required_type, [])
        return actual_type in allowed_types
    
    def _edge_type_matches(self, edge: GraphEdge, allowed_types: List[str]) -> bool:
        """Check if edge type matches (supports flexible matching)"""
        actual_type = edge.edge_type.value
        
        # Direct match
        if actual_type in allowed_types:
            return True
        
        # Flexible matching mapping - aligned with actual EdgeType enum values
        flexible_mapping = {
            "NavTo": ["NavTo", "contains", "web_navigation"],
            "Contains": ["contains", "NavTo", "web_layout"],
            "Controls": ["Controls", "contains", "web_interaction"],
            "Fills": ["Fills", "contains", "web_form_submit"],
            "Opens": ["Opens", "NavTo", "web_click_trigger"],
            "Filters": ["Filters", "contains", "web_interaction"],
            "RefersTo": ["refers_to", "NavTo", "web_data_flow"],
            "nav_to": ["NavTo", "contains", "web_navigation"],
            "contains": ["contains", "NavTo", "web_layout"],
            "controls": ["Controls", "contains", "web_interaction"],
            "fills": ["Fills", "contains", "web_form_submit"],
            "opens": ["Opens", "NavTo", "web_click_trigger"],
            "filters": ["Filters", "contains", "web_interaction"],
            "refers_to": ["refers_to", "NavTo", "web_data_flow"]
        }
        
        for allowed_type in allowed_types:
            allowed_alternatives = flexible_mapping.get(allowed_type, [])
            if actual_type in allowed_alternatives:
                return True
        
        return False
    
    def _calculate_match_quality(self, match: GraphRegexMatch, pattern: GraphRegexPattern, subgraph: SubgraphSample) -> float:
        """Calculate match quality score"""
        score = 0.0
        
        # Base score: number of matched components
        total_components = len(pattern.pattern_components)
        matched_components = 0
        
        for component in pattern.pattern_components:
            if component["type"] == "node":
                variable = component["variable"]
                if variable in match.variable_bindings:
                    matched_components += 1
            elif component["type"] == "edge":
                if match.matched_edges:
                    matched_components += 1
        
        if total_components > 0:
            score += (matched_components / total_components) * 0.6
        
        # Node quality score: check quality of matched nodes
        node_quality = 0.0
        for node_id in match.matched_nodes:
            if node_id in subgraph.nodes:
                node = subgraph.nodes[node_id]
                # Check if node is interactive
                if self._is_node_interactive(node):
                    node_quality += 0.3
                # Check if node has useful text content
                if hasattr(node, 'metadata') and node.metadata and node.metadata.text_content:
                    node_quality += 0.2
        
        if match.matched_nodes:
            score += (node_quality / len(match.matched_nodes)) * 0.3
        
        # Connectivity score: check if matched edges form coherent paths
        if match.matched_edges:
            connectivity_score = self._calculate_connectivity_score(match, subgraph)
            score += connectivity_score * 0.1
        
        return min(score, 1.0)
    
    def _is_node_interactive(self, node: GraphNode) -> bool:
        """Check if node is interactive"""
        # Check node type
        if node.node_type in [NodeType.BUTTON, NodeType.NAVIGATION, NodeType.SEARCH_BOX, NodeType.INPUT]:
            return True
        
        # Check metadata
        if hasattr(node, 'metadata') and node.metadata:
            if hasattr(node.metadata, 'is_clickable') and node.metadata.is_clickable:
                return True
            if hasattr(node.metadata, 'is_input') and node.metadata.is_input:
                return True
        
        return False
    
    def _calculate_connectivity_score(self, match: GraphRegexMatch, subgraph: SubgraphSample) -> float:
        """Calculate connectivity score"""
        if not match.matched_edges:
            return 0.0
        
        # Check if edges form coherent paths
        connected_nodes = set()
        for edge_id in match.matched_edges:
            if edge_id in subgraph.edges:
                edge = subgraph.edges[edge_id]
                connected_nodes.add(edge.source_node_id)
                connected_nodes.add(edge.target_node_id)
        
        # Connectivity score = connected nodes / total matched nodes
        if match.matched_nodes:
            return len(connected_nodes) / len(set(match.matched_nodes))
        else:
            return 0.0
    
    def _validate_match(self, match: GraphRegexMatch, pattern: GraphRegexPattern, subgraph: SubgraphSample) -> bool:
        """Validate if match is valid"""
        # Check basic requirements
        if not match.matched_nodes:
            return False
        
        # Check quality threshold - lowered threshold to allow more matches
        if match.match_quality < 0.1:
            return False
        
        # Check if there is at least one interactive node
        has_interactive = False
        for node_id in match.matched_nodes:
            if node_id in subgraph.nodes:
                node = subgraph.nodes[node_id]
                if self._is_node_interactive(node):
                    has_interactive = True
                    break
        
        if not has_interactive:
            return False
        
        # Check pattern-specific constraints
        return self._validate_pattern_constraints(match, pattern, subgraph)
    
    def _validate_pattern_constraints(self, match: GraphRegexMatch, pattern: GraphRegexPattern, subgraph: SubgraphSample) -> bool:
        """Validate pattern-specific constraints"""
        # Perform specific validation based on different pattern types
        if pattern.pattern_id.startswith("business_data_"):
            # Business data pattern: ensure business data nodes exist
            return len(match.matched_nodes) >= 2  # At least 2 nodes required
        
        elif pattern.pattern_id.startswith("general_"):
            # General interaction pattern: ensure interactive elements exist
            return len(match.matched_nodes) >= 1
        
        elif pattern.pattern_id.startswith("simple_"):
            # Simple interaction pattern: ensure basic elements exist
            return len(match.matched_nodes) >= 1
        
        elif pattern.pattern_id.startswith("multi_"):
            # Multi-step pattern: ensure multiple elements exist
            return len(match.matched_nodes) >= 2
        
        elif pattern.pattern_id.startswith("link_"):
            # Link pattern: ensure link elements exist
            return len(match.matched_nodes) >= 1
        
        # Default validation
        return True
    
    def find_all_matches(self, subgraph: SubgraphSample) -> List[GraphRegexMatch]:
        """Find all pattern matches in subgraph"""
        all_matches = []
        
        for pattern in self.patterns.values():
            matches = self.match_pattern(pattern, subgraph)
            all_matches.extend(matches)
        
        # Sort by quality score
        all_matches.sort(key=lambda m: m.match_quality, reverse=True)
        
        return all_matches
    
    def get_best_match_for_seed(self, subgraph: SubgraphSample, seed_type: TaskSeedType) -> Optional[GraphRegexMatch]:
        """Find best match for specific task seed type"""
        best_match = None
        best_score = 0.0
        
        for pattern in self.patterns.values():
            if pattern.seed_type == seed_type:
                matches = self.match_pattern(pattern, subgraph)
                for match in matches:
                    if match.match_quality > best_score:
                        best_match = match
                        best_score = match.match_quality
        
        return best_match
    
    def get_business_patterns(self) -> Dict[str, GraphRegexPattern]:
        """Get business data patterns (high priority)"""
        business_patterns = {}
        for pattern_id, pattern in self.patterns.items():
            if pattern_id.startswith("business_data_") or pattern_id.startswith("business_context_"):
                business_patterns[pattern_id] = pattern
        return business_patterns
    
    def get_general_patterns(self) -> Dict[str, GraphRegexPattern]:
        """Get general interaction patterns (medium priority)"""
        general_patterns = {}
        for pattern_id, pattern in self.patterns.items():
            if pattern_id.startswith("general_") or pattern_id.startswith("multi_hop_"):
                general_patterns[pattern_id] = pattern
        return general_patterns
    
    def get_fallback_patterns(self) -> Dict[str, GraphRegexPattern]:
        """Get basic interaction patterns (low priority/fallback)"""
        fallback_patterns = {}
        for pattern_id, pattern in self.patterns.items():
            if (pattern_id.startswith("simple_") or 
                pattern_id.startswith("multi_") or 
                pattern_id.startswith("link_") or
                pattern_id in ["data_extraction", "information_aggregation", "content_exploration"]):
                fallback_patterns[pattern_id] = pattern
        return fallback_patterns
    
    def find_matches_by_priority(self, subgraph: SubgraphSample, business_data_available: bool = False) -> List[GraphRegexMatch]:
        """Find matches by priority hierarchy"""
        all_matches = []
        
        # 1. Prioritize business data patterns (if business data available)
        if business_data_available:
            business_patterns = self.get_business_patterns()
            logger.debug(f"🎯 Matching {len(business_patterns)} business patterns")
            for pattern in business_patterns.values():
                matches = self.match_pattern(pattern, subgraph)
                all_matches.extend(matches)
                if matches:
                    logger.debug(f"🎯 Found {len(matches)} matches for business pattern: {pattern.pattern_id}")
        
        # 2. If business data matches are insufficient, use general interaction patterns
        if len(all_matches) < 2:
            general_patterns = self.get_general_patterns()
            logger.debug(f"🎯 Matching {len(general_patterns)} general patterns")
            for pattern in general_patterns.values():
                matches = self.match_pattern(pattern, subgraph)
                all_matches.extend(matches)
                if matches:
                    logger.debug(f"🎯 Found {len(matches)} matches for general pattern: {pattern.pattern_id}")
        
        # 3. If still insufficient, use basic interaction patterns (fallback)
        if len(all_matches) < 1:
            fallback_patterns = self.get_fallback_patterns()
            logger.debug(f"🎯 Matching {len(fallback_patterns)} fallback patterns")
            for pattern in fallback_patterns.values():
                matches = self.match_pattern(pattern, subgraph)
                all_matches.extend(matches)
                if matches:
                    logger.debug(f"🎯 Found {len(matches)} matches for fallback pattern: {pattern.pattern_id}")
        
        # Sort by quality score
        all_matches.sort(key=lambda m: m.match_quality, reverse=True)
        
        logger.info(f"🎯 Total matches found: {len(all_matches)} (business_data_available: {business_data_available})")
        return all_matches
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get pattern statistics"""
        business_patterns = self.get_business_patterns()
        general_patterns = self.get_general_patterns()
        fallback_patterns = self.get_fallback_patterns()
        
        return {
            "total_patterns": len(self.patterns),
            "business_patterns": {
                "count": len(business_patterns),
                "patterns": [pattern.pattern_id for pattern in business_patterns.values()]
            },
            "general_patterns": {
                "count": len(general_patterns),
                "patterns": [pattern.pattern_id for pattern in general_patterns.values()]
            },
            "fallback_patterns": {
                "count": len(fallback_patterns),
                "patterns": [pattern.pattern_id for pattern in fallback_patterns.values()]
            }
        }


