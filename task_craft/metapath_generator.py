"""
Metapath Generator - Metapath/Pattern-driven Generation
Define pattern syntax for each task type, transforming "task generation" into structure matching + slot filling
Support graph regex syntax for flexible matching and dynamic composition
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import re
from loguru import logger
from graph_rag.graph_builder import TaskGraph
from graph_rag.edge_types import GraphEdge, EdgeType
from graph_rag.node_types import GraphNode, NodeType
from .task_seeds import TaskSeedPattern, TaskSeedType
from .web_subgraph_sampler import SubgraphSample
from .graph_regex_engine import GraphRegexEngine, GraphRegexMatch

@dataclass
class MetapathPattern:
    """Metapath pattern - defines structural patterns for tasks"""
    pattern_id: str
    name: str
    description: str
    
    # Pattern syntax (graph query template)
    pattern_syntax: str
    
    # Slot definitions
    slots: Dict[str, str] = field(default_factory=dict)  # slot_name -> slot_type
    
    # Constraint conditions
    constraints: List[str] = field(default_factory=list)
    
    # Corresponding task seed type
    seed_type: TaskSeedType = TaskSeedType.BUSINESS_SEARCH_FILTER

@dataclass
class MetapathInstance:
    """Metapath instance - specific instantiation of patterns"""
    pattern: MetapathPattern
    slot_bindings: Dict[str, str] = field(default_factory=dict)  # slot_name -> node_id
    matched_nodes: List[str] = field(default_factory=list)
    matched_edges: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern.pattern_id,
            "slot_bindings": self.slot_bindings,
            "matched_nodes": self.matched_nodes,
            "matched_edges": self.matched_edges
        }

class MetapathGenerator:
    """Metapath generator - supports flexible matching with graph regex"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.regex_engine = GraphRegexEngine()
    
    def _initialize_patterns(self) -> Dict[str, MetapathPattern]:
        """Initialize metapath patterns - supports dual seed design"""
        patterns = {}
        
        # ===== BUSINESS DATA PATTERNS (High-Value Tasks) =====
        
        # 1. Business data search pattern - unified version after merging
        patterns["business_search_pattern"] = MetapathPattern(
            pattern_id="business_search_pattern",
            name="Business Search Pattern",
            description="Search→Filter→View Details pattern based on business data",
            pattern_syntax="""
            Page(∋SearchBox S) -[Fills]-> BusinessQuery -[Controls]-> SearchBtn → 
            Page(Results R) -[Contains]-> BusinessData BD -[Filters]-> BusinessFilter F → 
            Page(FilteredResults) -[Opens]-> BusinessDetail D
            """,
            slots={
                "S": "SearchBox",
                "BusinessQuery": "Input",
                "SearchBtn": "Button",
                "R": "List",
                "BD": "BusinessData",  # Must contain business data
                "F": "Filter",
                "D": "Detail"
            },
            constraints=[
                "S must be visible and enabled",
                "BD must contain actual business data (names, IDs, amounts)",
                "BusinessQuery should target specific business entities",
                "F must filter based on business criteria",
                "D must show business-specific details"
            ],
            seed_type=TaskSeedType.BUSINESS_SEARCH_FILTER
        )
        
        
        
        # 5. Business navigation pattern
        patterns["business_navigation_pattern"] = MetapathPattern(
            pattern_id="business_navigation_pattern",
            name="Business Navigation Pattern",
            description="Navigation pattern based on business context",
            pattern_syntax="""
            Page(Current) -[Contains]-> BusinessContext BC -[NavTo]-> NavigationElement NE → 
            Page(Target) -[Contains]-> BusinessData BD
            """,
            slots={
                "BC": "BusinessData",
                "NE": "Navigation",
                "BD": "BusinessData"
            },
            constraints=[
                "BC must provide business context for navigation",
                "NE must be relevant to business context",
                "BD must be related to business context"
            ],
            seed_type=TaskSeedType.BUSINESS_NAVIGATION
        )
        
        # ===== INTERACTION PATTERNS (Fallback Tasks) =====
        
        # 1. General search pattern
        patterns["search_pattern"] = MetapathPattern(
            pattern_id="search_pattern",
            name="Search Pattern",
            description="Search→Filter→View Details pattern",
            pattern_syntax="""
            Page(∋SearchBox S) -[Fills]-> Query -[Controls]-> Btn → 
            Page(Results R) -[Contains]-> Content -[Filters]-> Filter F → 
            Page(FilteredResults) -[Opens]-> Detail D
            """,
            slots={
                "S": "SearchBox",
                "Query": "Input",
                "Btn": "Button",
                "R": "List",
                "F": "Filter",
                "D": "Detail"
            },
            constraints=[
                "S must be visible and enabled",
                "R must contain content for meaningful search",
                "F must be accessible from R",
                "D must be reachable from FilteredResults"
            ],
            seed_type=TaskSeedType.BUSINESS_SEARCH_FILTER
        )
        
        # 2. Button interaction pattern
        patterns["button_pattern"] = MetapathPattern(
            pattern_id="button_pattern",
            name="Button Interaction Pattern",
            description="Button click→Response verification pattern",
            pattern_syntax="""
            Page(Button B) -[Controls]-> Action → 
            (Toast|Modal|Page|Content)
            """,
            slots={
                "B": "Button",
                "Action": "Action",
                "Toast": "Toast",
                "Modal": "Modal",
                "Page": "Page",
                "Content": "Content"
            },
            constraints=[
                "Button must be clickable",
                "Action must be triggered by button",
                "Response must be visible after click"
            ],
            seed_type=TaskSeedType.BUTTON_INTERACTION
        )
        
        
        # 4. Navigation pattern (simplified version)
        patterns["navigation_pattern"] = MetapathPattern(
            pattern_id="navigation_pattern",
            name="Navigation Pattern",
            description="Multi-hop navigation pattern",
            pattern_syntax="""
            Navigation -[NavTo]-> Navigation -[Contains]-> Button
            """,
            slots={
                "Navigation": "Navigation",
                "Button": "Button"
            },
            constraints=[
                "Navigation path must be connected",
                "Button must be clickable"
            ],
            seed_type=TaskSeedType.MULTI_HOP_NAVIGATION
        )
        
        # 5. Content browsing pattern (simplified version)
        patterns["content_browsing_pattern"] = MetapathPattern(
            pattern_id="content_browsing_pattern",
            name="Content Browsing Pattern",
            description="Content browsing pattern",
            pattern_syntax="""
            Navigation -[Contains]-> Button -[NavTo]-> Navigation
            """,
            slots={
                "Navigation": "Navigation",
                "Button": "Button"
            },
            constraints=[
                "Navigation must be accessible",
                "Button must be clickable"
            ],
            seed_type=TaskSeedType.CONTENT_BROWSING
        )
        
        
        # ===== General Interaction Patterns (Fallback Tasks) =====
        
        # 9. Basic navigation pattern
        patterns["basic_navigation_pattern"] = MetapathPattern(
            pattern_id="basic_navigation_pattern",
            name="Basic Navigation Pattern",
            description="Basic navigation pattern",
            pattern_syntax="""
            Navigation -[NavTo]-> Page
            """,
            slots={
                "Navigation": "Navigation",
                "Page": "Page"
            },
            constraints=[
                "Navigation must be accessible",
                "Page must be reachable"
            ],
            seed_type=TaskSeedType.BASIC_NAVIGATION
        )
        
        # 10. Button interaction pattern
        patterns["button_interaction_pattern"] = MetapathPattern(
            pattern_id="button_interaction_pattern",
            name="Button Interaction Pattern",
            description="Button interaction pattern",
            pattern_syntax="""
            Button -[Controls]-> Action
            """,
            slots={
                "Button": "Button",
                "Action": "Action"
            },
            constraints=[
                "Button must be clickable",
                "Action must be triggered by button"
            ],
            seed_type=TaskSeedType.BUTTON_INTERACTION
        )
        
        
        # 12. Menu exploration pattern
        patterns["menu_exploration_pattern"] = MetapathPattern(
            pattern_id="menu_exploration_pattern",
            name="Menu Exploration Pattern",
            description="Menu exploration pattern",
            pattern_syntax="""
            Menu -[Contains]-> MenuItem -[Opens]-> Submenu
            """,
            slots={
                "Menu": "Menu",
                "MenuItem": "MenuItem",
                "Submenu": "Menu"
            },
            constraints=[
                "Menu must be expandable",
                "MenuItem must be clickable",
                "Submenu must be accessible"
            ],
            seed_type=TaskSeedType.MENU_EXPLORATION
        )
        
        # 13. Tab switching pattern
        patterns["tab_switching_pattern"] = MetapathPattern(
            pattern_id="tab_switching_pattern",
            name="Tab Switching Pattern",
            description="Tab switching pattern",
            pattern_syntax="""
            TabContainer -[Contains]-> Tab -[Opens]-> TabContent
            """,
            slots={
                "TabContainer": "TabContainer",
                "Tab": "Tab",
                "TabContent": "Content"
            },
            constraints=[
                "TabContainer must have multiple tabs",
                "Tab must be clickable",
                "TabContent must be visible after click"
            ],
            seed_type=TaskSeedType.TAB_SWITCHING
        )
        
        # 14. Modal interaction pattern
        patterns["modal_interaction_pattern"] = MetapathPattern(
            pattern_id="modal_interaction_pattern",
            name="Modal Interaction Pattern",
            description="Modal interaction pattern",
            pattern_syntax="""
            Button -[Opens]-> Modal -[Contains]-> ModalContent
            """,
            slots={
                "Button": "Button",
                "Modal": "Modal",
                "ModalContent": "Content"
            },
            constraints=[
                "Button must trigger modal",
                "Modal must be visible",
                "ModalContent must be accessible"
            ],
            seed_type=TaskSeedType.MODAL_INTERACTION
        )
        
        # 15. Breadcrumb navigation pattern
        patterns["breadcrumb_navigation_pattern"] = MetapathPattern(
            pattern_id="breadcrumb_navigation_pattern",
            name="Breadcrumb Navigation Pattern",
            description="Breadcrumb navigation pattern",
            pattern_syntax="""
            Breadcrumb -[Contains]-> BreadcrumbItem -[NavTo]-> Page
            """,
            slots={
                "Breadcrumb": "Breadcrumb",
                "BreadcrumbItem": "Navigation",
                "Page": "Page"
            },
            constraints=[
                "Breadcrumb must show current path",
                "BreadcrumbItem must be clickable",
                "Page must be reachable"
            ],
            seed_type=TaskSeedType.BREADCRUMB_NAVIGATION
        )
        
        # 16. Pagination browsing pattern
        patterns["pagination_browsing_pattern"] = MetapathPattern(
            pattern_id="pagination_browsing_pattern",
            name="Pagination Browsing Pattern",
            description="Pagination browsing pattern",
            pattern_syntax="""
            List -[Contains]-> Pagination -[Controls]-> NextPage
            """,
            slots={
                "List": "List",
                "Pagination": "Navigation",
                "NextPage": "Page"
            },
            constraints=[
                "List must have multiple pages",
                "Pagination must be functional",
                "NextPage must be accessible"
            ],
            seed_type=TaskSeedType.PAGINATION_BROWSING
        )
        
        
        
        return patterns
    
    def match_patterns(self, subgraph: SubgraphSample) -> List[MetapathInstance]:
        """Match patterns in subgraph - supports dual seed design"""
        instances = []
        
        # Debug info: analyze subgraph structure
        logger.debug(f"🎯 Analyzing subgraph for metapath matching with Dual Seeds Design...")
        logger.debug(f"🎯 Subgraph has {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges")
        
        # Analyze node type distribution
        node_types = {}
        for node_id, node in subgraph.nodes.items():
            node_type = node.node_type.value
            node_types[node_type] = node_types.get(node_type, 0) + 1
        logger.debug(f"🎯 Node types: {node_types}")
        
        # Statistics on business data availability
        business_data_nodes = [node for node in subgraph.nodes.values() 
                             if node.node_type in [NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA, NodeType.ORDER_DATA]]
        business_data_count = len(business_data_nodes)
        business_data_available = business_data_count > 0
        
        logger.info(f"🎯 Business data analysis: {business_data_count} business data nodes available")
        
        # ===== Prioritize matching business data patterns (High-Value Tasks) =====
        if business_data_available and business_data_count >= 2:
            business_data_instances = self._match_business_data_patterns(subgraph)
            instances.extend(business_data_instances)
            logger.info(f"🎯 Matched {len(business_data_instances)} business data patterns")
        
        # ===== Match general interaction patterns (Fallback Tasks) =====
        interaction_instances = []
        
        # 1. Use hierarchical graph regex engine for matching
        logger.debug(f"🎯 Using hierarchical graph regex engine for pattern matching...")
        
        # Check business data availability
        business_data_nodes = []
        for node_id, node in subgraph.nodes.items():
            if (node.node_type in [NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA, 
                                  NodeType.ORDER_DATA, NodeType.CONTENT_DATA, NodeType.FINANCIAL_DATA,
                                  NodeType.LOCATION_DATA, NodeType.TIME_DATA] or
                getattr(node, 'element_type', '') == 'business_data'):
                business_data_nodes.append((node_id, node))
        
        business_data_available = len(business_data_nodes) >= 2  # At least 2 business data nodes needed
        logger.debug(f"🎯 Business data analysis: {len(business_data_nodes)} business data nodes, available: {business_data_available}")
        
        # Use hierarchical matching strategy
        regex_matches = self.regex_engine.find_matches_by_priority(subgraph, business_data_available)
        logger.debug(f"🎯 Found {len(regex_matches)} regex matches using hierarchical strategy")
        
        # Record pattern statistics
        pattern_stats = self.regex_engine.get_pattern_statistics()
        logger.info(f"🎯 Graph Regex Pattern Statistics:")
        logger.info(f"   - Total patterns: {pattern_stats['total_patterns']}")
        logger.info(f"   - Business patterns: {pattern_stats['business_patterns']['count']} ({', '.join(pattern_stats['business_patterns']['patterns'])})")
        logger.info(f"   - General patterns: {pattern_stats['general_patterns']['count']} ({', '.join(pattern_stats['general_patterns']['patterns'])})")
        logger.info(f"   - Fallback patterns: {pattern_stats['fallback_patterns']['count']} ({', '.join(pattern_stats['fallback_patterns']['patterns'])})")
        
        # Lower quality threshold to allow more matches
        min_quality = 0.1  # Lower threshold
        
        for regex_match in regex_matches:
            # Check match quality threshold
            if regex_match.match_quality < min_quality:
                logger.debug(f"🎯 Regex match filtered out: quality={regex_match.match_quality:.2f} < {min_quality}")
                continue
                
            # Convert to metapath instance
            metapath_instance = self._convert_regex_match_to_metapath(regex_match, subgraph)
            if metapath_instance:
                interaction_instances.append(metapath_instance)
                logger.debug(f"🎯 Added regex metapath instance: {metapath_instance.pattern.name} (pattern_id: {regex_match.pattern.pattern_id})")
        
        # 2. If graph regex matching is insufficient, use traditional pattern matching as supplement
        if len(interaction_instances) < 2:  # At least 2 interaction instances needed
            logger.debug(f"🎯 Not enough regex matches ({len(interaction_instances)} < 2), trying traditional patterns...")
            traditional_instances = self._match_traditional_patterns(subgraph)
            interaction_instances.extend(traditional_instances)
            logger.debug(f"🎯 Added {len(traditional_instances)} traditional instances")
        
        instances.extend(interaction_instances)
        logger.info(f"🎯 Matched {len(interaction_instances)} interaction patterns")
        
        # Sort by priority
        instances.sort(key=lambda x: self._get_pattern_priority(x.pattern), reverse=True)
        
        logger.debug(f"🎯 Total metapath instances found: {len(instances)}")
        return instances
    
    def _match_business_data_patterns(self, subgraph: SubgraphSample) -> List[MetapathInstance]:
        """Specifically match business data related patterns"""
        instances = []
        
        # Identify business data nodes
        business_data_nodes = []
        for node_id, node in subgraph.nodes.items():
            # Check if it's a business data node
            if (node.node_type in [NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA, 
                                  NodeType.ORDER_DATA, NodeType.CONTENT_DATA, NodeType.FINANCIAL_DATA,
                                  NodeType.LOCATION_DATA, NodeType.TIME_DATA] or
                getattr(node, 'element_type', '') == 'business_data'):
                business_data_nodes.append((node_id, node))
        
        logger.debug(f"🎯 Found {len(business_data_nodes)} business data nodes")
        
        if not business_data_nodes:
            return instances
        
        # Create specific pattern instances for each business data node
        for node_id, node in business_data_nodes:
            # Select corresponding pattern based on business data type
            pattern = self._select_business_data_pattern(node)
            if pattern:
                instance = MetapathInstance(
                    pattern=pattern,
                    slot_bindings={"BD": node_id},
                    matched_nodes=[node_id],
                    matched_edges=[]
                )
                instances.append(instance)
                logger.debug(f"🎯 Created business data pattern instance for {node.node_type.value}")
        
        return instances
    
    def _select_business_data_pattern(self, node: GraphNode) -> Optional[MetapathPattern]:
        """Select corresponding pattern based on business data node"""
        node_type = node.node_type
        
        # Select pattern based on node type
        if node_type in [NodeType.USER_DATA, NodeType.PRODUCT_DATA, NodeType.ORDER_DATA]:
            # These are specific business data types, suitable for search and filter patterns
            return self.patterns.get("business_search_pattern")
        elif node_type == NodeType.BUSINESS_DATA:
            # General business data, suitable for search pattern
            return self.patterns.get("business_search_pattern")
        elif node_type in [NodeType.CONTENT_DATA, NodeType.FINANCIAL_DATA]:
            # Content data, suitable for navigation pattern
            return self.patterns.get("business_navigation_pattern")
        else:
            # Other types, use business search pattern
            return self.patterns.get("business_search_pattern")
    
    def _match_traditional_patterns(self, subgraph: SubgraphSample) -> List[MetapathInstance]:
        """Match traditional general interaction patterns"""
        instances = []
        
        # Analyze node types in subgraph
        node_type_counts = {}
        for node_id, node in subgraph.nodes.items():
            node_type = node.node_type.value
            if node_type not in node_type_counts:
                node_type_counts[node_type] = []
            node_type_counts[node_type].append(node_id)
        
        logger.debug(f"🎯 Node type distribution: {node_type_counts}")
        
        # Select most suitable pattern based on node type distribution
        if "Navigation" in node_type_counts and len(node_type_counts["Navigation"]) >= 2:
            # Multiple navigation nodes, suitable for navigation pattern
            pattern = self.patterns.get("basic_navigation_pattern")
            if pattern:
                instance = MetapathInstance(
                    pattern=pattern,
                    slot_bindings={"Navigation": node_type_counts["Navigation"][0]},
                    matched_nodes=node_type_counts["Navigation"][:2],
                    matched_edges=[]
                )
                instances.append(instance)
                logger.debug(f"🎯 Created navigation pattern instance")
        
        if "Button" in node_type_counts:
            # Has button nodes, suitable for button interaction pattern
            pattern = self.patterns.get("button_interaction_pattern")
            if pattern:
                instance = MetapathInstance(
                    pattern=pattern,
                    slot_bindings={"Button": node_type_counts["Button"][0]},
                    matched_nodes=node_type_counts["Button"][:1],
                    matched_edges=[]
                )
                instances.append(instance)
                logger.debug(f"🎯 Created button interaction pattern instance")
        
        if "SearchBox" in node_type_counts:
            # Has search box, suitable for search pattern
            pattern = self.patterns.get("search_pattern")
            if pattern:
                instance = MetapathInstance(
                    pattern=pattern,
                    slot_bindings={"S": node_type_counts["SearchBox"][0]},
                    matched_nodes=node_type_counts["SearchBox"][:1],
                    matched_edges=[]
                )
                instances.append(instance)
                logger.debug(f"🎯 Created search pattern instance")
        
        if "Page" in node_type_counts:
            # Has page nodes, suitable for content browsing pattern
            pattern = self.patterns.get("content_browsing_pattern")
            if pattern:
                instance = MetapathInstance(
                    pattern=pattern,
                    slot_bindings={"Navigation": node_type_counts.get("Navigation", [None])[0]},
                    matched_nodes=node_type_counts["Page"][:1],
                    matched_edges=[]
                )
                instances.append(instance)
                logger.debug(f"🎯 Created content browsing pattern instance")
        
        # If no specific pattern found, create general pattern
        if not instances and node_type_counts:
            # Select first available pattern as general pattern
            pattern = self.patterns.get("basic_navigation_pattern")
            if pattern:
                # Use first two nodes
                all_nodes = []
                for nodes in node_type_counts.values():
                    all_nodes.extend(nodes)
                
                if len(all_nodes) >= 2:
                    instance = MetapathInstance(
                pattern=pattern,
                        slot_bindings={},
                        matched_nodes=all_nodes[:2],
                matched_edges=[]
            )
                    instances.append(instance)
                    logger.debug(f"🎯 Created generic pattern instance")
        
        # If still no pattern found, force create a basic pattern
        if not instances:
            logger.debug(f"🎯 No patterns found, creating fallback pattern")
            # Use any available nodes to create basic navigation pattern
            all_nodes = list(subgraph.nodes.keys())
            if len(all_nodes) >= 2:
                pattern = self.patterns.get("basic_navigation_pattern")
                if pattern:
                    instance = MetapathInstance(
                        pattern=pattern,
                        slot_bindings={},
                        matched_nodes=all_nodes[:2],
                        matched_edges=[]
                    )
                    instances.append(instance)
                    logger.debug(f"🎯 Created fallback pattern instance")
        
        logger.debug(f"🎯 Created {len(instances)} traditional pattern instances")
        return instances
    
    def _node_matches_pattern_slot(self, node: GraphNode, pattern: MetapathPattern) -> bool:
        """Check if node matches pattern slot"""
        # Simple matching logic: check if node type is in pattern slots
        node_type = node.node_type.value
        
        # Check all slots in pattern
        for slot_name, slot_type in pattern.slots.items():
            if slot_type == node_type or slot_type == "Any":
                return True
        
        return False
    

    
    def _match_single_pattern(self, subgraph: SubgraphSample, pattern: MetapathPattern) -> Optional[MetapathInstance]:
        """Match single pattern"""
        # Parse pattern syntax
        parsed_pattern = self._parse_pattern_syntax(pattern.pattern_syntax)
        
        # Try to bind slots
        slot_bindings = self._bind_slots(subgraph, pattern, parsed_pattern)
        if not slot_bindings:
            return None
        
        # Validate constraint conditions
        if not self._validate_constraints(subgraph, pattern, slot_bindings):
            return None
        
        # Create instance
        return MetapathInstance(
            pattern=pattern,
            slot_bindings=slot_bindings,
            matched_nodes=list(slot_bindings.values()),
            matched_edges=self._find_matching_edges(subgraph, parsed_pattern, slot_bindings)
        )
    
    def _parse_pattern_syntax(self, syntax: str) -> List[Dict[str, Any]]:
        """Parse pattern syntax"""
        # Simplified syntax parser
        # Format: NodeType(Slot) -[EdgeType]-> NodeType(Slot)
        parsed = []
        
        # Split syntax into steps
        steps = syntax.strip().split('\n')
        
        for step in steps:
            step = step.strip()
            if not step:
                continue
            
            # Parse nodes and edges
            # Example: Page(∋SearchBox S) -[Fills]-> Query -[Controls]-> Btn
            parts = step.split('->')
            
            for i, part in enumerate(parts):
                part = part.strip()
                
                # Parse node
                node_match = re.search(r'(\w+)\(([^)]+)\)', part)
                if node_match:
                    node_type = node_match.group(1)
                    slot_info = node_match.group(2)
                    
                    # Parse slot information
                    if '∋' in slot_info:
                        # Containment relationship: ∋SearchBox S
                        slot_parts = slot_info.split('∋')
                        slot_type = slot_parts[1].split()[0]
                        slot_name = slot_parts[1].split()[1] if len(slot_parts[1].split()) > 1 else slot_type
                    else:
                        # Simple slot: S
                        slot_name = slot_info
                        slot_type = None
                    
                    parsed.append({
                        'type': 'node',
                        'node_type': node_type,
                        'slot_name': slot_name,
                        'slot_type': slot_type
                    })
                
                # Parse edge (between nodes)
                if i < len(parts) - 1:
                    edge_match = re.search(r'-\[([^\]]+)\]->', step)
                    if edge_match:
                        edge_type = edge_match.group(1)
                        parsed.append({
                            'type': 'edge',
                            'edge_type': edge_type
                        })
        
        return parsed
    
    def _bind_slots(self, subgraph: SubgraphSample, pattern: MetapathPattern, parsed_pattern: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """Bind slots to actual nodes (simplified version)"""
        slot_bindings = {}
        
        # Simplified binding logic: consider it matchable if subgraph contains required node types
        for slot_name, required_type in pattern.slots.items():
            # Find nodes of matching type
            available_nodes = []
            for node_id, node in subgraph.nodes.items():
                if node.node_type.value == required_type:
                    available_nodes.append(node_id)
            
            if available_nodes:
                # Select first available node
                slot_bindings[slot_name] = available_nodes[0]
            else:
                # If no exact match found, try using similar types
                # For example: Navigation can be replaced by Button
                fallback_mapping = {
                    "Navigation": ["Button", "SearchBox"],
                    "Button": ["Navigation"],
                    "SearchBox": ["Input", "Button"]
                }
                
                fallback_types = fallback_mapping.get(required_type, [])
                for fallback_type in fallback_types:
                    for node_id, node in subgraph.nodes.items():
                        if node.node_type.value == fallback_type:
                            slot_bindings[slot_name] = node_id
                            break
                    if slot_name in slot_bindings:
                        break
        
        # If at least one slot is bound, consider it a successful match
        return slot_bindings if slot_bindings else None
    
    def _analyze_subgraph_structure(self, subgraph: SubgraphSample) -> Dict[str, Any]:
        """Analyze subgraph structure"""
        analysis = {
            "node_types": defaultdict(list),
            "edge_types": defaultdict(list),
            "interactive_elements": [],
            "navigation_paths": [],
            "form_elements": [],
            "search_elements": []
        }
        
        # Analyze node types
        for node_id, node in subgraph.nodes.items():
            analysis["node_types"][node.node_type.value].append(node_id)
            
            # Classify interactive elements
            if node.node_type in [NodeType.BUTTON, NodeType.NAVIGATION]:
                analysis["interactive_elements"].append(node_id)
            elif node.node_type == NodeType.SEARCH_BOX:
                analysis["search_elements"].append(node_id)
            elif node.node_type == NodeType.INPUT:
                analysis["form_elements"].append(node_id)
        
        # Analyze edge types and navigation paths
        for edge_id, edge in subgraph.edges.items():
            analysis["edge_types"][edge.edge_type.value].append(edge_id)
            
            if edge.edge_type == EdgeType.NAV_TO:
                analysis["navigation_paths"].append({
                    "source": edge.source_node_id,
                    "target": edge.target_node_id,
                    "edge_id": edge_id
                })
        
        return analysis
    
    
    def _generate_navigation_patterns(self, subgraph: SubgraphSample, analysis: Dict[str, Any]) -> List[MetapathInstance]:
        """Generate navigation patterns"""
        instances = []
        
        # Quality check: ensure sufficient navigation elements
        if len(analysis["navigation_paths"]) < 1 or len(analysis["interactive_elements"]) < 2:
            return instances
        
        # Generate multi-hop navigation patterns
        for i, path in enumerate(analysis["navigation_paths"][:3]):  # Limit quantity
            # Create dynamic pattern
            pattern = MetapathPattern(
                pattern_id=f"nav_pattern_{i}",
                name=f"Navigation Pattern {i+1}",
                description="Multi-hop navigation through interactive elements",
                pattern_syntax=f"Navigation({path['source']}) -[NavTo]-> Navigation({path['target']})",
                slots={
                    "source": "Navigation",
                    "target": "Navigation"
                },
                seed_type=TaskSeedType.MULTI_HOP_NAVIGATION
            )
            
            # Create instance
            slot_bindings = {
                "source": path["source"],
                "target": path["target"]
            }
            
            instance = MetapathInstance(
                pattern=pattern,
                slot_bindings=slot_bindings,
                matched_nodes=[path["source"], path["target"]],
                matched_edges=[path["edge_id"]]
            )
            
            instances.append(instance)
        
        return instances
    
    def _generate_browsing_patterns(self, subgraph: SubgraphSample, analysis: Dict[str, Any]) -> List[MetapathInstance]:
        """Generate content browsing patterns"""
        instances = []
        
        # Quality check: ensure interactive elements exist
        if len(analysis["interactive_elements"]) < 2:
            return instances
        
        # Generate browsing patterns: click interactive elements
        interactive_nodes = analysis["interactive_elements"][:3]  # Limit quantity
        
        for i, node_id in enumerate(interactive_nodes):
            pattern = MetapathPattern(
                pattern_id=f"browse_pattern_{i}",
                name=f"Content Browsing Pattern {i+1}",
                description="Browse content by interacting with elements",
                pattern_syntax=f"Page -[Contains]-> Interactive({node_id})",
                slots={
                    "interactive": "Button"
                },
                seed_type=TaskSeedType.CONTENT_BROWSING
            )
            
            slot_bindings = {"interactive": node_id}
            
            instance = MetapathInstance(
                pattern=pattern,
                slot_bindings=slot_bindings,
                matched_nodes=[node_id],
                matched_edges=[]
            )
            
            instances.append(instance)
        
        return instances
    
    def _generate_form_patterns(self, subgraph: SubgraphSample, analysis: Dict[str, Any]) -> List[MetapathInstance]:
        """Generate form patterns"""
        instances = []
        
        # Quality check: ensure form elements exist
        if len(analysis["form_elements"]) < 1 or len(analysis["interactive_elements"]) < 1:
            return instances
        
        # Generate form filling patterns
        for i, form_node in enumerate(analysis["form_elements"][:2]):
            for j, button_node in enumerate(analysis["interactive_elements"][:2]):
                pattern = MetapathPattern(
                    pattern_id=f"form_pattern_{i}_{j}",
                    name=f"Form Pattern {i+1}-{j+1}",
                    description="Fill form and submit",
                    pattern_syntax=f"Input({form_node}) -[Controls]-> Button({button_node})",
                    slots={
                        "input": "Input",
                        "button": "Button"
                    },
                    seed_type=TaskSeedType.BUTTON_INTERACTION
                )
                
                slot_bindings = {
                    "input": form_node,
                    "button": button_node
                }
                
                instance = MetapathInstance(
                    pattern=pattern,
                    slot_bindings=slot_bindings,
                    matched_nodes=[form_node, button_node],
                    matched_edges=[]
                )
                
                instances.append(instance)
        
        return instances
    
    def _generate_search_patterns(self, subgraph: SubgraphSample, analysis: Dict[str, Any]) -> List[MetapathInstance]:
        """Generate search patterns"""
        instances = []
        
        # Quality check: ensure search elements exist
        if len(analysis["search_elements"]) < 1:
            return instances
        
        # Generate search patterns
        for i, search_node in enumerate(analysis["search_elements"]):
            pattern = MetapathPattern(
                pattern_id=f"search_pattern_{i}",
                name=f"Search Pattern {i+1}",
                description="Search functionality",
                pattern_syntax=f"SearchBox({search_node}) -[Fills]-> Query",
                slots={
                    "search": "SearchBox"
                },
                seed_type=TaskSeedType.BUSINESS_SEARCH_FILTER
            )
            
            slot_bindings = {"search": search_node}
            
            instance = MetapathInstance(
                pattern=pattern,
                slot_bindings=slot_bindings,
                matched_nodes=[search_node],
                matched_edges=[]
            )
            
            instances.append(instance)
        
        return instances
    
    def _validate_constraints(self, subgraph: SubgraphSample, pattern: MetapathPattern, slot_bindings: Dict[str, str]) -> bool:
        """Validate constraint conditions (quality-oriented)"""
        if len(slot_bindings) == 0:
            return False
        
        # Quality check 1: ensure bound nodes are valid
        for slot_name, node_id in slot_bindings.items():
            if node_id not in subgraph.nodes:
                return False
            
            node = subgraph.nodes[node_id]
            
            # Quality check 2: ensure node is interactive
            if not self._is_node_interactive(node):
                return False
        
        # Quality check 3: ensure pattern has sufficient complexity
        if len(slot_bindings) < 1:
            return False
        
        return True
    
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
    
    def _evaluate_constraint(self, subgraph: SubgraphSample, constraint: str, slot_bindings: Dict[str, str]) -> bool:
        """Evaluate single constraint condition"""
        # Simplified constraint evaluator
        if "must be visible" in constraint:
            # Check if node is visible
            for slot_name, node_id in slot_bindings.items():
                if slot_name in constraint:
                    node = subgraph.nodes[node_id]
                    if not node.metadata.is_visible:
                        return False
        
        elif "must be enabled" in constraint:
            # Check if node is enabled
            for slot_name, node_id in slot_bindings.items():
                if slot_name in constraint:
                    node = subgraph.nodes[node_id]
                    if not node.metadata.is_enabled:
                        return False
        
        elif "must be reachable" in constraint:
            # Check reachability (simplified version)
            # More complex reachability checks can be implemented here
            pass
        
        return True
    
    def _find_matching_edges(self, subgraph: SubgraphSample, parsed_pattern: List[Dict[str, Any]], slot_bindings: Dict[str, str]) -> List[str]:
        """Find matching edges"""
        matching_edges = []
        
        for item in parsed_pattern:
            if item['type'] == 'edge':
                edge_type = item['edge_type']
                
                # Find edges of matching type
                for edge_id, edge in subgraph.edges.items():
                    if edge.edge_type.value == edge_type:
                        matching_edges.append(edge_id)
        
        return matching_edges
    
    def generate_task_from_instance(self, instance: MetapathInstance, subgraph: SubgraphSample) -> Dict[str, Any]:
        """Generate task from metapath instance"""
        task = {
            "task_id": f"task_{instance.pattern.pattern_id}",
            "task_type": instance.pattern.seed_type.value,
            "pattern": instance.pattern.name,
            "description": instance.pattern.description,
            "steps": [],
            "slot_bindings": instance.slot_bindings
        }
        
        # Generate steps based on pattern
        if instance.pattern.pattern_id == "search_pattern":
            task["steps"] = self._generate_search_steps(instance, subgraph)
        elif instance.pattern.pattern_id == "button_pattern":
            task["steps"] = self._generate_button_steps(instance, subgraph)
        elif instance.pattern.pattern_id == "navigation_pattern":
            task["steps"] = self._generate_navigation_steps(instance, subgraph)
        elif instance.pattern.pattern_id == "business_search_pattern":
            task["steps"] = self._generate_search_steps(instance, subgraph)
        elif instance.pattern.pattern_id == "business_navigation_pattern":
            task["steps"] = self._generate_navigation_steps(instance, subgraph)

        
        return task
    
    def _generate_search_steps(self, instance: MetapathInstance, subgraph: SubgraphSample) -> List[Dict[str, Any]]:
        """Generate steps for search tasks"""
        steps = []
        
        # Step 1: Input in search box
        if "S" in instance.slot_bindings:
            search_node = subgraph.nodes[instance.slot_bindings["S"]]
            steps.append({
                "step_type": "input",
                "target_som_mark": search_node.metadata.som_mark,
                "action_description": f"Input search term in {search_node.metadata.text_content or 'search box'}",
                "input_value": "search_term"
            })
        
        # Step 2: Click search button
        if "Btn" in instance.slot_bindings:
            btn_node = subgraph.nodes[instance.slot_bindings["Btn"]]
            steps.append({
                "step_type": "click",
                "target_som_mark": btn_node.metadata.som_mark,
                "action_description": f"Click {btn_node.metadata.text_content or 'search button'}"
            })
        
        # Step 3: Apply filter
        if "F" in instance.slot_bindings:
            filter_node = subgraph.nodes[instance.slot_bindings["F"]]
            steps.append({
                "step_type": "click",
                "target_som_mark": filter_node.metadata.som_mark,
                "action_description": f"Apply filter: {filter_node.metadata.text_content}"
            })
        
        # Step 4: View details
        if "D" in instance.slot_bindings:
            detail_node = subgraph.nodes[instance.slot_bindings["D"]]
            steps.append({
                "step_type": "click",
                "target_som_mark": detail_node.metadata.som_mark,
                "action_description": f"View details of {detail_node.metadata.text_content}"
            })
        
        return steps
    
    def _generate_button_steps(self, instance: MetapathInstance, subgraph: SubgraphSample) -> List[Dict[str, Any]]:
        """Generate steps for button interaction tasks"""
        steps = []
        
        # Step 1: Click button
        if "B" in instance.slot_bindings:
            button_node = subgraph.nodes[instance.slot_bindings["B"]]
            steps.append({
                "step_type": "click",
                "target_som_mark": button_node.metadata.som_mark,
                "action_description": f"Click {button_node.metadata.text_content or 'button'}"
            })
        
        # Step 2: Verify response
        steps.append({
            "step_type": "verify",
            "target_som_mark": "",
            "action_description": "Verify the button response (page change, modal, toast, etc.)"
        })
        
        return steps
    
    
    def _generate_navigation_steps(self, instance: MetapathInstance, subgraph: SubgraphSample) -> List[Dict[str, Any]]:
        """Generate steps for navigation tasks"""
        steps = []
        
        # Step 1: Click menu
        if "Menu" in instance.slot_bindings:
            menu_node = subgraph.nodes[instance.slot_bindings["Menu"]]
            steps.append({
                "step_type": "click",
                "target_som_mark": menu_node.metadata.som_mark,
                "action_description": f"Navigate to {menu_node.metadata.text_content}"
            })
        
        # Step 2: Select section
        if "Section" in instance.slot_bindings:
            section_node = subgraph.nodes[instance.slot_bindings["Section"]]
            steps.append({
                "step_type": "click",
                "target_som_mark": section_node.metadata.som_mark,
                "action_description": f"Select section: {section_node.metadata.text_content}"
            })
        
        # Step 3: Select subsection
        if "Subsection" in instance.slot_bindings:
            subsection_node = subgraph.nodes[instance.slot_bindings["Subsection"]]
            steps.append({
                "step_type": "click",
                "target_som_mark": subsection_node.metadata.som_mark,
                "action_description": f"Select subsection: {subsection_node.metadata.text_content}"
            })
        
        # Step 4: View item
        if "Item" in instance.slot_bindings:
            item_node = subgraph.nodes[instance.slot_bindings["Item"]]
            steps.append({
                "step_type": "click",
                "target_som_mark": item_node.metadata.som_mark,
                "action_description": f"View item: {item_node.metadata.text_content}"
            })
        
        return steps
    
    
    def _convert_regex_match_to_metapath(self, regex_match: GraphRegexMatch, subgraph: SubgraphSample) -> Optional[MetapathInstance]:
        """Convert graph regex match to metapath instance"""
        try:
            # Create corresponding metapath pattern
            pattern = MetapathPattern(
                pattern_id=regex_match.pattern.pattern_id,
                name=regex_match.pattern.name,
                description=regex_match.pattern.description,
                pattern_syntax=regex_match.pattern.regex_syntax,
                slots=regex_match.variable_bindings,
                seed_type=regex_match.pattern.seed_type
            )
            
            # Create metapath instance
            instance = MetapathInstance(
                pattern=pattern,
                slot_bindings=regex_match.variable_bindings,
                matched_nodes=regex_match.matched_nodes,
                matched_edges=regex_match.matched_edges
            )
            
            # Add quality score
            instance.quality_score = regex_match.match_quality
            
            return instance
            
        except Exception as e:
            # If conversion fails, return None
            return None
    
    def _match_traditional_patterns(self, subgraph: SubgraphSample) -> List[MetapathInstance]:
        """Use traditional method to match patterns (fallback mechanism)"""
        instances = []
        
        for pattern in self.patterns.values():
            # Check if pattern is suitable for subgraph's seed type
            if pattern.seed_type != subgraph.task_seed.seed_type:
                continue
            
            # Try to match pattern
            instance = self._match_single_pattern(subgraph, pattern)
            if instance:
                instances.append(instance)
        
        return instances
    
    def _deduplicate_instances(self, instances: List[MetapathInstance]) -> List[MetapathInstance]:
        """Deduplicate metapath instances"""
        unique_instances = []
        seen_patterns = set()
        
        for instance in instances:
            # Use pattern ID and matched nodes as unique identifier
            pattern_signature = (
                instance.pattern.pattern_id,
                tuple(sorted(instance.matched_nodes))
            )
            
            if pattern_signature not in seen_patterns:
                seen_patterns.add(pattern_signature)
                unique_instances.append(instance)
        
        return unique_instances
    
    def get_pattern_for_seed_type(self, seed_type: TaskSeedType, subgraph: SubgraphSample) -> Optional[MetapathInstance]:
        """Get best pattern match for specific seed type"""
        # Use graph regex engine
        best_match = self.regex_engine.get_best_match_for_seed(subgraph, seed_type)
        
        if best_match:
            return self._convert_regex_match_to_metapath(best_match, subgraph)
        
        # Fallback to traditional matching
        for pattern in self.patterns.values():
            if pattern.seed_type == seed_type:
                instance = self._match_single_pattern(subgraph, pattern)
                if instance:
                    return instance
        
        return None
    
    def _get_pattern_priority(self, pattern: MetapathPattern) -> int:
        """Get pattern priority - hierarchical priority allocation"""
        pattern_id = pattern.pattern_id
        
        # Business data patterns - layered by value
        if pattern_id == "business_search_pattern":
            return 10  # Search related - most valuable
        elif pattern_id == "business_navigation_pattern":
            return 9   # Business navigation - high value
        
        # General interaction patterns - unified priority
        else:
            return 5   # Standard priority for interaction patterns
    
