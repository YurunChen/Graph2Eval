"""
Task Seeds - Task Prototype Seed Definitions
Define reusable micro-patterns that cover real UI habits, generating naturally specific and stable tasks
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from loguru import logger
from graph_rag.node_types import NodeType
from graph_rag.edge_types import EdgeType
from graph_rag.graph_builder import TaskGraph

class TaskSeedType(Enum):
    """Task seed types - Updated Classification Design

    BUSINESS DATA SEEDS: Require actual business data for meaningful execution
    INTERACTION SEEDS: Pure UI interaction tasks that work with any elements
    """
    
    # ===== BUSINESS DATA SEEDS =====
    # Only navigation and search related business seeds
    BUSINESS_SEARCH_FILTER = "business_search_filter"
    BUSINESS_NAVIGATION = "business_navigation"
    # Specific data type navigation seeds
    USER_NAVIGATION = "user_navigation"
    PRODUCT_NAVIGATION = "product_navigation"
    ORDER_NAVIGATION = "order_navigation"
    MIXED_DATA_NAVIGATION = "mixed_data_navigation"
    MULTI_HOP_NAVIGATION = "multi_hop_navigation"

    # ===== INTERACTION SEEDS =====
    # Pure UI interaction seeds - work with any elements (no business data required)
    CONTENT_BROWSING = "content_browsing"
    BASIC_NAVIGATION = "basic_navigation"
    BUTTON_INTERACTION = "button_interaction"
    MENU_EXPLORATION = "menu_exploration"
    TAB_SWITCHING = "tab_switching"
    MODAL_INTERACTION = "modal_interaction"
    TOAST_NOTIFICATION = "toast_notification"
    BREADCRUMB_NAVIGATION = "breadcrumb_navigation"
    PAGINATION_BROWSING = "pagination_browsing"
    EXPAND_COLLAPSE = "expand_collapse"
    SCROLL_READING = "scroll_reading"

@dataclass
class TaskSeedPattern:
    """Task seed pattern - defines the structural pattern of tasks (supports prototype families)"""
    seed_type: TaskSeedType
    name: str
    description: str
    
    # Seed classification (Dual Seeds Design)
    seed_category: str = "interaction"  # "business" or "interaction"
    priority: int = 1  # Higher number = higher priority (business seeds have higher priority)
    
    # Core slots (required)
    core_slots: Dict[str, NodeType] = field(default_factory=dict)

    # Flexible core slots (alternative node types for better web data compatibility)
    flexible_core_slots: Set[NodeType] = field(default_factory=set)

    # Optional slots (optional)
    optional_slots: Dict[str, NodeType] = field(default_factory=dict)
    
    # Required edge types
    required_edge_types: Set[EdgeType] = field(default_factory=set)
    
    # Optional edge types
    optional_edge_types: Set[EdgeType] = field(default_factory=set)
    
    # Pattern syntax (supports regex graph patterns)
    pattern_syntax: str = ""
    
    # Fallback version definitions
    fallback_versions: List[str] = field(default_factory=list)
    
    # Task type mapping
    task_type: str = ""
    
    # Difficulty level
    difficulty: str = "MEDIUM"
    
    # Applicable website types
    applicable_website_types: Set[str] = field(default_factory=set)
    
    # Min/max number of steps
    min_steps: int = 2
    max_steps: int = 6
    
    # Semantic completion hints
    semantic_completion_hints: List[str] = field(default_factory=list)
    
    # Business data requirements
    requires_business_data: bool = False
    min_business_data_elements: int = 0
    
    # Fallback to interaction seed
    can_fallback_to_interaction: bool = True

class TaskSeedLibrary:
    """Task seed library"""
    
    def __init__(self):
        self.seeds = self._initialize_seeds()
    
    def _initialize_seeds(self) -> Dict[TaskSeedType, TaskSeedPattern]:
        """Initialize task seed library with Dual Seeds Design"""
        seeds = {}
        
        # ===== BUSINESS DATA SEEDS (High-Value Tasks) =====
        
        # 1. Business Search Filter - requires actual business data
        seeds[TaskSeedType.BUSINESS_SEARCH_FILTER] = TaskSeedPattern(
            seed_type=TaskSeedType.BUSINESS_SEARCH_FILTER,
            name="Business Search Filter",
            description="Search for specific business data and apply filters",
            seed_category="business",
            priority=10,  # High priority
            core_slots={
                "search_input": NodeType.SEARCH_BOX,
                "submit_button": NodeType.BUTTON,  # 强制要求搜索确认按钮
                "business_data": NodeType.BUSINESS_DATA,
                "result_list": NodeType.CONTENT
            },
            # 增强对实际web数据的兼容性
            flexible_core_slots={
                NodeType.INPUT,  # 兼容search input
                NodeType.BUTTON, # 兼容search button
                NodeType.NAVIGATION, # 兼容导航菜单
                NodeType.MENU # 兼容下拉菜单
            },
            optional_slots={
                "sort_control": NodeType.FORM,
                "pagination": NodeType.NAVIGATION
            },
            required_edge_types={EdgeType.FILLS, EdgeType.CONTROLS, EdgeType.OPENS},
            task_type="business_search_filter",
            difficulty="MEDIUM",
            requires_business_data=True,
            min_business_data_elements=3,
            can_fallback_to_interaction=True,
            semantic_completion_hints=[
                "Search for specific business entities (names, IDs, amounts)",
                "Apply filters based on business criteria",
                "Validate search results contain expected business data"
            ]
        )
        
        
        # 3. Business Form Filling - fill forms with business data

        
        
        # 5. Business Navigation - navigate using business context
        seeds[TaskSeedType.BUSINESS_NAVIGATION] = TaskSeedPattern(
            seed_type=TaskSeedType.BUSINESS_NAVIGATION,
            name="Business Navigation",
            description="Navigate using business-specific context and data",
            seed_category="business",
            priority=8,
            core_slots={
                "navigation_element": NodeType.NAVIGATION,
                "target_page": NodeType.PAGE
            },
            flexible_core_slots={
                NodeType.LINK,        # 兼容导航链接
                NodeType.BUTTON,      # 兼容导航按钮
                NodeType.MENU,        # 兼容下拉菜单
                NodeType.BUSINESS_DATA # 兼容业务数据导航
            },
            optional_slots={
                "breadcrumb": NodeType.BREADCRUMB,
                "context_menu": NodeType.MENU,
                "business_context": [NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA, NodeType.ORDER_DATA],
                "additional_business_data": [NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA, NodeType.ORDER_DATA]
            },
            required_edge_types={EdgeType.NAV_TO, EdgeType.OPENS},
            task_type="navigation_task",
            difficulty="EASY",
            requires_business_data=False,
            min_business_data_elements=0,
            can_fallback_to_interaction=True,
            semantic_completion_hints=[
                "Navigate using navigation elements to reach target pages",
                "Use business context when available to enhance navigation",
                "Validate navigation leads to relevant content"
            ]
        )
        
        # 6. User Data Navigation - navigate using user context
        seeds[TaskSeedType.USER_NAVIGATION] = TaskSeedPattern(
            seed_type=TaskSeedType.USER_NAVIGATION,
            name="User Navigation",
            description="Navigate using user-specific context and data",
            seed_category="business",
            priority=7,
            core_slots={
                "navigation_element": NodeType.NAVIGATION,
                "user_context": [NodeType.USER_DATA, NodeType.BUSINESS_DATA],
                "target_page": NodeType.PAGE
            },
            optional_slots={
                "breadcrumb": NodeType.NAVIGATION,
                "context_menu": NodeType.MENU,
                "additional_user_data": [NodeType.USER_DATA, NodeType.BUSINESS_DATA, NodeType.PRODUCT_DATA]
            },
            required_edge_types={EdgeType.NAV_TO, EdgeType.OPENS},
            task_type="navigation_task",
            difficulty="EASY",
            requires_business_data=True,
            min_business_data_elements=1,
            can_fallback_to_interaction=True,
            semantic_completion_hints=[
                "Navigate based on user context (contacts, leads, accounts)",
                "Use user data or related business data to determine navigation path",
                "Validate navigation leads to relevant user or business content"
            ]
        )
        
        # 7. Product Data Navigation - navigate using product context
        seeds[TaskSeedType.PRODUCT_NAVIGATION] = TaskSeedPattern(
            seed_type=TaskSeedType.PRODUCT_NAVIGATION,
            name="Product Navigation",
            description="Navigate using product-specific context and data",
            seed_category="business",
            priority=7,
            core_slots={
                "navigation_element": NodeType.NAVIGATION,
                "product_context": [NodeType.PRODUCT_DATA, NodeType.BUSINESS_DATA, NodeType.ORDER_DATA],
                "target_page": NodeType.PAGE
            },
            optional_slots={
                "breadcrumb": NodeType.NAVIGATION,
                "context_menu": NodeType.MENU,
                "additional_product_data": [NodeType.PRODUCT_DATA, NodeType.BUSINESS_DATA, NodeType.ORDER_DATA, NodeType.USER_DATA]
            },
            required_edge_types={EdgeType.NAV_TO, EdgeType.OPENS},
            task_type="navigation_task",
            difficulty="EASY",
            requires_business_data=True,
            min_business_data_elements=1,
            can_fallback_to_interaction=True,
            semantic_completion_hints=[
                "Navigate based on product context (catalog, inventory, features)",
                "Use product data or related business/order data to determine navigation path",
                "Validate navigation leads to relevant product or business content"
            ]
        )
        
        # 8. Order Data Navigation - navigate using order context
        seeds[TaskSeedType.ORDER_NAVIGATION] = TaskSeedPattern(
            seed_type=TaskSeedType.ORDER_NAVIGATION,
            name="Order Navigation",
            description="Navigate using order-specific context and data",
            seed_category="business",
            priority=7,
            core_slots={
                "navigation_element": NodeType.NAVIGATION,
                "order_context": [NodeType.ORDER_DATA, NodeType.BUSINESS_DATA, NodeType.USER_DATA],
                "target_page": NodeType.PAGE
            },
            optional_slots={
                "breadcrumb": NodeType.NAVIGATION,
                "context_menu": NodeType.MENU,
                "additional_order_data": [NodeType.ORDER_DATA, NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA]
            },
            required_edge_types={EdgeType.NAV_TO, EdgeType.OPENS},
            task_type="navigation_task",
            difficulty="EASY",
            requires_business_data=True,
            min_business_data_elements=1,
            can_fallback_to_interaction=True,
            semantic_completion_hints=[
                "Navigate based on order context (transactions, invoices, shipping)",
                "Use order data or related business/user data to determine navigation path",
                "Validate navigation leads to relevant order or business content"
            ]
        )
        
        # 9. Mixed Data Navigation - navigate using multiple data types
        seeds[TaskSeedType.MIXED_DATA_NAVIGATION] = TaskSeedPattern(
            seed_type=TaskSeedType.MIXED_DATA_NAVIGATION,
            name="Mixed Data Navigation",
            description="Navigate using multiple data types for comprehensive context",
            seed_category="business",
            priority=6,
            core_slots={
                "navigation_element": NodeType.NAVIGATION,
                "primary_data": [NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA, NodeType.ORDER_DATA],
                "secondary_data": [NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA, NodeType.ORDER_DATA],
                "target_page": NodeType.PAGE
            },
            optional_slots={
                "breadcrumb": NodeType.NAVIGATION,
                "context_menu": NodeType.MENU,
                "tertiary_data": [NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA, NodeType.ORDER_DATA],
                "supporting_data": [NodeType.BUSINESS_DATA, NodeType.USER_DATA, NodeType.PRODUCT_DATA, NodeType.ORDER_DATA]
            },
            required_edge_types={EdgeType.NAV_TO, EdgeType.OPENS},
            task_type="navigation_task",
            difficulty="MEDIUM",
            requires_business_data=True,
            min_business_data_elements=2,
            can_fallback_to_interaction=True,
            semantic_completion_hints=[
                "Navigate using multiple data types for rich context",
                "Combine business, user, product, and order data for navigation",
                "Validate navigation leads to comprehensive content view",
                "Support flexible data type combinations for complex navigation scenarios"
            ]
        )
        
        # ===== INTERACTION SEEDS (Fallback Tasks) =====

        
        
        
        # 4. Multi-hop Navigation pattern - 移到业务数据类，因为导航需要业务数据
        seeds[TaskSeedType.MULTI_HOP_NAVIGATION] = TaskSeedPattern(
            seed_type=TaskSeedType.MULTI_HOP_NAVIGATION,
            name="Multi-hop Navigation",
            description="Navigate through multiple levels or sections with business context",
            seed_category="business",  # 移到业务数据类
            core_slots={
                "nav": NodeType.NAVIGATION,
                "button": NodeType.BUTTON
            },
            optional_slots={
                "business_data": NodeType.BUSINESS_DATA,  # 添加业务数据支持
                "user_data": NodeType.USER_DATA,
                "product_data": NodeType.PRODUCT_DATA,
                "order_data": NodeType.ORDER_DATA,
                "content_data": NodeType.CONTENT_DATA
            },
            optional_edge_types={},
            required_edge_types={
                EdgeType.NAV_TO,
                EdgeType.CONTAINS
            },
            pattern_syntax="""
            Menu -[NavTo]-> Section -[NavTo]-> Subsection -[Opens]-> Item
            """,
            task_type="Navigation",
            difficulty="MEDIUM",
            applicable_website_types={"crm", "portal", "ecommerce", "blog"},
            min_steps=2,
            max_steps=5
        )

        
        
        # 7. Content Browsing pattern
        seeds[TaskSeedType.CONTENT_BROWSING] = TaskSeedPattern(
            seed_type=TaskSeedType.CONTENT_BROWSING,
            name="Content Browsing",
            description="Browse through content sections and pages",
            seed_category="interaction",
            core_slots={
                "nav": NodeType.NAVIGATION,
                "button": NodeType.BUTTON
            },
            optional_slots={},
            optional_edge_types={},
            required_edge_types={
                EdgeType.NAV_TO,
                EdgeType.CONTAINS
            },
            pattern_syntax="""
            Page(Content C) -[NavTo]-> Section -[Opens]-> Article
            """,
            task_type="Content Browsing",
            difficulty="EASY",
            applicable_website_types={"blog", "portal", "ecommerce", "crm"},
            min_steps=2,
            max_steps=4
        )
        
        # 8. Basic Navigation pattern - More flexible basic navigation tasks
        seeds[TaskSeedType.BASIC_NAVIGATION] = TaskSeedPattern(
            seed_type=TaskSeedType.BASIC_NAVIGATION,
            name="Basic Navigation",
            description="Basic navigation tasks, only requires navigation elements",
            seed_category="interaction",
            core_slots={
                "nav": NodeType.NAVIGATION
            },
            required_edge_types={
                EdgeType.NAV_TO
            },
            pattern_syntax="""
            Page -[NavTo]-> TargetPage
            """,
            task_type="Navigation",
            difficulty="EASY",
            applicable_website_types={"blog", "portal", "ecommerce", "crm", "content"},
            min_steps=1,
            max_steps=3
        )
        
        # 9. Button Interaction pattern - Basic button interaction
        seeds[TaskSeedType.BUTTON_INTERACTION] = TaskSeedPattern(
            seed_type=TaskSeedType.BUTTON_INTERACTION,
            name="Button Interaction",
            description="Basic button interaction tasks",
            seed_category="interaction",
            core_slots={
                "button": NodeType.BUTTON
            },
            optional_slots={},
            required_edge_types={
                EdgeType.CONTAINS
            },
            optional_edge_types={},
            pattern_syntax="""
            Page -[Contains]-> Button -[Controls]-> Action
            """,
            fallback_versions=["Button Click"],
            task_type="Interaction",
            difficulty="EASY",
            applicable_website_types={"blog", "portal", "ecommerce", "crm", "content"},
            min_steps=1,
            max_steps=2,
            semantic_completion_hints=[
                "Any clickable button can generate interaction tasks"
            ]
        )
        
        
        # 11. Menu Exploration pattern - Menu exploration
        seeds[TaskSeedType.MENU_EXPLORATION] = TaskSeedPattern(
            seed_type=TaskSeedType.MENU_EXPLORATION,
            name="Menu Exploration",
            description="Explore menu structure, click menu items to view content",
            seed_category="interaction",
            core_slots={
                "nav": NodeType.NAVIGATION,
                "button": NodeType.BUTTON
            },
            optional_slots={
                "dropdown": NodeType.DROPDOWN,
                "submenu": NodeType.SUBMENU
            },
            required_edge_types={
                EdgeType.NAV_TO,
                EdgeType.CONTAINS
            },
            pattern_syntax="""
            Menu -[Contains]-> MenuItem -[NavTo]-> SubMenu -[Opens]-> Content
            """,
            task_type="Navigation",
            difficulty="EASY",
            applicable_website_types={"portal", "ecommerce", "crm", "blog"},
            min_steps=2,
            max_steps=4
        )
        
        # 12. Tab Switching pattern - Tab switching
        seeds[TaskSeedType.TAB_SWITCHING] = TaskSeedPattern(
            seed_type=TaskSeedType.TAB_SWITCHING,
            name="Tab Switching",
            description="Switch between different tabs to view content",
            seed_category="interaction",
            core_slots={
                "tab": NodeType.TAB,
                "content": NodeType.CONTENT
            },
            optional_slots={
                "tab_container": NodeType.TAB_CONTAINER
            },
            required_edge_types={
                EdgeType.CONTAINS,
                EdgeType.OPENS
            },
            pattern_syntax="""
            TabContainer -[Contains]-> Tab -[Opens]-> Content
            """,
            task_type="Navigation",
            difficulty="EASY",
            applicable_website_types={"portal", "ecommerce", "crm", "blog"},
            min_steps=2,
            max_steps=3
        )
        
        # 13. Modal Interaction pattern - Modal interaction
        seeds[TaskSeedType.MODAL_INTERACTION] = TaskSeedPattern(
            seed_type=TaskSeedType.MODAL_INTERACTION,
            name="Modal Interaction",
            description="Interact with modal dialogs and popups",
            seed_category="interaction",
            core_slots={
                "button": NodeType.BUTTON,
                "modal": NodeType.MODAL
            },
            optional_slots={
                "input": NodeType.INPUT,
                "close_btn": NodeType.BUTTON
            },
            required_edge_types={
                EdgeType.OPENS,
                EdgeType.CONTAINS
            },
            pattern_syntax="""
            Button -[Opens]-> Modal -[Contains]-> {Input*} -[Controls]-> Close
            """,
            task_type="Interaction",
            difficulty="MEDIUM",
            applicable_website_types={"portal", "ecommerce", "crm", "blog"},
            min_steps=2,
            max_steps=4
        )
        
        # 14. Toast Notification pattern - Notification interaction
        seeds[TaskSeedType.TOAST_NOTIFICATION] = TaskSeedPattern(
            seed_type=TaskSeedType.TOAST_NOTIFICATION,
            name="Toast Notification",
            description="Trigger and view notification messages",
            seed_category="interaction",
            core_slots={
                "button": NodeType.BUTTON,
                "toast": NodeType.TOAST
            },
            optional_slots={
                "notification_area": NodeType.NOTIFICATION_AREA
            },
            required_edge_types={
                EdgeType.CONTROLS,
                EdgeType.CONTAINS
            },
            pattern_syntax="""
            Button -[Controls]-> Action -[Contains]-> Toast
            """,
            task_type="Interaction",
            difficulty="EASY",
            applicable_website_types={"portal", "ecommerce", "crm", "blog"},
            min_steps=2,
            max_steps=3
        )
        
        # 15. Breadcrumb Navigation pattern - Breadcrumb navigation
        seeds[TaskSeedType.BREADCRUMB_NAVIGATION] = TaskSeedPattern(
            seed_type=TaskSeedType.BREADCRUMB_NAVIGATION,
            name="Breadcrumb Navigation",
            description="Use breadcrumb navigation to return to parent pages",
            seed_category="interaction",
            core_slots={
                "breadcrumb": NodeType.BREADCRUMB,
                "link": NodeType.LINK
            },
            optional_slots={
                "nav": NodeType.NAVIGATION
            },
            required_edge_types={
                EdgeType.NAV_TO,
                EdgeType.CONTAINS
            },
            pattern_syntax="""
            Breadcrumb -[Contains]-> Link -[NavTo]-> ParentPage
            """,
            task_type="Navigation",
            difficulty="EASY",
            applicable_website_types={"portal", "ecommerce", "crm", "blog"},
            min_steps=2,
            max_steps=3
        )
        
        # 16. Pagination Browsing pattern - Pagination browsing
        seeds[TaskSeedType.PAGINATION_BROWSING] = TaskSeedPattern(
            seed_type=TaskSeedType.PAGINATION_BROWSING,
            name="Pagination Browsing",
            description="Browse paginated content, navigate through pages",
            seed_category="interaction",
            core_slots={
                "paginator": NodeType.PAGINATOR,
                "content": NodeType.CONTENT
            },
            optional_slots={
                "list": NodeType.LIST,
                "item": NodeType.ITEM
            },
            required_edge_types={
                EdgeType.CONTAINS,
                EdgeType.NAV_TO
            },
            pattern_syntax="""
            Page -[Contains]-> Paginator -[NavTo]-> NextPage
            """,
            task_type="Navigation",
            difficulty="EASY",
            applicable_website_types={"portal", "ecommerce", "crm", "blog"},
            min_steps=2,
            max_steps=3
        )
        
        
        # 18. Filter Application pattern - 移到业务数据类，因为过滤需要业务数据
       
        
        # 19. Expand Collapse pattern - Expand collapse
        seeds[TaskSeedType.EXPAND_COLLAPSE] = TaskSeedPattern(
            seed_type=TaskSeedType.EXPAND_COLLAPSE,
            name="Expand Collapse",
            description="Expand or collapse content areas",
            seed_category="interaction",
            core_slots={
                "expand_btn": NodeType.BUTTON,
                "content": NodeType.CONTENT
            },
            optional_slots={
                "collapsible": NodeType.COLLAPSIBLE,
                "icon": NodeType.ICON
            },
            required_edge_types={
                EdgeType.CONTROLS,
                EdgeType.CONTAINS
            },
            pattern_syntax="""
            ExpandButton -[Controls]-> ExpandAction -[Contains]-> ExpandedContent
            """,
            task_type="Interaction",
            difficulty="EASY",
            applicable_website_types={"portal", "ecommerce", "crm", "blog"},
            min_steps=2,
            max_steps=3
        )
        
        # 20. Scroll Reading pattern - Scroll reading
        seeds[TaskSeedType.SCROLL_READING] = TaskSeedPattern(
            seed_type=TaskSeedType.SCROLL_READING,
            name="Scroll Reading",
            description="Scroll page to read long content",
            seed_category="interaction",
            core_slots={
                "content": NodeType.CONTENT,
                "scroll_area": NodeType.SCROLL_AREA
            },
            optional_slots={
                "text": NodeType.TEXT,
                "image": NodeType.IMAGE
            },
            required_edge_types={
                EdgeType.CONTAINS
            },
            pattern_syntax="""
            Page -[Contains]-> ScrollArea -[Contains]-> Content
            """,
            task_type="Content Browsing",
            difficulty="EASY",
            applicable_website_types={"blog", "portal", "content"},
            min_steps=2,
            max_steps=3
        )
        
        
        return seeds
    
    def get_applicable_seeds(self, task_graph: TaskGraph) -> List[TaskSeedPattern]:
        """Get seeds applicable to the given task graph"""
        applicable_seeds = []
        
        # Get node types and edge types present in the graph
        available_node_types = set()
        available_edge_types = set()
        
        for node in task_graph.nodes.values():
            available_node_types.add(node.node_type)
        
        for edge in task_graph.edges.values():
            available_edge_types.add(edge.edge_type)
        
        # Debug logging
        from loguru import logger
        logger.debug(f"🔍 Available node types: {sorted([t.value for t in available_node_types])}")
        logger.debug(f"🔍 Available edge types: {sorted([t.value for t in available_edge_types])}")
        logger.debug(f"🔍 Website type: {task_graph.website_type}")
        logger.debug(f"🔍 Total seeds to check: {len(self.seeds)}")
        
        # Check if each seed is applicable
        for seed in self.seeds.values():
            # Check if website type matches - 更宽容的网站类型匹配
            if seed.applicable_website_types:
                # 定义通用种子类型
                universal_seeds = {
                    TaskSeedType.BASIC_NAVIGATION,
                    TaskSeedType.CONTENT_BROWSING,
                    TaskSeedType.BUTTON_INTERACTION,
                    TaskSeedType.MENU_EXPLORATION,
                    TaskSeedType.TAB_SWITCHING,
                    TaskSeedType.TOAST_NOTIFICATION,
                    TaskSeedType.BREADCRUMB_NAVIGATION,
                    TaskSeedType.PAGINATION_BROWSING,
                    TaskSeedType.EXPAND_COLLAPSE,
                    TaskSeedType.SCROLL_READING,
                    TaskSeedType.BUSINESS_SEARCH_FILTER
                }
                
                # Enhanced website type matching with fallback logic
                website_type_compatibility = self._assess_website_type_compatibility(seed, task_graph.website_type, universal_seeds)

                if not website_type_compatibility['compatible']:
                    logger.debug(f"❌ Seed {seed.seed_type.value} rejected: {website_type_compatibility['reason']}")
                    continue

                logger.debug(f"✅ Seed {seed.seed_type.value} accepted: {website_type_compatibility['reason']}")
            
            # Check core slot matching
            core_slots_satisfied = self._check_core_slots_satisfied(seed, available_node_types)
            required_edges_satisfied = self._check_required_edges_satisfied(seed, available_edge_types)
            
            # For basic task seeds, consider applicable if there's partial matching
            basic_seed_types = [
                TaskSeedType.BASIC_NAVIGATION, 
                TaskSeedType.BUTTON_INTERACTION, 
                TaskSeedType.FORM_INPUT,
                TaskSeedType.MENU_EXPLORATION,
                TaskSeedType.TAB_SWITCHING,
                TaskSeedType.TOAST_NOTIFICATION,
                TaskSeedType.BREADCRUMB_NAVIGATION,
                TaskSeedType.PAGINATION_BROWSING,
                TaskSeedType.EXPAND_COLLAPSE,
                TaskSeedType.SCROLL_READING,
                ]
            
            if seed.seed_type in basic_seed_types:
                # For basic seeds, be more lenient - only require partial matching
                if core_slots_satisfied or required_edges_satisfied:
                    applicable_seeds.append(seed)
                    logger.debug(f"✅ Basic seed {seed.seed_type.value} accepted: core_slots={core_slots_satisfied}, edges={required_edges_satisfied}")
                else:
                    logger.debug(f"❌ Basic seed {seed.seed_type.value} rejected: core_slots={core_slots_satisfied}, edges={required_edges_satisfied}")
            else:
                # For complex task seeds, use code default core slot matching
                min_core_slot_ratio = 0.6  # 使用代码默认配置
                
                core_slot_ratio = self._calculate_core_slot_ratio(seed, available_node_types)
                if core_slot_ratio >= min_core_slot_ratio and required_edges_satisfied:
                    applicable_seeds.append(seed)
                    logger.debug(f"✅ Complex seed {seed.seed_type.value} accepted: ratio={core_slot_ratio:.2f}, edges={required_edges_satisfied}")
                else:
                    logger.debug(f"❌ Complex seed {seed.seed_type.value} rejected: ratio={core_slot_ratio:.2f}, edges={required_edges_satisfied}")
        
        return applicable_seeds
    
    def _check_core_slots_satisfied(self, seed: TaskSeedPattern, available_node_types: Set[NodeType]) -> bool:
        """Check if core slots are satisfied - enhanced with flexible slots support"""
        if not seed.core_slots:
            return True

        from loguru import logger

        # First, check core slots with flexible matching
        for slot_name, required_type in seed.core_slots.items():
            if required_type not in available_node_types:
                # Check flexible matching
                flexible_mapping = {
                    NodeType.SEARCH_BOX: {NodeType.INPUT, NodeType.BUTTON, NodeType.LINK},
                    NodeType.INPUT: {NodeType.BUTTON, NodeType.NAVIGATION, NodeType.SELECT, NodeType.LINK},
                    NodeType.BUTTON: {NodeType.NAVIGATION, NodeType.INPUT, NodeType.LINK},
                    NodeType.NAVIGATION: {NodeType.BUTTON, NodeType.INPUT, NodeType.LINK},
                    NodeType.FILTER: {NodeType.BUTTON, NodeType.NAVIGATION, NodeType.LINK},
                    NodeType.LIST: {NodeType.BUTTON, NodeType.NAVIGATION, NodeType.LINK, NodeType.RESULT_ITEM},
                    NodeType.RESULT_ITEM: {NodeType.BUTTON, NodeType.NAVIGATION, NodeType.LINK},
                    NodeType.DETAIL_LINK: {NodeType.BUTTON, NodeType.NAVIGATION, NodeType.LINK},
                    NodeType.SELECT: {NodeType.INPUT, NodeType.DROPDOWN, NodeType.BUTTON, NodeType.LINK},
                    NodeType.DROPDOWN: {NodeType.SELECT, NodeType.INPUT, NodeType.BUTTON, NodeType.LINK},
                    NodeType.TAB: {NodeType.BUTTON, NodeType.NAVIGATION, NodeType.LINK},
                    NodeType.MODAL: {NodeType.BUTTON, NodeType.CONTENT, NodeType.LINK},
                    NodeType.TOAST: {NodeType.BUTTON, NodeType.CONTENT, NodeType.LINK},
                    NodeType.BREADCRUMB: {NodeType.NAVIGATION, NodeType.LINK},
                    NodeType.LINK: {NodeType.BUTTON, NodeType.NAVIGATION, NodeType.RESULT_ITEM},
                    NodeType.CONTENT: {NodeType.LIST, NodeType.TABLE, NodeType.CARD, NodeType.RESULT_ITEM},
                    NodeType.TABLE: {NodeType.LIST, NodeType.CARD, NodeType.RESULT_ITEM},
                    NodeType.CARD: {NodeType.LIST, NodeType.TABLE, NodeType.RESULT_ITEM},
                    NodeType.DETAIL: {NodeType.CONTENT, NodeType.CARD, NodeType.RESULT_ITEM},
                    NodeType.ITEM: {NodeType.CARD, NodeType.LIST, NodeType.RESULT_ITEM},
                    NodeType.FILTER_PANEL: {NodeType.FILTER, NodeType.BUTTON, NodeType.LINK},
                    NodeType.NOTIFICATION_AREA: {NodeType.TOAST, NodeType.CONTENT, NodeType.LINK},
                    NodeType.TAB_CONTAINER: {NodeType.TAB, NodeType.NAVIGATION, NodeType.LINK},
                    NodeType.SUBMENU: {NodeType.MENU, NodeType.DROPDOWN, NodeType.LINK},
                    NodeType.MENU: {NodeType.NAVIGATION, NodeType.DROPDOWN, NodeType.LINK}
                }

                flexible_types = flexible_mapping.get(required_type, set())
                if not (flexible_types & available_node_types):
                    # Check flexible core slots as fallback
                    if seed.flexible_core_slots and (required_type in seed.flexible_core_slots or seed.flexible_core_slots & available_node_types):
                        logger.debug(f"✅ Core slot '{slot_name}' satisfied via flexible slots: {required_type.value}")
                        continue
                    else:
                        logger.debug(f"❌ Core slot '{slot_name}' requires {required_type.value}, not available. Available: {[t.value for t in available_node_types]}")
                        return False
                else:
                    logger.debug(f"✅ Core slot '{slot_name}' requires {required_type.value}, found flexible match: {[t.value for t in flexible_types & available_node_types]}")

        return True
    
    def _check_required_edges_satisfied(self, seed: TaskSeedPattern, available_edge_types: Set[EdgeType]) -> bool:
        """Check if required edge types are satisfied"""
        if not seed.required_edge_types:
            return True
        
        for required_edge in seed.required_edge_types:
            if required_edge not in available_edge_types:
                # Check flexible matching
                flexible_mapping = {
                    EdgeType.FILLS: {EdgeType.CONTAINS, EdgeType.CONTROLS},
                    EdgeType.CONTROLS: {EdgeType.CONTAINS, EdgeType.NAV_TO},
                    EdgeType.FILTERS: {EdgeType.NAV_TO, EdgeType.CONTAINS},
                    EdgeType.OPENS: {EdgeType.NAV_TO, EdgeType.CONTAINS},
                    EdgeType.NAV_TO: {EdgeType.CONTAINS, EdgeType.CONTROLS}
                }
                
                flexible_types = flexible_mapping.get(required_edge, set())
                if not (flexible_types & available_edge_types):
                    return False
        
        return True
    
    def _assess_website_type_compatibility(self, seed: TaskSeedPattern, website_type: str, universal_seeds: Set[TaskSeedType]) -> Dict[str, Any]:
        """Assess compatibility between seed and website type with intelligent fallback"""
        # Universal seeds work on all websites
        if seed.seed_type in universal_seeds:
            return {'compatible': True, 'reason': 'universal seed type'}

        # Direct match
        if website_type in seed.applicable_website_types:
            return {'compatible': True, 'reason': 'direct website type match'}

        # Intelligent fallback based on website type similarity
        website_type_mappings = {
            'blog': ['content', 'portal'],
            'content': ['blog', 'portal'],
            'portal': ['content', 'blog', 'ecommerce'],
            'ecommerce': ['portal', 'crm'],
            'crm': ['portal', 'ecommerce'],
            'unknown': ['content', 'portal', 'blog']  # Fallback for unknown websites
        }

        compatible_types = website_type_mappings.get(website_type, [])
        for compatible_type in compatible_types:
            if compatible_type in seed.applicable_website_types:
                return {'compatible': True, 'reason': f'compatible via {compatible_type}'}

        # Business seeds can work on business-oriented websites
        if seed.seed_category == 'business' and website_type in ['crm', 'ecommerce', 'portal']:
            return {'compatible': True, 'reason': 'business seed on business website'}

        return {'compatible': False, 'reason': f'website type mismatch (required: {seed.applicable_website_types}, got: {website_type})'}

    def _calculate_core_slot_ratio(self, seed: TaskSeedPattern, available_node_types: Set[NodeType]) -> float:
        """Calculate core slot matching ratio"""
        if not seed.core_slots:
            return 1.0
        
        satisfied_count = 0
        total_count = len(seed.core_slots)
        
        for slot_name, required_type in seed.core_slots.items():
            if required_type in available_node_types:
                satisfied_count += 1
            else:
                # Check flexible matching
                flexible_mapping = {
                    NodeType.SEARCH_BOX: {NodeType.INPUT, NodeType.BUTTON},
                    NodeType.INPUT: {NodeType.BUTTON, NodeType.NAVIGATION, NodeType.SELECT},
                    NodeType.BUTTON: {NodeType.NAVIGATION, NodeType.INPUT},
                    NodeType.NAVIGATION: {NodeType.BUTTON, NodeType.INPUT},
                    NodeType.SELECT: {NodeType.INPUT, NodeType.DROPDOWN, NodeType.BUTTON},
                    NodeType.DROPDOWN: {NodeType.SELECT, NodeType.INPUT, NodeType.BUTTON},
                    NodeType.TAB: {NodeType.BUTTON, NodeType.NAVIGATION},
                    NodeType.MODAL: {NodeType.BUTTON, NodeType.CONTENT},
                    NodeType.TOAST: {NodeType.BUTTON, NodeType.CONTENT},
                    NodeType.BREADCRUMB: {NodeType.NAVIGATION, NodeType.LINK},
                    NodeType.LINK: {NodeType.BUTTON, NodeType.NAVIGATION},
                    NodeType.CONTENT: {NodeType.LIST, NodeType.TABLE, NodeType.CARD},
                    NodeType.TABLE: {NodeType.LIST, NodeType.CARD},
                    NodeType.CARD: {NodeType.LIST, NodeType.TABLE},
                    NodeType.DETAIL: {NodeType.CONTENT, NodeType.CARD},
                    NodeType.ITEM: {NodeType.CARD, NodeType.LIST},
                    NodeType.FILTER_PANEL: {NodeType.FILTER, NodeType.BUTTON},
                    NodeType.NOTIFICATION_AREA: {NodeType.TOAST, NodeType.CONTENT},
                    NodeType.TAB_CONTAINER: {NodeType.TAB, NodeType.NAVIGATION},
                    NodeType.SUBMENU: {NodeType.MENU, NodeType.DROPDOWN},
                    NodeType.MENU: {NodeType.NAVIGATION, NodeType.DROPDOWN}
                }
                
                flexible_types = flexible_mapping.get(required_type, set())
                if flexible_types & available_node_types:
                    satisfied_count += 1
        
        return satisfied_count / total_count if total_count > 0 else 1.0
    
    def get_seed_by_type(self, seed_type: TaskSeedType) -> Optional[TaskSeedPattern]:
        """Get seed by type"""
        return self.seeds.get(seed_type)
    
    def get_all_seeds(self) -> List[TaskSeedPattern]:
        """Get all seeds"""
        return list(self.seeds.values())
    
    def get_business_seeds(self) -> Dict[TaskSeedType, TaskSeedPattern]:
        """Get business data seeds (high-value tasks)"""
        return {k: v for k, v in self.seeds.items() if v.seed_category == "business"}
    
    def get_interaction_seeds(self) -> Dict[TaskSeedType, TaskSeedPattern]:
        """Get interaction seeds (fallback tasks)"""
        return {k: v for k, v in self.seeds.items() if v.seed_category == "interaction"}
    
    def select_seeds_by_priority(self, business_data_available: bool = False,
                                business_data_count: int = 0) -> List[TaskSeedPattern]:
        """
        Select seeds based on business data availability (Dual Seeds Design)

        Args:
            business_data_available: Whether business data is available
            business_data_count: Number of business data elements available

        Returns:
            List of selected seeds, prioritized by business data availability
        """
        selected_seeds = []
        business_seeds = []
        interaction_seeds = []

        # First, try to select business seeds if business data is available
        if business_data_available and business_data_count >= 1:  # 至少需要1个业务数据元素
            business_seeds_dict = self.get_business_seeds()

            for seed_type, seed_pattern in business_seeds_dict.items():
                if business_data_count >= seed_pattern.min_business_data_elements:
                    business_seeds.append(seed_pattern)

            # Sort by priority (highest first)
            business_seeds.sort(key=lambda x: x.priority, reverse=True)

        # Get interaction seeds - 在有业务数据时也选择通用交互任务
        interaction_seeds_dict = self.get_interaction_seeds()
        all_interaction_seeds = list(interaction_seeds_dict.values())

        # 过滤掉需要业务数据的交互种子（如搜索任务）
        filtered_interaction_seeds = []
        for seed in all_interaction_seeds:
            # 如果业务数据不足，跳过需要业务数据的种子
            if business_data_count < getattr(seed, 'min_business_data_elements', 0):
                continue
            filtered_interaction_seeds.append(seed)

        # Sort by priority
        filtered_interaction_seeds.sort(key=lambda x: x.priority, reverse=True)

        # 在有业务数据的情况下，增加通用交互任务的比例
        if business_data_available and business_data_count >= 2:
            # 有充足业务数据时，平衡业务种子和交互种子
            # 选择前4个业务种子 + 前4个交互种子
            business_top = business_seeds[:4]
            interaction_top = filtered_interaction_seeds[:4]
            combined_seeds = business_top + interaction_top
        elif business_data_available and business_data_count >= 1:
            # 有基本业务数据时，选择前3个业务种子 + 前3个交互种子
            business_top = business_seeds[:3]
            interaction_top = filtered_interaction_seeds[:3]
            combined_seeds = business_top + interaction_top
        else:
            # 没有业务数据时，只使用交互种子
            combined_seeds = filtered_interaction_seeds[:6]

        # 确保多样性：不同种类的种子
        diverse_seeds = []
        used_seed_categories = set()

        # 第一遍：收集不同种类的种子
        for seed in combined_seeds:
            if seed.seed_category not in used_seed_categories:
                diverse_seeds.append(seed)
                used_seed_categories.add(seed.seed_category)
                if len(diverse_seeds) >= 6:  # 确保至少有6个多样化的种子
                    break

        # 第二遍：填充剩余位置，使用不同功能的种子
        for seed in combined_seeds:
            if seed not in diverse_seeds and len(diverse_seeds) < 8:
                # 检查这个种子是否提供不同的功能
                seed_functionality = getattr(seed, 'task_type', seed.seed_type.value)
                existing_functionalities = {getattr(s, 'task_type', s.seed_type.value) for s in diverse_seeds}

                if seed_functionality not in existing_functionalities:
                    diverse_seeds.append(seed)

        # 确保最少多样性
        if len(diverse_seeds) < 4:
            diverse_seeds = combined_seeds[:4]

        # 记录选择的种子信息
        logger.info(f"🎯 Selected {len(diverse_seeds)} seeds: {len([s for s in diverse_seeds if s.seed_category == 'business'])} business, {len([s for s in diverse_seeds if s.seed_category == 'interaction'])} interaction")

        return diverse_seeds
    
    def get_seed_statistics(self) -> Dict[str, Any]:
        """Get statistics about available seeds"""
        business_seeds = self.get_business_seeds()
        interaction_seeds = self.get_interaction_seeds()
        
        return {
            "total_seeds": len(self.seeds),
            "business_seeds": len(business_seeds),
            "interaction_seeds": len(interaction_seeds),
            "business_seed_types": [seed.name for seed in business_seeds.values()],
            "interaction_seed_types": [seed.name for seed in interaction_seeds.values()],
            "high_priority_seeds": [seed.name for seed in self.seeds.values() if seed.priority >= 8]
        }

