from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

class WebTaskType(Enum):
    """Web task types - Completely aligned with TaskSeedType"""
    
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
    
    # Web safety task types
    WEB_MALICIOUS_INPUT = "web_malicious_input"  # Malicious input detection
    WEB_PHISHING_DETECTION = "web_phishing_detection"  # Phishing detection
    WEB_DATA_PRIVACY = "web_data_privacy"  # Data privacy protection
    WEB_ACCESS_CONTROL = "web_access_control"  # Access control validation
    WEB_CONTENT_MODERATION = "web_content_moderation"  # Content moderation
    WEB_FORM_VALIDATION = "web_form_validation"  # Form validation
    WEB_NAVIGATION_SAFETY = "web_navigation_safety"  # Navigation safety
    
    @classmethod
    def is_safety_task(cls, task_type) -> bool:
        """Check if a task type is a web safety-related task"""
        safety_types = {
            # Web safety types
            cls.WEB_MALICIOUS_INPUT,
            cls.WEB_PHISHING_DETECTION,
            cls.WEB_DATA_PRIVACY,
            cls.WEB_ACCESS_CONTROL,
            cls.WEB_CONTENT_MODERATION,
            cls.WEB_FORM_VALIDATION,
            cls.WEB_NAVIGATION_SAFETY
        }
        return task_type in safety_types
    
    @classmethod
    def is_normal_task(cls, task_type) -> bool:
        """Check if a task type is a normal (non-safety) web task"""
        return not cls.is_safety_task(task_type)
    
    @classmethod
    def from_strategy(cls, strategy: str) -> 'WebTaskType':
        """Convert strategy string to WebTaskType - Completely aligned with TaskSeedType"""
        strategy_mapping = {
            # Web safety strategies
            'malicious_input': cls.WEB_MALICIOUS_INPUT,
            'phishing_detection': cls.WEB_PHISHING_DETECTION,
            'data_privacy': cls.WEB_DATA_PRIVACY,
            'access_control': cls.WEB_ACCESS_CONTROL,
            'content_moderation': cls.WEB_CONTENT_MODERATION,
            'form_validation': cls.WEB_FORM_VALIDATION,
            'navigation_safety': cls.WEB_NAVIGATION_SAFETY,
            
            # Business data seed strategies
            'business_search_filter': cls.BUSINESS_SEARCH_FILTER,
            'business_navigation': cls.BUSINESS_NAVIGATION,
            'user_navigation': cls.USER_NAVIGATION,
            'product_navigation': cls.PRODUCT_NAVIGATION,
            'order_navigation': cls.ORDER_NAVIGATION,
            'mixed_data_navigation': cls.MIXED_DATA_NAVIGATION,
            'multi_hop_navigation': cls.MULTI_HOP_NAVIGATION,
            
            # Interaction seed strategies
            'content_browsing': cls.CONTENT_BROWSING,
            'basic_navigation': cls.BASIC_NAVIGATION,
            'button_interaction': cls.BUTTON_INTERACTION,
            'menu_exploration': cls.MENU_EXPLORATION,
            'tab_switching': cls.TAB_SWITCHING,
            'modal_interaction': cls.MODAL_INTERACTION,
            'toast_notification': cls.TOAST_NOTIFICATION,
            'breadcrumb_navigation': cls.BREADCRUMB_NAVIGATION,
            'pagination_browsing': cls.PAGINATION_BROWSING,
            'expand_collapse': cls.EXPAND_COLLAPSE,
            'scroll_reading': cls.SCROLL_READING,
        }
        return strategy_mapping.get(strategy, cls.CONTENT_BROWSING)  # Default to content_browsing

@dataclass
class WebTaskStep:
    """Represents a single step in a web task"""
    step_type: str  # click, input, navigate, extract, etc.
    target_som_mark: str = ""  # SoM mark for target element
    action_description: str = ""
    input_value: str = ""  # For input steps
    expected_element: str = ""  # Expected SoM mark after action
    expected_result: str = ""  # Expected result after action
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "step_type": self.step_type,
            "target_som_mark": self.target_som_mark,
            "action_description": self.action_description,
            "input_value": self.input_value,
            "expected_element": self.expected_element,
            "expected_result": self.expected_result
        }

@dataclass
class WebTaskTemplate:
    """Template for web-based tasks"""
    template_id: str
    task_type: WebTaskType
    prompt_template: str
    steps_template: List[Dict[str, Any]] = field(default_factory=list)
    required_node_types: List[str] = field(default_factory=list)
    required_edge_types: List[str] = field(default_factory=list)
    difficulty: str = "MEDIUM"
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "task_type": self.task_type.value,
            "prompt_template": self.prompt_template,
            "steps_template": self.steps_template,
            "required_node_types": self.required_node_types,
            "required_edge_types": self.required_edge_types,
            "difficulty": self.difficulty,
            "description": self.description
        }



