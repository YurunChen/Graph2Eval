"""
Graph node types for GraphRAG
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Set
from enum import Enum
import json
import numpy as np


class NodeType(Enum):
    """Types of nodes in the graph - Document and Web types"""
    # Document-focused types
    PARAGRAPH = "paragraph"
    TABLE = "table" 
    HEADING = "heading"
    FIGURE = "figure"
    ENTITY = "entity"
    CHUNK = "chunk"
    
    # Web-specific node types
    PAGE = "Page"
    SECTION = "Section"
    
    # Form-related nodes
    FORM = "Form"
    INPUT = "Input"
    BUTTON = "Button"
    SUBMIT = "Submit"
    SELECT = "Select"
    TEXTAREA = "Textarea"
    
    # Content display nodes
    CARD = "Card"
    LIST = "List"
    DETAIL = "Detail"
    ITEM = "Item"
    CONTENT = "Content"
    
    # Interactive control nodes
    FILTER = "Filter"
    PAGINATOR = "Paginator"
    SEARCH_BOX = "SearchBox"
    
    # Modal and notification nodes
    MODAL = "Modal"
    TOAST = "Toast"
    
    # Navigation-related nodes
    BREADCRUMB = "Breadcrumb"
    MENU = "Menu"
    NAVIGATION = "Navigation"
    LINK = "Link"
    
    # Dashboard nodes
    DASHBOARD = "Dashboard"
    
    # Result and link nodes
    RESULT_ITEM = "ResultItem"
    DETAIL_LINK = "DetailLink"
    
    # Additional node types
    TAB = "Tab"
    TAB_CONTAINER = "TabContainer"
    DROPDOWN = "Dropdown"
    SUBMENU = "Submenu"
    NOTIFICATION_AREA = "NotificationArea"
    FILTER_PANEL = "FilterPanel"
    COLLAPSIBLE = "Collapsible"
    ICON = "Icon"
    SCROLL_AREA = "ScrollArea"
    TEXT = "Text"
    IMAGE = "Image"
    
    # Business data node types - Generic data content
    BUSINESS_DATA = "business_data"  # Generic business data
    USER_DATA = "user_data"          # User information (name, email, phone, etc.)
    PRODUCT_DATA = "product_data"    # Product information (name, price, description, etc.)
    ORDER_DATA = "order_data"        # Order information (order number, status, amount, etc.)
    CONTENT_DATA = "content_data"    # Content information (title, author, date, etc.)
    FINANCIAL_DATA = "financial_data" # Financial information (amount, currency, date, etc.)
    LOCATION_DATA = "location_data"  # Location information (address, city, country, etc.)
    TIME_DATA = "time_data"          # Time information (date, time, duration, etc.)


class BusinessTag(Enum):
    """业务标签 - 标识节点的业务含义"""
    PRODUCT = "product"
    CONTACT = "contact"
    ORDER = "order"
    TICKET = "ticket"
    ACCOUNT = "account"
    REPORT = "report"
    SETTING = "setting"
    DASHBOARD = "dashboard"


@dataclass
class NodeMetadata:
    """节点元信息"""
    # 可见性和交互性
    is_visible: bool = True
    is_clickable: bool = False
    is_input: bool = False
    is_enabled: bool = True
    
    # 定位信息
    text_anchor: str = ""  # 文本锚点
    xpath: str = ""
    css_selector: str = ""
    som_mark: str = ""  # SoM标记
    
    # 视觉信息
    screenshot_region: Optional[Dict[str, int]] = None  # {x, y, width, height}
    
    # 业务标签
    business_tags: Set[BusinessTag] = field(default_factory=set)
    
    # 内容信息
    text_content: str = ""
    placeholder: str = ""
    input_type: str = ""
    
    # HTML属性信息
    tag_name: str = ""  # HTML标签名
    href: str = ""  # 链接地址
    css_classes: List[str] = field(default_factory=list)  # CSS类名
    role: str = ""  # ARIA角色
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_visible": self.is_visible,
            "is_clickable": self.is_clickable,
            "is_input": self.is_input,
            "is_enabled": self.is_enabled,
            "text_anchor": self.text_anchor,
            "xpath": self.xpath,
            "css_selector": self.css_selector,
            "som_mark": self.som_mark,
            "screenshot_region": self.screenshot_region,
            "business_tags": [tag.value for tag in self.business_tags],
            "text_content": self.text_content,
            "placeholder": self.placeholder,
            "input_type": self.input_type,
            "tag_name": self.tag_name,
            "href": self.href,
            "css_classes": self.css_classes,
            "role": self.role
        }


@dataclass
class GraphNode:
    """图节点 - 代表可交互的UI元素"""
    node_id: str
    node_type: NodeType
    url: str = ""
    metadata: NodeMetadata = field(default_factory=NodeMetadata)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "url": self.url,
            "metadata": self.metadata.to_dict()
        }
        # 添加element_type字段（如果存在）
        if hasattr(self, 'element_type'):
            # 确保element_type是字符串，如果是NodeType枚举则转换为字符串
            if hasattr(self.element_type, 'value'):
                result["element_type"] = self.element_type.value
            else:
                result["element_type"] = str(self.element_type)
        return result


@dataclass
class Node(ABC):
    """Abstract base class for all graph nodes"""
    node_id: str
    node_type: NodeType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    source_file: Optional[str] = None
    page_num: Optional[int] = None
    bbox: Optional[tuple] = None  # (x0, y0, x1, y1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        data = {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "source_file": self.source_file,
            "page_num": self.page_num,
            "bbox": self.bbox
        }
        
        # Handle embedding separately due to numpy array
        if self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create node from dictionary"""
        # This will be implemented in subclasses
        raise NotImplementedError
    
    def get_text_for_embedding(self) -> str:
        """Get text representation for embedding generation"""
        return self.content
    
    def get_display_text(self, max_length: int = 200) -> str:
        """Get truncated text for display"""
        text = self.content
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text


@dataclass
class ChunkNode(Node):
    """Node representing a document chunk"""
    chunk_size: int = 0
    chunk_index: int = 0
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.node_type != NodeType.CHUNK:
            self.node_type = NodeType.CHUNK
        if not self.chunk_size:
            self.chunk_size = len(self.content)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkNode':
        """Create ChunkNode from dictionary"""
        embedding = None
        if "embedding" in data and data["embedding"]:
            embedding = np.array(data["embedding"])
        
        return cls(
            node_id=data["node_id"],
            node_type=NodeType.CHUNK,
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=embedding,
            source_file=data.get("source_file"),
            page_num=data.get("page_num"),
            bbox=data.get("bbox"),
            chunk_size=data.get("chunk_size", len(data["content"])),
            chunk_index=data.get("chunk_index", 0),
            parent_chunk_id=data.get("parent_chunk_id"),
            child_chunk_ids=data.get("child_chunk_ids", [])
        )


@dataclass
class ParagraphNode(Node):
    """Node representing a paragraph"""
    word_count: int = 0
    sentence_count: int = 0
    paragraph_index: int = 0
    
    def __post_init__(self):
        if self.node_type != NodeType.PARAGRAPH:
            self.node_type = NodeType.PARAGRAPH
        if not self.word_count:
            self.word_count = len(self.content.split())
        if not self.sentence_count:
            # Simple sentence count
            self.sentence_count = len([s for s in self.content.split('.') if s.strip()])
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParagraphNode':
        embedding = None
        if "embedding" in data and data["embedding"]:
            embedding = np.array(data["embedding"])
        
        return cls(
            node_id=data["node_id"],
            node_type=NodeType.PARAGRAPH,
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=embedding,
            source_file=data.get("source_file"),
            page_num=data.get("page_num"),
            bbox=data.get("bbox"),
            word_count=data.get("word_count", len(data["content"].split())),
            sentence_count=data.get("sentence_count", 0),
            paragraph_index=data.get("paragraph_index", 0)
        )


@dataclass
class TableNode(Node):
    """Node representing a table"""
    table_data: List[List[str]] = field(default_factory=list)
    rows: int = 0
    cols: int = 0
    has_header: bool = False
    table_caption: str = ""
    
    def __post_init__(self):
        if self.node_type != NodeType.TABLE:
            self.node_type = NodeType.TABLE
        if self.table_data and not self.rows:
            self.rows = len(self.table_data)
            self.cols = len(self.table_data[0]) if self.table_data else 0
    
    def get_text_for_embedding(self) -> str:
        """Get text representation including table structure"""
        text_parts = []
        
        if self.table_caption:
            text_parts.append(f"Table: {self.table_caption}")
        
        # Add table content
        if self.table_data:
            # Add header if exists
            if self.has_header and self.table_data:
                headers = " | ".join(self.table_data[0])
                text_parts.append(f"Headers: {headers}")
                
                # Add data rows
                for row in self.table_data[1:]:
                    row_text = " | ".join(str(cell) for cell in row)
                    text_parts.append(row_text)
            else:
                for row in self.table_data:
                    row_text = " | ".join(str(cell) for cell in row)
                    text_parts.append(row_text)
        else:
            text_parts.append(self.content)
        
        return "\n".join(text_parts)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TableNode':
        embedding = None
        if "embedding" in data and data["embedding"]:
            embedding = np.array(data["embedding"])
        
        return cls(
            node_id=data["node_id"],
            node_type=NodeType.TABLE,
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=embedding,
            source_file=data.get("source_file"),
            page_num=data.get("page_num"),
            bbox=data.get("bbox"),
            table_data=data.get("table_data", []),
            rows=data.get("rows", 0),
            cols=data.get("cols", 0),
            has_header=data.get("has_header", False),
            table_caption=data.get("table_caption", "")
        )


@dataclass 
class HeadingNode(Node):
    """Node representing a heading"""
    level: int = 1
    section_number: str = ""
    subsection_count: int = 0
    
    def __post_init__(self):
        if self.node_type != NodeType.HEADING:
            self.node_type = NodeType.HEADING
    
    def get_text_for_embedding(self) -> str:
        """Include hierarchical context in embedding text"""
        text_parts = [f"Heading Level {self.level}"]
        
        if self.section_number:
            text_parts.append(f"Section {self.section_number}")
        
        text_parts.append(self.content)
        
        return ": ".join(text_parts)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HeadingNode':
        embedding = None
        if "embedding" in data and data["embedding"]:
            embedding = np.array(data["embedding"])
        
        return cls(
            node_id=data["node_id"],
            node_type=NodeType.HEADING,
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=embedding,
            source_file=data.get("source_file"),
            page_num=data.get("page_num"),
            bbox=data.get("bbox"),
            level=data.get("level", 1),
            section_number=data.get("section_number", ""),
            subsection_count=data.get("subsection_count", 0)
        )


@dataclass
class FigureNode(Node):
    """Node representing a figure/image"""
    figure_type: str = "image"  # image, chart, diagram, etc.
    caption: str = ""
    alt_text: str = ""
    image_path: Optional[str] = None
    extracted_text: str = ""  # OCR text if available
    
    def __post_init__(self):
        if self.node_type != NodeType.FIGURE:
            self.node_type = NodeType.FIGURE
    
    def get_text_for_embedding(self) -> str:
        """Get text representation including all textual elements"""
        text_parts = [f"Figure ({self.figure_type})"]
        
        if self.caption:
            text_parts.append(f"Caption: {self.caption}")
        
        if self.alt_text:
            text_parts.append(f"Alt text: {self.alt_text}")
        
        if self.extracted_text:
            text_parts.append(f"Extracted text: {self.extracted_text}")
        
        if self.content and self.content != self.caption:
            text_parts.append(self.content)
        
        return "\n".join(text_parts)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FigureNode':
        embedding = None
        if "embedding" in data and data["embedding"]:
            embedding = np.array(data["embedding"])
        
        return cls(
            node_id=data["node_id"],
            node_type=NodeType.FIGURE,
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=embedding,
            source_file=data.get("source_file"),
            page_num=data.get("page_num"),
            bbox=data.get("bbox"),
            figure_type=data.get("figure_type", "image"),
            caption=data.get("caption", ""),
            alt_text=data.get("alt_text", ""),
            image_path=data.get("image_path"),
            extracted_text=data.get("extracted_text", "")
        )


@dataclass
class EntityNode(Node):
    """Node representing a named entity"""
    entity_type: str = "PERSON"  # PERSON, ORG, LOC, MISC, etc.
    entity_subtype: str = ""
    canonical_name: str = ""
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0
    mentions: List[Dict[str, Any]] = field(default_factory=list)  # List of mention contexts
    
    def __post_init__(self):
        if self.node_type != NodeType.ENTITY:
            self.node_type = NodeType.ENTITY
        if not self.canonical_name:
            self.canonical_name = self.content
    
    def get_text_for_embedding(self) -> str:
        """Get text representation including entity context"""
        text_parts = [f"{self.entity_type} Entity: {self.canonical_name}"]
        
        if self.aliases:
            text_parts.append(f"Also known as: {', '.join(self.aliases)}")
        
        # Add mention contexts
        if self.mentions:
            contexts = []
            for mention in self.mentions[:3]:  # Limit to first 3 mentions
                if "context" in mention:
                    contexts.append(mention["context"])
            if contexts:
                text_parts.append(f"Mentioned in context: {' | '.join(contexts)}")
        
        return "\n".join(text_parts)
    
    def add_mention(self, context: str, start_pos: int = 0, end_pos: int = 0, 
                   source_node_id: Optional[str] = None):
        """Add a mention of this entity"""
        mention = {
            "context": context,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "source_node_id": source_node_id,
            "mention_text": self.content
        }
        self.mentions.append(mention)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityNode':
        embedding = None
        if "embedding" in data and data["embedding"]:
            embedding = np.array(data["embedding"])
        
        return cls(
            node_id=data["node_id"],
            node_type=NodeType.ENTITY,
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=embedding,
            source_file=data.get("source_file"),
            page_num=data.get("page_num"),
            bbox=data.get("bbox"),
            entity_type=data.get("entity_type", "PERSON"),
            entity_subtype=data.get("entity_subtype", ""),
            canonical_name=data.get("canonical_name", data["content"]),
            aliases=data.get("aliases", []),
            confidence=data.get("confidence", 1.0),
            mentions=data.get("mentions", [])
        )


# Factory function for creating nodes
def create_node(node_type: NodeType, **kwargs) -> Node:
    """Factory function to create appropriate node type"""
    node_classes = {
        NodeType.CHUNK: ChunkNode,
        NodeType.PARAGRAPH: ParagraphNode,
        NodeType.TABLE: TableNode,
        NodeType.HEADING: HeadingNode,
        NodeType.FIGURE: FigureNode,
        NodeType.ENTITY: EntityNode
    }
    
    node_class = node_classes.get(node_type)
    if not node_class:
        raise ValueError(f"Unknown node type: {node_type}")
    
    return node_class(node_type=node_type, **kwargs)


def node_from_dict(data: Dict[str, Any], is_web_graph: bool = False) -> Node:
    """Create node from dictionary representation"""
    node_type = NodeType(data["node_type"])

    # Document node classes (for text/document graphs)
    document_node_classes = {
        NodeType.CHUNK: ChunkNode,
        NodeType.PARAGRAPH: ParagraphNode,
        NodeType.TABLE: TableNode,
        NodeType.HEADING: HeadingNode,
        NodeType.FIGURE: FigureNode,
        NodeType.ENTITY: EntityNode
    }

    # For web graphs, only use GraphNode to avoid mixing document and web node types
    if is_web_graph:
        return GraphNode(
            node_id=data["node_id"],
            node_type=node_type,
            url=data.get("url", ""),
            metadata=NodeMetadata(**data.get("metadata", {}))
        )
    
    # For document graphs, use appropriate document node classes
    node_class = document_node_classes.get(node_type)
    if node_class:
        return node_class.from_dict(data)
    else:
        # For unsupported node types, use GraphNode
        return GraphNode(
            node_id=data["node_id"],
            node_type=node_type,
            url=data.get("url", ""),
            metadata=NodeMetadata(**data.get("metadata", {}))
        )



