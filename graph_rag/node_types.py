"""
Graph node types for GraphRAG
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import json
import numpy as np


class NodeType(Enum):
    """Types of nodes in the graph"""
    PARAGRAPH = "paragraph"
    TABLE = "table" 
    HEADING = "heading"
    FIGURE = "figure"
    ENTITY = "entity"
    CHUNK = "chunk"
    # Web-specific node types
    WEB_PAGE = "web_page"
    WEB_ELEMENT = "web_element"
    WEB_FORM = "web_form"
    WEB_BUTTON = "web_button"
    WEB_INPUT = "web_input"
    WEB_LINK = "web_link"
    WEB_TABLE = "web_table"
    WEB_IMAGE = "web_image"


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


def node_from_dict(data: Dict[str, Any]) -> Node:
    """Create node from dictionary representation"""
    node_type = NodeType(data["node_type"])
    
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
    
    return node_class.from_dict(data)


@dataclass
class WebPageNode(Node):
    """Node representing a web page"""
    url: str = ""
    title: str = ""
    page_type: str = ""  # homepage, product, form, etc.
    load_time: float = 0.0
    page_size: int = 0
    
    def __post_init__(self):
        if self.node_type != NodeType.WEB_PAGE:
            self.node_type = NodeType.WEB_PAGE
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebPageNode':
        """Create WebPageNode from dictionary"""
        embedding = None
        if "embedding" in data and data["embedding"]:
            embedding = np.array(data["embedding"])
        
        return cls(
            node_id=data["node_id"],
            node_type=NodeType.WEB_PAGE,
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=embedding,
            source_file=data.get("source_file"),
            page_num=data.get("page_num"),
            bbox=data.get("bbox"),
            url=data.get("url", ""),
            title=data.get("title", ""),
            page_type=data.get("page_type", ""),
            load_time=data.get("load_time", 0.0),
            page_size=data.get("page_size", 0)
        )


@dataclass
class WebElementNode(Node):
    """Node representing a web page element"""
    element_type: str = ""  # button, input, table, link, etc.
    tag_name: str = ""
    text_content: str = ""
    placeholder: str = ""
    value: str = ""
    href: str = ""
    src: str = ""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    css_classes: List[str] = field(default_factory=list)
    css_selector: str = ""
    is_clickable: bool = False
    is_input: bool = False
    is_visible: bool = True
    is_enabled: bool = True
    input_type: str = ""
    required: bool = False
    options: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.node_type != NodeType.WEB_ELEMENT:
            self.node_type = NodeType.WEB_ELEMENT
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebElementNode':
        """Create WebElementNode from dictionary"""
        embedding = None
        if "embedding" in data and data["embedding"]:
            embedding = np.array(data["embedding"])
        
        return cls(
            node_id=data["node_id"],
            node_type=NodeType.WEB_ELEMENT,
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=embedding,
            source_file=data.get("source_file"),
            page_num=data.get("page_num"),
            bbox=data.get("bbox"),
            element_type=data.get("element_type", ""),
            tag_name=data.get("tag_name", ""),
            text_content=data.get("text_content", ""),
            placeholder=data.get("placeholder", ""),
            value=data.get("value", ""),
            href=data.get("href", ""),
            src=data.get("src", ""),
            x=data.get("x", 0),
            y=data.get("y", 0),
            width=data.get("width", 0),
            height=data.get("height", 0),
            css_classes=data.get("css_classes", []),
            css_selector=data.get("css_selector", ""),
            is_clickable=data.get("is_clickable", False),
            is_input=data.get("is_input", False),
            is_visible=data.get("is_visible", True),
            is_enabled=data.get("is_enabled", True),
            input_type=data.get("input_type", ""),
            required=data.get("required", False),
            options=data.get("options", [])
        )


@dataclass
class WebFormNode(WebElementNode):
    """Node representing a web form"""
    form_elements: List[str] = field(default_factory=list)  # element IDs
    form_action: str = ""
    form_method: str = "GET"
    
    def __post_init__(self):
        if self.node_type != NodeType.WEB_FORM:
            self.node_type = NodeType.WEB_FORM


@dataclass
class WebButtonNode(WebElementNode):
    """Node representing a web button"""
    button_type: str = ""  # submit, reset, button
    button_text: str = ""
    
    def __post_init__(self):
        if self.node_type != NodeType.WEB_BUTTON:
            self.node_type = NodeType.WEB_BUTTON


@dataclass
class WebInputNode(WebElementNode):
    """Node representing a web input field"""
    input_validation: Dict[str, Any] = field(default_factory=dict)
    input_pattern: str = ""
    
    def __post_init__(self):
        if self.node_type != NodeType.WEB_INPUT:
            self.node_type = NodeType.WEB_INPUT


@dataclass
class WebLinkNode(WebElementNode):
    """Node representing a web link"""
    link_text: str = ""
    target_url: str = ""
    link_type: str = ""  # internal, external, download
    
    def __post_init__(self):
        if self.node_type != NodeType.WEB_LINK:
            self.node_type = NodeType.WEB_LINK


@dataclass
class WebTableNode(WebElementNode):
    """Node representing a web table"""
    table_rows: int = 0
    table_columns: int = 0
    table_headers: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.node_type != NodeType.WEB_TABLE:
            self.node_type = NodeType.WEB_TABLE


@dataclass
class WebImageNode(WebElementNode):
    """Node representing a web image"""
    image_alt: str = ""
    image_title: str = ""
    image_size: tuple = (0, 0)  # (width, height)
    
    def __post_init__(self):
        if self.node_type != NodeType.WEB_IMAGE:
            self.node_type = NodeType.WEB_IMAGE


def create_web_node(node_type: NodeType, **kwargs) -> Node:
    """Factory function to create web nodes"""
    node_classes = {
        NodeType.WEB_PAGE: WebPageNode,
        NodeType.WEB_ELEMENT: WebElementNode,
        NodeType.WEB_FORM: WebFormNode,
        NodeType.WEB_BUTTON: WebButtonNode,
        NodeType.WEB_INPUT: WebInputNode,
        NodeType.WEB_LINK: WebLinkNode,
        NodeType.WEB_TABLE: WebTableNode,
        NodeType.WEB_IMAGE: WebImageNode,
    }
    
    if node_type not in node_classes:
        raise ValueError(f"Unknown web node type: {node_type}")
    
    return node_classes[node_type](**kwargs)
