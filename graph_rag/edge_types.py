"""
Graph edge types for GraphRAG relationships
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import json
from loguru import logger


class EdgeType(Enum):
    """Types of edges in the graph"""
    # Document-focused edge types
    SEQUENCE = "sequence"  # next/prev sequential relationship
    CONTAINS = "contains"  # hierarchical containment
    REFERS_TO = "refers_to"  # explicit reference
    SEMANTIC_SIM = "semantic_sim"  # semantic similarity
    ENTITY_RELATION = "entity_relation"  # entity relationships
    CO_REFERENCE = "co_reference"  # coreference resolution
    CROSS_DOC = "cross_doc"  # cross-document relationships
    TABLE_CONTEXT = "table_context"  # table-paragraph relationships
    FIGURE_CONTEXT = "figure_context"  # figure-text relationships
    
    # Web-specific edge types
    WEB_NAVIGATION = "web_navigation"  # page navigation
    WEB_INTERACTION = "web_interaction"  # user interaction
    WEB_FORM_SUBMIT = "web_form_submit"  # form submission
    WEB_CLICK_TRIGGER = "web_click_trigger"  # click triggers action
    WEB_DATA_FLOW = "web_data_flow"  # data flow between elements
    WEB_LAYOUT = "web_layout"  # spatial layout relationships
    
    # Task-specific edge types
    NAV_TO = "NavTo"  # 页面跳转
    FILLS = "Fills"  # 输入→表单字段
    CONTROLS = "Controls"  # 按钮→表单/提交
    OPENS = "Opens"  # 打开详情/弹窗
    FILTERS = "Filters"  # 过滤关系
    SAME_ENTITY = "SameEntity"  # 同一实体不同视图
    TRIGGERS = "Triggers"  # 点击触发其他元素
    DATA_FLOW = "DataFlow"  # 数据流关系


@dataclass
class Edge(ABC):
    """Abstract base class for all graph edges"""
    edge_id: str
    edge_type: EdgeType
    source_node_id: str
    target_node_id: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    bidirectional: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for serialization"""
        return {
            "edge_id": self.edge_id,
            "edge_type": self.edge_type.value,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "weight": self.weight,
            "metadata": self.metadata,
            "bidirectional": self.bidirectional
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Edge':
        """Create edge from dictionary - implemented in subclasses"""
        raise NotImplementedError
    
    def reverse(self) -> 'Edge':
        """Create reverse edge if bidirectional"""
        if not self.bidirectional:
            raise ValueError("Cannot reverse unidirectional edge")
        
        # Create new edge with swapped source/target
        reversed_edge = self.__class__(
            edge_id=f"{self.edge_id}_reverse",
            edge_type=self.edge_type,
            source_node_id=self.target_node_id,
            target_node_id=self.source_node_id,
            weight=self.weight,
            metadata={**self.metadata, "is_reverse": True},
            bidirectional=self.bidirectional
        )
        return reversed_edge


@dataclass
class SequenceEdge(Edge):
    """Edge representing sequential relationship (next/previous)"""
    sequence_type: str = "next"  # next, prev, follows
    distance: int = 1  # how many steps apart
    
    def __post_init__(self):
        if self.edge_type != EdgeType.SEQUENCE:
            self.edge_type = EdgeType.SEQUENCE
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SequenceEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.SEQUENCE,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False),
            sequence_type=data.get("sequence_type", "next"),
            distance=data.get("distance", 1)
        )


@dataclass
class ContainsEdge(Edge):
    """Edge representing containment relationship"""
    containment_type: str = "contains"  # contains, part_of, belongs_to
    level_difference: int = 1  # hierarchical levels apart
    
    def __post_init__(self):
        if self.edge_type != EdgeType.CONTAINS:
            self.edge_type = EdgeType.CONTAINS
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContainsEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.CONTAINS,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False),
            containment_type=data.get("containment_type", "contains"),
            level_difference=data.get("level_difference", 1)
        )


@dataclass
class ReferenceEdge(Edge):
    """Edge representing explicit references"""
    reference_type: str = "mentions"  # mentions, cites, refers_to, points_to
    reference_text: str = ""  # the actual reference text
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.edge_type != EdgeType.REFERS_TO:
            self.edge_type = EdgeType.REFERS_TO
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReferenceEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.REFERS_TO,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False),
            reference_type=data.get("reference_type", "mentions"),
            reference_text=data.get("reference_text", ""),
            confidence=data.get("confidence", 1.0)
        )


@dataclass
class SemanticEdge(Edge):
    """Edge representing semantic similarity"""
    similarity_score: float = 0.0
    similarity_method: str = "cosine"  # cosine, jaccard, semantic, etc.
    threshold_used: float = 0.7
    
    def __post_init__(self):
        if self.edge_type != EdgeType.SEMANTIC_SIM:
            self.edge_type = EdgeType.SEMANTIC_SIM
        # Weight should be similarity score
        if self.similarity_score > 0:
            self.weight = self.similarity_score
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.SEMANTIC_SIM,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", True),  # Similarity is typically bidirectional
            similarity_score=data.get("similarity_score", 0.0),
            similarity_method=data.get("similarity_method", "cosine"),
            threshold_used=data.get("threshold_used", 0.7)
        )


@dataclass
class EntityRelationEdge(Edge):
    """Edge representing relationships between entities"""
    relation_type: str = "RELATED_TO"  # WORKS_FOR, LOCATED_IN, PART_OF, etc.
    relation_confidence: float = 1.0
    relation_context: str = ""  # context where relation was found
    
    def __post_init__(self):
        if self.edge_type != EdgeType.ENTITY_RELATION:
            self.edge_type = EdgeType.ENTITY_RELATION
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityRelationEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.ENTITY_RELATION,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False),
            relation_type=data.get("relation_type", "RELATED_TO"),
            relation_confidence=data.get("relation_confidence", 1.0),
            relation_context=data.get("relation_context", "")
        )


@dataclass
class CoReferenceEdge(Edge):
    """Edge representing coreference relationships"""
    coreference_type: str = "pronoun"  # pronoun, nominal, proper_noun
    mention_text: str = ""
    resolution_confidence: float = 1.0
    
    def __post_init__(self):
        if self.edge_type != EdgeType.CO_REFERENCE:
            self.edge_type = EdgeType.CO_REFERENCE
        self.bidirectional = True  # Coreference is bidirectional
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoReferenceEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.CO_REFERENCE,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", True),
            coreference_type=data.get("coreference_type", "pronoun"),
            mention_text=data.get("mention_text", ""),
            resolution_confidence=data.get("resolution_confidence", 1.0)
        )


@dataclass
class CrossDocEdge(Edge):
    """Edge representing cross-document relationships"""
    doc1_id: str = ""
    doc2_id: str = ""
    match_type: str = "entity_match"  # entity_match, topic_match, citation
    match_confidence: float = 1.0
    
    def __post_init__(self):
        if self.edge_type != EdgeType.CROSS_DOC:
            self.edge_type = EdgeType.CROSS_DOC
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrossDocEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.CROSS_DOC,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False),
            doc1_id=data.get("doc1_id", ""),
            doc2_id=data.get("doc2_id", ""),
            match_type=data.get("match_type", "entity_match"),
            match_confidence=data.get("match_confidence", 1.0)
        )


@dataclass
class TableContextEdge(Edge):
    """Edge connecting tables with their context"""
    context_type: str = "explains"  # explains, introduces, summarizes, references
    context_position: str = "before"  # before, after, above, below
    distance_sentences: int = 0
    
    def __post_init__(self):
        if self.edge_type != EdgeType.TABLE_CONTEXT:
            self.edge_type = EdgeType.TABLE_CONTEXT
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TableContextEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.TABLE_CONTEXT,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False),
            context_type=data.get("context_type", "explains"),
            context_position=data.get("context_position", "before"),
            distance_sentences=data.get("distance_sentences", 0)
        )


@dataclass
class FigureContextEdge(Edge):
    """Edge connecting figures with their context"""
    context_type: str = "caption"  # caption, reference, description
    context_position: str = "below"  # above, below, left, right
    
    def __post_init__(self):
        if self.edge_type != EdgeType.FIGURE_CONTEXT:
            self.edge_type = EdgeType.FIGURE_CONTEXT
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FigureContextEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.FIGURE_CONTEXT,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False),
            context_type=data.get("context_type", "caption"),
            context_position=data.get("context_position", "below")
        )


# Factory function for creating edges
def create_edge(edge_type: EdgeType, **kwargs) -> Edge:
    """Factory function to create appropriate edge type"""
    edge_classes = {
        EdgeType.SEQUENCE: SequenceEdge,
        EdgeType.CONTAINS: ContainsEdge,
        EdgeType.REFERS_TO: ReferenceEdge,
        EdgeType.SEMANTIC_SIM: SemanticEdge,
        EdgeType.ENTITY_RELATION: EntityRelationEdge,
        EdgeType.CO_REFERENCE: CoReferenceEdge,
        EdgeType.CROSS_DOC: CrossDocEdge,
        EdgeType.TABLE_CONTEXT: TableContextEdge,
        EdgeType.FIGURE_CONTEXT: FigureContextEdge
    }
    
    edge_class = edge_classes.get(edge_type)
    if not edge_class:
        raise ValueError(f"Unknown edge type: {edge_type}")
    
    return edge_class(edge_type=edge_type, **kwargs)


def edge_from_dict(data: Dict[str, Any]) -> Edge:
    """Create edge from dictionary representation"""
    edge_type = EdgeType(data["edge_type"])
    
    edge_classes = {
        EdgeType.SEQUENCE: SequenceEdge,
        EdgeType.CONTAINS: ContainsEdge,
        EdgeType.REFERS_TO: ReferenceEdge,
        EdgeType.SEMANTIC_SIM: SemanticEdge,
        EdgeType.ENTITY_RELATION: EntityRelationEdge,
        EdgeType.CO_REFERENCE: CoReferenceEdge,
        EdgeType.CROSS_DOC: CrossDocEdge,
        EdgeType.TABLE_CONTEXT: TableContextEdge,
        EdgeType.FIGURE_CONTEXT: FigureContextEdge
    }
    
    edge_class = edge_classes.get(edge_type)
    if edge_class:
        return edge_class.from_dict(data)
    else:
        # For unsupported edge types, create a basic Edge
        return Edge(
            edge_id=data["edge_id"],
            edge_type=edge_type,
            source_node_id=data.get("source_node_id", data.get("source_node", "")),
            target_node_id=data.get("target_node_id", data.get("target_node", "")),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {})
        )


# Graph motif patterns for task generation
@dataclass
class GraphMotif:
    """Represents a graph pattern/motif for task generation"""
    motif_id: str
    name: str
    description: str
    node_pattern: List[str]  # Required node types
    edge_pattern: List[str]  # Required edge types
    min_nodes: int = 2
    max_nodes: int = 10
    complexity_score: float = 1.0
    
    def matches_subgraph(self, nodes: List[str], edges: List[str]) -> bool:
        """Check if a subgraph matches this motif pattern"""
        # Simple pattern matching - can be enhanced
        node_types_present = set(nodes)
        edge_types_present = set(edges)
        
        required_nodes = set(self.node_pattern)
        required_edges = set(self.edge_pattern)
        
        return (required_nodes.issubset(node_types_present) and 
                required_edges.issubset(edge_types_present) and
                self.min_nodes <= len(nodes) <= self.max_nodes)


# Common motif patterns
COMMON_MOTIFS = [
    GraphMotif(
        motif_id="sequential_path",
        name="Sequential Path",
        description="A sequence of connected nodes",
        node_pattern=["paragraph", "paragraph"],
        edge_pattern=["sequence"],
        min_nodes=2,
        max_nodes=5,
        complexity_score=1.0
    ),
    GraphMotif(
        motif_id="table_with_context",
        name="Table with Context",
        description="Table connected to explanatory text",
        node_pattern=["table", "paragraph"],
        edge_pattern=["table_context"],
        min_nodes=2,
        max_nodes=4,
        complexity_score=2.0
    ),
    GraphMotif(
        motif_id="entity_cluster",
        name="Entity Cluster",
        description="Multiple entities with relationships",
        node_pattern=["entity", "entity"],
        edge_pattern=["entity_relation"],
        min_nodes=2,
        max_nodes=6,
        complexity_score=2.5
    ),
    GraphMotif(
        motif_id="hierarchical_section",
        name="Hierarchical Section",
        description="Heading with subordinate content",
        node_pattern=["heading", "paragraph"],
        edge_pattern=["contains"],
        min_nodes=2,
        max_nodes=8,
        complexity_score=1.5
    ),
    GraphMotif(
        motif_id="semantic_cluster",
        name="Semantic Cluster",
        description="Semantically similar content",
        node_pattern=["paragraph", "paragraph"],
        edge_pattern=["semantic_sim"],
        min_nodes=3,
        max_nodes=7,
        complexity_score=2.0
    ),
    GraphMotif(
        motif_id="cross_reference",
        name="Cross Reference",
        description="Nodes with explicit references",
        node_pattern=["paragraph", "paragraph"],
        edge_pattern=["refers_to"],
        min_nodes=2,
        max_nodes=5,
        complexity_score=1.8
    )
]


@dataclass
class WebNavigationEdge(Edge):
    """Edge representing web page navigation"""
    navigation_type: str = "link"  # link, form_submit, button_click
    target_url: str = ""
    navigation_method: str = "GET"  # GET, POST, etc.
    
    def __post_init__(self):
        if self.edge_type != EdgeType.WEB_NAVIGATION:
            self.edge_type = EdgeType.WEB_NAVIGATION
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebNavigationEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.WEB_NAVIGATION,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False),
            navigation_type=data.get("navigation_type", "link"),
            target_url=data.get("target_url", ""),
            navigation_method=data.get("navigation_method", "GET")
        )


@dataclass
class WebInteractionEdge(Edge):
    """Edge representing user interaction with web elements"""
    interaction_type: str = "click"  # click, input, hover, etc.
    interaction_data: Dict[str, Any] = field(default_factory=dict)
    interaction_result: str = ""  # success, failure, redirect, etc.
    
    def __post_init__(self):
        if self.edge_type != EdgeType.WEB_INTERACTION:
            self.edge_type = EdgeType.WEB_INTERACTION
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebInteractionEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.WEB_INTERACTION,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False),
            interaction_type=data.get("interaction_type", "click"),
            interaction_data=data.get("interaction_data", {}),
            interaction_result=data.get("interaction_result", "")
        )


@dataclass
class WebFormSubmitEdge(Edge):
    """Edge representing form submission"""
    form_data: Dict[str, Any] = field(default_factory=dict)
    submit_method: str = "POST"
    validation_result: str = "success"  # success, validation_error, etc.
    
    def __post_init__(self):
        if self.edge_type != EdgeType.WEB_FORM_SUBMIT:
            self.edge_type = EdgeType.WEB_FORM_SUBMIT
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebFormSubmitEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.WEB_FORM_SUBMIT,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False),
            form_data=data.get("form_data", {}),
            submit_method=data.get("submit_method", "POST"),
            validation_result=data.get("validation_result", "success")
        )


@dataclass
class WebClickTriggerEdge(Edge):
    """Edge representing click-triggered actions"""
    trigger_action: str = ""  # show_modal, hide_element, update_content, etc.
    trigger_condition: str = ""  # always, on_valid_input, etc.
    
    def __post_init__(self):
        if self.edge_type != EdgeType.WEB_CLICK_TRIGGER:
            self.edge_type = EdgeType.WEB_CLICK_TRIGGER
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebClickTriggerEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.WEB_CLICK_TRIGGER,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False),
            trigger_action=data.get("trigger_action", ""),
            trigger_condition=data.get("trigger_condition", "")
        )


@dataclass
class WebDataFlowEdge(Edge):
    """Edge representing data flow between elements"""
    data_type: str = ""  # text, number, selection, etc.
    data_transformation: str = ""  # copy, validate, format, etc.
    data_validation: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.edge_type != EdgeType.WEB_DATA_FLOW:
            self.edge_type = EdgeType.WEB_DATA_FLOW
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebDataFlowEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.WEB_DATA_FLOW,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False),
            data_type=data.get("data_type", ""),
            data_transformation=data.get("data_transformation", ""),
            data_validation=data.get("data_validation", {})
        )


@dataclass
class WebLayoutEdge(Edge):
    """Edge representing spatial layout relationships"""
    layout_type: str = ""  # above, below, left, right, inside, etc.
    distance: float = 0.0  # spatial distance
    alignment: str = ""  # horizontal, vertical, diagonal
    
    def __post_init__(self):
        if self.edge_type != EdgeType.WEB_LAYOUT:
            self.edge_type = EdgeType.WEB_LAYOUT
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebLayoutEdge':
        return cls(
            edge_id=data["edge_id"],
            edge_type=EdgeType.WEB_LAYOUT,
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False),
            layout_type=data.get("layout_type", ""),
            distance=data.get("distance", 0.0),
            alignment=data.get("alignment", "")
        )


def create_web_edge(edge_type: EdgeType, **kwargs) -> Edge:
    """Factory function to create web edges"""
    edge_classes = {
        EdgeType.WEB_NAVIGATION: WebNavigationEdge,
        EdgeType.WEB_INTERACTION: WebInteractionEdge,
        EdgeType.WEB_FORM_SUBMIT: WebFormSubmitEdge,
        EdgeType.WEB_CLICK_TRIGGER: WebClickTriggerEdge,
        EdgeType.WEB_DATA_FLOW: WebDataFlowEdge,
        EdgeType.WEB_LAYOUT: WebLayoutEdge,
    }
    
    if edge_type not in edge_classes:
        raise ValueError(f"Unknown web edge type: {edge_type}")
    
    return edge_classes[edge_type](**kwargs)


class GraphEdge:
    """图边 - 代表节点间的交互关系"""
    
    def __init__(self, edge_id: str, source_node_id: str, target_node_id: str, 
                 edge_type: EdgeType, weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        self.edge_id = edge_id
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.edge_type = edge_type
        self.weight = weight
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node": self.source_node_id,
            "target_node": self.target_node_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata
        }


 
