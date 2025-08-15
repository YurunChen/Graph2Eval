"""
Graph storage backends for GraphRAG
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from pathlib import Path
import pickle

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from loguru import logger

from .node_types import Node, NodeType, node_from_dict
from .edge_types import Edge, EdgeType, edge_from_dict


class GraphStorage(ABC):
    """Abstract base class for graph storage backends"""
    
    @abstractmethod
    def add_node(self, node: Node):
        """Add a node to storage"""
        pass
    
    @abstractmethod
    def add_edge(self, edge: Edge):
        """Add an edge to storage"""
        pass
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID"""
        pass
    
    @abstractmethod
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get edge by ID"""
        pass
    
    @abstractmethod
    def get_nodes(self, node_ids: List[str]) -> List[Node]:
        """Get multiple nodes by IDs"""
        pass
    
    @abstractmethod
    def get_edges(self, edge_ids: List[str]) -> List[Edge]:
        """Get multiple edges by IDs"""
        pass
    
    @abstractmethod
    def find_nodes(self, **criteria) -> List[Node]:
        """Find nodes matching criteria"""
        pass
    
    @abstractmethod
    def find_edges(self, **criteria) -> List[Edge]:
        """Find edges matching criteria"""
        pass
    
    @abstractmethod
    def get_neighbors(self, node_id: str, edge_types: Optional[List[EdgeType]] = None) -> List[Tuple[Node, Edge]]:
        """Get neighboring nodes and connecting edges"""
        pass
    
    @abstractmethod
    def get_subgraph(self, node_ids: List[str], max_hops: int = 1) -> Tuple[List[Node], List[Edge]]:
        """Get subgraph around given nodes"""
        pass
    
    @abstractmethod
    def remove_node(self, node_id: str):
        """Remove node and associated edges"""
        pass
    
    @abstractmethod
    def remove_edge(self, edge_id: str):
        """Remove edge"""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all data"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save graph to file"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load graph from file"""
        pass


class JSONStorage(GraphStorage):
    """JSON-based graph storage for lightweight usage"""
    
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.nodes = {}  # node_id -> Node
        self.edges = {}  # edge_id -> Edge
        self.node_edges = {}  # node_id -> set of edge_ids
        
        # Only auto-load if file_path is a file (not a directory)
        if file_path and Path(file_path).exists() and Path(file_path).is_file():
            self.load(file_path)
    
    def add_node(self, node: Node):
        """Add a node to storage"""
        self.nodes[node.node_id] = node
        if node.node_id not in self.node_edges:
            self.node_edges[node.node_id] = set()
    
    def add_edge(self, edge: Edge):
        """Add an edge to storage"""
        self.edges[edge.edge_id] = edge
        
        # Update node-edge mappings
        if edge.source_node_id not in self.node_edges:
            self.node_edges[edge.source_node_id] = set()
        if edge.target_node_id not in self.node_edges:
            self.node_edges[edge.target_node_id] = set()
        
        self.node_edges[edge.source_node_id].add(edge.edge_id)
        self.node_edges[edge.target_node_id].add(edge.edge_id)
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID"""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get edge by ID"""
        return self.edges.get(edge_id)
    
    def get_nodes(self, node_ids: List[str]) -> List[Node]:
        """Get multiple nodes by IDs"""
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_edges(self, edge_ids: List[str]) -> List[Edge]:
        """Get multiple edges by IDs"""
        return [self.edges[eid] for eid in edge_ids if eid in self.edges]
    
    def find_nodes(self, **criteria) -> List[Node]:
        """Find nodes matching criteria"""
        results = []
        
        for node in self.nodes.values():
            match = True
            
            for key, value in criteria.items():
                if key == "node_type":
                    if isinstance(value, str):
                        value = NodeType(value)
                    if node.node_type != value:
                        match = False
                        break
                elif key == "source_file":
                    if node.source_file != value:
                        match = False
                        break
                elif key == "page_num":
                    if node.page_num != value:
                        match = False
                        break
                elif key == "content_contains":
                    if value.lower() not in node.content.lower():
                        match = False
                        break
                elif hasattr(node, key):
                    if getattr(node, key) != value:
                        match = False
                        break
            
            if match:
                results.append(node)
        
        return results
    
    def find_edges(self, **criteria) -> List[Edge]:
        """Find edges matching criteria"""
        results = []
        
        for edge in self.edges.values():
            match = True
            
            for key, value in criteria.items():
                if key == "edge_type":
                    if isinstance(value, str):
                        value = EdgeType(value)
                    if edge.edge_type != value:
                        match = False
                        break
                elif key == "source_node_id":
                    if edge.source_node_id != value:
                        match = False
                        break
                elif key == "target_node_id":
                    if edge.target_node_id != value:
                        match = False
                        break
                elif hasattr(edge, key):
                    if getattr(edge, key) != value:
                        match = False
                        break
            
            if match:
                results.append(edge)
        
        return results
    
    def get_neighbors(self, node_id: str, edge_types: Optional[List[EdgeType]] = None) -> List[Tuple[Node, Edge]]:
        """Get neighboring nodes and connecting edges"""
        neighbors = []
        
        if node_id not in self.node_edges:
            return neighbors
        
        for edge_id in self.node_edges[node_id]:
            edge = self.edges.get(edge_id)
            if not edge:
                continue
            
            # Filter by edge type
            if edge_types and edge.edge_type not in edge_types:
                continue
            
            # Get neighbor node
            neighbor_id = edge.target_node_id if edge.source_node_id == node_id else edge.source_node_id
            neighbor = self.nodes.get(neighbor_id)
            
            if neighbor:
                neighbors.append((neighbor, edge))
        
        return neighbors
    
    def get_subgraph(self, node_ids: List[str], max_hops: int = 1) -> Tuple[List[Node], List[Edge]]:
        """Get subgraph around given nodes"""
        visited_nodes = set(node_ids)
        current_nodes = set(node_ids)
        
        # Expand by hops
        for hop in range(max_hops):
            next_nodes = set()
            
            for node_id in current_nodes:
                neighbors = self.get_neighbors(node_id)
                for neighbor, _ in neighbors:
                    if neighbor.node_id not in visited_nodes:
                        next_nodes.add(neighbor.node_id)
            
            visited_nodes.update(next_nodes)
            current_nodes = next_nodes
            
            if not current_nodes:  # No more nodes to expand
                break
        
        # Get all nodes and edges in subgraph
        subgraph_nodes = [self.nodes[nid] for nid in visited_nodes if nid in self.nodes]
        
        subgraph_edges = []
        for edge in self.edges.values():
            if (edge.source_node_id in visited_nodes and 
                edge.target_node_id in visited_nodes):
                subgraph_edges.append(edge)
        
        return subgraph_nodes, subgraph_edges
    
    def remove_node(self, node_id: str):
        """Remove node and associated edges"""
        if node_id not in self.nodes:
            return
        
        # Remove associated edges
        if node_id in self.node_edges:
            edges_to_remove = list(self.node_edges[node_id])
            for edge_id in edges_to_remove:
                self.remove_edge(edge_id)
        
        # Remove node
        del self.nodes[node_id]
        if node_id in self.node_edges:
            del self.node_edges[node_id]
    
    def remove_edge(self, edge_id: str):
        """Remove edge"""
        if edge_id not in self.edges:
            return
        
        edge = self.edges[edge_id]
        
        # Update node-edge mappings
        if edge.source_node_id in self.node_edges:
            self.node_edges[edge.source_node_id].discard(edge_id)
        if edge.target_node_id in self.node_edges:
            self.node_edges[edge.target_node_id].discard(edge_id)
        
        # Remove edge
        del self.edges[edge_id]
    
    def clear(self):
        """Clear all data"""
        self.nodes.clear()
        self.edges.clear()
        self.node_edges.clear()
    
    def save(self, path: str):
        """Save graph to JSON file"""
        try:
            graph_data = {
                "nodes": [node.to_dict() for node in self.nodes.values()],
                "edges": [edge.to_dict() for edge in self.edges.values()],
                "metadata": {
                    "total_nodes": len(self.nodes),
                    "total_edges": len(self.edges),
                    "node_types": {nt.value: len([n for n in self.nodes.values() if n.node_type == nt]) 
                                  for nt in NodeType},
                    "edge_types": {et.value: len([e for e in self.edges.values() if e.edge_type == et]) 
                                  for et in EdgeType}
                }
            }
            
            # Custom JSON encoder to handle numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, 'item'):  # numpy scalars
                        return obj.item()
                    elif hasattr(obj, 'tolist'):  # numpy arrays
                        return obj.tolist()
                    return super().default(obj)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.info(f"Saved graph to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            raise
    
    def load(self, path: str):
        """Load graph from JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # Clear existing data
            self.clear()
            
            # Load nodes
            for node_data in graph_data.get("nodes", []):
                node = node_from_dict(node_data)
                self.add_node(node)
            
            # Load edges
            for edge_data in graph_data.get("edges", []):
                edge = edge_from_dict(edge_data)
                self.add_edge(edge)
            
            logger.info(f"Loaded graph from {path}")
            logger.info(f"Nodes: {len(self.nodes)}, Edges: {len(self.edges)}")
            
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": {nt.value: len([n for n in self.nodes.values() if n.node_type == nt]) 
                          for nt in NodeType},
            "edge_types": {et.value: len([e for e in self.edges.values() if e.edge_type == et]) 
                          for et in EdgeType}
        }


class SQLiteStorage(GraphStorage):
    """SQLite-based graph storage for better query performance"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Nodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                source_file TEXT,
                page_num INTEGER,
                bbox TEXT,
                embedding BLOB
            )
        ''')
        
        # Edges table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                edge_type TEXT NOT NULL,
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata TEXT,
                bidirectional BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (source_node_id) REFERENCES nodes (node_id),
                FOREIGN KEY (target_node_id) REFERENCES nodes (node_id)
            )
        ''')
        
        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes (node_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes (source_file)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_type ON edges (edge_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_source ON edges (source_node_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target_node_id)')
        
        self.conn.commit()
    
    def add_node(self, node: Node):
        """Add a node to storage"""
        cursor = self.conn.cursor()
        
        # Serialize complex data
        metadata_json = json.dumps(node.metadata) if node.metadata else None
        bbox_json = json.dumps(node.bbox) if node.bbox else None
        embedding_blob = pickle.dumps(node.embedding) if node.embedding is not None else None
        
        cursor.execute('''
            INSERT OR REPLACE INTO nodes 
            (node_id, node_type, content, metadata, source_file, page_num, bbox, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            node.node_id,
            node.node_type.value,
            node.content,
            metadata_json,
            node.source_file,
            node.page_num,
            bbox_json,
            embedding_blob
        ))
        
        self.conn.commit()
    
    def add_edge(self, edge: Edge):
        """Add an edge to storage"""
        cursor = self.conn.cursor()
        
        metadata_json = json.dumps(edge.metadata) if edge.metadata else None
        
        cursor.execute('''
            INSERT OR REPLACE INTO edges 
            (edge_id, edge_type, source_node_id, target_node_id, weight, metadata, bidirectional)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            edge.edge_id,
            edge.edge_type.value,
            edge.source_node_id,
            edge.target_node_id,
            edge.weight,
            metadata_json,
            edge.bidirectional
        ))
        
        self.conn.commit()
    
    def _row_to_node(self, row) -> Node:
        """Convert database row to Node object"""
        node_data = {
            "node_id": row["node_id"],
            "node_type": row["node_type"],
            "content": row["content"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "source_file": row["source_file"],
            "page_num": row["page_num"],
            "bbox": json.loads(row["bbox"]) if row["bbox"] else None
        }
        
        node = node_from_dict(node_data)
        
        # Load embedding
        if row["embedding"]:
            node.embedding = pickle.loads(row["embedding"])
        
        return node
    
    def _row_to_edge(self, row) -> Edge:
        """Convert database row to Edge object"""
        edge_data = {
            "edge_id": row["edge_id"],
            "edge_type": row["edge_type"],
            "source_node_id": row["source_node_id"],
            "target_node_id": row["target_node_id"],
            "weight": row["weight"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "bidirectional": bool(row["bidirectional"])
        }
        
        return edge_from_dict(edge_data)
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM nodes WHERE node_id = ?', (node_id,))
        row = cursor.fetchone()
        
        return self._row_to_node(row) if row else None
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get edge by ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM edges WHERE edge_id = ?', (edge_id,))
        row = cursor.fetchone()
        
        return self._row_to_edge(row) if row else None
    
    def get_nodes(self, node_ids: List[str]) -> List[Node]:
        """Get multiple nodes by IDs"""
        if not node_ids:
            return []
        
        cursor = self.conn.cursor()
        placeholders = ','.join(['?'] * len(node_ids))
        cursor.execute(f'SELECT * FROM nodes WHERE node_id IN ({placeholders})', node_ids)
        
        return [self._row_to_node(row) for row in cursor.fetchall()]
    
    def get_edges(self, edge_ids: List[str]) -> List[Edge]:
        """Get multiple edges by IDs"""
        if not edge_ids:
            return []
        
        cursor = self.conn.cursor()
        placeholders = ','.join(['?'] * len(edge_ids))
        cursor.execute(f'SELECT * FROM edges WHERE edge_id IN ({placeholders})', edge_ids)
        
        return [self._row_to_edge(row) for row in cursor.fetchall()]
    
    def find_nodes(self, **criteria) -> List[Node]:
        """Find nodes matching criteria"""
        cursor = self.conn.cursor()
        
        where_clauses = []
        params = []
        
        for key, value in criteria.items():
            if key == "node_type":
                if isinstance(value, NodeType):
                    value = value.value
                where_clauses.append("node_type = ?")
                params.append(value)
            elif key == "source_file":
                where_clauses.append("source_file = ?")
                params.append(value)
            elif key == "page_num":
                where_clauses.append("page_num = ?")
                params.append(value)
            elif key == "content_contains":
                where_clauses.append("content LIKE ?")
                params.append(f"%{value}%")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        query = f"SELECT * FROM nodes WHERE {where_clause}"
        
        cursor.execute(query, params)
        return [self._row_to_node(row) for row in cursor.fetchall()]
    
    def find_edges(self, **criteria) -> List[Edge]:
        """Find edges matching criteria"""
        cursor = self.conn.cursor()
        
        where_clauses = []
        params = []
        
        for key, value in criteria.items():
            if key == "edge_type":
                if isinstance(value, EdgeType):
                    value = value.value
                where_clauses.append("edge_type = ?")
                params.append(value)
            elif key == "source_node_id":
                where_clauses.append("source_node_id = ?")
                params.append(value)
            elif key == "target_node_id":
                where_clauses.append("target_node_id = ?")
                params.append(value)
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        query = f"SELECT * FROM edges WHERE {where_clause}"
        
        cursor.execute(query, params)
        return [self._row_to_edge(row) for row in cursor.fetchall()]
    
    def get_neighbors(self, node_id: str, edge_types: Optional[List[EdgeType]] = None) -> List[Tuple[Node, Edge]]:
        """Get neighboring nodes and connecting edges"""
        cursor = self.conn.cursor()
        
        # Build query with edge type filter
        edge_type_filter = ""
        params = [node_id, node_id]
        
        if edge_types:
            edge_type_values = [et.value for et in edge_types]
            placeholders = ','.join(['?'] * len(edge_type_values))
            edge_type_filter = f" AND e.edge_type IN ({placeholders})"
            params.extend(edge_type_values)
        
        query = f'''
            SELECT n.*, e.* FROM nodes n
            JOIN edges e ON (
                (e.source_node_id = ? AND e.target_node_id = n.node_id) OR
                (e.target_node_id = ? AND e.source_node_id = n.node_id)
            )
            WHERE n.node_id != ?{edge_type_filter}
        '''
        params.append(node_id)
        
        cursor.execute(query, params)
        
        neighbors = []
        for row in cursor.fetchall():
            # Split row data for node and edge
            node_data = dict(row)
            edge_data = {
                "edge_id": row["edge_id"],
                "edge_type": row["edge_type"],
                "source_node_id": row["source_node_id"],
                "target_node_id": row["target_node_id"],
                "weight": row["weight"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "bidirectional": bool(row["bidirectional"])
            }
            
            node = self._row_to_node(row)
            edge = edge_from_dict(edge_data)
            
            neighbors.append((node, edge))
        
        return neighbors
    
    def get_subgraph(self, node_ids: List[str], max_hops: int = 1) -> Tuple[List[Node], List[Edge]]:
        """Get subgraph around given nodes"""
        visited_nodes = set(node_ids)
        current_nodes = set(node_ids)
        
        # Expand by hops
        for hop in range(max_hops):
            if not current_nodes:
                break
                
            cursor = self.conn.cursor()
            placeholders = ','.join(['?'] * len(current_nodes))
            
            # Get all neighbors of current nodes
            query = f'''
                SELECT DISTINCT 
                    CASE 
                        WHEN source_node_id IN ({placeholders}) THEN target_node_id
                        ELSE source_node_id
                    END as neighbor_id
                FROM edges 
                WHERE source_node_id IN ({placeholders}) OR target_node_id IN ({placeholders})
            '''
            
            params = list(current_nodes) * 3
            cursor.execute(query, params)
            
            next_nodes = set()
            for row in cursor.fetchall():
                neighbor_id = row["neighbor_id"]
                if neighbor_id not in visited_nodes:
                    next_nodes.add(neighbor_id)
            
            visited_nodes.update(next_nodes)
            current_nodes = next_nodes
        
        # Get all nodes and edges in subgraph
        subgraph_nodes = self.get_nodes(list(visited_nodes))
        
        cursor = self.conn.cursor()
        placeholders = ','.join(['?'] * len(visited_nodes))
        cursor.execute(f'''
            SELECT * FROM edges 
            WHERE source_node_id IN ({placeholders}) AND target_node_id IN ({placeholders})
        ''', list(visited_nodes) * 2)
        
        subgraph_edges = [self._row_to_edge(row) for row in cursor.fetchall()]
        
        return subgraph_nodes, subgraph_edges
    
    def remove_node(self, node_id: str):
        """Remove node and associated edges"""
        cursor = self.conn.cursor()
        
        # Remove associated edges
        cursor.execute('DELETE FROM edges WHERE source_node_id = ? OR target_node_id = ?', 
                      (node_id, node_id))
        
        # Remove node
        cursor.execute('DELETE FROM nodes WHERE node_id = ?', (node_id,))
        
        self.conn.commit()
    
    def remove_edge(self, edge_id: str):
        """Remove edge"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM edges WHERE edge_id = ?', (edge_id,))
        self.conn.commit()
    
    def clear(self):
        """Clear all data"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM edges')
        cursor.execute('DELETE FROM nodes')
        self.conn.commit()
    
    def save(self, path: str):
        """Save database to file"""
        # SQLite database is already persisted
        logger.info(f"SQLite database already saved to {self.db_path}")
    
    def load(self, path: str):
        """Load database from file"""
        # Close current connection and open new one
        self.conn.close()
        self.db_path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Loaded SQLite database from {path}")
    
    def close(self):
        """Close database connection"""
        self.conn.close()


if NEO4J_AVAILABLE:
    class Neo4jStorage(GraphStorage):
        """Neo4j-based graph storage for advanced graph queries"""
        
        def __init__(self, uri: str, user: str, password: str):
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self._create_constraints()
        
        def _create_constraints(self):
            """Create database constraints"""
            with self.driver.session() as session:
                # Create uniqueness constraints
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.node_id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR ()-[e:Edge]-() REQUIRE e.edge_id IS UNIQUE")
                
                # Create indices
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.node_type)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.source_file)")
        
        def add_node(self, node: Node):
            """Add a node to Neo4j"""
            with self.driver.session() as session:
                session.run('''
                    MERGE (n:Node {node_id: $node_id})
                    SET n.node_type = $node_type,
                        n.content = $content,
                        n.metadata = $metadata,
                        n.source_file = $source_file,
                        n.page_num = $page_num,
                        n.bbox = $bbox
                ''', {
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "content": node.content,
                    "metadata": json.dumps(node.metadata),
                    "source_file": node.source_file,
                    "page_num": node.page_num,
                    "bbox": json.dumps(node.bbox) if node.bbox else None
                })
        
        def add_edge(self, edge: Edge):
            """Add an edge to Neo4j"""
            with self.driver.session() as session:
                session.run('''
                    MATCH (source:Node {node_id: $source_id})
                    MATCH (target:Node {node_id: $target_id})
                    MERGE (source)-[e:Edge {edge_id: $edge_id}]->(target)
                    SET e.edge_type = $edge_type,
                        e.weight = $weight,
                        e.metadata = $metadata,
                        e.bidirectional = $bidirectional
                ''', {
                    "source_id": edge.source_node_id,
                    "target_id": edge.target_node_id,
                    "edge_id": edge.edge_id,
                    "edge_type": edge.edge_type.value,
                    "weight": edge.weight,
                    "metadata": json.dumps(edge.metadata),
                    "bidirectional": edge.bidirectional
                })
        
        # Other methods would be implemented similarly...
        # For brevity, showing just the basic structure
        
        def close(self):
            """Close Neo4j driver"""
            self.driver.close()
else:
    class Neo4jStorage(GraphStorage):
        """Placeholder for Neo4j storage when neo4j package is not available"""
        
        def __init__(self, *args, **kwargs):
            raise ImportError("Neo4j package not available. Install with: pip install neo4j")
        
        # Implement abstract methods with NotImplementedError
        def add_node(self, node: Node):
            raise NotImplementedError
        
        def add_edge(self, edge: Edge):
            raise NotImplementedError
        
        def get_node(self, node_id: str) -> Optional[Node]:
            raise NotImplementedError
        
        def get_edge(self, edge_id: str) -> Optional[Edge]:
            raise NotImplementedError
        
        def get_nodes(self, node_ids: List[str]) -> List[Node]:
            raise NotImplementedError
        
        def get_edges(self, edge_ids: List[str]) -> List[Edge]:
            raise NotImplementedError
        
        def find_nodes(self, **criteria) -> List[Node]:
            raise NotImplementedError
        
        def find_edges(self, **criteria) -> List[Edge]:
            raise NotImplementedError
        
        def get_neighbors(self, node_id: str, edge_types: Optional[List[EdgeType]] = None) -> List[Tuple[Node, Edge]]:
            raise NotImplementedError
        
        def get_subgraph(self, node_ids: List[str], max_hops: int = 1) -> Tuple[List[Node], List[Edge]]:
            raise NotImplementedError
        
        def remove_node(self, node_id: str):
            raise NotImplementedError
        
        def remove_edge(self, edge_id: str):
            raise NotImplementedError
        
        def clear(self):
            raise NotImplementedError
        
        def save(self, path: str):
            raise NotImplementedError
        
        def load(self, path: str):
            raise NotImplementedError
