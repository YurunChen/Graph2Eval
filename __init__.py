"""
GraphRAG + TaskCraft Agent Framework

A universal agent framework for automated task generation, evaluation, and optimization
of agentic LLMs with focus on context understanding, task utility, safety, and attribution.
"""

__version__ = "0.1.0"
__author__ = "Yu Runchen"

from .agent_framework import Agent
from .graph_rag import GraphRAG
from .task_craft import TaskGenerator, SafetyTaskGenerator

__all__ = ["Agent", "GraphRAG", "TaskGenerator", "SafetyTaskGenerator"]
