"""
Skill-3D Core Module

This module contains the core components of the Skill-3D architecture:
- SPAgent: Main agent class for problem solving
- Tool: Base class for external expert tools
- Model: Base class for VLLM model wrappers
- DataCollector: Training data collection for multimodal models
"""

from .agent import SPAgent
from .tool import Tool, ToolRegistry
from .model import Model
from .data_collector import DataCollector, InferenceSample, SessionData
from .skill_learning import AdaptiveSkillManager
from .skill_retrieval import SkillMemoryUnit, SkillRetrievalIndex

__all__ = [
    'SPAgent',
    'Tool',
    'ToolRegistry',
    'Model',
    'DataCollector',
    'InferenceSample',
    'SessionData',
    'AdaptiveSkillManager',
    'SkillMemoryUnit',
    'SkillRetrievalIndex',
]
