"""
Access Control DAG Processing Package

This package provides modular components for processing Access Control DAG images
into structured knowledge graphs using vision-language models.
"""

from .config import (
    ProcessingConfig,
    APIConfig,
    ImageConfig,
    Entity,
    RelationshipResult,
    KnowledgeGraph,
    EvaluationMetrics,
    ProcessingResult,
    BatchProcessingResult
)

from .core_processor import AccessControlProcessor
from .processing_strategies import ProcessingStrategyFactory

__all__ = [
    # Configuration and models
    'ProcessingConfig',
    'APIConfig',
    'ImageConfig',
    'Entity',
    'RelationshipResult',
    'KnowledgeGraph',
    'EvaluationMetrics',
    'ProcessingResult',
    'BatchProcessingResult',

    # Core components
    'AccessControlProcessor',
    'ProcessingStrategyFactory',
]