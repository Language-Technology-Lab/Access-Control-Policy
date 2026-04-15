"""Configuration, constants, and data models for the Access Control pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from pydantic import BaseModel, Field


# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths (single source of truth)
DATASETS_DIR = PROJECT_ROOT / "datasets"
SUBGRAPHS_IMAGES_DIR = DATASETS_DIR / "SubgraphsWithTriplesImages"
SUBGRAPHS_JSON_DIR = DATASETS_DIR / "SubgraphsWithTriplesJSON"
# Subset names for ground truth; image dirs are {subset} and {subset}_wo_legend → same JSON dir
DATASET_SUBSETS = ["subgraphs_01", "subgraphs_001", "subgraphs_06"]

DEFAULT_INPUT_PATH = str(DATASETS_DIR)
DEFAULT_OUTPUT_PATH = str(PROJECT_ROOT / "experiments")
DEFAULT_MODEL = "gpt-5-nano"
# Vision models supported for DAG image analysis
SUPPORTED_VISION_MODELS = ["gpt-5-nano", "gpt-5-mini", "gpt-4o-mini", "gpt-4o"]
# When primary model returns empty, try these in order; set to [] to disable fallback
FALLBACK_VISION_MODELS = []
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2048"))

VALID_METHODS = ["extract_entities", "relation_classification", "relation_extraction", "extract_relation", "enumerate_paths", "path_generation"]
VALID_FEW_SHOT_MODES = ["zero", "few"]
VALID_RELATION_SOURCES = ["ground_truth", "predicted"]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """Represents an entity in the access control graph"""
    label: str = Field(..., description="Entity label/name")
    type: str = Field(..., description="Entity type")


class RelationshipResult(BaseModel):
    """Result of binary relationship classification"""
    exists: str = Field(..., description="Whether relationship exists (Yes/No)")
    confidence: str = Field(..., description="Confidence level")
    explanation: Optional[str] = Field(None, description="Explanation")
    source_image: str = Field(..., description="Source image filename")
    timestamp: str = Field(..., description="Processing timestamp")
    method: str = Field(default="relation_classification")
    entity1: str = Field(..., description="First entity label")
    entity1_type: str = Field(..., description="First entity type (user_attributes, object_attributes, policy_classes)")
    entity2: str = Field(..., description="Second entity label")
    entity2_type: str = Field(..., description="Second entity type (user_attributes, object_attributes, policy_classes)")
    relation: str = Field(..., description="Relation type (assign, permit, prohibit)")
    subrelations: Optional[List[str]] = Field(default_factory=list, description="Permission weights or subrelations (action subtypes)")
    groundtruth: Optional[str] = Field(None, description="Ground truth value (Yes/No) indicating if prediction matches ground truth")


class KnowledgeGraphNode(BaseModel):
    """Node in the knowledge graph"""
    id: str = Field(..., description="Unique node identifier")
    label: str = Field(..., description="Node display label")
    type: str = Field(..., description="Node type")


class KnowledgeGraphEdge(BaseModel):
    """Edge in the knowledge graph"""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    relation_type: Optional[str] = Field(None, description="Type of relationship")
    source_name: str = Field(..., description="Source node label")
    target_name: str = Field(..., description="Target node label")
    subrelations: Optional[List[str]] = Field(default_factory=list, description="Permission weights or subrelations")


@dataclass
class GraphPath:
    """Represents a path in the knowledge graph"""
    nodes: List[str] = field(default_factory=list)
    edges: List[Dict] = field(default_factory=list)
    description: Optional[str] = None


class KnowledgeGraph(BaseModel):
    """Complete knowledge graph structure"""
    nodes: List[KnowledgeGraphNode] = Field(default_factory=list)
    edges: List[KnowledgeGraphEdge] = Field(default_factory=list)
    paths: List[GraphPath] = Field(default_factory=list)
    graph_metadata: Dict = Field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    precision: float
    recall: float
    f1: Union[float, str]
    accuracy: Optional[float] = None
    confusion_matrix: Optional[Dict[str, int]] = None


@dataclass
class ProcessingResult:
    """Result of a processing operation"""
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    entities_extracted: Optional[int] = None
    metrics: Optional[EvaluationMetrics] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class BatchProcessingResult:
    """Result of batch processing operations"""
    level: str
    method: str
    timestamp: str
    processed: List[Dict]
    failed: List[Dict]
    evaluated: List[Dict]
    summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProcessingConfig:
    """Configuration for processing operations"""
    input_path: str = field(default=DEFAULT_INPUT_PATH)
    output_path: str = field(default=DEFAULT_OUTPUT_PATH)
    method: str = field(default="extract_entities")
    few_shot_mode: str = field(default="zero")
    relation_source: str = field(default="ground_truth")
    entities_input: Optional[str] = field(default=None)
    gt_input: Optional[str] = field(default=None)
    subset_size: Optional[int] = field(default=None)
    comprehensive_eval: bool = field(default=False)
    with_legend: bool = field(default=True)
    fuzzy_matching: bool = field(default=False)
    max_workers: int = field(default=4)

    # Derived properties
    few_shot_examples: Optional[List[Dict]] = None
    print_full_prompt: bool = False
    is_subgraphs_dataset: bool = False

    def __post_init__(self):
        """Initialize derived properties after dataclass creation."""
        self.print_full_prompt = (self.few_shot_mode == "few")
        self.is_subgraphs_dataset = "datasets" in Path(self.input_path).parts or "SubgraphsWithTriples" in Path(self.input_path).parts

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.method not in VALID_METHODS:
            raise ValueError(f"Invalid method: {self.method}. Must be one of {VALID_METHODS}")

        if self.few_shot_mode not in VALID_FEW_SHOT_MODES:
            raise ValueError(f"Invalid few_shot_mode: {self.few_shot_mode}. Must be 'zero' or 'few'")

        if self.relation_source not in VALID_RELATION_SOURCES:
            raise ValueError(f"Invalid relation_source: {self.relation_source}. Must be one of {VALID_RELATION_SOURCES}")

    def get_processing_description(self) -> str:
        """Get human-readable description of the processing configuration."""
        mode_desc = "Few-shot (Context7 sequential)" if self.few_shot_mode == "few" else "Zero-shot"
        return f"{mode_desc} {self.method.replace('_', ' ').title()}"


@dataclass
class APIConfig:
    """Configuration for API interactions"""
    api_key: Optional[str] = None
    model: str = field(default=DEFAULT_MODEL)
    max_tokens: int = 16384  # completion limit; larger helps avoid empty/truncated JSON for big graphs
    temperature: float = 1.0  # Use 1.0; many newer models only support default (1)
    image_detail: str = field(default="low")  # "low" (cost-efficient) or "high" (higher quality, more expensive)

    def __post_init__(self):
        """Initialize API key from .env file and environment if not provided."""
        if not self.api_key:
            # Load from .env file first
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass  # dotenv not available, continue with env var check

            # Then check environment variable
            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in .env file or environment variable")


@dataclass
class ImageConfig:
    """Configuration for image processing"""
    max_image_side: int = field(default=MAX_IMAGE_SIDE)
    supported_formats: List[str] = field(default_factory=lambda: ['png', 'jpg', 'jpeg', 'gif', 'webp'])
    image_detail: str = field(default="low")  # "low" or "high" - controls vision API detail level


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def create_output_directory(base_path: str, subdir: Optional[str] = None) -> Path:
    """Create and return output directory path"""
    output_path = Path(base_path)
    if subdir:
        output_path = output_path / subdir
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
