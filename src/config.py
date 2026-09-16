"""StructStd-RAG global configuration."""

import os
from pathlib import Path
from dataclasses import dataclass, field

# ============================================================
# Path configuration
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MAINTQA_DIR = DATA_DIR / "maintqa"
INDEX_DIR = DATA_DIR / "index"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

# ============================================================
# Model configuration
# ============================================================
EMBEDDING_MODEL = "BAAI/bge-m3"  # Multilingual embedding model for cross-lingual retrieval
EMBEDDING_DIM = 1024

# LLM API configuration (DashScope / OpenAI-compatible)
LLM_CONFIG = {
    "provider": "dashscope",
    "name": "qwen-turbo",  # Model: qwen-turbo / qwen-max
    "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "temperature": 0.0,
    "max_tokens": 2048,
}

# ============================================================
# Hyperparameter configuration
# ============================================================
@dataclass
class ChunkConfig:
    """Chunking hyperparameters."""
    fixed_size: int = 512
    fixed_overlap: int = 50
    recursive_size: int = 512
    recursive_overlap: int = 50
    semantic_threshold: float = 0.5
    adaptive_min_size: int = 128
    adaptive_max_size: int = 1024
    # StdChunk specific
    maint_section_max: int = 2048    # Layer 1: Section Block token limit
    maint_paragraph_max: int = 512   # Layer 2: Paragraph Block token limit
    maint_constraint_max: int = 256  # Layer 3: Constraint Block token limit


@dataclass
class RetrieverConfig:
    """Retriever hyperparameters."""
    top_k: int = 5
    recall_k: int = 50  # Stage 1 dense recall pool size
    w_dense: float = 0.6
    w_bm25: float = 0.4
    # CatRetriever specific
    type_match_boost: float = 0.3     # Category match reward (corresponds to paper's beta)
    type_mismatch_penalty: float = -0.2  # Category mismatch penalty (corresponds to paper's gamma)
    alpha: float = 1.0  # bge-m3 dev-set sweep optimum (was 0.5 for bge-large-zh); trust strong dense, keep category rerank
    beta: float = 0.5   # Global category-term scale (paper's lambda); effective reward = beta*type_match_boost = 0.15


@dataclass
class GeneratorConfig:
    """Generator hyperparameters."""
    temperature: float = 0.0
    max_tokens: int = 2048
    # BoundGrounder specific
    w_source: float = 0.2        # Source tracing weight
    w_plausibility: float = 0.5  # Physical plausibility weight (primary verification signal)
    w_consistency: float = 0.3   # Multi-source consistency weight
    confidence_threshold: float = 0.4  # Confidence threshold for graceful fallback


@dataclass
class ExperimentConfig:
    """Aggregated experiment configuration."""
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    seed: int = 42
    num_workers: int = 4


# Default configuration instance
DEFAULT_CONFIG = ExperimentConfig()
