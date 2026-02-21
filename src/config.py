"""
config.py – load and validate a corpus config YAML.

Usage:
    from src.config import load_config
    cfg = load_config("configs/execution.yaml")
    print(cfg.corpus, cfg.chunk.chunk_size)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class IngestConfig(BaseModel):
    file_types: List[str] = [".pdf", ".txt", ".md"]


class ChunkConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64


class EmbedConfig(BaseModel):
    model: str = "text-embedding-3-small"
    batch_size: int = 64


class RetrieveConfig(BaseModel):
    top_k: int = 5
    score_threshold: float = 0.0


class AnswerConfig(BaseModel):
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.2
    max_tokens: int = 1024
    system_prompt: str = "You are a helpful assistant. Answer using only the provided context."


class EvalConfig(BaseModel):
    queries_file: str = "eval/queries/execution.csv"
    metrics: List[str] = ["hit_rate", "mrr", "faithfulness"]


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class Config(BaseModel):
    corpus: str
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    # Derived paths (not in YAML)
    @property
    def raw_dir(self) -> Path:
        return Path("data/raw") / self.corpus

    @property
    def processed_dir(self) -> Path:
        return Path("data/processed") / self.corpus

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def index_path(self) -> Path:
        return self.processed_dir / "faiss.index"

    @property
    def meta_path(self) -> Path:
        return self.processed_dir / "chunk_meta.json"

    @property
    def feedback_path(self) -> Path:
        return self.processed_dir / "feedback.csv"


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and return a validated Config object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)


def list_corpora(configs_dir: str | Path = "configs") -> List[str]:
    """
    Scan configs_dir for *.yaml files and return sorted corpus names (stems).
    Used by the Streamlit dropdown and any CLI that needs to enumerate corpora.
    """
    d = Path(configs_dir)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))
