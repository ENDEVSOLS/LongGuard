"""Detector registry — re-exports all loop detectors."""

from .base import AbstractLoopDetector, DetectionResult
from .dead_end import DeadEndDriftDetector
from .semantic_osc import (
    Embedder,
    HashBasedEmbedder,
    SemanticOscillationDetector,
    SentenceTransformerEmbedder,
)
from .token_velocity import TokenVelocityDetector
from .tool_repeat import ToolRepeatDetector

__all__ = [
    "AbstractLoopDetector",
    "DetectionResult",
    "ToolRepeatDetector",
    "SemanticOscillationDetector",
    "HashBasedEmbedder",
    "SentenceTransformerEmbedder",
    "Embedder",
    "DeadEndDriftDetector",
    "TokenVelocityDetector",
]
