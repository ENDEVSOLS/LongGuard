"""LongGuard core — step models, detectors, circuit breaker, and reporting."""

from .breaker import BreakerDecision, BreakerState, CircuitBreaker
from .detectors import (
    AbstractLoopDetector,
    DeadEndDriftDetector,
    DetectionResult,
    HashBasedEmbedder,
    SemanticOscillationDetector,
    SentenceTransformerEmbedder,
    TokenVelocityDetector,
    ToolRepeatDetector,
)
from .pivot import PIVOT_TEMPLATES, ReflectAndPivotInjector
from .reporter import GuardReport
from .step import AgentStep

__all__ = [
    "AgentStep",
    "AbstractLoopDetector",
    "DetectionResult",
    "ToolRepeatDetector",
    "SemanticOscillationDetector",
    "DeadEndDriftDetector",
    "TokenVelocityDetector",
    "HashBasedEmbedder",
    "SentenceTransformerEmbedder",
    "CircuitBreaker",
    "BreakerState",
    "BreakerDecision",
    "ReflectAndPivotInjector",
    "PIVOT_TEMPLATES",
    "GuardReport",
]
