"""SemanticOscillationDetector — detects when CoT reasoning cycles through the same thoughts."""

from __future__ import annotations

import hashlib
import struct
from collections import deque
from typing import Protocol, runtime_checkable

import numpy as np

from ..step import AgentStep
from .base import AbstractLoopDetector, DetectionResult


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding providers used by SemanticOscillationDetector."""

    def embed(self, text: str) -> list[float]:
        """Embed a text string into a float vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding.
        """
        ...


class HashBasedEmbedder:
    """Deterministic pseudo-embedding using hash functions.

    This is the default embedder — it works without any ML dependencies
    and produces deterministic vectors suitable for detecting semantic
    oscillation. It uses multiple hash rounds to generate a vector of
    the requested dimension.

    Not suitable for general-purpose semantic similarity, but effective
    for detecting near-identical or highly similar text where exact
    string matching is too brittle.

    Args:
        dimension: The dimensionality of the output vector. Default 64.
    """

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic pseudo-embedding from text.

        Uses SHA-256 with varying seeds to produce multiple float values,
        then normalizes the resulting vector.

        Args:
            text: The text to pseudo-embed.

        Returns:
            A list of ``dimension`` floats in approximately [-1, 1].
        """
        vector = np.zeros(self.dimension, dtype=np.float64)
        text_bytes = text.encode("utf-8")

        for i in range(self.dimension):
            # Vary the seed per dimension
            seed = struct.pack("!I", i)
            h = hashlib.sha256(seed + text_bytes).digest()
            # Convert 8 bytes to a float in [-1, 1]
            value = struct.unpack("!q", h[:8])[0] / (2**63)
            vector[i] = value

        # Normalize to unit vector for cosine-like behavior
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return [float(x) for x in vector.tolist()]


class SentenceTransformerEmbedder:
    """Embedder using sentence-transformers for high-quality semantic embeddings.

    Requires the ``sentence-transformers`` package. Install with:

        pip install longguard[embeddings]

    Args:
        model_name: The sentence-transformer model to use. Default is
            ``"all-MiniLM-L6-v2"`` — fast and effective for this use case.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install it with: pip install longguard[embeddings]"
            )
        self._model = SentenceTransformer(model_name)
        self._dimension: int | None = None

    def embed(self, text: str) -> list[float]:
        """Embed text using the sentence-transformer model.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding.
        """
        embedding = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in embedding.tolist()]


class SemanticOscillationDetector(AbstractLoopDetector):
    """Detects when the agent's chain-of-thought reasoning oscillates semantically.

    When an agent is stuck, its reasoning text often cycles through the same
    semantic territory — not identical text, but similar meaning expressed
    differently. This detector embeds each thought and tracks the variance
    across recent embeddings. Low variance means the agent is re-hashing the
    same reasoning without making progress.

    The default ``HashBasedEmbedder`` works without ML dependencies and is
    effective for detecting near-identical reasoning. For more nuanced
    semantic detection, use ``SentenceTransformerEmbedder``.

    Args:
        window: Number of recent thoughts to analyze. Default 8.
        variance_threshold: Mean variance below which oscillation is detected.
            Lower = more sensitive. Default 0.15.
        embedder: An Embedder instance. Defaults to HashBasedEmbedder.
    """

    def __init__(
        self,
        window: int = 8,
        variance_threshold: float = 0.15,
        embedder: Embedder | None = None,
    ) -> None:
        self.window = window
        self.variance_threshold = variance_threshold
        self.embedder: Embedder = embedder if embedder is not None else HashBasedEmbedder()
        self._thought_history: deque[list[float]] = deque(maxlen=window * 2)

    def analyze(self, step: AgentStep) -> DetectionResult:
        """Analyze a step for semantic oscillation.

        Embeds the step's thought text, adds it to the sliding window,
        and checks whether the variance across recent embeddings is
        below the threshold.

        Args:
            step: The agent step to analyze.

        Returns:
            DetectionResult with pattern ``"semantic_oscillation"`` if the
            mean variance of recent thought embeddings is below the threshold.
        """
        # Compute and cache the embedding
        vector = self.embedder.embed(step.thought)
        step.thought_vector = vector
        self._thought_history.append(vector)

        # Not enough data yet
        if len(self._thought_history) < self.window:
            return DetectionResult(detected=False)

        # Take the most recent `window` embeddings
        recent = np.array(list(self._thought_history)[-self.window :])

        # Compute variance across the thought embedding space
        # Low variance = thoughts are clustered = oscillation
        variance = float(np.mean(np.var(recent, axis=0)))

        if variance < self.variance_threshold:
            confidence = 1.0 - (variance / self.variance_threshold)
            return DetectionResult(
                detected=True,
                pattern="semantic_oscillation",
                confidence=confidence,
                evidence={
                    "thought_variance": variance,
                    "threshold": self.variance_threshold,
                    "window_size": self.window,
                    "steps_analyzed": len(self._thought_history),
                },
            )

        return DetectionResult(detected=False)

    def reset(self) -> None:
        """Clear the thought history."""
        self._thought_history.clear()

    @property
    def name(self) -> str:
        return "SemanticOscillationDetector"
