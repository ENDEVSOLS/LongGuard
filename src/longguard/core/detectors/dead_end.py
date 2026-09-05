"""DeadEndDriftDetector — detects when the agent makes no progress toward the goal."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..step import AgentStep
from .base import AbstractLoopDetector, DetectionResult

if TYPE_CHECKING:
    from .semantic_osc import Embedder


def _tokenize(text: str) -> set[str]:
    """Simple whitespace tokenizer that lowercases tokens.

    Args:
        text: The text to tokenize.

    Returns:
        A set of lowercase tokens.
    """
    return set(text.lower().split())


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Returns 0.0 if both sets are empty.

    Args:
        set_a: First set of tokens.
        set_b: Second set of tokens.

    Returns:
        Jaccard similarity in [0.0, 1.0].
    """
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity in [-1.0, 1.0]. Returns 0.0 if either vector is zero.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class DeadEndDriftDetector(AbstractLoopDetector):
    """Detects when the agent has drifted into a dead-end reasoning path.

    A dead-end drift occurs when the agent's recent steps show no meaningful
    progress toward the goal. Progress is measured by three heuristics:

    1. **Observation novelty**: A new observation that is significantly different
       from the previous one indicates new information, which counts as progress.
       When an ``embedder`` is provided, novelty is measured by cosine similarity
       on semantic embeddings (catches meaning-level similarity). Otherwise,
       Jaccard similarity on word sets is used as a fallback.
    2. **Action change**: Switching to a different tool suggests the agent is
       trying a new approach, which counts as progress.
    3. **Thought novelty**: New concepts in the thought text suggest new reasoning.
       When an ``embedder`` is provided, cosine similarity is used. Otherwise,
       token-set overlap is used as a fallback.

    If none of these progress signals are detected for ``dead_end_threshold``
    consecutive steps, the detector flags a dead-end drift.

    Args:
        dead_end_threshold: Number of consecutive no-progress steps before
            detection. Default is 5.
        progress_threshold: Similarity threshold. If the similarity between
            consecutive observations is **above** this value (i.e., observations
            are very similar), it's considered no-progress.
            Default is 0.6. Lower this to make the detector more sensitive.
        embedder: An optional Embedder instance for semantic similarity.
            When provided, cosine similarity on embeddings is used instead of
            Jaccard similarity. Pass a ``SentenceTransformerEmbedder`` for
            production-grade semantic accuracy. Defaults to None (Jaccard fallback).
    """

    def __init__(
        self,
        dead_end_threshold: int = 5,
        progress_threshold: float = 0.6,
        embedder: Embedder | None = None,
    ) -> None:
        self.dead_end_threshold = dead_end_threshold
        self.progress_threshold = progress_threshold
        self._embedder = embedder
        self._consecutive_no_progress: int = 0
        self._last_action: str | None = None
        self._last_progressive_step: int = 0
        self._step_count: int = 0

        # Semantic mode state (when embedder is provided)
        self._last_observation_vector: list[float] | None = None
        self._last_thought_vector: list[float] | None = None

        # Lexical fallback state (when no embedder)
        self._last_observation_tokens: set[str] | None = None
        self._last_thought_tokens: set[str] | None = None

    @property
    def _use_semantic(self) -> bool:
        """Whether to use semantic (embedding-based) similarity."""
        return self._embedder is not None

    def _check_progress(self, step: AgentStep) -> bool:
        """Check if a step shows any progress signals.

        Args:
            step: The agent step to check.

        Returns:
            True if progress is detected, False otherwise.
        """
        progress = False

        # Signal 1: Action change — different tool = new approach
        if step.action is not None and step.action != self._last_action:
            progress = True

        # Signal 2: Observation novelty
        if step.observation is not None:
            if self._use_semantic and self._embedder is not None:
                obs_vector = self._embedder.embed(step.observation)
                if self._last_observation_vector is not None:
                    similarity = _cosine_similarity(
                        obs_vector, self._last_observation_vector
                    )
                    if similarity < self.progress_threshold:
                        progress = True
                else:
                    progress = True
                self._last_observation_vector = obs_vector
            else:
                obs_tokens = _tokenize(step.observation)
                if self._last_observation_tokens is not None:
                    similarity = _jaccard_similarity(
                        obs_tokens, self._last_observation_tokens
                    )
                    if similarity < self.progress_threshold:
                        progress = True
                else:
                    progress = True
                self._last_observation_tokens = obs_tokens

        # Signal 3: Thought novelty
        if step.thought:
            if self._use_semantic and self._embedder is not None:
                thought_vector = self._embedder.embed(step.thought)
                if self._last_thought_vector is not None:
                    similarity = _cosine_similarity(
                        thought_vector, self._last_thought_vector
                    )
                    # Use a slightly lower threshold for thoughts (more
                    # permissive) to match the original "partial credit" design.
                    if similarity < (self.progress_threshold - 0.1):
                        progress = True
                self._last_thought_vector = thought_vector
            else:
                thought_tokens = _tokenize(step.thought)
                if self._last_thought_tokens is not None:
                    new_tokens = thought_tokens - self._last_thought_tokens
                    if (
                        len(thought_tokens) > 0
                        and len(new_tokens) / len(thought_tokens) > 0.4
                    ):
                        progress = True
                self._last_thought_tokens = thought_tokens

        # Update last action
        if step.action is not None:
            self._last_action = step.action

        return progress

    def analyze(self, step: AgentStep) -> DetectionResult:
        """Analyze a step for dead-end drift.

        Args:
            step: The agent step to analyze.

        Returns:
            DetectionResult with pattern ``"dead_end_drift"`` if the agent
            has had no progress for ``dead_end_threshold`` consecutive steps.
        """
        self._step_count += 1

        has_progress = self._check_progress(step)

        if has_progress:
            self._consecutive_no_progress = 0
            self._last_progressive_step = self._step_count
        else:
            self._consecutive_no_progress += 1

        if self._consecutive_no_progress >= self.dead_end_threshold:
            confidence = min(
                self._consecutive_no_progress / self.dead_end_threshold, 1.0
            )
            return DetectionResult(
                detected=True,
                pattern="dead_end_drift",
                confidence=confidence,
                evidence={
                    "steps_without_progress": self._consecutive_no_progress,
                    "last_progressive_step": self._last_progressive_step,
                    "total_steps_analyzed": self._step_count,
                    "threshold": self.dead_end_threshold,
                    "similarity_mode": (
                        "semantic" if self._use_semantic else "lexical"
                    ),
                },
            )

        return DetectionResult(detected=False)

    def reset(self) -> None:
        """Clear all tracking state."""
        self._consecutive_no_progress = 0
        self._last_observation_tokens = None
        self._last_observation_vector = None
        self._last_action = None
        self._last_thought_tokens = None
        self._last_thought_vector = None
        self._last_progressive_step = 0
        self._step_count = 0

    @property
    def name(self) -> str:
        return "DeadEndDriftDetector"
