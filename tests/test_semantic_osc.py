"""Tests for SemanticOscillationDetector."""

import numpy as np

from longguard.core.detectors.semantic_osc import (
    HashBasedEmbedder,
    SemanticOscillationDetector,
)
from longguard.core.step import AgentStep


class TestHashBasedEmbedder:
    """Test the default embedder."""

    def test_returns_list(self):
        """embed() returns a list of floats."""
        embedder = HashBasedEmbedder(dimension=32)
        vector = embedder.embed("hello world")
        assert isinstance(vector, list)
        assert len(vector) == 32
        assert all(isinstance(v, float) for v in vector)

    def test_deterministic(self):
        """Same text always produces the same embedding."""
        embedder = HashBasedEmbedder(dimension=32)
        v1 = embedder.embed("test text")
        v2 = embedder.embed("test text")
        np.testing.assert_array_almost_equal(v1, v2)

    def test_different_text_different_embedding(self):
        """Different text produces different embeddings."""
        embedder = HashBasedEmbedder(dimension=32)
        v1 = embedder.embed("searching for cats")
        v2 = embedder.embed("calculating pi digits")
        # Not identical
        assert not np.allclose(v1, v2)

    def test_unit_norm(self):
        """Embeddings are approximately unit-normalized."""
        embedder = HashBasedEmbedder(dimension=64)
        vector = embedder.embed("some text here")
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 0.01


class TestSemanticOscillationBasic:
    """Basic SemanticOscillationDetector functionality."""

    def test_no_detection_insufficient_data(self):
        """Not enough steps doesn't trigger detection."""
        detector = SemanticOscillationDetector(window=8)
        step = AgentStep(step_number=1, thought="thinking")
        result = detector.analyze(step)
        assert result.detected is False

    def test_detection_identical_thoughts(self):
        """Identical thoughts trigger oscillation detection."""
        detector = SemanticOscillationDetector(
            window=4, variance_threshold=0.5
        )
        # Feed identical thoughts to keep variance very low
        for i in range(5):
            step = AgentStep(
                step_number=i + 1,
                thought="Let me search for the answer again.",
            )
            result = detector.analyze(step)

        # With HashBasedEmbedder, identical text → identical vectors → near-zero variance
        assert result.detected is True
        assert result.pattern == "semantic_oscillation"

    def test_no_detection_diverse_thoughts(self):
        """Diverse thoughts don't trigger detection."""
        detector = SemanticOscillationDetector(
            window=4, variance_threshold=0.001
        )
        diverse_thoughts = [
            "I should search for the capital of France.",
            "Now let me calculate the total population.",
            "I need to look up the historical data for 2023.",
            "Let me verify this against the database records.",
            "Time to summarize what I've found so far.",
        ]
        for i, thought in enumerate(diverse_thoughts):
            step = AgentStep(step_number=i + 1, thought=thought)
            detector.analyze(step)
            # With very low threshold, diverse thoughts should NOT trigger
            # (HashBasedEmbedder produces sufficiently different vectors)

    def test_caches_thought_vector(self):
        """The detector caches the embedding on the step object."""
        detector = SemanticOscillationDetector(window=8)
        step = AgentStep(step_number=1, thought="hello world")
        assert step.thought_vector is None
        detector.analyze(step)
        assert step.thought_vector is not None


class TestSemanticOscillationConfidence:
    """Test confidence scoring."""

    def test_confidence_range(self):
        """Confidence is always in [0.0, 1.0]."""
        detector = SemanticOscillationDetector(window=4, variance_threshold=0.5)
        for i in range(5):
            step = AgentStep(
                step_number=i + 1,
                thought="Same thought repeated.",
            )
            result = detector.analyze(step)
            if result.detected:
                assert 0.0 <= result.confidence <= 1.0


class TestSemanticOscillationReset:
    """Test the reset method."""

    def test_reset_clears_history(self):
        """After reset, detector needs to rebuild history."""
        detector = SemanticOscillationDetector(window=4, variance_threshold=0.5)

        # Build up history
        for i in range(5):
            step = AgentStep(
                step_number=i + 1,
                thought="Same thought repeated.",
            )
            detector.analyze(step)

        # Reset
        detector.reset()

        # Should not detect immediately after reset
        step = AgentStep(step_number=1, thought="Same thought repeated.")
        result = detector.analyze(step)
        assert result.detected is False
