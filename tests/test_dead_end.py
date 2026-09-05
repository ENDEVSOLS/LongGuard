"""Tests for DeadEndDriftDetector."""

from longguard.core.detectors.dead_end import (
    DeadEndDriftDetector,
    _cosine_similarity,
    _jaccard_similarity,
    _tokenize,
)
from longguard.core.detectors.semantic_osc import HashBasedEmbedder
from longguard.core.step import AgentStep


class TestJaccardSimilarity:
    """Test the Jaccard similarity helper."""

    def test_identical_sets(self):
        """Identical sets have similarity 1.0."""
        assert _jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        """Disjoint sets have similarity 0.0."""
        assert _jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        """Partial overlap gives a value between 0 and 1."""
        sim = _jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert 0.0 < sim < 1.0

    def test_empty_sets(self):
        """Two empty sets have similarity 0.0."""
        assert _jaccard_similarity(set(), set()) == 0.0


class TestCosineSimilarity:
    """Test the cosine similarity helper."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        vec = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        assert abs(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-9

    def test_zero_vector(self):
        """Zero vector returns 0.0."""
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_partial_similarity(self):
        """Partially aligned vectors give a value between 0 and 1."""
        sim = _cosine_similarity([1.0, 1.0], [1.0, 0.0])
        assert 0.0 < sim < 1.0


class TestTokenize:
    """Test the tokenizer helper."""

    def test_basic(self):
        assert _tokenize("Hello World") == {"hello", "world"}

    def test_empty(self):
        assert _tokenize("") == set()


class TestDeadEndBasic:
    """Basic DeadEndDriftDetector functionality."""

    def test_no_detection_early(self):
        """First few steps don't trigger detection."""
        detector = DeadEndDriftDetector(dead_end_threshold=3)
        step = AgentStep(
            step_number=1,
            thought="Let me search.",
            action="search",
            action_input="query",
            observation="No results found",
        )
        result = detector.analyze(step)
        assert result.detected is False

    def test_detection_no_progress(self):
        """Consecutive no-progress steps trigger detection."""
        detector = DeadEndDriftDetector(dead_end_threshold=3, progress_threshold=0.6)

        # Feed steps with same action and similar observations
        for i in range(5):
            step = AgentStep(
                step_number=i + 1,
                thought="Let me try the same thing again.",
                action="search",
                action_input="same query",
                observation="No new results found",  # Very similar observation
            )
            result = detector.analyze(step)

        assert result.detected is True
        assert result.pattern == "dead_end_drift"

    def test_no_detection_with_progress(self):
        """Steps that make progress don't trigger detection."""
        detector = DeadEndDriftDetector(dead_end_threshold=3)

        for i in range(10):
            step = AgentStep(
                step_number=i + 1,
                thought=f"Using tool {i}",
                action=f"tool_{i % 3}",  # Different tools
                action_input=f"arg_{i}",
                observation=f"Unique result {i} with new data",
            )
            result = detector.analyze(step)
            assert result.detected is False


class TestDeadEndProgressSignals:
    """Test the three progress signal heuristics."""

    def test_action_change_is_progress(self):
        """Switching to a different tool counts as progress."""
        detector = DeadEndDriftDetector(dead_end_threshold=5)

        # Step 1: search
        step1 = AgentStep(
            step_number=1, thought="search", action="search",
            action_input="q", observation="result1",
        )
        detector.analyze(step1)

        # Step 2: different tool = progress
        step2 = AgentStep(
            step_number=2, thought="lookup", action="lookup",
            action_input="q", observation="result1",  # Same observation
        )
        result = detector.analyze(step2)
        assert result.detected is False

    def test_observation_novelty_is_progress(self):
        """A very different observation counts as progress."""
        detector = DeadEndDriftDetector(dead_end_threshold=5, progress_threshold=0.6)

        step1 = AgentStep(
            step_number=1, thought="search", action="search",
            action_input="q", observation="The quick brown fox jumps over the lazy dog",
        )
        detector.analyze(step1)

        # Very different observation = progress
        step2 = AgentStep(
            step_number=2, thought="search", action="search",
            action_input="q2",
            observation="Python programming language was created by Guido van Rossum",
        )
        result = detector.analyze(step2)
        assert result.detected is False

    def test_thought_novelty_is_partial_progress(self):
        """New concepts in the thought text can count as progress."""
        detector = DeadEndDriftDetector(dead_end_threshold=5)

        step1 = AgentStep(
            step_number=1,
            thought="I need to find the answer to this question about physics.",
            action="search",
            action_input="q",
            observation="Similar result",
        )
        detector.analyze(step1)

        # Very different thought content = new concepts = progress
        step2 = AgentStep(
            step_number=2,
            thought="Chemistry biology mathematics astronomy geography are different fields.",
            action="search",
            action_input="q",
            observation="Similar result",
        )
        result = detector.analyze(step2)
        # Should not be dead-end because thought has new tokens
        assert result.detected is False


class TestDeadEndReset:
    """Test the reset method."""

    def test_reset_clears_state(self):
        """After reset, previous no-progress steps don't count."""
        detector = DeadEndDriftDetector(dead_end_threshold=3)

        # Build up no-progress steps
        for i in range(4):
            step = AgentStep(
                step_number=i + 1,
                thought="Same approach",
                action="search",
                action_input="q",
                observation="Same result",
            )
            detector.analyze(step)

        # Reset
        detector.reset()

        # Should not detect immediately
        step = AgentStep(
            step_number=1,
            thought="Same approach",
            action="search",
            action_input="q",
            observation="Same result",
        )
        result = detector.analyze(step)
        assert result.detected is False

    def test_reset_clears_semantic_state(self):
        """Reset clears embedding vectors when using semantic mode."""
        embedder = HashBasedEmbedder()
        detector = DeadEndDriftDetector(dead_end_threshold=3, embedder=embedder)

        step = AgentStep(
            step_number=1,
            thought="Some thought",
            action="search",
            action_input="q",
            observation="Some result",
        )
        detector.analyze(step)

        detector.reset()
        assert detector._last_observation_vector is None
        assert detector._last_thought_vector is None


class TestDeadEndEvidence:
    """Test that detection evidence is informative."""

    def test_evidence_contains_key_fields(self):
        """Detected events have useful evidence."""
        detector = DeadEndDriftDetector(dead_end_threshold=3, progress_threshold=0.9)

        for i in range(5):
            step = AgentStep(
                step_number=i + 1,
                thought="thinking",
                action="search",
                action_input="q",
                observation="same same same",
            )
            result = detector.analyze(step)

        if result.detected:
            assert "steps_without_progress" in result.evidence
            assert result.evidence["steps_without_progress"] >= 3

    def test_evidence_includes_similarity_mode_lexical(self):
        """Evidence reports 'lexical' mode when no embedder is used."""
        detector = DeadEndDriftDetector(dead_end_threshold=2, progress_threshold=0.9)

        for i in range(3):
            step = AgentStep(
                step_number=i + 1,
                thought="thinking",
                action="search",
                action_input="q",
                observation="same same same",
            )
            result = detector.analyze(step)

        if result.detected:
            assert result.evidence["similarity_mode"] == "lexical"

    def test_evidence_includes_similarity_mode_semantic(self):
        """Evidence reports 'semantic' mode when an embedder is used."""
        embedder = HashBasedEmbedder()
        detector = DeadEndDriftDetector(
            dead_end_threshold=2, progress_threshold=0.9, embedder=embedder
        )

        for i in range(3):
            step = AgentStep(
                step_number=i + 1,
                thought="thinking",
                action="search",
                action_input="q",
                observation="same same same",
            )
            result = detector.analyze(step)

        if result.detected:
            assert result.evidence["similarity_mode"] == "semantic"


class TestDeadEndSemanticMode:
    """Test the semantic (embedding-based) similarity mode."""

    def test_semantic_mode_enabled(self):
        """When an embedder is provided, _use_semantic is True."""
        embedder = HashBasedEmbedder()
        detector = DeadEndDriftDetector(embedder=embedder)
        assert detector._use_semantic is True

    def test_lexical_mode_default(self):
        """When no embedder is provided, _use_semantic is False."""
        detector = DeadEndDriftDetector()
        assert detector._use_semantic is False

    def test_semantic_detects_identical_observations(self):
        """Semantic mode detects identical observations as no-progress."""
        embedder = HashBasedEmbedder()
        detector = DeadEndDriftDetector(
            dead_end_threshold=3, embedder=embedder
        )

        for i in range(5):
            step = AgentStep(
                step_number=i + 1,
                thought="Same thought",
                action="search",
                action_input="q",
                observation="Identical result that never changes",
            )
            result = detector.analyze(step)

        assert result.detected is True
        assert result.pattern == "dead_end_drift"

    def test_semantic_progress_on_different_observations(self):
        """Semantic mode recognizes very different observations as progress."""
        embedder = HashBasedEmbedder()
        detector = DeadEndDriftDetector(
            dead_end_threshold=3, embedder=embedder
        )

        observations = [
            "The weather in Tokyo is sunny today",
            "Python was created by Guido van Rossum in 1991",
            "Machine learning uses neural networks for predictions",
            "The stock market closed higher on Monday",
        ]

        for i, obs in enumerate(observations):
            step = AgentStep(
                step_number=i + 1,
                thought=f"Investigating topic {i}",
                action="search",
                action_input=f"query_{i}",
                observation=obs,
            )
            result = detector.analyze(step)
            assert result.detected is False

    def test_backward_compatibility_no_embedder(self):
        """Without embedder, behavior is identical to original Jaccard-based."""
        detector = DeadEndDriftDetector(dead_end_threshold=3, progress_threshold=0.6)

        # Same test as test_detection_no_progress — should still work
        for i in range(5):
            step = AgentStep(
                step_number=i + 1,
                thought="Let me try the same thing again.",
                action="search",
                action_input="same query",
                observation="No new results found",
            )
            result = detector.analyze(step)

        assert result.detected is True
        assert result.pattern == "dead_end_drift"
