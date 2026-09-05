"""Tests for AgentStep dataclass."""

from longguard.core.step import AgentStep


class TestAgentStepCreation:
    """Test AgentStep construction and basic properties."""

    def test_minimal_creation(self):
        """Can create a step with just required fields."""
        step = AgentStep(step_number=1, thought="Hello")
        assert step.step_number == 1
        assert step.thought == "Hello"
        assert step.action is None
        assert step.action_input is None
        assert step.observation is None
        assert step.tokens_used == 0
        assert step.latency_ms == 0.0

    def test_full_creation(self):
        """Can create a step with all fields."""
        step = AgentStep(
            step_number=5,
            thought="I need to search.",
            action="search",
            action_input={"query": "python"},
            observation="Results for python",
            tokens_used=250,
            latency_ms=150.0,
        )
        assert step.step_number == 5
        assert step.action == "search"
        assert step.action_input == {"query": "python"}
        assert step.observation == "Results for python"
        assert step.tokens_used == 250
        assert step.latency_ms == 150.0


class TestActionHash:
    """Test the action_hash property for tool-repeat detection."""

    def test_same_action_same_hash(self):
        """Identical tool calls produce the same hash."""
        step1 = AgentStep(step_number=1, thought="a", action="search", action_input="x")
        step2 = AgentStep(step_number=2, thought="b", action="search", action_input="x")
        assert step1.action_hash == step2.action_hash

    def test_different_action_different_hash(self):
        """Different tool names produce different hashes."""
        step1 = AgentStep(step_number=1, thought="a", action="search", action_input="x")
        step2 = AgentStep(step_number=2, thought="a", action="lookup", action_input="x")
        assert step1.action_hash != step2.action_hash

    def test_different_input_different_hash(self):
        """Same tool but different arguments produce different hashes."""
        step1 = AgentStep(step_number=1, thought="a", action="search", action_input="x")
        step2 = AgentStep(step_number=2, thought="a", action="search", action_input="y")
        assert step1.action_hash != step2.action_hash

    def test_no_action_hash(self):
        """Steps without a tool call still produce a hash (based on None, None)."""
        step1 = AgentStep(step_number=1, thought="thinking")
        step2 = AgentStep(step_number=2, thought="also thinking")
        # Both have action=None, action_input=None → same hash
        assert step1.action_hash == step2.action_hash

    def test_thought_does_not_affect_hash(self):
        """Different thought text doesn't change the action hash."""
        step1 = AgentStep(step_number=1, thought="alpha", action="search", action_input="q")
        step2 = AgentStep(step_number=2, thought="beta", action="search", action_input="q")
        assert step1.action_hash == step2.action_hash

    def test_dict_input_consistent_hash(self):
        """Dict inputs are consistently serialized regardless of key order."""
        step1 = AgentStep(
            step_number=1, thought="a", action="search",
            action_input={"z": 1, "a": 2},
        )
        step2 = AgentStep(
            step_number=2, thought="a", action="search",
            action_input={"a": 2, "z": 1},
        )
        assert step1.action_hash == step2.action_hash


class TestThoughtVector:
    """Test the thought_vector property."""

    def test_initially_none(self):
        """Thought vector starts as None."""
        step = AgentStep(step_number=1, thought="hello")
        assert step.thought_vector is None

    def test_set_and_get(self):
        """Can set and retrieve the thought vector."""
        step = AgentStep(step_number=1, thought="hello")
        vector = [0.1, 0.2, 0.3]
        step.thought_vector = vector
        assert step.thought_vector == vector

    def test_set_to_none(self):
        """Can reset the thought vector to None."""
        step = AgentStep(step_number=1, thought="hello")
        step.thought_vector = [0.1]
        step.thought_vector = None
        assert step.thought_vector is None


class TestProperties:
    """Test has_tool_call and has_observation properties."""

    def test_has_tool_call_true(self):
        step = AgentStep(step_number=1, thought="a", action="search")
        assert step.has_tool_call is True

    def test_has_tool_call_false(self):
        step = AgentStep(step_number=1, thought="a")
        assert step.has_tool_call is False

    def test_has_observation_true(self):
        step = AgentStep(step_number=1, thought="a", observation="result")
        assert step.has_observation is True

    def test_has_observation_false(self):
        step = AgentStep(step_number=1, thought="a")
        assert step.has_observation is False


class TestSerialization:
    """Test to_dict and from_dict round-trip."""

    def test_roundtrip(self):
        """Serialization and deserialization produce equivalent objects."""
        original = AgentStep(
            step_number=3,
            thought="searching",
            action="search",
            action_input="test",
            observation="found",
            tokens_used=200,
            latency_ms=50.0,
        )
        d = original.to_dict()
        restored = AgentStep.from_dict(d)

        assert restored.step_number == original.step_number
        assert restored.thought == original.thought
        assert restored.action == original.action
        assert restored.action_input == original.action_input
        assert restored.observation == original.observation
        assert restored.tokens_used == original.tokens_used
        assert restored.latency_ms == original.latency_ms

    def test_thought_vector_not_serialized(self):
        """The thought_vector is not included in to_dict output."""
        step = AgentStep(step_number=1, thought="test")
        step.thought_vector = [1.0, 2.0]
        d = step.to_dict()
        assert "_thought_vector" not in d
        assert "thought_vector" not in d


class TestStr:
    """Test the __str__ representation."""

    def test_with_action(self):
        step = AgentStep(
            step_number=1,
            thought="Let me search",
            action="search",
            action_input="query",
            observation="Result",
        )
        s = str(step)
        assert "Step 1" in s
        assert "search" in s

    def test_without_action(self):
        step = AgentStep(step_number=1, thought="Just thinking")
        s = str(step)
        assert "Step 1" in s
