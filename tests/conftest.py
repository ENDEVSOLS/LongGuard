"""Shared test fixtures for LongGuard."""

import pytest

from longguard.config import GuardConfig
from longguard.core.step import AgentStep


@pytest.fixture
def default_config() -> GuardConfig:
    """Default GuardConfig for testing."""
    return GuardConfig()


@pytest.fixture
def strict_config() -> GuardConfig:
    """Strict GuardConfig with low thresholds for easier testing."""
    return GuardConfig(
        tool_repeat_threshold=2,
        tool_repeat_window=4,
        semantic_variance_threshold=0.5,
        semantic_window=4,
        dead_end_threshold=3,
        max_tokens_per_run=10_000,
        max_steps=15,
        max_reflections=1,
    )


@pytest.fixture
def make_step():
    """Factory fixture for creating AgentStep instances with defaults."""

    def _make_step(
        step_number: int = 1,
        thought: str = "I need to search for information.",
        action: str = "search",
        action_input: str = "query",
        observation: str = "Result found",
        tokens_used: int = 100,
    ) -> AgentStep:
        return AgentStep(
            step_number=step_number,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
            tokens_used=tokens_used,
        )

    return _make_step


@pytest.fixture
def repeating_steps():
    """A sequence of steps where the same tool is called repeatedly."""
    steps = []
    for i in range(6):
        steps.append(
            AgentStep(
                step_number=i + 1,
                thought=f"Let me try searching again. Attempt {i + 1}.",
                action="search",
                action_input="same query",
                observation="No new results",
                tokens_used=100,
            )
        )
    return steps


@pytest.fixture
def diverse_steps():
    """A sequence of steps with different actions and observations."""
    steps = [
        AgentStep(
            step_number=1,
            thought="I should search for the capital of France.",
            action="search",
            action_input="capital of France",
            observation="Paris is the capital of France.",
            tokens_used=120,
        ),
        AgentStep(
            step_number=2,
            thought="Now let me look up the population.",
            action="lookup",
            action_input="population of Paris",
            observation="Paris has 2.1 million inhabitants.",
            tokens_used=150,
        ),
        AgentStep(
            step_number=3,
            thought="Let me calculate the total area.",
            action="calculator",
            action_input="2.1 * 1000",
            observation="2100",
            tokens_used=80,
        ),
    ]
    return steps
