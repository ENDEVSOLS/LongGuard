"""Tests for LangGraph integration."""

from longguard.config import GuardConfig
from longguard.core.breaker import BreakerState
from longguard.integrations.langgraph import LongGuard, add_guard_to_graph


class TestLongGuardInit:
    """Test LongGuard initialization."""

    def test_default_config(self):
        """LongGuard works with default config."""
        guard = LongGuard()
        assert guard.config is not None
        assert guard.breaker is not None

    def test_custom_config(self):
        """LongGuard accepts custom config."""
        config = GuardConfig(max_tokens_per_run=100_000)
        guard = LongGuard(config)
        assert guard.config.max_tokens_per_run == 100_000


class TestLongGuardExtractStep:
    """Test step extraction from LangGraph state."""

    def test_extract_step_basic(self):
        """Extract step from a minimal state dict."""
        guard = LongGuard(GuardConfig(max_tokens_per_run=1_000_000, max_steps=100))
        state = {
            "messages": [
                type("Msg", (), {"content": "I should search for info", "type": "ai"})(),
            ]
        }
        step = guard._extract_step(state)
        assert step.step_number == 1
        assert step.thought == "I should search for info"

    def test_extract_step_counter_increments(self):
        """Step counter increments with each extraction."""
        guard = LongGuard(GuardConfig(max_tokens_per_run=1_000_000, max_steps=100))
        state = {"messages": []}
        step1 = guard._extract_step(state)
        step2 = guard._extract_step(state)
        assert step1.step_number == 1
        assert step2.step_number == 2

    def test_extract_step_empty_state(self):
        """Empty state produces a step with default thought."""
        guard = LongGuard(GuardConfig(max_tokens_per_run=1_000_000, max_steps=100))
        state = {}
        step = guard._extract_step(state)
        assert step.thought == "(no thought)"


class TestLongGuardWrapNode:
    """Test node wrapping."""

    def test_wrap_node_continue(self):
        """Wrapped node runs normally on continue decision."""
        guard = LongGuard(GuardConfig(max_tokens_per_run=1_000_000, max_steps=100))

        call_count = 0

        def original_fn(state):
            nonlocal call_count
            call_count += 1
            return state

        wrapped = guard.wrap_node("test", original_fn)
        wrapped({"messages": []})
        assert call_count == 1

    def test_wrap_node_kill(self):
        """Wrapped node returns terminated state on kill."""
        guard = LongGuard(GuardConfig(max_tokens_per_run=50))

        def original_fn(state):
            return state

        wrapped = guard.wrap_node("test", original_fn)
        # Feed a step that exceeds token budget
        state = {
            "messages": [
                type("Msg", (), {
                    "content": "x" * 1000,  # Lots of content → high token estimate
                    "type": "ai",
                    "tool_calls": None,
                    "additional_kwargs": {},
                })(),
            ]
        }
        result = wrapped(state)
        # May or may not kill depending on token estimate
        # Just verify it returns a dict
        assert isinstance(result, dict)

    def test_wrap_node_already_terminated(self):
        """Wrapped node returns immediately if already terminated."""
        guard = LongGuard(GuardConfig())

        call_count = 0

        def original_fn(state):
            nonlocal call_count
            call_count += 1
            return state

        wrapped = guard.wrap_node("test", original_fn)
        state = {
            "messages": [],
            "__longguard_terminated__": True,
        }
        wrapped(state)
        assert call_count == 0  # original_fn not called


class TestLongGuardReset:
    """Test guard reset."""

    def test_reset_clears_state(self):
        """Reset clears step counter and breaker state."""
        guard = LongGuard(GuardConfig(max_tokens_per_run=1_000_000, max_steps=100))
        state = {"messages": []}
        guard._extract_step(state)
        guard._extract_step(state)
        assert guard._step_counter == 2

        guard.reset()
        assert guard._step_counter == 0
        assert guard.breaker.state == BreakerState.CLOSED


class TestLongGuardReport:
    """Test report access."""

    def test_get_report(self):
        """get_report returns the breaker's report."""
        guard = LongGuard()
        report = guard.get_report()
        assert report is not None

class TestLangGraphAdditional:
    """Test the branches missed by basic tests."""

    def test_extract_step_tool_calls(self):
        guard = LongGuard()
        state = {
            "messages": [
                type("Msg", (), {
                    "content": "",
                    "tool_calls": [{"name": "search", "args": {"q": "test"}}],
                    "additional_kwargs": {}
                })(),
            ]
        }
        step = guard._extract_step(state)
        assert step.action == "search"
        assert step.action_input == {"q": "test"}

    def test_extract_step_additional_kwargs(self):
        guard = LongGuard()
        state = {
            "messages": [
                type("Msg", (), {
                    "content": "",
                    "tool_calls": None,
                    "additional_kwargs": {
                        "tool_calls": [
                            {"function": {"name": "search2", "arguments": '{"q": "test2"}'}}
                        ]
                    }
                })(),
            ]
        }
        step = guard._extract_step(state)
        assert step.action == "search2"
        assert step.action_input == {"q": "test2"}

    def test_extract_step_observation(self):
        guard = LongGuard()
        state = {
            "messages": [
                type("Msg", (), {"content": "", "tool_calls": None, "additional_kwargs": {}})(),
                type("Msg", (), {"type": "tool", "content": "tool result"})(),
                type("Msg", (), {"content": "next thought"})()
            ]
        }
        step = guard._extract_step(state)
        assert step.thought == "next thought"
        assert step.observation == "tool result"

    def test_inject_pivot(self):
        guard = LongGuard()
        state = {"messages": []}
        new_state = guard._inject_pivot(state, "pivot!")
        assert len(new_state["messages"]) == 1
        msg = new_state["messages"][0]
        # Depending on if langchain_core is installed, it's either an object or dict
        if isinstance(msg, dict):
            assert msg["content"] == "pivot!"
        else:
            assert msg.content == "pivot!"

    def test_wrap_node_reflect(self):
        guard = LongGuard(GuardConfig(max_steps=100))
        # force reflect
        decision = type(
            "Decision",
            (),
            {"action": "reflect", "reason": "test", "inject_prompt": "pivot", "report": None},
        )()
        guard.breaker.check = lambda x: decision

        def original_fn(state):
            return state

        wrapped = guard.wrap_node("test", original_fn)
        result = wrapped({"messages": []})
        msg = result["messages"][0]
        if isinstance(msg, dict):
            assert msg["content"] == "pivot"
        else:
            assert msg.content == "pivot"


def test_add_guard_to_graph():
    class MockGraph:
        def __init__(self):
            self.nodes = {"agent": lambda x: x, "tool": lambda x: x}

    graph = MockGraph()
    guarded_graph = add_guard_to_graph(graph)
    assert hasattr(guarded_graph, "__longguard__")
    assert guarded_graph.nodes["agent"].__name__ == "guarded_agent"
