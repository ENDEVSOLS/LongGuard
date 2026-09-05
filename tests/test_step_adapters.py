"""Tests for AgentStep.from_openai_response and from_anthropic_response adapters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from longguard.core.step import AgentStep

# ---------------------------------------------------------------------------
# Helpers: mock response objects (duck-typed, no real SDK needed)
# ---------------------------------------------------------------------------

def _make_openai_response(
    content: str | None = "I should search for this.",
    tool_name: str | None = None,
    tool_args: str | None = None,
    total_tokens: int = 500,
) -> Any:
    """Build a minimal mock object matching the OpenAI ChatCompletion shape."""
    tool_calls = []
    if tool_name is not None:
        fn = SimpleNamespace(name=tool_name, arguments=tool_args or "{}")
        tool_calls = [SimpleNamespace(function=fn)]

    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls if tool_calls else None,
    )
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(total_tokens=total_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_anthropic_response(
    text: str | None = "Let me think about this.",
    tool_name: str | None = None,
    tool_input: Any | None = None,
    input_tokens: int = 200,
    output_tokens: int = 150,
) -> Any:
    """Build a minimal mock object matching the Anthropic Message shape."""
    blocks: list[Any] = []
    if text is not None:
        blocks.append(SimpleNamespace(type="text", text=text))
    if tool_name is not None:
        blocks.append(
            SimpleNamespace(type="tool_use", name=tool_name, input=tool_input or {})
        )
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=blocks, usage=usage)


# ===========================================================================
# from_openai_response
# ===========================================================================


class TestFromOpenAIResponse:
    def test_basic_text_response(self) -> None:
        resp = _make_openai_response(content="Some thought here.")
        step = AgentStep.from_openai_response(resp, step_number=1)

        assert step.step_number == 1
        assert step.thought == "Some thought here."
        assert step.action is None
        assert step.action_input is None
        assert step.tokens_used == 500

    def test_tool_call_response(self) -> None:
        resp = _make_openai_response(
            content=None,
            tool_name="web_search",
            tool_args='{"query": "python asyncio"}',
        )
        step = AgentStep.from_openai_response(resp, step_number=2)

        assert step.action == "web_search"
        assert step.action_input == {"query": "python asyncio"}
        assert step.thought == ""

    def test_tool_call_with_text(self) -> None:
        resp = _make_openai_response(
            content="I'll search for it.",
            tool_name="calculator",
            tool_args='{"expr": "2+2"}',
        )
        step = AgentStep.from_openai_response(resp, step_number=3)

        assert step.thought == "I'll search for it."
        assert step.action == "calculator"
        assert step.action_input == {"expr": "2+2"}

    def test_invalid_json_args_kept_as_string(self) -> None:
        resp = _make_openai_response(
            tool_name="tool",
            tool_args="not valid json!!!",
        )
        step = AgentStep.from_openai_response(resp, step_number=1)
        assert step.action == "tool"
        assert step.action_input == "not valid json!!!"

    def test_token_count(self) -> None:
        resp = _make_openai_response(total_tokens=1234)
        step = AgentStep.from_openai_response(resp, step_number=1)
        assert step.tokens_used == 1234

    def test_observation_passed_through(self) -> None:
        resp = _make_openai_response()
        step = AgentStep.from_openai_response(resp, step_number=1, observation="tool result")
        assert step.observation == "tool result"

    def test_latency_ms_passed_through(self) -> None:
        resp = _make_openai_response()
        step = AgentStep.from_openai_response(resp, step_number=1, latency_ms=123.4)
        assert step.latency_ms == 123.4

    def test_no_usage_defaults_to_zero(self) -> None:
        resp = _make_openai_response()
        resp.usage = None
        step = AgentStep.from_openai_response(resp, step_number=1)
        assert step.tokens_used == 0

    def test_empty_tool_calls_list(self) -> None:
        resp = _make_openai_response(content="thinking", tool_name=None)
        # Explicitly set tool_calls to empty list
        resp.choices[0].message.tool_calls = []
        step = AgentStep.from_openai_response(resp, step_number=1)
        assert step.action is None

    def test_multimodal_content_list(self) -> None:
        """Content as a list of parts (vision models)."""
        part1 = SimpleNamespace(type="text", text="First part")
        part2 = SimpleNamespace(type="image_url", url="https://example.com/img.png")
        part3 = SimpleNamespace(type="text", text="Second part")
        resp = _make_openai_response(content=[part1, part2, part3])
        step = AgentStep.from_openai_response(resp, step_number=1)
        assert step.thought == "First part Second part"

    def test_step_number_assigned_correctly(self) -> None:
        resp = _make_openai_response()
        for n in [1, 5, 100]:
            step = AgentStep.from_openai_response(resp, step_number=n)
            assert step.step_number == n

    def test_has_tool_call_property(self) -> None:
        resp_with_tool = _make_openai_response(tool_name="fn", tool_args="{}")
        step = AgentStep.from_openai_response(resp_with_tool, step_number=1)
        assert step.has_tool_call is True

        resp_no_tool = _make_openai_response(tool_name=None)
        step2 = AgentStep.from_openai_response(resp_no_tool, step_number=1)
        assert step2.has_tool_call is False


# ===========================================================================
# from_anthropic_response
# ===========================================================================


class TestFromAnthropicResponse:
    def test_basic_text_response(self) -> None:
        resp = _make_anthropic_response(text="Let me reason through this.")
        step = AgentStep.from_anthropic_response(resp, step_number=1)

        assert step.step_number == 1
        assert step.thought == "Let me reason through this."
        assert step.action is None
        assert step.action_input is None
        assert step.tokens_used == 350  # 200 + 150

    def test_tool_use_block(self) -> None:
        resp = _make_anthropic_response(
            text=None,
            tool_name="bash",
            tool_input={"command": "ls -la"},
        )
        step = AgentStep.from_anthropic_response(resp, step_number=2)

        assert step.action == "bash"
        assert step.action_input == {"command": "ls -la"}
        assert step.thought == ""

    def test_text_and_tool_use_together(self) -> None:
        resp = _make_anthropic_response(
            text="I'll run a command.",
            tool_name="bash",
            tool_input={"command": "pwd"},
        )
        step = AgentStep.from_anthropic_response(resp, step_number=3)

        assert step.thought == "I'll run a command."
        assert step.action == "bash"
        assert step.action_input == {"command": "pwd"}

    def test_token_count(self) -> None:
        resp = _make_anthropic_response(input_tokens=100, output_tokens=300)
        step = AgentStep.from_anthropic_response(resp, step_number=1)
        assert step.tokens_used == 400

    def test_observation_passed_through(self) -> None:
        resp = _make_anthropic_response()
        step = AgentStep.from_anthropic_response(resp, step_number=1, observation="result")
        assert step.observation == "result"

    def test_latency_ms_passed_through(self) -> None:
        resp = _make_anthropic_response()
        step = AgentStep.from_anthropic_response(resp, step_number=1, latency_ms=50.0)
        assert step.latency_ms == 50.0

    def test_no_usage(self) -> None:
        resp = _make_anthropic_response()
        resp.usage = None
        step = AgentStep.from_anthropic_response(resp, step_number=1)
        assert step.tokens_used == 0

    def test_empty_content_list(self) -> None:
        resp = _make_anthropic_response(text=None, tool_name=None)
        resp.content = []
        step = AgentStep.from_anthropic_response(resp, step_number=1)
        assert step.thought == ""
        assert step.action is None

    def test_multiple_text_blocks_concatenated(self) -> None:
        block1 = SimpleNamespace(type="text", text="Part one.")
        block2 = SimpleNamespace(type="text", text="Part two.")
        resp = _make_anthropic_response(text=None)
        resp.content = [block1, block2]
        step = AgentStep.from_anthropic_response(resp, step_number=1)
        assert step.thought == "Part one. Part two."

    def test_first_tool_use_wins(self) -> None:
        """Only the first tool_use block should be extracted as the action."""
        tool1 = SimpleNamespace(type="tool_use", name="first_tool", input={"k": "v1"})
        tool2 = SimpleNamespace(type="tool_use", name="second_tool", input={"k": "v2"})
        resp = _make_anthropic_response(text=None)
        resp.content = [tool1, tool2]
        step = AgentStep.from_anthropic_response(resp, step_number=1)
        assert step.action == "first_tool"

    def test_has_tool_call_property(self) -> None:
        resp_with_tool = _make_anthropic_response(tool_name="fn")
        step = AgentStep.from_anthropic_response(resp_with_tool, step_number=1)
        assert step.has_tool_call is True

        resp_no_tool = _make_anthropic_response(tool_name=None)
        step2 = AgentStep.from_anthropic_response(resp_no_tool, step_number=1)
        assert step2.has_tool_call is False

    def test_unknown_block_types_ignored(self) -> None:
        """Future block types should not crash the parser."""
        future_block = SimpleNamespace(type="image", source="data:image/png;base64,...")
        text_block = SimpleNamespace(type="text", text="Normal thought.")
        resp = _make_anthropic_response(text=None)
        resp.content = [future_block, text_block]
        step = AgentStep.from_anthropic_response(resp, step_number=1)
        assert step.thought == "Normal thought."
