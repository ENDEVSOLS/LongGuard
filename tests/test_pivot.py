"""Tests for ReflectAndPivotInjector."""

from longguard.core.detectors.base import DetectionResult
from longguard.core.pivot import _SENTINEL, PIVOT_TEMPLATES, ReflectAndPivotInjector
from longguard.core.step import AgentStep


class TestPivotTemplates:
    """Test the default PIVOT_TEMPLATES."""

    def test_all_patterns_have_templates(self):
        """Every expected pattern has a template."""
        for pattern in ["tool_repeat", "semantic_oscillation", "dead_end_drift", "token_velocity"]:
            assert pattern in PIVOT_TEMPLATES

    def test_templates_are_non_empty(self):
        """All templates are non-empty strings."""
        for pattern, template in PIVOT_TEMPLATES.items():
            assert isinstance(template, str)
            assert len(template) > 0


class TestPivotGeneration:
    """Test prompt generation from detection results."""

    def test_tool_repeat_prompt(self):
        """Tool repeat detection generates a prompt with tool name and count."""
        injector = ReflectAndPivotInjector()
        result = DetectionResult(
            detected=True,
            pattern="tool_repeat",
            confidence=0.9,
            evidence={"tool": "search", "repeated_times": 3, "hash": "abc123"},
        )
        step = AgentStep(step_number=1, thought="test", action="search")
        prompt = injector.generate(result, step)

        assert "LOOP DETECTED" in prompt
        assert "search" in prompt
        assert "3" in prompt

    def test_semantic_oscillation_prompt(self):
        """Semantic oscillation detection generates a prompt."""
        injector = ReflectAndPivotInjector()
        result = DetectionResult(
            detected=True,
            pattern="semantic_oscillation",
            confidence=0.8,
            evidence={"thought_variance": 0.05, "window_size": 8},
        )
        step = AgentStep(step_number=1, thought="test")
        prompt = injector.generate(result, step)

        assert "REASONING LOOP" in prompt

    def test_dead_end_drift_prompt(self):
        """Dead end drift detection generates a prompt with step count."""
        injector = ReflectAndPivotInjector()
        result = DetectionResult(
            detected=True,
            pattern="dead_end_drift",
            confidence=0.7,
            evidence={"steps_without_progress": 5},
        )
        step = AgentStep(step_number=1, thought="test")
        prompt = injector.generate(result, step)

        assert "DEAD END" in prompt
        assert "5" in prompt

    def test_token_velocity_prompt(self):
        """Token velocity detection generates a prompt with velocity info."""
        injector = ReflectAndPivotInjector()
        result = DetectionResult(
            detected=True,
            pattern="token_velocity",
            confidence=0.85,
            evidence={
                "current_velocity": 5000.0,
                "baseline_velocity": 100.0,
                "velocity_ratio": 5.0,
            },
        )
        step = AgentStep(step_number=1, thought="test")
        prompt = injector.generate(result, step)

        assert "COST SPIKE" in prompt
        assert "5000" in prompt


class TestCustomTemplates:
    """Test custom template registration."""

    def test_custom_template_in_constructor(self):
        """Custom templates passed to constructor override defaults."""
        custom = {"tool_repeat": "CUSTOM: {tool} repeated {count} times!"}
        injector = ReflectAndPivotInjector(templates=custom)

        result = DetectionResult(
            detected=True,
            pattern="tool_repeat",
            confidence=0.9,
            evidence={"tool": "search", "repeated_times": 5, "hash": "x"},
        )
        step = AgentStep(step_number=1, thought="test", action="search")
        prompt = injector.generate(result, step)

        assert "CUSTOM: search repeated 5 times!" in prompt
        assert prompt.startswith(_SENTINEL)

    def test_add_template(self):
        """add_template registers a new template."""
        injector = ReflectAndPivotInjector()
        injector.add_template("custom_pattern", "Custom alert for {tool}!")

        result = DetectionResult(
            detected=True,
            pattern="custom_pattern",
            confidence=0.9,
            evidence={"tool": "my_tool"},
        )
        step = AgentStep(step_number=1, thought="test", action="my_tool")
        prompt = injector.generate(result, step)

        assert "Custom alert for my_tool!" in prompt
        assert prompt.startswith(_SENTINEL)

    def test_get_template(self):
        """get_template returns the template for a known pattern."""
        injector = ReflectAndPivotInjector()
        template = injector.get_template("tool_repeat")
        assert template is not None
        assert "LOOP DETECTED" in template

    def test_get_template_unknown(self):
        """get_template returns None for unknown patterns."""
        injector = ReflectAndPivotInjector()
        assert injector.get_template("nonexistent") is None


class TestFallbackPrompt:
    """Test fallback prompt for unknown patterns."""

    def test_unknown_pattern_uses_fallback(self):
        """Unknown pattern names get a generic fallback prompt."""
        injector = ReflectAndPivotInjector()
        result = DetectionResult(
            detected=True,
            pattern="weird_new_pattern",
            confidence=0.75,
            evidence={"detail": "something"},
        )
        step = AgentStep(step_number=1, thought="test")
        prompt = injector.generate(result, step)

        assert "weird_new_pattern" in prompt
        assert "75%" in prompt or "0.75" in prompt

    def test_templates_property(self):
        """templates property returns a copy of all templates."""
        injector = ReflectAndPivotInjector()
        templates = injector.templates
        assert isinstance(templates, dict)
        assert "tool_repeat" in templates
        # Modifying the copy shouldn't affect the injector
        templates["new_key"] = "test"
        assert "new_key" not in injector.templates
