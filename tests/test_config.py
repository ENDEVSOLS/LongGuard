"""Tests for GuardConfig."""

import pytest

from longguard.config import GuardConfig


class TestGuardConfigDefaults:
    """Test default configuration values."""

    def test_default_values(self):
        """Default config has expected values."""
        config = GuardConfig()
        assert config.tool_repeat_threshold == 3
        assert config.tool_repeat_window == 6
        assert config.semantic_variance_threshold == 0.15
        assert config.semantic_window == 8
        assert config.dead_end_threshold == 5
        assert config.max_tokens_per_run == 50_000
        assert config.max_steps == 30
        assert config.max_reflections == 2
        assert config.log_level == "WARNING"
        assert config.emit_events is True


class TestGuardConfigValidation:
    """Test configuration validation."""

    def test_valid_config_passes(self):
        """Valid configuration doesn't raise."""
        config = GuardConfig(
            tool_repeat_threshold=2,
            tool_repeat_window=5,
        )
        assert config.tool_repeat_threshold == 2

    def test_zero_repeat_threshold_raises(self):
        """Zero repeat threshold raises ValueError."""
        with pytest.raises(ValueError, match="tool_repeat_threshold"):
            GuardConfig(tool_repeat_threshold=0)

    def test_window_less_than_threshold_raises(self):
        """Window smaller than threshold raises ValueError."""
        with pytest.raises(ValueError, match="tool_repeat_window"):
            GuardConfig(tool_repeat_threshold=5, tool_repeat_window=3)

    def test_negative_variance_raises(self):
        """Negative variance threshold raises ValueError."""
        with pytest.raises(ValueError, match="semantic_variance_threshold"):
            GuardConfig(semantic_variance_threshold=-0.1)

    def test_small_semantic_window_raises(self):
        """Semantic window less than 2 raises ValueError."""
        with pytest.raises(ValueError, match="semantic_window"):
            GuardConfig(semantic_window=1)

    def test_zero_dead_end_threshold_raises(self):
        """Zero dead end threshold raises ValueError."""
        with pytest.raises(ValueError, match="dead_end_threshold"):
            GuardConfig(dead_end_threshold=0)

    def test_low_velocity_multiplier_raises(self):
        """Velocity multiplier <= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="token_velocity_multiplier"):
            GuardConfig(token_velocity_multiplier=0.5)

    def test_zero_max_tokens_raises(self):
        """Zero max tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens_per_run"):
            GuardConfig(max_tokens_per_run=0)

    def test_zero_max_steps_raises(self):
        """Zero max steps raises ValueError."""
        with pytest.raises(ValueError, match="max_steps"):
            GuardConfig(max_steps=0)

    def test_zero_max_reflections_raises(self):
        """Zero max reflections raises ValueError."""
        with pytest.raises(ValueError, match="max_reflections"):
            GuardConfig(max_reflections=0)


class TestGuardConfigFromDict:
    """Test from_dict factory method."""

    def test_from_dict_full(self):
        """from_dict creates config from a full dictionary."""
        d = {
            "tool_repeat_threshold": 5,
            "tool_repeat_window": 10,
            "max_tokens_per_run": 100_000,
        }
        config = GuardConfig.from_dict(d)
        assert config.tool_repeat_threshold == 5
        assert config.tool_repeat_window == 10
        assert config.max_tokens_per_run == 100_000

    def test_from_dict_ignores_unknown_keys(self):
        """Unknown keys in the dictionary are silently ignored."""
        d = {
            "tool_repeat_threshold": 5,
            "unknown_key": "ignored",
        }
        config = GuardConfig.from_dict(d)
        assert config.tool_repeat_threshold == 5

    def test_from_dict_empty(self):
        """Empty dictionary produces default config."""
        config = GuardConfig.from_dict({})
        assert config.tool_repeat_threshold == 3  # default


class TestGuardConfigToDict:
    """Test to_dict serialization."""

    def test_to_dict(self):
        """to_dict produces a complete dictionary."""
        config = GuardConfig()
        d = config.to_dict()
        assert "tool_repeat_threshold" in d
        assert "max_tokens_per_run" in d
        assert "pivot_templates" in d

    def test_roundtrip(self):
        """to_dict → from_dict roundtrip preserves values."""
        original = GuardConfig(
            tool_repeat_threshold=5,
            max_tokens_per_run=200_000,
        )
        d = original.to_dict()
        restored = GuardConfig.from_dict(d)
        assert restored.tool_repeat_threshold == 5
        assert restored.max_tokens_per_run == 200_000


class TestGuardConfigMerge:
    """Test the merge method."""

    def test_merge_with_kwargs(self):
        """merge with kwargs overrides specific fields."""
        base = GuardConfig()
        merged = base.merge(max_tokens_per_run=999_999)
        assert merged.max_tokens_per_run == 999_999
        assert base.max_tokens_per_run == 50_000  # original unchanged

    def test_merge_with_other_config(self):
        """merge with another config uses its non-default values."""
        base = GuardConfig()
        other = GuardConfig(max_tokens_per_run=999_999)
        merged = base.merge(other)
        assert merged.max_tokens_per_run == 999_999

    def test_merge_none_other(self):
        """merge with None other just applies kwargs."""
        base = GuardConfig()
        merged = base.merge(None, max_steps=50)
        assert merged.max_steps == 50
