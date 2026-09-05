"""
LongGuard: Custom Detector Example

This example shows how to create a custom loop detector by subclassing
AbstractLoopDetector, and register it with the CircuitBreaker.

Custom detectors let you add domain-specific loop detection logic
beyond the built-in patterns (tool_repeat, semantic_oscillation,
dead_end_drift, token_velocity).

Usage:
    python custom_detector.py
"""

from longguard import GuardConfig, CircuitBreaker, AgentStep
from longguard.core.detectors.base import AbstractLoopDetector, DetectionResult
from longguard.core.breaker import BreakerState


class ErrorMessageDetector(AbstractLoopDetector):
    """Detects when the agent repeatedly receives error messages.

    This is a common pattern: the agent calls a tool, gets an error,
    retries the same call, gets the same error, and loops forever.
    Unlike ToolRepeatDetector which checks for identical calls, this
    detector focuses on the observation content — detecting when
    the agent keeps hitting errors regardless of which tool it uses.

    Args:
        error_threshold: Number of consecutive error observations to trigger.
        window: Sliding window size.
        error_patterns: List of substrings that indicate an error response.
    """

    def __init__(
        self,
        error_threshold: int = 3,
        window: int = 5,
        error_patterns: list | None = None,
    ):
        self.error_threshold = error_threshold
        self.window = window
        self.error_patterns = error_patterns or [
            "error",
            "failed",
            "not found",
            "invalid",
            "unauthorized",
            "timeout",
        ]
        self._error_count = 0

    @property
    def name(self) -> str:
        return "ErrorMessageDetector"

    def analyze(self, step: AgentStep) -> DetectionResult:
        """Check if the step's observation contains an error pattern."""
        if step.observation is None:
            return DetectionResult(detected=False)

        obs_lower = step.observation.lower()
        is_error = any(pattern in obs_lower for pattern in self.error_patterns)

        if is_error:
            self._error_count += 1
        else:
            self._error_count = 0

        if self._error_count >= self.error_threshold:
            return DetectionResult(
                detected=True,
                pattern="error_message_loop",
                confidence=min(self._error_count / self.error_threshold, 1.0),
                evidence={
                    "consecutive_errors": self._error_count,
                    "last_observation": step.observation[:100],
                    "threshold": self.error_threshold,
                },
            )

        return DetectionResult(detected=False)

    def reset(self) -> None:
        """Clear the error counter."""
        self._error_count = 0


class StateRegressionDetector(AbstractLoopDetector):
    """Detects when the agent returns to a previous reasoning state.

    This detector keeps a history of thought summaries and checks
    if a new thought is semantically identical to a previous one,
    indicating the agent has circled back to an earlier conclusion.

    Uses simple Jaccard similarity on token sets as a lightweight
    alternative to embedding-based similarity.

    Args:
        similarity_threshold: Jaccard similarity above which two thoughts
            are considered the same reasoning state.
        regression_threshold: Number of regressions to trigger detection.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        regression_threshold: int = 2,
    ):
        self.similarity_threshold = similarity_threshold
        self.regression_threshold = regression_threshold
        self._thought_history: list[set[str]] = []
        self._regression_count = 0

    @property
    def name(self) -> str:
        return "StateRegressionDetector"

    def _tokenize(self, text: str) -> set[str]:
        return set(text.lower().split())

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 0.0
        return len(a & b) / len(a | b)

    def analyze(self, step: AgentStep) -> DetectionResult:
        """Check if the current thought regresses to a previous state."""
        current_tokens = self._tokenize(step.thought)

        # Check against all previous thoughts
        for i, prev_tokens in enumerate(self._thought_history):
            similarity = self._jaccard(current_tokens, prev_tokens)
            if similarity > self.similarity_threshold:
                self._regression_count += 1
                if self._regression_count >= self.regression_threshold:
                    return DetectionResult(
                        detected=True,
                        pattern="state_regression",
                        confidence=similarity,
                        evidence={
                            "similar_to_step": i + 1,
                            "similarity": similarity,
                            "regression_count": self._regression_count,
                        },
                    )
                break

        self._thought_history.append(current_tokens)
        return DetectionResult(detected=False)

    def reset(self) -> None:
        """Clear thought history and regression counter."""
        self._thought_history.clear()
        self._regression_count = 0


def demo_custom_detector():
    """Demonstrate custom detector usage."""
    print("=" * 60)
    print("LongGuard Custom Detector Example")
    print("=" * 60)
    print()

    # Create a breaker with our custom detector
    config = GuardConfig(
        max_tokens_per_run=50_000,
        max_steps=20,
        max_reflections=2,
    )
    breaker = CircuitBreaker(config)

    # Register the custom detector
    error_detector = ErrorMessageDetector(error_threshold=2)
    breaker.add_detector(error_detector)

    # Also register a custom pivot template for the new pattern
    breaker.pivot.add_template(
        "error_message_loop",
        "SYSTEM OVERRIDE — ERROR LOOP DETECTED:\n"
        "You have received {consecutive_errors} consecutive error responses.\n"
        "The current approach is not working.\n\n"
        "REQUIRED: Try a different tool or reformulate your request.\n"
        "Do NOT repeat the same request that produced the error.",
    )

    print("Registered custom detector:", error_detector.name)
    print("Registered custom pivot template for 'error_message_loop'")
    print()

    # Simulate agent steps with errors
    steps = [
        AgentStep(
            step_number=1,
            thought="Let me query the database.",
            action="sql_query",
            action_input="SELECT * FROM users",
            observation="Error: Table 'users' not found",
            tokens_used=100,
        ),
        AgentStep(
            step_number=2,
            thought="Let me try the query again with a different format.",
            action="sql_query",
            action_input="SELECT * FROM Users",
            observation="Error: Table 'Users' not found",
            tokens_used=120,
        ),
        AgentStep(
            step_number=3,
            thought="Maybe the table exists with a slightly different name.",
            action="sql_query",
            action_input="select * from USERS",
            observation="Error: Table 'USERS' not found",
            tokens_used=110,
        ),
        AgentStep(
            step_number=4,
            thought="I should try listing available tables first.",
            action="sql_query",
            action_input="SHOW TABLES",
            observation="Available tables: orders, products, customers",
            tokens_used=90,
        ),
    ]

    for step in steps:
        decision = breaker.check(step)
        print(f"Step {step.step_number}: action={step.action} → decision={decision.action}")

        if decision.action == "reflect":
            print(f"  ⚠️  Pattern: {decision.reason}")
            print(f"  Pivot: {decision.inject_prompt[:80]}...")

        elif decision.action == "kill":
            print(f"  🛑 Killed: {decision.reason}")

        print()

    # Print the final report
    print("Run Report:")
    print(breaker.report.summary())


if __name__ == "__main__":
    demo_custom_detector()
