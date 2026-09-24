"""Universal @guarded decorator for Python agent functions and loops.

Provides a 1-line runtime circuit breaker wrapper for custom agent functions,
SDK calls, generators, and async loops without needing framework-specific code.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Generator
from typing import Any, TypeVar, cast, overload

from ..config import GuardConfig
from .breaker import BreakerDecision, CircuitBreaker, CircuitBreakerTrippedError
from .reporter import GuardReport
from .step import AgentStep

F = TypeVar("F", bound=Callable[..., Any])


def _extract_step_from_args_kwargs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    step_number: int,
) -> AgentStep | None:
    """Attempt to extract or construct an AgentStep from function arguments."""
    # 1. Direct AgentStep in args
    for arg in args:
        if isinstance(arg, AgentStep):
            return arg

    # 2. Direct AgentStep in kwargs
    for val in kwargs.values():
        if isinstance(val, AgentStep):
            return val

    # 3. Known step keys in kwargs
    step_keys = {"thought", "action", "action_input", "observation", "tokens_used"}
    if any(k in kwargs for k in step_keys):
        return AgentStep(
            step_number=step_number,
            thought=str(kwargs.get("thought", "")),
            action=kwargs.get("action"),
            action_input=kwargs.get("action_input"),
            observation=kwargs.get("observation"),
            tokens_used=int(kwargs.get("tokens_used", 0)),
        )

    return None


def _extract_step_from_result(
    result: Any,
    step_number: int,
    observation: Any = None,
) -> AgentStep | None:
    """Attempt to extract an AgentStep from a function return value."""
    if isinstance(result, AgentStep):
        return result

    # OpenAI ChatCompletion object (has choices)
    if hasattr(result, "choices"):
        try:
            return AgentStep.from_openai_response(
                result,
                step_number=step_number,
                observation=str(observation) if observation is not None else None,
            )
        except Exception:
            pass

    # Anthropic Message object (has content and role or type)
    if hasattr(result, "content") and (hasattr(result, "role") or hasattr(result, "type")):
        try:
            return AgentStep.from_anthropic_response(
                result,
                step_number=step_number,
                observation=str(observation) if observation is not None else None,
            )
        except Exception:
            pass

    # Dictionary representation
    if isinstance(result, dict):
        step_keys = {"thought", "action", "action_input", "observation", "tokens_used"}
        if any(k in result for k in step_keys):
            return AgentStep(
                step_number=step_number,
                thought=str(result.get("thought", "")),
                action=result.get("action"),
                action_input=result.get("action_input"),
                observation=result.get("observation"),
                tokens_used=int(result.get("tokens_used", 0)),
            )

    return None


class GuardedFunctionWrapper:
    """Wrapper that attaches CircuitBreaker control to a decorated callable."""

    def __init__(
        self,
        func: Callable[..., Any],
        breaker: CircuitBreaker,
        on_kill: str | Callable[..., Any] = "raise",
        step_extractor: Callable[..., AgentStep | None] | None = None,
    ) -> None:
        self.func = func
        self.breaker = breaker
        self.on_kill = on_kill
        self.step_extractor = step_extractor
        self.last_reflection_prompt: str | None = None
        functools.update_wrapper(self, func)

    @property
    def report(self) -> GuardReport:
        """The GuardReport of the underlying CircuitBreaker."""
        return self.breaker.report

    def reset(self) -> None:
        """Reset the breaker state and telemetry for a fresh run."""
        self.breaker.reset()
        self.last_reflection_prompt = None

    def _handle_kill(
        self,
        decision: BreakerDecision,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if callable(self.on_kill):
            return self.on_kill(decision, *args, **kwargs)
        if self.on_kill == "return":
            return f"Agent terminated by LongGuard: {decision.reason}"
        raise CircuitBreakerTrippedError(
            reason=decision.reason,
            report=self.breaker.report,
            decision=decision,
        )

    def _check_and_handle(
        self,
        step: AgentStep,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[BreakerDecision, Any | None]:
        decision = self.breaker.check(step)
        if decision.action == "reflect":
            self.last_reflection_prompt = decision.inject_prompt
        elif decision.action == "kill":
            kill_res = self._handle_kill(decision, args, kwargs)
            return decision, kill_res
        return decision, None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        next_step_num = len(self.breaker.report.steps) + 1

        # 1. Custom or pre-call step extraction from args/kwargs
        pre_step: AgentStep | None = None
        if self.step_extractor is not None:
            pre_step = self.step_extractor(*args, **kwargs)
        if pre_step is None:
            pre_step = _extract_step_from_args_kwargs(args, kwargs, step_number=next_step_num)

        if pre_step is not None:
            decision, kill_res = self._check_and_handle(pre_step, args, kwargs)
            if decision.action == "kill":
                return kill_res

        # 2. Invoke wrapped function
        result = self.func(*args, **kwargs)

        # 3. Post-call extraction from return value if not already extracted
        if pre_step is None:
            post_step: AgentStep | None = None
            if self.step_extractor is not None:
                post_step = self.step_extractor(*args, result=result, **kwargs)
            if post_step is None:
                obs = kwargs.get("observation")
                post_step = _extract_step_from_result(
                    result, step_number=next_step_num, observation=obs
                )

            if post_step is not None:
                decision, kill_res = self._check_and_handle(post_step, args, kwargs)
                if decision.action == "kill":
                    return kill_res

        return result


@overload
def guarded(func: F) -> F:
    ...


@overload
def guarded(
    func: None = None,
    *,
    config: GuardConfig | None = None,
    breaker: CircuitBreaker | None = None,
    on_kill: str | Callable[..., Any] = "raise",
    step_extractor: Callable[..., AgentStep | None] | None = None,
    **config_kwargs: Any,
) -> Callable[[F], F]:
    ...


def guarded(
    func: F | None = None,
    *,
    config: GuardConfig | None = None,
    breaker: CircuitBreaker | None = None,
    on_kill: str | Callable[..., Any] = "raise",
    step_extractor: Callable[..., AgentStep | None] | None = None,
    **config_kwargs: Any,
) -> Any:
    """Decorate an agent function, loop step, or generator with LongGuard protection.

    Usage::

        # 1. Simple kwargs configuration
        @guarded(max_cost_usd=0.25, tool_repeat_threshold=2)
        def run_step(thought: str, action: str, action_input: dict, observation: str):
            ...

        # 2. Decorating an OpenAI SDK call
        @guarded(model="gpt-4o", max_cost_usd=1.0)
        def call_llm(messages: list[dict]):
            return client.chat.completions.create(model="gpt-4o", messages=messages)

        # 3. Handling termination gracefully without raising
        @guarded(on_kill="return", tool_repeat_threshold=3)
        def execute_tool(action: str, action_input: str):
            ...

    Args:
        func: The function to decorate.
        config: GuardConfig instance. If omitted, built from config_kwargs or defaults.
        breaker: Existing CircuitBreaker instance.
        on_kill: Termination behavior. "raise" (default) raises CircuitBreakerTrippedError,
            "return" returns a fallback error string, or a custom callable
            ``f(decision, *args, **kwargs)``.
        step_extractor: Optional callable `(args, kwargs, result=None) -> AgentStep`.
        **config_kwargs: Passed to GuardConfig if config is not explicitly provided.
    """

    def decorator(fn: F) -> Any:
        # Resolve config and breaker
        if breaker is not None:
            cb = breaker
        elif config is not None:
            cb = CircuitBreaker(config)
        else:
            cb = CircuitBreaker(GuardConfig(**config_kwargs))

        # Generator function
        if inspect.isgeneratorfunction(fn):

            @functools.wraps(fn)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Generator[Any, Any, Any]:
                gen = fn(*args, **kwargs)
                wrapper_obj: Any = gen_wrapper
                while True:
                    try:
                        item = next(gen)
                    except StopIteration as stop:
                        return stop.value

                    step_num = len(cb.report.steps) + 1
                    step = None
                    if step_extractor is not None:
                        step = step_extractor(*args, result=item, **kwargs)
                    if step is None:
                        step = _extract_step_from_result(item, step_number=step_num)
                    if step is None:
                        step = _extract_step_from_args_kwargs(args, kwargs, step_number=step_num)

                    if step is not None:
                        decision = cb.check(step)
                        if decision.action == "reflect":
                            wrapper_obj.last_reflection_prompt = decision.inject_prompt
                        elif decision.action == "kill":
                            if callable(on_kill):
                                yield on_kill(decision, *args, **kwargs)
                                return
                            if on_kill == "return":
                                yield f"Agent terminated by LongGuard: {decision.reason}"
                                return
                            raise CircuitBreakerTrippedError(
                                reason=decision.reason,
                                report=cb.report,
                                decision=decision,
                            )
                    yield item

            gen_wrapper.breaker = cb  # type: ignore[attr-defined]
            gen_wrapper.report = cb.report  # type: ignore[attr-defined]
            gen_wrapper.reset = cb.reset  # type: ignore[attr-defined]
            gen_wrapper.last_reflection_prompt = None  # type: ignore[attr-defined]
            return cast(F, gen_wrapper)

        # Async function
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                wrapper_obj: Any = async_wrapper
                next_step_num = len(cb.report.steps) + 1
                pre_step = None
                if step_extractor is not None:
                    pre_step = step_extractor(*args, **kwargs)
                if pre_step is None:
                    pre_step = _extract_step_from_args_kwargs(
                        args, kwargs, step_number=next_step_num
                    )

                if pre_step is not None:
                    decision = cb.check(pre_step)
                    if decision.action == "reflect":
                        wrapper_obj.last_reflection_prompt = decision.inject_prompt
                    elif decision.action == "kill":
                        if callable(on_kill):
                            return on_kill(decision, *args, **kwargs)
                        if on_kill == "return":
                            return f"Agent terminated by LongGuard: {decision.reason}"
                        raise CircuitBreakerTrippedError(
                            reason=decision.reason,
                            report=cb.report,
                            decision=decision,
                        )

                result = await fn(*args, **kwargs)

                if pre_step is None:
                    post_step = None
                    if step_extractor is not None:
                        post_step = step_extractor(*args, result=result, **kwargs)
                    if post_step is None:
                        post_step = _extract_step_from_result(
                            result, step_number=next_step_num
                        )
                    if post_step is not None:
                        decision = cb.check(post_step)
                        if decision.action == "reflect":
                            wrapper_obj.last_reflection_prompt = decision.inject_prompt
                        elif decision.action == "kill":
                            if callable(on_kill):
                                return on_kill(decision, *args, **kwargs)
                            if on_kill == "return":
                                return f"Agent terminated by LongGuard: {decision.reason}"
                            raise CircuitBreakerTrippedError(
                                reason=decision.reason,
                                report=cb.report,
                                decision=decision,
                            )

                return result

            async_wrapper.breaker = cb  # type: ignore[attr-defined]
            async_wrapper.report = cb.report  # type: ignore[attr-defined]
            async_wrapper.reset = cb.reset  # type: ignore[attr-defined]
            async_wrapper.last_reflection_prompt = None  # type: ignore[attr-defined]
            return cast(F, async_wrapper)

        # Standard synchronous function
        sync_wrapper = GuardedFunctionWrapper(
            func=fn,
            breaker=cb,
            on_kill=on_kill,
            step_extractor=step_extractor,
        )
        return cast(F, sync_wrapper)

    if func is not None:
        return decorator(func)
    return decorator
