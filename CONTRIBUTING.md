# Contributing to LongGuard

Thank you for your interest in contributing to **LongGuard**! We welcome bug reports, feature requests, documentation improvements, and pull requests.

LongGuard is an open-source project by [EnDevSols](https://github.com/ENDEVSOLS).

---

## Development Setup

LongGuard uses [`uv`](https://github.com/astral-sh/uv) for fast and deterministic Python environment management.

### 1. Clone the repository

```bash
git clone https://github.com/ENDEVSOLS/LongGuard.git
cd LongGuard
```

### 2. Install dependencies

```bash
# Install package with all dev dependencies
uv sync --extra dev
```

### 3. Run Tests

```bash
# Run all unit tests
uv run pytest tests/ -v

# Run with test coverage
uv run pytest tests/ --cov=longguard --cov-report=term-missing
```

### 4. Code Quality Checks

We enforce formatting and linting with **Ruff** and strict static typing with **MyPy**:

```bash
# Run linter
uv run ruff check src/ tests/

# Run type checker
uv run mypy src/
```

Ensure all tests, lint checks, and type checks pass before submitting a Pull Request.

---

## Pull Request Guidelines

1. **Focus on solving real agent failure modes**: If adding a new detector, provide synthetic or real-world traces in `tests/` demonstrating both positive detection and non-interference on legitimate tasks.
2. **Keep overhead minimal**: Loop detection runs synchronously on each agent step; algorithms should execute in sub-millisecond time.
3. **Preserve framework neutrality**: Core detection mechanisms (`src/longguard/core/`) must remain independent of specific agent frameworks (LangGraph, LangChain, etc.). Framework integrations belong in `src/longguard/integrations/`.
4. **Follow Semantic Versioning**: Major for breaking changes, minor for new detectors/adapters, patch for bug fixes.

---

## License

By contributing to LongGuard, you agree that your contributions will be licensed under the [MIT License](LICENSE).
