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
# Install package with all dev & doc dependencies
uv sync --extra dev --extra docs
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

### 5. Preview Documentation Locally

```bash
uv run mkdocs serve
```
Open `http://127.0.0.1:8000` in your browser.

---

## License

By contributing to LongGuard, you agree that your contributions will be licensed under the [MIT License](https://github.com/ENDEVSOLS/LongGuard/blob/main/LICENSE).
