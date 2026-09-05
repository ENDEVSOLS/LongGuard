# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.3] - 2026-09-05

### 🚀 Added
- **Raw Client Integration (OpenAI & Anthropic)**: Added `AgentStep.from_openai_response()` and `AgentStep.from_anthropic_response()` class methods. Enables using LongGuard directly with standard `openai` and `anthropic` SDK client loops without requiring LangGraph or LangChain.
- **Dollar Cost Tracking**: Built-in pricing engine (`PRICING_TABLE`, `compute_cost`, `list_supported_models`) supporting 40+ major LLM models (OpenAI GPT-4o/o1/o3, Anthropic Claude 3.5/4, Gemini 1.5/2.0/2.5, LLaMA 3.1/3.3, Mistral, Cohere).
- **Budget Hard Cap (`max_cost_usd`)**: Configure hard dollar spend limits on agent runs (`max_cost_usd`) to immediately trip the circuit breaker and prevent runaway API bills.
- **Custom Token Pricing Overrides**: Added `cost_per_input_token` and `cost_per_output_token` in `GuardConfig` for private/fine-tuned or unlisted models.
- **Report Persistence (`save` / `load`)**: Persist telemetry to disk via `report.save("run.json")` and restore full state using `GuardReport.load("run.json")`. Supports JSON natively and YAML (when `pyyaml` is installed).
- **Runnable Examples & Docs**: Added standalone example in `examples/openai_raw_guard.py` and documentation in `docs/integrations/raw-client.md`.

### 🔄 Changed
- **GuardReport Telemetry**: `GuardReport.summary()` now displays `Estimated Cost: $X.XXXX USD (model)` when cost tracking is configured.
- **Expanded Test Suite**: Added 76 new test cases (255 tests total, 100% passing) covering SDK response parsing, pricing calculations, breaker cost limits, and report serialization.
- **Automated GitHub Releases**: Enhanced GitHub Actions workflow to parse release notes directly from `CHANGELOG.md` and exclude internal build metadata from release assets.

---

## [0.1.2] - 2026-09-05

### 🎨 Changed
- **Dual-Theme Branding**: Added dual-theme vector logos (`logo-light.svg` and `logo-dark.svg`) ensuring crystal-clear visibility across GitHub Dark, GitHub Light, and PyPI.
- **Clean Architecture Diagram**: Replaced architecture diagram with high-contrast, professional SVG asset.
- **Contact Updates**: Updated repository contact and security email addresses to `technology@endevsols.com`.
- **Branding Alignment**: Refactored README and documentation to match the EnDevSols Long Suite design standards.

---

## [0.1.1] - 2026-09-05

### 🚀 Added
- **Initial Public Release** of LongGuard: Production-grade in-flight circuit breaker for autonomous AI agents.
- **Circuit Breaker State Machine**: 4-state lifecycle (`CLOSED`, `REFLECTING`, `HALF_OPEN`, `OPEN`).
- **4 Autonomous Loop Detectors**:
  - `ToolRepeatDetector`: Identifies repetitive tool invocations with identical parameters.
  - `SemanticOscillationDetector`: Detects thought cycles and semantic loops using embedding variance.
  - `DeadEndDriftDetector`: Flags repeated empty, uninformative, or error-laden tool observations via Jaccard similarity.
  - `TokenVelocityDetector`: Monitors token consumption spikes against dynamic baselines.
- **Reflect & Pivot Recovery**: Dynamic prompt injection mechanism that guides stuck agents out of loops before halting.
- **Framework Integrations**:
  - LangGraph 1.0+ drop-in (`add_guard_to_graph`, `LongGuard`).
  - LangChain `GuardedAgentExecutor` wrapper.
  - Experimental Strands agent framework support (`StrandsGuard`).
- **Full Observability**: Comprehensive telemetry and run analytics via `GuardReport`.
- **CI/CD & Documentation**: Automated GitHub Actions test matrix across Python 3.10–3.12 and Material for MkDocs documentation site.
