# Framework Coverage & Trust: development evidence

> [!NOTE]
> **Historical Audit Log:** This document records the original baseline verification evidence from the `0.2.0` framework trust sprint (2026-10-08). For current release verification and compatibility status, see the [Upstream Compatibility Matrix](compatibility.md) and [Publishing Standards](publishing-standards.md).

Status: **Historical `0.2.0.dev0` baseline log** on `feat/v020-framework-trust`.
Implementation baseline: `7b02104b47277cb5948c84e9ca4830e5f27df8c4` (`v0.1.4`).
Local verification date: 2026-10-08. Release publication remains a maintainer decision.

## Starting audit

The nested repository began clean at the exact roadmap release SHA. Package and import
versions were `0.1.4`. Contribution guidance, source, tests, examples and workflows were
inspected; no applicable `AGENTS.md` was present. The configured origin is the Abdullahbinmushtaq
fork. Its tag query returned no tags. Upstream `main` and `v0.1.4` both matched the
starting SHA; upstream tags also include v0.1.3 and v0.1.1. Direct PyPI metadata and upstream GitHub release API
checks still reported `0.1.4`; the latest GitHub release was published 2026-09-24. Upstream
open pull-request and issue queries returned empty lists. The development version was unused
on PyPI when checked; no tag or package was published.

A source export of the baseline passed **271 tests** using Python 3.12 with the required async
test plugin. System Python 3.14 without that plugin initially failed an async test; development
environments now use the documented dev extra. The release's README fixed count was stale;
the README now describes the test command without a frozen count.

Core policy remains in `core/breaker.py`; `config.py` owns defaults, `core/step.py` normalization,
`core/detectors/` detector interfaces, `core/reporter.py` telemetry, `core/pivot.py` recovery prompts,
and `integrations/` framework execution. Existing `guarded`, provider conversions, pricing,
report persistence, lazy integrations and `py.typed` are retained.

## Behavior changes and practical effect

| Area | Before v0.1.4 | Development preview | Effect |
|---|---|---|---|
| Terminal circuit | Clean checks could return continue after a detector trip | OPEN keeps returning the original kill without changing counters/history | Stops accidental continuation after termination |
| Reflection allowance | Recovering to CLOSED could permit further advice beyond the run allowance | Reflection count remains bounded across clean recovery | Prevents an unbounded advice/recovery cycle |
| CrewAI recovery | Callback stored/logged advice | Execution-scoped hook delivers advice before next supported model request | Local real agent recovers when it consumes advice |
| CrewAI callbacks | Agent callbacks replaced; execution event duplication unproven | Prior callbacks retained; complete actions counted once | Preserves application notifications without inflated step/usage totals |
| CrewAI lifecycle | Callback attachment alone did not establish run boundaries | Fresh report per kickoff; concurrent/repeated attachment controlled | Independent accounting and clear unsupported-mode errors |
| Tool dispatch | Excluded LangGraph tool nodes could still route after termination | Excluded nodes are gated; controlled runner stops before killed turn's tools | Prevents later guarded dispatch |
| Pending calls | Termination text could leave requests unanswered | Original IDs get truthful skipped/error results; completed results retained | Valid supported tool conversation at continuation boundary |
| Resume | Reset discarded usage/history; no explicit graph contract | Matching checkpoint, correction, finite reflection grant; cumulative accounting | Continues completed work without replaying completed tools in tested mode |
| Reports | Telemetry without resume/cleanup metadata | Additive run IDs, continuations, dispositions and usage warnings | Explains stop/continuation; old files still load |
| Detector quality | No reproducible benchmark evidence | Fixed synthetic corpus, baseline/current results, raw counts and timing | Exposes false positives and misses; approval remains pending |
| Release workflow | Unquoted Markdown, unchecked claims, candidates stable/latest, manual tag source mismatch | Literal generator, actual gates, requested tag checkout, prerelease flags, draft release | Makes artifacts and claims reviewable before publication |

## Executed verification

- Baseline Python 3.12 suite: 271 passed.
- Core-only locked Python 3.10.21: 362 passed, 3 optional/release-script modules skipped;
  strict MyPy passed. Release tooling runs on Python 3.12.
- Core-only locked Python 3.11.16: 365 passed, 2 optional framework modules skipped;
  strict MyPy passed.
- Clean wheel and sdist installs passed outside the repository; bare imports load no optional
  framework/provider modules and include `py.typed`. An actual published `0.1.4` install was
  upgraded to the local preview wheel and its saved report loaded successfully. Both quickstarts
  ran outside the source tree using the installed wheel and real framework extras.
- Combined locked frameworks (LangGraph 1.2.11, LangChain 1.3.18/core 1.6.1,
  CrewAI 1.15.22): **402 passed**, **97.60% statement coverage**, strict MyPy passed with the preserved Python 3.10 target.
- Combined newest frameworks: **402 passed**, **97.60% statement coverage**
  (1,915 measured statements; 46 uncovered). Ruff, strict MyPy (Python 3.12 target), strict MkDocs and
  wheel/sdist build passed. Exact commands/output are recorded in the
  local `release-evidence.json` and `release-coverage.json` generated by
  `scripts/verify_release.py`. Framework CI and release checks enforce at least **95%**
  aggregate statement coverage; optional dependencies are required for these checks.
- Real graph fixtures check normal completion, delivered recovery, trips, original-ID cleanup,
  cumulative resume, counted effects, independent threads, concurrent resume, changed checkpoints
  and the legacy dispatch path. Provider cleanup is parsed by the real LangChain message
  converter for OpenAI and Anthropic histories, including an existing failed result. Real CrewAI fixtures check normal/recovered/stopped executions,
  callbacks, fresh reports, caps, reattachment, shared-agent isolation and unsupported-mode rejection.
- Provider, decorator, pricing, historical reporting, lazy imports and original contracts retain
  their existing regression tests. A fixture was saved by the installed public v0.1.4 package.
- Benchmark hash/Jaccard baseline/current counts match; thresholds and algorithms were not tuned.
  The corpus has 32 held-out traces and zero reported execution errors. Human label review is pending.

The compatible locked and newest framework environments are enumerated in
[the compatibility matrix](compatibility.md). Latest NumPy typing stubs use Python 3.12 syntax,
so the framework strict-typing check uses Python 3.12; core strict checks now target Python 3.11 after raising the minimum supported version.
Warnings include CrewAI's deprecated public executor and callback serialization. Tests isolate
credential-file storage and unrelated event listeners without replacing framework execution.
No live provider requests or credentials are needed. The graph example was run directly; the
CrewAI example was run with temporary credential storage because the workspace cannot write
CrewAI's normal home storage location.

## Coverage follow-up

Coverage increased from **89% / 306 tests** to **97.60% / 402 tests** without excluding
production code. The 96 additional cases exercise public exports, configuration validation,
decorator extraction and termination policies, framework normalization, checkpoint rejection,
CrewAI unsupported modes and concurrent execution, and custom-detector failure isolation.

These tests exposed two decorator defects. Once the circuit is OPEN, synchronous, asynchronous
and generator wrappers now stop before further body execution, including post-observation
functions. Async and generator resets now expose the new report and clear old reflection
advice. Regression tests assert side-effect counts, original reports and all kill policies.

Coverage measures executed statements, not complete branch coverage or detector accuracy.
Optional embedding-model execution, live provider acceptance and release approval remain
separate validation work.

## Reproduce engineering gates

```bash
uv sync --python 3.12 --extra dev --extra langgraph --extra langchain --extra crewai
uv run --no-sync python scripts/verify_release.py
uv run --no-sync python scripts/release_notes.py --tag v0.2.0.dev0   --evidence release-evidence.json --output release_body.md
uv run --no-sync python examples/langgraph_resume.py
uv run --no-sync python examples/crewai_guarded.py
uv run --no-sync python tests/benchmarks/run.py --output tests/benchmarks/results/current-hash.json
```

The check record captures actual commands, exit statuses, dependency/platform versions, dirty
checkout status and artifact hashes. A local edited checkout is not a tagged release artifact.
The GitHub workflow checks the tag's commit and package version, then creates a draft. It has
no PyPI publishing step.

## Maintainer delivery and feedback

Engineering work is local and reviewable. No GitHub PR was opened: no GitHub CLI or connector is available
and no GitHub API credential is present. The local draft description is
`.github/PR_v0.2.0_DRAFT.md`; maintainers can split review into core/protocol, framework, and
benchmark/release changes. No main push, merge, stable tag or publication was performed.

Before promoting this preview to an RC or stable version:

1. Review labels/split leakage in `tests/benchmarks/LABEL_REVIEW.md`, then rerun the evaluation and
   record the quality decision, especially legitimate polling and semantic false positives.
2. Execute hosted CI, including macOS; local Linux results do not prove that matrix.
3. Review supported framework mode restrictions and deprecated CrewAI executor behavior.
4. Choose an unused RC version, update package/import/lockfile/changelog together, verify the
   exact source tag and built wheel/sdist, and perform candidate clean-install/upgrade checks.
5. Maintainer publishes the candidate through the project's chosen PyPI credentials/procedure
   (no existing automated PyPI workflow was found), then checks published metadata and fresh installs.
6. Collect minimized/redacted reports from candidate users: successful useful work, false
   positives, recovery failures, replay concerns and framework versions. Record evidence and
   priority; defer unrelated feature requests. No candidate/community feedback is claimed here.
7. Confirm acceptance and promote only after maintainer approval. The stable version must be
   unused and the final artifacts rebuilt from the reviewed source.

Sentence-transformers quality, live provider acceptance, community feedback, hosted CI and
human label review remain explicit gaps. Reports are local telemetry, not cross-process
execution checkpoints or billing statements.

## Expected local example output

The graph example prints `Finished using existing results.`, completed tool IDs
`search_0` and `search_1`, and a CLOSED report with four cumulative steps and one reflection.
Its resumed run does not add a third tool side effect. Its text-derived token count is an
estimate. The CrewAI example prints `recovered` and a CLOSED report with three steps and
one reflection. The local model exposes no usage, so its compatibility zero token count
has a missing-usage warning; it is not proof of free provider execution.

For feedback, use the repository's existing GitHub Issues destination with a minimized,
redacted reproduction, tested versions and expected/actual behavior. External user feedback
and the maintainer's exact PyPI publishing procedure remain unconfirmed.

## Local review units

The work is committed on the feature branch in focused core/protocol, graph continuation,
CrewAI and benchmark/release units. The saved benchmark identifies source commit
`ba62f8748ff82eb62f70ad115fe4dcb8a0815f49` from a clean checkout. The coverage follow-up changes decorator lifecycle
code, so this saved package-source digest describes that earlier snapshot. Detector algorithms
and thresholds remain unchanged.
Review the branch against `7b02104b47277cb5948c84e9ca4830e5f27df8c4`.

The preview proposes qualifying the old blanket latency wording with measured workload/backend
evidence. Maintainer approval of public quality/performance claims remains part of the release decision.

## Python minimum update

The minimum supported Python version was subsequently raised to **3.11**. Python 3.10
results above describe earlier compatibility checks; the current package metadata and CI
require Python 3.11 or newer.

## Benchmark location and CI repair — 2026-10-09

Benchmark source, corpus, review instructions and saved results are tracked under
`tests/benchmarks/`. CI, the benchmark test and reproduction instructions now use this
location. Release lint checks `tests/`, which includes the benchmark runner. The runner
finds its default corpus beside its script; it no longer depends on the old root folder.
A regression test executes the CLI with its bundled default corpus.

After this repair, the combined framework environment passed **403 tests** with
**97.60% statement coverage**. Ruff, strict MyPy, strict MkDocs and the offline wheel/sdist
build passed through `scripts/verify_release.py --offline`. Core-only Python 3.11 passed
**366 tests**, with two optional framework modules skipped. The configured CI benchmark
command evaluated 32 held-out traces with zero detector execution errors. These are local
Linux results; hosted CI must rerun after the fix is pushed. Saved historical benchmark
results retain their original source and command metadata.
