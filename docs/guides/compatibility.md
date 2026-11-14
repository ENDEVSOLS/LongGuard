# Tested execution and compatibility

[![Upstream Compatibility Canary](https://github.com/ENDEVSOLS/LongGuard/actions/workflows/canary.yml/badge.svg)](https://github.com/ENDEVSOLS/LongGuard/actions/workflows/canary.yml)

LongGuard 0.3.x extends the existing circuit breaker. Optional framework dependency
lower bounds are tested and verified against the execution contracts shown below. Bare `import longguard`
keeps framework dependencies optional.

## Upstream Compatibility Matrix

LongGuard continuously verifies compatibility through scheduled nightly and on-demand canary CI workflows that execute integration test suites against both the lowest declared dependency bounds and the latest upstream releases.

| Framework / Library | Declared Minimum | Verified Latest | Canary Status | Verified Integration Contracts |
|---|---|---|---|---|
| **LangGraph** | `>=1.0.0` | `1.2.14` | [![Canary](https://github.com/ENDEVSOLS/LongGuard/actions/workflows/canary.yml/badge.svg)](https://github.com/ENDEVSOLS/LongGuard/actions/workflows/canary.yml) | Synchronous 2-node graph execution, `compile_guarded_graph`, `GuardedGraph.resume` checkpoint recovery |
| **LangChain Core** | `>=1.0.0` | `1.6.7` | Passing | Tool execution wrappers, message normalization, `AIMessage`/`ToolMessage` histories |
| **LangChain** | `>=1.0.0` | `1.4.3` | Passing | `GuardedAgentExecutor`, callback-driven step and loop interception |
| **CrewAI** | `>=1.15.19` | `1.15.25` (`<1.15.29`) | Passing | Sequential ReAct `Crew.kickoff` execution, `add_guard_to_crew`, execution-scoped hook API (`crewai.hooks.dispatch`) |
| **OpenAI / Anthropic Client** | Unpinned | Latest | Passing | Direct client decoration (`@guarded`), unresolved tool call cleanup, usage extraction |

## Execution modes

| Interface | Supported behavior | Boundary or unsupported mode |
|---|---|---|
| `CircuitBreaker.check` | Existing synchronous policy and additive explicit resume | Application owns model/tool dispatch and checkpoint integrity |
| `guarded`, provider converters, `GuardedAgentExecutor` | Existing contracts retained and regression tested | Decorator advice must be consumed by the application |
| `add_guard_to_graph` | Existing wrapping; excluded tools are gated after OPEN | One mutable guard; reset for a new run; use controlled runner for thread isolation/resume |
| `compile_guarded_graph` | Synchronous `START -> agent -> tools/END`, `tools -> agent`; saved node boundaries | Exactly two nodes; no async, streaming, subgraphs, parallel nodes, namespaces or checkpoint branching |
| `GuardedGraph.resume` | Same controller/process; matching stopped checkpoint; explicit correction and finite reflection grant | Hard caps, completed/failed runs, unknown threads, concurrent continuation and changed checkpoints rejected |
| `add_guard_to_crew` | Sequential, synchronous ReAct `Crew.kickoff`; public `CrewAgentExecutor`; separate report per kickoff | Async/batch/streaming/hierarchical/native tool batches, delegation, reasoning, planning, memory and CrewAI checkpoint continuation rejected |
| Callback-only `CrewGuard` | Existing callback API, reports and `last_reflection` | No automatic advice delivery or full execution guarantee; attach through `add_guard_to_crew` for tested recovery |
| Provider cleanup helper | Valid OpenAI/Anthropic dictionaries and LangChain message histories | Caller must block dispatch; running tools retain unresolved status; invalid IDs rejected |

CrewAI's public `CrewAgentExecutor` is deprecated in the tested release. Its alternative
executor is outside this preview's proven mode. A future compatibility change needs real
contract tests before widening support. The adapter installs execution-scoped hooks rather
than registering global handlers.

## Version evidence

| Local environment | Python | LangGraph | LangChain / core | CrewAI |
|---|---|---|---|---|
| Lockfile graph verification | 3.12.14 | 1.2.11 | 1.3.18 / 1.6.1 | Separately tested 1.15.22 |
| Combined environment before CrewAI upgrade | 3.12.14 | 1.2.14 | 1.4.3 / 1.6.7 | 1.15.22 |
| Latest stable verification | 3.12.14 | 1.2.14 | 1.4.3 / 1.6.7 | 1.15.25 |

Latest-version results and exact commands are recorded in the development evidence after
verification. Only executed checks establish support. CI runs the 9-cell Python
3.11–3.13 matrix across Ubuntu, macOS, and Windows plus separate locked/newest framework jobs; configuration
of a CI job is not evidence that those platforms passed locally.

## Usage and side effects

Guards observe completed requests. A token/step cap trips on `>` and a known estimated-cost
cap on `>=`, as before. The crossing request has already incurred usage. CrewAI's complete
action callback follows its tool result: that turn's tool may already have executed before
the decision. Subsequent guarded requests stop after a trip.

Actual model usage is used when available. Graph text estimates and CrewAI missing-usage
zero defaults are identified in `usage_warnings`. Cost retains the approximate half-input,
half-output token split; it is not provider billing. Guarded continuation retains previous
consumption. Changing the budget or importing a report does not authorize execution.

Callbacks and arbitrary tools remain application code: LongGuard cannot undo their side
effects or guarantee exactly-once behavior after an application crash. The supported graph
continuation avoids replaying completed tools at its saved boundary in the same process.

Newest NumPy 2.5 typing stubs require Python 3.12 syntax. Framework jobs use strict MyPy
with a Python 3.12 target; core jobs use the Python 3.11 target and locked dependencies.

Controlled continuation was verified with LangGraph `InMemorySaver`; other saver implementations
are unverified. Pass an unwrapped graph to the controller. Do not share guarded agents or the
same guard between crews. Shared provider usage counters across crews are outside tested accounting.
