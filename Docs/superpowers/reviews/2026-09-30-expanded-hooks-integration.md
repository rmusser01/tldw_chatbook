# Expanded hooks: current-dev integration

Base: `origin/dev` at `10b34ffdd5698a64de9c285ef899e8c610055ff0`.
Branch: `codex/expanded-hooks`. The clean original `codex/managed-plugins`
checkpoint `cdeb1687b6` is preserved and has no published PR. Its completed
implementation tasks were still marked To Do on dev; this stream reuses and
qualifies each implementation before adding the remaining capability work.

## TASK-32676 — v2 validation

Reused H1 checkpoint `7e99298cf2`. The current legacy inventory, row enablement,
shared validation and persistent exact-definition consent are retained. Explicit
v2 declarations have a separate lazy projection, including bounded invalid
required/event-policy metadata. No new v2 execution entry is activated by H1.

The unchanged current-dev baseline returned **81 passed, 1 failed**. The failure
was the old real-config test changing its profile after source-bound recovery
admission. The test now uses the existing selected private profile fixture.
Public-loader RED returned **2 failed, 2 passed**: missing v2 definitions and
required failure metadata, with disabled-legacy and real-save controls passing.

Final covering command:

```sh
.venv/bin/python -m pytest Tests/Agents/test_hooks_v2_validation.py Tests/Agents/test_run_hooks.py Tests/Chat/test_run_hooks_metadata.py Tests/Agents/test_hook_permissions.py Tests/Agents/test_hook_config_inventory.py Tests/UI/test_settings_hooks.py -q --timeout=120 --basetemp=/private/tmp/expanded-hooks-h1-qualified --junitxml=/private/tmp/expanded-hooks-h1-qualified.xml
```

**302 passed in 50.69s**, no skips. Final fixture alias/import cleanup was followed
by the exact metadata file: **6 passed**, separately recorded and overlapping.
The actual imported package is this worktree, Python 3.12.11/Pydantic 2.12.5.
Eight authored Python files pass full Ruff and formatting; all nine changed/new
Python files parse. Shared config.py retains exactly the same 168 existing Ruff
diagnostics. `git diff --check` passes.

Existing [ADR-163](../../../backlog/decisions/163-expanded-console-hook-runtime.md),
[ADR-148](../../../backlog/decisions/148-console-run-hooks.md), and
[ADR-197](../../../backlog/decisions/197-console-hook-configuration-review.md)
apply. H1 self-review covers all four ACs. No full suite or live-provider run,
new storage schema, dependency or permission owner is claimed. Command execution,
tool pipelines, lifecycle events and plugin/MCP composition remain separate tasks.

## TASK-32677 — bounded command ownership

Reused reviewed H2 checkpoint `17ee201cec`. The runtime integration preserves
current consent/recovery/voice owners and exact-session close fences. Shared
budgets count queued, active, suspended and cleanup-pending deliveries; process
custody survives cancellation and bounded shutdown. No lifecycle producer or
plugin authority is activated by H2.

Baseline shutdown/viewless: **48 passed**. Missing runtime registration RED:
**1 failed, 1 passed**, with normalization as its same-run control. Integrated
H2 core: **85 passed**. A capacity-pressure regression confirmed rejected runtime
IDs retained empty counters (**2 failed, 4 passed**); reserve now inserts only
successfully admitted counters.

Final covering command:

```sh
.venv/bin/python -m pytest Tests/Agents/test_hooks_v2_execution.py Tests/Agents/test_hooks_v2_budgets.py Tests/Chat/test_console_runtime_shutdown.py Tests/Chat/test_console_viewless_hooks.py Tests/Agents/test_run_hooks.py Tests/Agents/test_hook_permissions.py Tests/Chat/test_run_hooks_metadata.py -q --timeout=120 --basetemp=/private/tmp/expanded-hooks-h2-qualified --junitxml=/private/tmp/expanded-hooks-h2-qualified.xml
```

**262 passed in 55.13s**, no skips. Seven new Python files pass full Ruff,
eight authored/test files pass formatting, and changed/new files parse. Existing
runtime/shutdown Ruff debt remains exactly 30/2 diagnostics; whitespace passes.
Real child commands use the committed isolated-profile/provenance bootstrap.
Darwin command execution was exercised; Windows remains explicitly unsupported,
and Linux/live providers/full dependency resolution were not qualified.
All four ACs self-reviewed. Existing ADR163/148/197 apply; H2 spec/ADR163
interfaces are updated. No new dependency, storage or permission owner.

## TASK-32678 — transformations and post-event checkpoints

Reused reviewed H3 checkpoint `8192f1fb1f`. Frozen transformed arguments pass
legacy guards and existing permission review; current catalog/schema identities
are checked again before dispatch. Required postevents hold both next model input
and terminal persistence. Current approval provenance, sensitive projections,
worktree tool policy and guarded native workers survive the integration.

Unchanged baseline initially stopped at **11 passed, 1 failed** after a switched
profile made hook admission unavailable. Correct selected-profile fixture:
**135 passed**. Public behavior/core metadata RED: **3 failed, 2 passed**; the
barrier was missing and schema packages were dev-only/undeclared. Core H3:
**116 passed**. Skill and capacity fixture failures reproduce against exported
pre-H3 `d7432e3396` source; their selected-profile markers are repaired. The
capacity test now establishes physical worker startup and holds resources until
explicit release; its artificial old ten-second hold could expire during real
recovery admission. The terminal hook test likewise allows bounded cold startup
for its six controlled children while retaining deterministic held acceptance.

Final covering command:

```sh
.venv/bin/python -m pytest Tests/Agents/test_hooks_v2_tool_pipeline.py Tests/Agents/test_hooks_v2_post_checkpoints.py Tests/Agents/test_hooks_v2_execution.py Tests/Packaging/test_hooks_v2_dependencies.py Tests/Agents/test_post_tool_dispatch_hook.py Tests/Chat/test_console_run_hooks_regressions.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_skill_tool_spawn.py Tests/Agents/test_agent_models.py Tests/Agents/test_agent_runtime_review_hook.py Tests/Agents/test_tool_catalog.py Tests/Agents/test_tool_catalog_owner_cache.py Tests/Agents/test_tool_catalog_concurrency.py Tests/Agents/test_tool_record_projection.py Tests/Agents/test_tool_timeout_wall_clamp.py Tests/Agents/test_tool_call_abandon.py Tests/Agents/test_tool_worker_capacity.py Tests/Agents/test_execution_capacity.py -x -q --tb=short --show-capture=no --timeout=120 --basetemp=/private/tmp/expanded-hooks-h3-qualified-final --junitxml=/private/tmp/expanded-hooks-h3-qualified-final.xml
```

**514 passed in 88.47s**, zero skips. Seven hook/new test files pass full
Ruff/format; Console regression tests pass full Ruff. Changed Python parses,
TOML and whitespace pass. Shared Ruff debt is unchanged (models/runtime/service/
catalog/bridge/ConsoleRuntime: 3/20/51/10/27/30); the skill fixture improves 2 to 1.
Offline validation rejects external refs without retrieval; base distribution
metadata declares the qualified jsonschema/referencing bounds.

Host carrier/copy/multimodal mixed-origin budgets are qualified directly. Native
Plugin context wrapper/graph composition remains the foundation task. H3 consumes
pinned sessions; H4 creates lifecycle producers. Existing ADR163/162/197 apply;
all four ACs self-reviewed. No full suite, live provider, graphical UI, Windows/
Linux execution or fresh complete dependency resolution is claimed.

## TASK-32679 — lifecycle boundaries and standalone v2 consent

Reused reviewed H4 checkpoint `32e336fe90`. Session initialization reserves a
reversible Console slot and publishes effects only after controlling requirements
succeed. Child admission narrows inherited tools/budgets; child settlement and
committed compaction feed the same checkpoint/context owner. Configuration and
workspace replacement occur at idle admission. Retired operation captures release
without reopening required failure gates; sealing respects the map/checkpoint
lock order.

Current-dev integration extends the existing HookPermissions owner to exact v2
identities, persistent grants, launch serialization and cached effect fences.
Legacy fingerprints/editor indices remain intact. Console review and canonical
Settings expose v2 definitions through the existing Advanced Config editor.
Malformed saved sources cannot masquerade as absence. Exact durable parent IDs
are published in the common store hydration path, preserving current batched
version reads and recovery behavior without extra per-message database reads.

RED: lifecycle admission **1 failed, 1 passed**; v2 consent **2 failed, 1 passed**;
exact-parent normal/recovery publication **2 failed** against immutable pre-H3
source. Malformed-source absence control also failed before its admission fix.

Final core covering command:

```sh
.venv/bin/python -m pytest Tests/Agents/test_hooks_v2_post_checkpoints.py Tests/Agents/test_hooks_v2_tool_pipeline.py Tests/Agents/test_hooks_v2_child_events.py Tests/Chat/test_hooks_v2_lifecycle.py Tests/Chat/test_hooks_v2_compaction.py Tests/Chat/test_console_context_compaction.py Tests/Agents/test_hooks_v2_validation.py Tests/Agents/test_hook_permissions.py::test_backup_recognizes_but_never_imports_hook_permission_authority Tests/Chat/test_console_run_hooks_regressions.py -q --tb=short --show-capture=no --basetemp=/private/tmp/expanded-hooks-h4-final-owned --junitxml=/private/tmp/expanded-hooks-h4-final-owned.xml
```

**404 passed in 105.02s**, no skips/warnings. Consent/command/Settings/Console UI:
**150 passed in 134.16s**, no skips/warnings (overlapping tests counted separately).
A frozen exact H4 production export passed the existing real-SQLite **1,000-turn**
cleanup stress test: **1 passed in 536.17s**, with a realistic bounded timeout.
The last ownership-fixture change also passes H4's three feature files:
**36 passed in 45.77s**, separately recorded and overlapping.

Broader durable/runtime/design-token neighbors returned **172 passed, 3 failed,
1 deselected** (the 1,000-turn case ran separately). The bridge startup failure
is fixed and included in the final 404-case pass. The two unchanged round-1
recovery tests fail identically against immutable pre-H3 source: issued-generation
rollback and provider-entry recovery. They are recorded baseline defects, not
passing qualification or silently deselected successes. Continuation work must
assess their uncertain-dispatch implications. Profile-switch failures were
reproduced against the baseline and repaired with the existing selected private
profile fixture. Native backup classification may report unsupported metadata;
its restrictive nonportable status must not import device-local hook grants.

Combined runs exposed descriptor growth, which per-test GC did not remove.
The new fixtures now dispose their runtime and direct agent calls use the existing
production worker guard to close newly owned worker-thread SQLite caches. No
production recovery/cleanup policy or warning threshold changed. Tiny artificial
startup windows are replaced with actual-entry barriers and bounded cold-start
allowances. The final combined run emits no descriptor warning.

Twenty-two authored/owned Python files pass full Ruff and formatting; all changed
Python parses and whitespace passes. Shared Ruff counts/codes are unchanged:
agent runtime/service 20/51; Controller/Store 205/118; Compaction/ConsoleRuntime
12/30; existing compaction/acceptance/round1/round2 tests 2/3/2/1. Design-token
neighbors execute successfully. ADR163/197 and the specification document the
current-dev integration; all five ACs self-reviewed. No new dependency, permission
store/schema, full sweep, live provider, graphical browser or Linux/Windows
execution qualification is claimed. Plugin-dependent initialization remains
qualified with its foundation/MCP tasks.

## TASK-32668 — native package inspection prerequisite

Reused reviewed F1 checkpoint `9dbc6ceb50`, including the subsequently reviewed
portable MCP syntax repair `0a69399424`. Immutable native inventory and bounded
descriptor-based capture/materialization remain separate from runtime activation.
The existing path/input validation and installed PyYAML/Pydantic are reused; no
new dependency, schema owner or execution entry is added.

RED on the intended inspection file stopped at the missing `Plugins` package
import, before any package execution. The complete targeted qualification:

```sh
.venv/bin/python -m pytest Tests/Plugins/test_native_inspection.py Tests/Plugins/test_package_files.py -q --tb=short --show-capture=no --basetemp=/private/tmp/expanded-hooks-f1-core --junitxml=/private/tmp/expanded-hooks-f1-core.xml
```

**92 passed in 1.04s**, no skips or warnings. Fourteen new Python files pass
full Ruff and formatting. Actual files, FIFO/link boundaries, substitution during
destination writes, executable bits, byte/depth/count limits, malformed recognized
constraints, independent successful siblings and immutable record defaults are
exercised. macOS collision tests report lexical controls when APFS itself prevents
a two-entry fixture. Windows capture fails closed; Linux/Windows/vendor execution
is not qualified. Fixtures are original AGPL content with pinned reference
provenance. Existing ADR162/163 apply; all four ACs self-reviewed.

## Remaining requested order

- TASK-32669 through TASK-32675: required plugin registry, authority and drain foundations.
- TASK-32680: Stop continuations and teardown, including native plugin drain qualification.
- TASK-32685: MCP hooks, including required MCP/plugin prerequisites.
- TASK-32686: native capabilities, required by the requested TASK-32687.
- TASK-32687: Cursor/Codex package and hook adapter qualification.
