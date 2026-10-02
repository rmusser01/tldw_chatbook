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

## TASK-32669 — private plugin registry and runtime owner

Reused reviewed F2 checkpoint `17f2325a77`. The private schema v1, exact reopening,
transaction authorizer and stable OS lock retain uncertain process evidence; no
PID-based termination or grant inference is added. Current-dev inventory entries
remain intact, with C93/C94 and one registered `plugins.registry` policy. SQL is
selected by existing package data, with no dependency additions.

RED: **33 failed** at the missing registry/runtime-owner modules. Final registry,
real process and repaired inventory disposition control: **34 passed in 9.28s**,
no skips or warnings (`/private/tmp/expanded-hooks-f2-qualified.xml`). F1 and full
private-inventory neighbors: **145 passed, 1 failed in 82.63s**; the single failure
was new documentation disposition wording, corrected and included in the final
pass. No production inventory assertion was relaxed. Child controls use the
repository checkout/profile/network-refusal bootstrap; the fork control runs in
a fresh single-threaded child with deprecation warnings treated as errors.

A wheel built without isolation or installation contains the exact packaged v1
SQL bytes (`/private/tmp/expanded-hooks-f2-wheel`). Four task-owned Python files
pass full Ruff/formatting; shared SQLite/inventory lint codes and messages match
HEAD. Changed Python syntax and whitespace pass. Existing ADR162/163 apply;
all four ACs self-reviewed. Ownership remains qualified only on 64-bit macOS local
APFS; unknown/network/synchronized storage fails closed. No full sweep or
Linux/Windows execution qualification is claimed.

## TASK-32670 — authenticated complete plugin authority

Reused reviewed F3 checkpoint `a259e61459`. Closed complete logical authority,
exact generation/operation/digest markers and disjoint plugin purpose keys reuse
the existing KDF, AES-GCM and private publication helpers. Prepared intent cannot
substitute for a separate commit certificate; standalone skill trust stays intact.
Plugin schema v2 upgrades exact v1 in one owned transaction and retains explicit
review state and independent uninstall tombstones. ADR009/162/163 apply.

RED: **10 failed** for missing authority modules/keys. Real crypto, every-field
tamper, marker/posture/reset/durability/migration and owner controls:
**172 passed in 47.10s**, no skips/warnings (`/private/tmp/expanded-hooks-f3-core.xml`).
Standalone skill trust and protected-path neighbors: **92 passed in 56.29s**,
no skips/warnings (`/private/tmp/expanded-hooks-f3-neighbors-qualified.xml`). The
initial protected-path run hit 21 stale config-binding failures; one reproduced
against frozen pre-feature H4 production. The existing selected-profile fixture
now keeps admission current, and the test-owned collision directory is removed
in finally. No production path/config guard or permission expectation changed.
The independent crypto child uses exact checkout/private-profile/network refusal.

Six task-owned Python files pass full Ruff/formatting. Shared crypto/protected-path
lint codes/messages remain unchanged (2/1 respectively); changed Python syntax
and whitespace pass. All four ACs self-reviewed. Local macOS/APFS durability
controls and an isolated marker backend qualify behavior; no real-keyring,
Linux/Windows, network filesystem, power-loss or full-sweep claim is made.

## TASK-32671 — reviewed installation commit and recovery

Reused reviewed F4 checkpoint `27514eea01`. The persistent storage worker owns
review, immutable materialization, guarded SQLite commit and subsequent certificate,
marker and projection publication. Recovery authenticates exact lineage and retained
bytes; it preserves unrelated installations and unresolved runtime evidence instead
of inferring grants or stopped processes. Protected transition discovery is bounded
and never prunes the current recovery material. Existing ADR162/163 apply.

RED: the intended missing recovery behavior failed and 15 stack entries reported
the missing coordinator. Final full commit/recovery files: **51 passed in 55.26s**,
no skips/warnings (`/private/tmp/expanded-hooks-f4-qualified.xml`). Real owner death
at all six milestones, fresh-process missing/rolled-back registry, ambiguous lineage,
missing certificate, stale review, full disk and same-ID retry are exercised. Crash
workers now use Python isolation and the repository network-refusal/null-keyring
bootstrap before app imports, with explicit private HOME/config/profile provenance.
The six crash boundaries also passed separately (**6 in 10.11s**, overlapping).

Authority/registry/runtime-owner neighbors: **175 passed in 34.07s**, no
skips/warnings (`/private/tmp/expanded-hooks-f4-neighbors.xml`). Ten task-owned
Python files pass full Ruff/formatting; changed syntax and whitespace pass.
All four ACs self-reviewed. Local macOS/APFS and isolated marker storage are
qualified; no full sweep, real keychain, hardware power-loss, Linux/Windows or
network/synchronized storage qualification is claimed.

## Remaining requested order

- TASK-32672 through TASK-32675: required plugin registry, authority and drain foundations.
- TASK-32680: Stop continuations and teardown, including native plugin drain qualification.
- TASK-32685: MCP hooks, including required MCP/plugin prerequisites.
- TASK-32686: native capabilities, required by the requested TASK-32687.
- TASK-32687: Cursor/Codex package and hook adapter qualification.


## F5 — TASK-32672 native Console skills

Integrated the reviewed F5 increment from `ae19daa8be` onto current dev, retaining
H3 mixed hook/plugin context carriers, H4 exact consent/lifecycle ownership, builtin
skill handling, actual approval provenance, and current service-wiring/lifecycle
modules. Schema v3 adds stable aliases without changing pre-alias snapshot bytes.
Installed packages require separate review and scoped activation; actual Console
manual/model/fork/file flows use the existing skill/tool/approval owners.

Additional behavioral REDs proved the live authority worker selected the wrong
protected directory and both native workspace and Console chain offloads leaked
worker-held caches. Fixed these at shared ownership entries with existing helpers.
Removed a conditional import shadowing the context carrier on continuation resume;
updated stale test catalog doubles and selected-profile markers, preserving the
production recovery guard and file-descriptor sentinel.

Qualification:
- Native admission, actual Console sends/agents, formatting/continuations, resolver,
  provider custody and schema migration: **193 passed, 227.66s**, no warnings/skips;
  `/private/tmp/hooks-f5-qualified.xml`.
- Existing standalone Skills services: **51 passed, 12.02s**, no warnings/skips;
  `/private/tmp/hooks-f5-skills-controls.xml`.
- Hook lifecycle/child/post-checkpoint controls: **51 passed** in the integration
  probe, plus all **9 automatic-work lineage checks passed, 24.47s** after correcting
  their selected-profile fixture; `/private/tmp/hooks-f5-hooks-controls.xml` and
  `/private/tmp/hooks-f5-lineage-qualified.xml`. The first probe's eight lineage
  failures were profile-admission refusals; every one executes in the final nine.
- Owned plugin Python: full Ruff and formatter pass. Shared-file diagnostics were
  compared against HEAD; no new lint findings. Changed Python parses and whitespace
  checks pass. Baseline shared formatting/lint debt was not rewritten.

Limits: real local macOS/APFS storage, crypto, native threads, controlled child
processes, actual Console/AgentService composition; provider transport and marker
backends are isolated doubles. No external provider/keychain, Windows/Linux, full
suite, power-loss or plugin MCP/native capability qualification is claimed here.


## F6 — TASK-32673 scoped stop and uninstall

Integrated the reviewed `a9f4efd792` increment onto current native owners. The
shared live fence invalidates only the requested scope and starts retained exact
cancellation before trust/storage access. Durable persistence, confirmed runtime
stop and pending cleanup have separate receipts; surviving or unknown processes
keep their leases. Uninstall commits tombstones and installation-owned removals
before descriptor-relative package cleanup, retaining data and independent
credentials. Fresh re-enable does not restore old approvals or callbacks.

The first real A/B activation and controlled-child test reached the missing
revocation module after successful preconditions (RED). All **45 scoped
revocation/persistence checks passed in 155.21s**, no warnings/skips;
`/private/tmp/hooks-f6-core.xml`. This includes actual Console provider custody,
stalled/failed durable boundaries, exact original retries, replacement-parent
refusal and live surviving children. Neighbor admission, native Console skills, coordinator/recovery and automatic-work lineage: **123 passed in 248.90s**, no warnings/skips; `/private/tmp/hooks-f6-neighbors.xml`.

All 37 plugin Python files pass full Ruff and formatter checks. Changed Python
parses and whitespace checks pass; the shared Console controller retains its
205 pre-existing lint findings with no additions. ACs were self-reviewed against
the real service and storage flows. ADR-162/163 govern these existing boundaries.

Limits: local macOS/APFS, real SQLite/crypto/threads and controlled children; marker
and provider backends are isolated doubles. No full suite, external provider,
keychain, Linux/Windows or hardware power-loss qualification. Cross-session issued
request lookup and rolling history retention remain F7.

F5 packaging follow-up: the built wheel
`/private/tmp/hooks-f5-wheel/tldw_chatbook-0.2.2-py3-none-any.whl` was inspected;
all three packaged registry migration resources exactly match source. No editable
environment was rebound or reinstalled.


## F7 — TASK-32674 revision drains, retention and managed resume

Integrated the reviewed `e0f0327200` increment while retaining current child
definition limits, hook consent/context carriers, exact worker custody and
continuation transport. Updates fence fresh admission, retain actual run and
process blockers, and distinguish proposal cancellation from cancelling work.
Rollback is a fresh current-policy review; mutable data and external effects
are not restored. Immutable revisions and authenticated rolling history retain
live/recovery material. Issued retries keep their original target and phase.
Managed continuation pins constrain fresh admission through existing Console,
fleet and archive owners; historical permissions never become new authority.

The first real activation/child and retention tests failed at the missing new
entries, then both passed. Core revision, retention, managed continuation and
revocation checks: **105 passed, 601.64s**, no warnings/skips;
`/private/tmp/hooks-f7-core.xml`. Neighbor authority, admission, coordinator,
recovery and provider codec: **319 passed** in the first probe; fleet coordinator
and native Console skills: **78 passed** in the native probe.

Existing fleet/Console/archive fixtures initially refused admission because they
selected a profile different from the runtime. Selected-profile fixture markers
resolved these refusals without changing production guards. The first corrected
125-case run then exposed unclosed real AgentRunsDB and archive-fixture databases.
Yield/finally and existing content-operation ownership close those actual handles.
Final fleet/Console/archive controls: **125 passed, 97.21s**, no warnings/skips;
`/private/tmp/hooks-f7-profile-qualified.xml`. Affected direct archive transport
controls: **2 passed, 8.23s**, no warnings/skips;
`/private/tmp/hooks-f7-owned-transport.xml`.

All 43 owned plugin Python files pass Ruff and formatter checks. Changed Python
parses and whitespace checks pass; shared diagnostics were compared against HEAD
with no additions. All ACs were reviewed against real service/storage flows.
ADR-162/163/063 govern these accepted boundaries.

Limits: local macOS/APFS, real SQLite/crypto/native threads and controlled children;
provider and marker backends are isolated doubles. No full suite, external
provider/keychain, Windows/Linux, hardware power loss or future MCP/root-cleanup
qualification. Native Stop continuation qualification remains H5.


## F8 — TASK-32675 exact-root custody and cleanup

Integrated the reviewed `e1af542e8a` increment onto current authority/runtime
owners. Registry v4 records original root grants for pending, active, reader
and idle-process lifetimes. Native Darwin boot and precise directory birth
identity bind every destructive step. A separate protected clean/dirty runtime
checkpoint refuses unproven restart quiescence, including coherent SQLite
rollback. Reviewed root creation, deletion, attachment and reconciliation retain
original identity, deadline and destructive phase. Successful cleanup advances
generation; uncertain writers and partial cleanup retain fences and receipts.
Shutdown seals ordinary admissions before final clean publication while allowing
actual terminal settlement after refused close. Managed resume checks actual
root membership/generation through existing F7 pins.

RED reached the missing root-creation review entry after a successful real
installation. The idle writer and cross-process native identity then passed.
Final core cleanup and surviving-process recovery: **55 passed, 289.47s**,
no warnings/skips; `/private/tmp/hooks-f8-core.xml`. Registry migrations, runtime
ownership, authority, managed continuation, revocation, update drains and retention:
**250 passed, 513.89s**, no warnings/skips; `/private/tmp/hooks-f8-neighbors.xml`.
Children use the existing isolated-checkout/profile/null-keyring/network-refusal
bootstrap; the existing fresh-process real fork guard remains intact.

All 46 owned plugin Python files pass Ruff/formatter; changed Python parses and
whitespace checks pass. The uninstalled wheel
`/private/tmp/hooks-f8-wheel/tldw_chatbook-0.2.2-py3-none-any.whl` contains all
four registry migrations with exact source bytes. All ACs were self-reviewed
against the live service/storage flow. ADR-162/163 govern these existing contracts.

Limits: local macOS/APFS with actual boot/native fields, SQLite, threads and
controlled surviving children; marker/provider doubles are isolated. Boot-change
cases simulate a changed OS value and do not establish real reboot/power-loss
behavior. No arbitrary external-writer containment, Windows/Linux, external
provider/keychain or full-suite claim. Hook/MCP actual grant producers land in
H6/M4; F8 qualifies their shared ownership seam.


## H5 — TASK-32680 bounded Stop continuation and teardown

Integrated reviewed `fb9bae2e7a` through the existing queue, actual accepted
parent turn, controller and runtime. Whole proposals combine in stable order
into one machine turn. Three-turn/120-second ceilings and inherited real agent
budgets remain enforced. A one-use live gate and v74 SQLite receipt share the
existing durable acceptance transaction; consumption never claims commit proof.
Foreground input, maintenance, revocation/drain, veto, closure and uncertainty
refuse stale work. Machine input retains its untrusted carrier, never becomes
human history or fires UserPromptSubmit, and never replays automatically.
Stop synchronously seals admissions and signals the retained cancellation owner.
Interrupt/SessionEnd have fixed observation windows and cannot delay host cleanup.

Real maintenance admission initially failed two cases; the shared current gate
now checks the existing maintenance pause through acceptance. Fresh worker
handles exposed by archive reads, hook configuration, run-log selection and
fleet history discovery now use existing operation-owned connection retirement.
Mounted harness teardown closes the real backing app and its replaced database.
No warning suppression, descriptor threshold increase or production timeout
increase was used. Mounted fixture waits allow cold parent startup separately
from the measured immediate Stop action; the full chain cap is independently
qualified by the real scheduler controls.

Final scheduler, mounted/viewless teardown, legacy hooks and migration: **105
passed, 453.52s**, no warnings/skips; `/private/tmp/hooks-h5-qualified.xml`.
Affected queue/dispatch/maintenance controls: **133 passed, 61.64s**, no warnings;
`/private/tmp/hooks-h5-neighbors-final.xml`.
Archive, fleet and log-reader neighbors: **85 passed, two baseline failures,
133.14s**; `/private/tmp/hooks-h5-owner-neighbors.xml`. The same fleet snapshot
identity and old log-root assumptions fail on frozen pre-H5 production:
`/private/tmp/hooks-h5-fleet-prior.xml`. Existing dispatch fixture/settlement and
Stop record failures also reproduce there; see `hooks-h5-prior-dispatch*.xml`
and `hooks-h5-prior-stop-ui.xml` under `/private/tmp`. These are baseline
failures, not passing checks or grounds to weaken ownership guards.

The 1,000-real-SQLite-turn retention benchmark passed in **465.05s** without
warnings; `/private/tmp/hooks-h5-1000-turns.xml`.
The uninstalled wheel `/private/tmp/hooks-h5-wheel/tldw_chatbook-0.2.2-py3-none-any.whl`
contains the exact v74 migration bytes. All 16 hooks/new qualification Python
files pass Ruff and formatter checks. Design-token governance: **8 passed, 17.27s**, no warnings;
`/private/tmp/hooks-h5-governance.xml`. All H5 changed Python parses, whitespace
checks pass, and shared-file Ruff diagnostics have no additions against HEAD.
All ACs were self-reviewed against real scheduler/storage owners. ADR-162/163/063
apply; no new scheduler, permission owner or recursive model dispatch was added.

Limits: local macOS, real SQLite/native threads/controlled processes and mounted
Textual harness; provider and selected hook authority fixtures are isolated.
No full-suite, external provider/keychain, Windows/Linux, real reboot/power-loss
or future MCP hook qualification claim.


## M1 — TASK-32681 complete typed MCP results

Integrated reviewed `74462424ac` through the existing stdio client, local and
unified services and MCP provider. Strict typed results retain all content,
structured content, error flags and metadata before explicit legacy projection.
Private exact wire-result spans, duplicate-key/depth/size validation and host
write/settlement observations cannot be supplied by remote fields or retained
by changed/copy models. Error results and transport failures use fixed body-free
diagnostics. One per-call publication claim coordinates service/provider audit
I/O; the existing bridge has one bounded best-effort writer and preserves
uncertainty after timeout or unacknowledged cancellation.

Preserved current producer/recovery decorators, same-task service deadlines and
native failed-request settlement. Current custody retires a timed-out stdio
child before releasing admission; fixtures explicitly reconnect rather than
assume that child is reusable. Constructor-bypassing fixtures now initialize
the actual lifetime owner. Audit race controls reach the actual append with an
event before asserting the race. Native wire instrumentation passes the actual
new dispatch observation parameter, and checks fixed diagnostics without remote
text. Direct positive/refusal controls qualify all three new typed entries
against real producer closure and storage pause before any tool write.

Actual stdio RED retained the successful legacy control, then failed at dropped
structured fields. Final complete-result, framing/catalog, client/provider,
control-plane, fixed error projection and native child/recovery qualification:
**453 passed, 172.38s**, no warnings/skips;
`/private/tmp/hooks-m1-qualified.xml`. Twenty-one deselections comprise four
known local-hub deadline cases and the existing 17-case inactive group; 16 of
that group already passed in the affected-owner probe. Its one inspection
failure and ten untouched durable-source fixture failures reproduce on frozen
pre-M1 production: `/private/tmp/hooks-m1-backup-prior.xml`. The four short local
hub deadline failures reproduce there as well:
`/private/tmp/hooks-m1-local-hub-prior.xml`. Baselines remain failures; guards
and production deadlines were preserved.

Both new Python files pass Ruff/formatter; all changed Python parses, whitespace
checks pass and shared Ruff diagnostics have no additions against HEAD. ACs
were self-reviewed against actual client/service/provider flow. ADR-162/163
apply; no new transport, permission store, schema or dependency was added.

Limits: controlled stdlib peers, actual local processes/threads and production
client/control-plane code; no third-party plugin execution, external MCP server,
HTTP transport, full-suite or cross-platform qualification claim. M2 owns direct
HTTP and protocol profiles; H6 owns MCP hook effect normalization.


## M2 — TASK-32682 qualified direct transports

Integrated reviewed `68ebe2accb` below the current client/store/control-plane
owners. Explicit profiles qualify stdio and Streamable HTTP for 2026-07-28,
2025-11-25 and 2025-03-26. Modern per-request metadata/routing and legacy
initialize/session/notification flow share bounded strict M1 results. Discovery
never invokes tools; JSON/SSE, bounded legacy GET resumption, pagination,
unsupported capabilities and versions produce explicit readiness diagnostics.
HTTPS or explicit numeric loopback development origins only, no redirects,
proxy inheritance, TLS weakening or invocation replay. Recognized legacy profile
stores migrate to schema 2 using the existing protected atomic writer; malformed
state is retained. ADR-162 reciprocally partially supersedes ADR-111 for the
accepted direct generic transport boundary; unrelated server/OAuth rules remain.

Preserved current producer/recovery/source decorators and actual stdio timeout
settlement. HTTP raw request/notify use those same guards. Maintenance qualifies
only concrete subprocess or HTTP client owners, retains accepted call chains and
one pool cleanup task, and requires positive native exit or successful actual
pool closure before resume. A closed HTTPX client bit or empty session map is
not proof. Failed lower closure stays fenced; stalled closure can rejoin its
original task. Ordinary resume requires fresh explicit discovery, never replay.

Actual direct-entry RED retained the successful M1 stdio control; four added
native HTTP maintenance controls failed before the current-owner adapter.
Final direct transport/profile/schema/control-plane suite: **180 passed, 42.49s**,
no warnings/skips; `/private/tmp/hooks-m2-qualified.xml`. Typed-result/catalog/tool
neighbors: **319 passed, 73.86s**, no warnings/skips, four documented pre-M1 local
hub deadline baselines deselected; `/private/tmp/hooks-m2-m1-neighbors.xml`.
Actual native stdio recovery/maintenance/unknown-owner/pending-child and upper
service settlement controls: **12 passed, 45.32s**, no warnings/skips;
`/private/tmp/hooks-m2-native-final.xml`. The controlled upper peer now advertises
its implemented capabilities and forwards the real dispatch observation rather
than assuming unadvertised catalogs are executable. Five new HTTP controls also
verify raw storage-pause refusal before wire, all three maintenance profiles,
retained stalled cleanup and failed lower closure. New Python passes Ruff and
formatter; all changed Python parses, whitespace passes, and shared Ruff has no
additions against pre-M2 HEAD. No dependency or second permission owner added.

Limits: controlled repository-owned stdio and loopback HTTP interoperability
fixtures on local macOS, not vendor certification. Official MCP versioned transport
sources were checked; earlier mcp-unified source inspection remains historical
bounded evidence, not a fresh vendor-release claim. No external server/provider,
full-suite, cross-platform, generic OAuth, MRTR, subscriptions or standalone
push-listener qualification. M3 owns credential mapping and H6 MCP hook effects.


## M3 — TASK-32683 stable credential bindings

Integrated reviewed `406371452a` through existing config/keyring safety, local
MCP profiles/transport and authenticated plugin recovery. Frozen metadata separates
reviewed reference/generation/principal/issuer/audience/origin/scopes from current
token bytes, expiry and storage revision. Verified unchanged renewal retains
authority; changed/unknown identity, opaque replacement and revocation advance
it. Missing references require a fresh UUID, tombstones never wrap generations.
Secure keyring-only, data-root-scoped records use existing portalocker/safety
checks; no plaintext or new dependency. Production has no generic MCP OAuth
adapter and reports `unsupported_authentication`, never imported vendor grants.

HTTP resolves current usable headers at dispatch for the exact selected origin.
Reserved headers, case collisions and unsupported octets refuse. A bounded single
retained credential worker keeps the event loop responsive without queued jobs,
secret caching, late HTTP dispatch or replay after cancellation. Proven worker
start failure now releases unowned capacity and returns a fixed body-free error;
the same real transport succeeds after retry. Profile schema 3 persists only
reference/generation, using the current protected writer and preserving malformed
source bytes. Recovery validates every complete mapping through the actual owner.
Current producer/source decorators and native HTTP/stdio cleanup were preserved.

Saved-profile RED failed at discarded credential reference with M2 discovery
passing. Final binding/current transport/sentinel/renewal/expiry/migration and
protected recovery suite: **124 passed, 38.68s**, no warnings/skips;
`/private/tmp/hooks-m3-qualified.xml`. Direct transport/store/lifecycle and actual
plugin coordinator/recovery neighbors: **234 passed, 121.18s**, no warnings/skips;
`/private/tmp/hooks-m3-neighbors.xml`. New Python passes Ruff/formatter; all changed
Python parses, whitespace passes and shared Ruff has no additions against pre-M3
HEAD. All ACs self-reviewed; ADR-162/163 apply. Modified client/local store/service,
HTTP, config factory, coordinator/recovery and added credential owner/tests/docs.

Limits: memory credential backends, real portalocker with fake keyring API,
controlled local peers and actual protected recovery on macOS. No real OS
keychain/vendor OAuth, external MCP server, full-suite or cross-platform claim.
M3 mapping fixtures qualify capture/reconstruction, not M4 publication/launch.


## M4 — TASK-32684 scoped owned MCP tools

Integrated reviewed M4 increment `406371452a..d98450ee01` into the current native MCP/source/root owners. Configuration is data-only schema 4; exact connection and discovered-tool mappings publish through existing authenticated review/commit. Owned tools use the existing registry, normal permission/persona/profile/parent ceilings, audit and M1 typed results. Scoped reuse requires equal authority plus explicit host request-independent qualification. Cancellation detaches A, preserves authorized B, and retains unknown requests/idle writer/root custody until positive original terminal or actual local teardown evidence. Changed/malformed bindings remain unavailable per component and immutable capture ceiling; authenticated recovery still validates the whole mapping set.

Current-owner adapters retain producer/source guards and native child/HTTP-pool proof. Actual retained launch/request tasks reacquire through the existing MCP worker-isolation seam. Direct profile admission refusal is false; owned launch requires actual true/session custody. The subclass preserves H3's unanswered approval metadata. Portable expansion recognizes only PLUGIN_ROOT/PLUGIN_DATA once in approved fields, preserves unknown literal text, and requires the explicit persistent data binding for stdio. The exact native shared-request adapter shields the original future and retains source/producer custody after its deadline while B remains attached; ordinary/separate/last-owner cleanup still kills and reaps. No replay, fabricated terminal proof or restoration of A acceptance occurs.

RED: actual scoped setup refused copied task admission before the adapter. Three normal detach/separate/permission controls passed before the portable-literal refusal (`hooks-m4-native-contract-red.xml`). The corrected real shared deadline control then proved B's child was killed (`hooks-m4-shared-deadline-final-red.xml`). Earlier mechanical merge/setup and serialized-provider fixture failures were not behavioral qualification.

Targeted covering evidence: the full M4/connection/normal provider/profile/credential run had 286 passes, zero skips/warnings and one five-second deletion-observer timeout (`/private/tmp/hooks-m4-qualified.xml`, 950.15 s). The original deletion runs three authenticated root phases after real exit; the observer now shields that same operation under a separate finite bound. Exact root removal, child exit and both latest native edge cases passed: 3 tests, 39.72 s, no warnings/skips (`/private/tmp/hooks-m4-final-controls.xml`). This completes the same 287-node covering set; reruns are not additive counts. All 144 stdio client regressions passed (`hooks-m4-stdio-client.xml`); after consuming late future exceptions, 21 affected cancellation/timeout/reap controls passed (`hooks-m4-stdio-settlement-final.xml`). Eight actual MCP/source/upper-route/maintenance controls passed after the final native change (`hooks-m4-native-final.xml`, 24.56 s).

HTTP/credential/native neighbors: 125 passed plus one short cancellation-control race (`hooks-m4-native-neighbors.xml`). The same cancellation test failed on exact frozen committed M3 production (`hooks-m4-auth-m3-baseline.xml`). The control now cancels its positively started backend call before unrelated peer I/O; actual timeout/no-late-wire controls keep 30 ms. All 14 auth tests passed (`hooks-m4-auth-final.xml`, 4.82 s). Production credential timeout/cancellation behavior was not changed.

Four new Python files pass Ruff and formatter. Changed Python parses, shared Ruff diagnostics have no additions, and git diff whitespace checks pass. ADR-162/163 and operation/spec/plan notes updated; the timing incident is recorded in the testing-evidence lesson.

Limits: actual macOS/POSIX native children, SQLite/crypto/protected source owners and controlled loopback HTTP peers. No external-server session-isolation, OS keychain/OAuth, foreign host, cross-platform or full-suite claim. These qualify the owned service/catalog API; native managed graph/application composition is I1 and absent declarations still refuse. No new transport/permission/lifecycle runtime or automatic MCP startup was added.


## H6 — TASK-32685 normal MCP hook invocation

Integrated reviewed H6 increment `d98450ee01..cdeb1687b6` through the current
Console, AgentService, ToolHookRun, checkpoint, MCP and native custody owners.
Strict original-wire results are error-first, bounded by the entire original
payload, and accept only the declared structured/single-text/exact-mirror/empty
forms. Request-local restrictions narrow the normal invocation; provisional
initialization grants no accepted root or extra permission. Recursion, original
approval provenance, nested context staging, required postevents, dependency
readiness, observation pools and bounded teardown use the existing owners.
Unknown managed graph requirements refuse until I1 supplies the native graph.

The current H3 approval-only dataclass projection initially discarded exact
capture identity: six real approved/session paths failed. The provider's one
internal projection helper now transfers only the exact captured object's
witness, under the same capture lock. Arbitrary replacements and late owned
refusals do not acquire it. Regressions cover all six approval projections.

Actual Console qualification exposed repeated full-catalog I/O in lifecycle
currentness. The probe now compares live workspace/project/profile/persona
fields; the normal MCP owner uses its existing exact live catalog resolver at
hook dispatch and result acceptance. Vanished, disconnected, stale or changed
definitions refuse and record a normal metadata-only audit. Permission and
kill-switch checks remain fresh. Real Interrupt and SessionEnd peers prove
one/three-second clocks were retained; no production timeout was extended.

RED artifacts: original-wire missing-feature control, six approval projections
(`/private/tmp/hooks-h6-approval-red.xml`), and a valid actual Console probe
(`/private/tmp/hooks-h6-currentness-valid-red.xml`). The first command-only probe
had no MCP executor and was a harness error, not behavioral evidence. Temporary
frame observers were removed. A faulthandler diagnostic itself stalled inside
CPython's cancellation lock; its exact owned children/process were terminated
and that diagnostic is not application qualification.

Targeted native/Console/post/result covering set: 162 nodes, zero skips;
`/private/tmp/hooks-h6-final-native.xml` had 159 passes and three observer/audit
expectation failures. The two custody observers now allow their same finite
five-second configured handler budget plus the existing five-second settlement
wait, and shield the original owner. Cancellation/revocation/deadline retain
that exact ticket and request/validator until actual completion: all four
controls passed, 83.21 s (`hooks-h6-custody-final.xml`). The definition refusal
now has its normal audit row and all MCP execution/result tests pass in the
final boundary run below. These reruns complete the same covering set; counts
are not additive. Earlier narrow actual Console controls passed 13/13 in
31.40 s (`hooks-h6-live-boundaries.xml`).

Final hook results/execution, ordinary MCP provider, scheduler and typed-result
neighbors: **247 passed, 74.15 s**, zero skips (`hooks-h6-final-boundaries.xml`).
Earlier interrupt/budget/provider neighbors: 123 passed. Pytest exit reported
shared-temp garbage cleanup warnings from pre-existing unrelated test trees;
no deletion of those trees or global leak-free claim was made. Authored Python
passes Ruff/formatter; changed Python parses, whitespace passes, and shared
Ruff diagnostics have no additions against pre-H6 HEAD. All four ACs self-
reviewed; ADR-162/163 and operation/spec/plan notes updated.

Limits: actual local macOS native storage/source owners, owned child processes,
and repository-controlled stdio/loopback HTTP peers. No external vendor host,
real OS keychain/OAuth, GUI, full-suite or cross-platform qualification. Native
managed application composition remains I1; no parallel permission/runtime,
implicit connection or grant bypass was added.


## I1 — TASK-32686 native capabilities

Selected manual commands, always/manual rules, agent presets, owned hooks and MCP
now join the existing immutable skill/run snapshot. Review captures exact host
builtin/owned-MCP and model references without granting permission. Typed live
context keeps package text in the user lane and applies EMPTY constraints to
actual inline/fork/child paths. Normal tool discovery, approval, provider routes,
source currentness and dependency checkpoints remain the owners. Native hooks
reserve actual F2 process/root custody before H2 launch and retain it on revocation
until real process/pipe settlement.

Behavioral RED caught actual EMPTY-child MCP execution, impossible approval prompts,
wrong MCP-source resolution when native mappings sort first, missing reviewed
references and incomplete embedded namespaces. Corrected harness failures are not
behavioral RED: a stale combined review, wrong hook envelope, an unavailable runtime
accessor, the Console's user-role tool results, omitted progressive tool discovery,
and a readiness assertion that accidentally selected an unrelated fixture skill.

The final covering command ran the native component, native skill flow, Console
substitution, owned MCP and admission files plus the older agent-service file:
`/private/tmp/hooks-i1-final.xml`. All **164 integration cases** passed; the combined
agent cases hit source-selection recovery refusal after private-profile binding.
The older file now selects its own existing bootstrap profile for every test. Its
stale cancellation control requested cancellation before dispatch while expecting
an in-flight tool-specific error; it now uses actual start/release/finish events.
Separate fresh final runs returned **138 passed in 77.98s** for agent service
(`/private/tmp/hooks-i1-agent-final.xml`), **85 passed in 65.08s** for command/hooks
and Console lifecycle (`/private/tmp/hooks-i1-final-hook.xml`), and **3 passed in
12.66s** for latest mapped readiness, missing DATA and EMPTY-child approval controls
(`/private/tmp/hooks-i1-final-readiness.xml`). No platform/coroutine skips.

Authored files pass Ruff lint/format. All changed Python parses; shared files add no
Ruff diagnostics versus HEAD and whitespace checks pass. Final formatting cleanup
was verified to preserve the entire Python AST and comment sequence.

Qualification uses real private SQLite/APFS authority, bounded native child commands,
real controlled MCP stdio peers and actual Console/AgentService flows. Provider
responses and protected marker storage are controlled test owners. This does not
certify OS Keychain, OAuth, arbitrary remote isolation, original vendor hosts, full
GUI behavior, Windows/Linux, a full test suite, or distribution packaging. Skill model
overrides remain explicitly unsupported; agent model mappings retain the parent route.


## I2 — TASK-32687 pinned Cursor/Codex interpretation

Adapters `chatbook-openai/2026-10-01.1` and `chatbook-cursor/2026-10-01.1`
interpret captured bytes through existing native validation and authority owners.
Inline OpenAI overlays replace compatibility wholesale; portable identity and
locations remain canonical. Explicit Cursor paths/empty exclusions replace default
scans. Supported instruction subsets retain manual-only policy, unconditional rules
and EMPTY agent constraints. Unavailable variables/apps, conditional rules,
undocumented Codex presets and source hook contracts remain explicit blockers.
Retained catalog fields participate in executable/content identity; materialization
and recovery preserve the chosen dialect without a second store or trust owner.

Behavioral RED: absent OpenAI adapter/standalone inventory
(`/private/tmp/hooks-i2-codex-red.xml`), missing dialect materialization argument
(`/private/tmp/hooks-i2-dialect-red.xml`), and dropped root/group/handler guard scope
plus explicit missing MCP file (`/private/tmp/hooks-i2-controls-red.xml`). The last
run also had two corrected harness assertions: a wrong schema URL and the protected
revision field name. These are not behavioral RED. Earlier parser/Console harness
mistakes likewise do not count as qualification. Unavailable manual mentions are
ordinary literal user messages; expecting the entire message to refuse was wrong.

Final adapter/native-inspection/capture run: **117 passed in 5.77s**, no skips,
`/private/tmp/hooks-i2-final.xml`. It exercises actual reviewed Console command/rule
content and guarded non-expansion, plus protected review/commit and registry-loss
recovery after deleting the original source for both OpenAI and retained Cursor
catalog interpretations. Neighboring native components, native skill flow,
coordinator and recovery: **127 passed in 226.21s**, no skips,
`/private/tmp/hooks-i2-neighbors.xml`.

Fixture inventories are independently authored original AGPL test data. Provenance
pins OpenAI plugins `5fd93af4cd0c623e020d0cc7e9ce178b4ac1f70f` and Cursor template
`46216072ac5750f782f95bb325b4d12b7c3ae9c9`; licenses, official primary links and
dated unversioned documentation observations are in the interop README. No vendor
asset/script/skill prose is copied or executed. Parsing evidence is recorded as
parsed, separate from the actual Chatbook behavior exercised above.

Deliberate implementation adjustment: `normalize_vendor_hook` returns a data-only
qualification proposal, not the plan's prototype HookHandler. No foreign mapping
qualifies the complete payload/cwd/timing/output/timeout contract, so **all foreign
hooks remain unsupported**. Required/unknown guards fence affected package material;
known optional observers remain visible without fencing unrelated material. Native
v2 hooks keep their existing qualified runtime. No original-host comparison,
Windows/Linux, real Keychain/OAuth, GUI, full-suite or distribution certification
is claimed. I3+ acquisition/marketplace/UI tasks are outside this requested work.

Authored final run after type/format cleanup: **25 passed in 6.11s**,
`/private/tmp/hooks-i2-authored-final.xml`. Two exact interpretation controls pass
after the final lint-only set/tuple corrections (`hooks-i2-lint-final.xml`). All
changed Python/fixture JSON parses, all nine new Python files pass Ruff lint/format,
changed ranges are formatted, shared Python adds no Ruff diagnostics versus I1 HEAD,
and `git diff --check` passes. No dependency or copied-source licensing change.


## Final latest-dev integration — 2026-10-01

TASK-32668 through TASK-32687 are Done with checked ACs and implementation notes.
Rebased their twenty ordered commits onto dev `ef831d9f383f58a806fe54d61a6fd75678ec73c0`.
The two conflicts preserve both final hook dispatch validation and ephemeral tool
output, and both typed MCP result/dispatch custody and request-local progress.
`git range-diff` confirms the remaining changes are clean replay/context changes.
The existing progress request stub now accepts the actual dispatch argument;
controlled real stdio progress also proves complete original typed wire evidence
and settled dispatch. The earlier stub timeout is harness failure, not production
qualification, and no production boundary was loosened.

Fresh post-rebase evidence:

- `hooks-rebase-tools.xml`: 126 passed / 1 stale request-stub failure, 47.85s;
  typed result, hook dispatch and output cases passed. Corrected progress file:
  `hooks-rebase-progress-final.xml`, 3 passed / 0 skips, 0.75s. Together every
  one of the original 127 selected cases passes; the combined failure is retained.
- `hooks-rebase-console.xml`: 82 passed, one empty-parameter upstream architecture
  case skipped, 94.15s. Actual controller guard, provider tool schemas, Console
  rejection/denial and tab-close cases execute and pass. Empty parameter sets are
  not qualification evidence.
- `hooks-rebase-native.xml`: 98 passed / 0 skips, 221.49s; actual native Console
  components/skill flow and all pinned source adapters on the latest base.
- `hooks-rebase-governance.xml`: 11 passed / 0 skips, 14.96s; design-token and
  hooks dependency governance. CSS rebuild produces no tracked difference; the
  119-file UI gate census is intact.

All 186 branch-changed Python files parse, whole-branch/working whitespace checks
pass, and the two conflict owners plus changed progress test add no Ruff diagnostics
against the predecessor/latest-dev baselines. Targeted checks only: this remains a
draft for review, without a full-suite, original-host or cross-platform claim.


## PR #2946 CI repair and current-dev rebase — 2026-10-01

Rebased all 21 PR commits onto dev
`83c2c9810d5d09406c85e1f027541b846ead93aa` before repairing the failed
fast-lane and derived-artifact checks. Kept both branches' evidence lessons,
reused dev's exact `ConsoleDurableTurnCommit.user_parent_message_id` instead of
a duplicate optional field, and preserved both pending-hook Stop and dev's
queue-disabled reason/Redirect reservation and focus behavior.

The published CI run `36848847083` passed its 1,183-case main group (plus one
empty parameterization), but its admission group failed two stale controller
fixtures. Both omitted the now-required `bind_turn_request` coordinator seam;
one then waited forever for a task that had already failed. The fixtures now
model the existing no-active-chain coordinator contract, and the startup wait
is bounded. Production binding remains required. Local paired reproduction:
`hooks-pr2946-runtime-red.xml`, two failures; corrected cases plus actual continuation
migration/index checks: `hooks-pr2946-fixes.xml`, **7 passed in 4.225s**.

The exact admission-sensitive CI selection now passes **123 cases in 137.445s**,
with the existing TASK-32873 captured-attach fixture xfail retained and excluded
from qualification (`hooks-pr2946-admission.xml`). Mounted composer Stop,
Enter/collapse, Redirect resize, focus typing, queue reason, fresh saved-fork
lineage and pending-hook cancellation controls: **15 passed in 266.384s**, no
skips (`hooks-pr2946-rebase-ui.xml`).

Owner coverage exposed a further production interaction with dev's denial-reason
projection: an approved, settled MCP result was copied even when no denial text
was appended, breaking the original owner's exact typed-evidence identity.
`hooks-pr2946-owners.xml` retains **178 passes / one failure**; the isolated
approval case also failed (`hooks-pr2946-approval-isolated.xml`). The shared MCP
owner now returns its original result when the error text is unchanged. Denial
text still uses its existing bounded projection. No permission, definition,
raw-result, currentness or evidence check was relaxed. Connected initializers
(all three branches), normal MCP provider and strict typed hook invocation:
**140 passed in 31.766s**, no skips (`hooks-pr2946-mcp-fixed.xml`).
Fresh covering hook pipeline/lifecycle and native inspection/coordinator/Cursor/
Codex owner run: **179 passed in 96.365s**, no skips or failures
(`hooks-pr2946-owners-final.xml`).

Derived-artifact repairs follow existing contracts:

- ADR-173's shared `utc_now_iso()` replaces six new offset-format writers in
  lifecycle events, tool events, inspection evidence and registry receipts;
  monotonic budgets and tolerant old timestamp reads are unchanged.
- The existing continuation table joins the SQL identifier allowlist. A real,
  populated foreign-key conversation-delete query plan proves use of its receipt
  index with no ANALYZE/statistics; the census points to that runnable test.
- Reviewed fixed diagnostic/exception-type messages and the encrypted authority
  snapshot's private atomic sink are included in the production inventory.
  The sink writes an already authenticated/encrypted snapshot, not plaintext
  source bodies. No exclusion or gate was weakened.

Fresh timestamp, schema-table, index-plan and diagnostic-inventory checks pass;
all generated CSS bundles reproduce under project Python 3.12.11. Design-token
checks: **8 passed in 21.22s**. All **189 branch/working Python files** parse,
changed ranges/new owners are formatted, shared files add no Ruff diagnostics
against rebased HEAD, and whitespace checks pass. Existing ADR-162/163/173 apply;
these repairs add no new runtime, authority or storage boundary. Local evidence
remains targeted and does not replace a new remote CI run or bot review.


## PR #2946 Qodo review and boot-budget repair — 2026-10-01

Reviewed Qodo's ten findings against published head
`17c4734241a76148534535c74281fa01c473160b`, tracing actual callers and the
existing authority owners before applying changes. Nine inline findings are
identified below by GitHub comment ID; finding 10 appeared in the summary only.
Existing ADR-162/163 govern these direct contract repairs. ADR-097 governs the
unchanged startup ratchet; no new authority, storage, dependency or runtime
boundary is introduced.

| Finding | Disposition and evidence |
| --- | --- |
| 1 / 4161447234, continuation consumption | Incorrect control-flow inference. Durable `submit_draft` returns through `_accept_durable_turn` before reaching explicit consumption; its acceptance contribution consumes the gate in the transaction. The later explicit consumption belongs to ephemeral submission. Retained both owners and added a clarifying comment. Four actual queued-Console cases cover both storage modes, their real gateway submissions and exactly one consumption for each gate. Removing the ephemeral guard would weaken admission. |
| 2 / 4161447184, hook cwd | Reused `validate_existing_absolute_directory` at definition validation and fresh command launch. Explicit user-selected cwd may be outside a workspace; it must be an existing absolute directory. Plugin-relative cwd remains contained by the existing native plugin owner. Retargeting a validated directory to a symlink is refused before launching the process. Actual child-marker controls verify the selected directory and no execution after retarget, with custody/tickets released. |
| 3 / 4161447190, quota iterator | `managed_usage` now owns `os.scandir` with a context manager. A real retained iterator proves deterministic closure when recursive sizing refuses a symlink. |
| 4 / 4161447195, UI-thread authority read | Moved initial `_hooks_v2_context_key` collection to `asyncio.to_thread`, retaining the operation-owned database connection and rechecking disposal/session fences after the await. A real accepted Console turn confirms the collection runs off the UI thread. Synchronous exact-currentness gates remain in their existing admission locations. |
| 5 / 4161447211, validator documentation | Documented the manifest validator's actual Args, Returns and fixed-code Raises contract. |
| 6 / 4161447217, public profile types | Annotated all `save_owned_profile` inputs using existing `PackageInspection`/`DataRoot` types behind `TYPE_CHECKING`. No eager plugin import or new public model. |
| 7 / 4161447246, definitive tool settlement | Fixed shared `ToolHookRun.install_result`, the common checkpoint installation owner. Only owner-retirement checkpoint errors are tolerated; current-owner errors still propagate. Event scopes and already planned checkpoints close/fail through their existing lifecycle. Real definitive AgentService calls prove the original settled result identity and both terminal notifications survive retirement before installation and between the two postevents; a live-owner control passes. Retired owners stay closed with no retained checkpoint entries. |
| 8 / 4161447255, missing Library skill | Optional plugin service and an absent owned row now produce the documented `ValueError`, using `next(..., None)`. Tests cover no service, an empty list and unrelated installed rows. |
| 9 / 4161447202, strict manifest model | Existing installed Pydantic validates known manifest/author fields strictly; unknown top-level inspection fields remain retained, and absent optional fields remain absent. The public dictionary API and bounded package parser remain unchanged. Validation failures become fixed manifest codes without external values or raw Pydantic error bodies. Existing native manifest controls cover accepted fields, rejected coercions and retained extension data. |
| 10, malformed vendor skill metadata | Added the missing dictionary check before metadata conversion, allowing the existing per-component failure path to retain a blocked skill alongside usable siblings. Eight independently authored Cursor/Codex fixtures cover null, string, list and numeric metadata. Each asserts the invalid component's fixed blocker and a valid sibling's retained usability. All foreign hook mappings remain unsupported. |

The remote UI latency run `36939210300` failed the module-count ratchet at
**1,034 against a 1,033 limit**. Reproduced that exact failure locally
(`hooks-pr2946-boot-red.xml`, 20.889s), then deferred `MCP.protocol_profiles`
loading until actual connection construction/initialization. Removed the unused
private connection class default; constructed connections still receive the
same instance profile and negotiate through the same code. No ratchet,
snapshot, workflow or exclusion was changed. The isolated census passes
(`hooks-pr2946-boot-green.xml`, 16.238s), and the exact six-file affected
startup selection passes **22 cases in 159.636s**, zero skips
(`hooks-pr2946-boot-final.xml`). Local optional dependencies are installed;
the earlier remote selection's two missing-dependency skips are not evidence.

Fresh covering checks:

- Hook validation/execution/pipeline, native/Cursor/Codex inspection,
  retention, owned MCP, protocol profiles, typed results and progress:
  **488 passed / one fixture failure in 937.601s**, no skips
  (`hooks-pr2946-qodo-neighbors.xml`). The remaining progress helper bypassed
  `__init__` and relied on the removed class profile. Five initialization
  variants independently reproduced the same missing instance state
  (`hooks-pr2946-bare-init-red.xml`). The existing shared bare-connection
  helper now sets exactly the real constructor's default instance profile;
  the separate direct `__new__` fixture reuses that helper. The full catalog
  pagination and progress files then pass **147 cases in 7.827s**, no skips
  (`hooks-pr2946-bare-fixture-final.xml`), including real concurrent stdio
  progress. Every case in the original covering group is now passing across
  the retained run and corrected-file run. No production fallback was added.
- Actual definitive tool and Library skill service files: **56 passed in
  13.487s**, no skips (`hooks-pr2946-qodo-services.xml`).
- Complete continuation/lifecycle Console files: **80 passed in 328.062s**,
  no skips (`hooks-pr2946-qodo-console.xml`).
- Native manifest controls plus real continuation/worker controls:
  **77 passed in 10.938s**, no skips (`hooks-pr2946-qodo-manifest.xml`).
- Authored behavioral regressions: **18 passed in 1.845s**
  (`hooks-pr2946-qodo-green2.xml`); the later 56-case service run includes
  the additional between-postevents retirement control. Counts overlap.
- Diagnostic inventory, timestamp guard and all generated CSS bundles
  reproduce without tracked artifact changes. All 20 changed Python files
  parse, no Ruff diagnostics are added against the published head, authored
  files/changed shared ranges are formatted, and whitespace checks pass.

Retained RED evidence is scoped honestly: the initial combined authored run
also contained five fixture setup failures (private bootstrap selection and
an attempted read-only property assignment), and two vendor assertions assumed
a global diagnostic instead of the actual component-blocker contract. Those
are harness mistakes, not demonstrated production failures. Corrected
constructor fixtures independently reproduce all three missing-skill errors
(`hooks-pr2946-skill-red.xml`); the actual definitive owner-retirement path
reproduces `HookCheckpointError` before repair
(`hooks-pr2946-qodo-owner-red.xml`); the exact boot ratchet independently
reproduces its failure as recorded above. No production guard was weakened to
make a fixture pass.

CodeRabbit's successful check explicitly skipped review because automatic
review is disabled for this non-default target branch; it is not clean-review
evidence. A new remote run and Qodo review must assess the repaired head.
Qualification remains targeted Darwin Console/SQLite/subprocess behavior.
No full-suite, full GUI, real Keychain/OAuth, foreign-host or Windows/Linux
runtime certification is claimed.


## Rebase onto dev 84247cb843 (2026-10-01)

Dev advanced while the repaired `fa988be533` head was qualifying and the PR
became conflicting. Rebased all 23 branch commits onto
`84247cb8435fcf59b6d8e2d97c6b2f0913934dd0`. The only conflict was duplicate
`cached_context_window` documentation in the wake gateway double; retained
dev's complete docstring and identical implementation. Range comparison shows
21 identical commits and two changed only by this duplicate/surrounding context;
the Qodo repair commit replays identically. AST comparison retains dev's provider
selection/default and resolved-system-prompt builders exactly.

The follow-up changes are test fixtures, not production policy: configuration
cases retain the existing collection-time private profile; the custody double
supplies the real coordinator's no-chain binding contract; custody and wake
assertions compare the complete frozen snapshot with the runtime-owned
`plugin_turn_id`. Acceptance-worker tests now wait on their real entry event
with a bounded 30-second hold rather than three-second polling. The mounted
pending-Stop command uses the existing permitted 60-second timeout so its
controlled process remains pending through actual tab navigation.

Interrupt remains an optional one-second observation under ADR-163. Tests pin
exactly one host emission/cancellation, suppression after revocation, settled
resource ownership and no replay during fresh Send. If the optional command
cannot execute, only one fixed `event_deadline` or `cancelled` diagnostic is
accepted; any body that executes must have the exact parent ID and occur once.
The separate mounted Interrupt execution control still requires an actual
child-written file. No runtime deadline, budget, authority gate or CI ratchet
was changed. Existing ADR-162/163/197 and ADR-097 apply; no new ADR is required.

New-base evidence:

- Exact affected boot selection: **22 passed in 132.923s**, no skips,
  three intentional headroom warnings. Census is **1,031/1,033**;
  pre-import **557/557** and boot **680/686**, with unchanged limits
  (`hooks-pr2946-current-dev-boot.xml`).
- Diagnostic inventory, timestamp sources and all generated CSS reproduce
  without tracked changes (`hooks-pr2946-current-dev-derived.log`).
- Complete Console overlap: **201 passed / two failures in 711.588s**,
  no skips (`hooks-pr2946-current-dev-console-final.xml`). The scheduled
  wake passes in the isolated 10-case wake/Interrupt run; the remaining wake
  failure was the stale snapshot-identity assertion, corrected and passing
  through the actual survivor/provider path in the sequential mounted run.
- Sequential wake/composer/full teardown: **17 passed / one optional-observer
  assertion failure in 557.778s**, no skips
  (`hooks-pr2946-current-dev-ui-sequential.xml`). Actual composer collapse,
  all three resize variants, mandatory mounted Interrupt execution and all
  four pending-Stop size/collapse controls pass. The final teardown assertions
  use the existing best-effort contract as above: **13 passed in 431.771s**, no skips (`hooks-pr2946-current-dev-teardown-final.xml`).
- All **192 branch Python files** parse; four fixture corrections add no Ruff
  diagnostics and authored ranges are formatted; whitespace checks pass.
  Shared inherited formatting is preserved.

Retained earlier aggregate failures are not represented as successful full
qualification: the first Console attempt was interrupted after 157 recorded
cases, and the concurrent UI run was interrupted with 22 recorded cases
(11 passing). Concurrent mounted startup exceeded existing five-second harness
waits and observer deadlines; the exact composer controls then passed
sequentially without changing those waits. The isolated wake/Interrupt run
passes nine of ten cases in 49.510s and produces four actual Interrupt payload
files; its one stale snapshot assertion is corrected as recorded above.

All applicable remote checks on published repair head `fa988be533` completed
successfully, including both fast lanes, derived artifacts and UI latency
(`36948561724`, `36948561739`). Expected skipped platform/signing jobs and
neutral/skipped code reviews are not runtime/review evidence. Qodo still has
only its original review of `17c4734241`; fresh checks and review must assess
the newly rebased head. The platform and scope limits above remain unchanged.


## Rebase onto dev 27e718f01d (2026-10-01)

Dev advanced again with PR #2939 / TASK-33621.3 while the previous rebased head
was conflicting. Replayed all 24 commits onto
`27e718f01d81b7093502c0ea1d2690e07b9f6362`. Dev now owns the v74 auxiliary
failure-reason migration. Preserve that migration unchanged and advance hook
continuation receipts through v74 to v75, using the packaged
`chachanotes_v74_to_v75_hook_continuation_receipts.sql` resource. Genuine v73/v74
upgrades retain the failure-reason field, including a populated v74 failure row;
failed receipt DDL rolls back to v74 and reopening completes v75.

Compaction retains dev's retry latch, structural no-cost checks and durable
parent-lineage fallback. Pass the existing hook owner into the compact-once
operation after the retry fence. AST comparison preserves
`compaction_retry_fence`, `effective_memory_identity`, `_FailedCompaction` and
`_durable_context_snapshots` exactly; the outer `compact` method matches dev
apart from its hook argument and forwarding. Required PreCompact refusal now
uses the existing `_end` helper to record its fixed failure reason and
`attempted=False`; manual and automatic refusal make no summary or main-provider
call and commit no memory. Cancellation retains its existing propagation and
records the fixed `cancelled` reason. No billed-retry latch, consent or resource
owner is weakened. Existing ADR-162/163/052 apply; no new ADR is required.

The existing recovery declarations must match the new actual schema. Fresh core,
standalone subscriptions and combined subscriptions captures came from their
real constructors, not version-label substitution or an inferred schema union.
Core adds exactly the two receipt SQL records; standalone subscriptions is
unchanged; combined subscriptions adds exactly those two records. No previous
catalog record is removed. Core and combined version validation now require v75
while subscriptions retains its own v2 stamp and exact-schema checks.

New-base evidence (all runs have zero skips):

- Required manual/automatic pre-hook controls first reproduce two missing
  failure-reason failures in **2.961s** (`hooks-pr2946-v75-red.xml`). The actual
  core constructor independently reproduces the stale v74 recovery declaration:
  **four passed / one failed in 3.894s** (`hooks-pr2946-v75-core-red.xml`).
- Compaction, dev retry/failure-copy/live-session neighbors, migration and full
  affected recovery files: **497 passed / two failed in 371.375s**
  (`hooks-pr2946-v75-integration.xml`). One new manual-refusal fixture wrongly
  expected acceptance; corrected to assert refusal. Full hook-compaction and
  receipt-migration files then pass **25 cases in 31.102s**
  (`hooks-pr2946-v75-final-controls.xml`), including both real-SQLite refusal
  paths, populated v74 preservation, rollback/reopen and uncertain-dispatch
  receipt/cascade controls. Counts overlap.
- The remaining failure is the existing dormant-pet constructor assertion at
  `Widgets/AppFooterStatus.py:244`. The same AST constructor scan on an immutable
  archive of exact dev `27e718f01d` reproduces that call site
  (`hooks-pr2946-v75-upstream-dormant-proof.log`). It is excluded as an unrelated
  upstream failure, not fixed, xfailed or counted as passing qualification.
- Actual core/standalone constructor captures pass **two cases in 2.638s**;
  actual combined constructors pass **one case in 2.267s**
  (`hooks-pr2946-v75-capture-proof.xml`, `hooks-pr2946-v75-combined.xml`). The
  temporary capture harness is retained as `/private/tmp/hooks-pr2946-v75-capture-source.py`
  and was removed from the repository after recording the declarations.
- Exact affected boot selection: **22 passed in 112.007s**, three intentional
  headroom warnings (`hooks-pr2946-v75-boot.xml`). Census is **1,032/1,033**;
  pre-import **557/557** and boot **680/686**. Ratchets, snapshots and workflows
  remain unchanged.
- Built wheel contains the exact new v75 receipt SQL bytes and dev's v74
  failure-reason SQL (`hooks-pr2946-v75-wheel.log`). Diagnostic inventory,
  all 117 pinned timestamp occurrences and all generated CSS reproduce without
  tracked artifact changes (`hooks-pr2946-v75-derived.log`). All **196 branch
  Python files** parse; seven changed Python files add no Ruff diagnostics;
  authored/changed shared ranges are formatted and whitespace checks pass.

The 499-case covering run is not a clean sweep; the corrected 25-case run and
frozen upstream proof explain both failures explicitly. Earlier evidence and
platform limits remain as recorded above. No full test suite was requested or
run. The previous published `77d0766328` head received no CI run while conflicting;
new remote checks and Qodo review must assess the newly pushed head.


## Merge closeout rebase onto dev ab4df99959 (2026-10-01)

The owner explicitly requested latest-dev rebase, Qodo dispositions and merge.
Fetched the live target independently of PR base metadata: dev is
`ab4df99959545e37d8d2048c1c1ce15fd914721d`, including PR #2928's accepted
ADR-210 amendments and PR #2903's warm-config fast paths. All 25 hook-branch
commits replay identically (`hooks-pr2946-ab4-range-diff.log`). The exact ASTs of
both public/guarded settings readers, both snapshot readers, path-posture checks,
cache-hit/owned-file checks and the config-participant guard match fetched dev.
The visible Hooks action and existing consent contract remain under ADR-197 as
amended by ADR-210. Existing ADR-126/163/197 apply; no new ADR or owner is required.

Fresh affected qualification, with no skips:

- Warm-config controls plus full actual hook-compaction/v75 migration files:
  **38 passed in 26.476s** (`hooks-pr2946-ab4-config-compaction.xml`).
- Full persistent hook-permission controls: **56 passed in 24.760s**
  (`hooks-pr2946-ab4-consent.xml`).
- Full mounted Console hook-review file: **eight passed in 47.974s**
  (`hooks-pr2946-ab4-console-review.xml`). These use the actual icon, next-Send
  review/cancellation and draft custody, partial approval, Settings navigation,
  current revoke/disable actions and dismissal during refresh. Pytest shutdown
  emitted its shared garbage-directory cleanup warnings; no manual cleanup of
  those unrelated directories was attempted.
- The exact GGUF lazy-selector handoff case that failed on the previous Windows
  runner passes locally: **one passed in 9.201s**
  (`hooks-pr2946-ab4-gguf-handoff.xml`). This is Darwin evidence, not a claim that
  Windows is fixed. The previous Windows failure remains recorded in
  `hooks-pr2946-gguf-windows-failed.log`; its main suite reports missing late
  inventory after a ten-second settle bound, and its separate startup diagnostic
  also outlived its own limit. No unrelated UI behavior, test deadline or workflow
  was changed. Fresh Windows CI must qualify this rebased head.
- All **196 branch Python files** parse; whitespace checks pass. The unchanged
  diagnostic inventory reproduces exactly. The previous head's fast lanes,
  derived artifacts and UI latency checks passed; fresh remote checks must assess
  the new head before merge. Earlier frozen upstream exclusions remain explicit.

Qodo's current persistent report lists zero active bugs/rule violations, with
nine findings marked resolved and the independently disproved one-use gate
finding marked outdated. All nine inline threads are resolved. Its last separate
GitHub review object remains the original review of `17c4734241`; these report
updates are not represented as a new full review. Recheck new comments, exact
current head, live dev tip and fresh CI before the requested head-pinned merge.


### Windows GGUF fixture race during merge closeout

On `145c823ca0`, native Windows again fails the delayed-selector inventory case,
this time `[False]` while `[True]` passes; the preceding published head failed
`[True]`. The other six full-app source nodes and all 31 native source/lifecycle
controls pass. Retain `hooks-pr2946-merge-windows-complete.log`. Do not classify
this as a runner outage or suppress the failed case.

The test injects nonempty inventory while the actual app-owned initial worker
can still deliver its legitimate empty result in the same generation. A held
real worker reproduces that replacement deterministically:
**one passed in 3.984s** (`hooks-pr2946-inventory-order-confirmed.xml`). The
first probe used the wrong `window` owner filter and failed before its behavioral
assertion; this was a harness mistake, not runtime RED. Correcting it to the
existing app owner proves the fixture race. The probe source is retained in
`/private/tmp/hooks-pr2946-inventory-order-probe-source.py`; the temporary module
is removed from the repository.

Settle only the existing app-owned `managed_gguf_inventory` worker before this
test injects its late discovery. Keep the actual nested-label hold, inactive
pane, exact selected references/rendered labels, handoff settlement and unchanged
claim authority assertions. No production code, timeout, permission guard or
workflow changes. Both changed private-profile nodes and the neighboring cached
hydration/fresh-read handoff pass: **three individual cases in 8.943, 8.708, 8.448s**
(`hooks-pr2946-inventory-fixed-0.xml` through `-2.xml`), no skips. AST, added-range
formatting, no-added-Ruff comparison and whitespace checks pass. Existing
ADR-025 governs; no new ADR is required. TASK-2062.2 AC #7 is reopened pending
fresh native Windows qualification; the earlier historical implementation remains
unchanged. Fresh CI must assess the newly published repair before merge.


Native requalification on published repair `583d2c4beaddddefb01dbf07d17625a44a66e03e`
passes [all three GGUF source jobs](https://github.com/rmusser01/tldw_chatbook/actions/runs/36974337203).
The [Windows job](https://github.com/rmusser01/tldw_chatbook/actions/runs/36974337203/job/110734792443)
passes all **31 source/lifecycle controls plus seven individually isolated
full-app cases**, with no skipped cases. Both repaired delayed-selector variants
pass in **56.42s and 56.88s**, retaining the original deadlines and authority
assertions. Retain `hooks-pr2946-windows-583d-success.log`. TASK-2062.2 AC #7 is
checked and status is Done through the Backlog CLI; the subsequent rebase retains
the exact GGUF fixture/runtime bytes. Final-head CI remains required before merge.

### Final rebase onto logging dev 92a95170a5

Live dev advanced during CI to `92a95170a5406b3ebdc9be6dc26a12f2741d756f`
(PR #2904 / TASK-33262). All **27 implementation/repair commits** replay; the
only range-diff change is append context in the testing-lessons conflict. Both
independent lessons are retained. Six upstream logging implementation files are
byte-identical to dev, the actual worker-state event owner's AST is identical,
and all **40 upstream DB log demotions** are retained. Existing runtime/receipt
and consent ADRs apply; this integration introduces no architectural decision.

Fresh targeted logging, hook-compaction and actual v75 migration controls:
**42 passed in 72.21s**, no skips (`hooks-pr2946-92a-logging-compaction.xml`).
The merged diagnostic inventory reproduces exactly at **630 owners, 1425 TASK-492
calls, 56 TASK-31551 calls, 7590 TASK-494 calls and 16 sink files**. All **117**
pinned timestamp occurrences pass. The full mounted Console hook-review file
and affected boot selection pass **30 cases in 175.19s**, with no skips and three
existing headroom warnings (`hooks-pr2946-92a-console-boot.xml`). Limits and
snapshots remain unchanged. All **197 branch-changed Python files** parse;
whitespace checks pass. Earlier static, authority, teardown and recovery evidence
retains its exact original head/base and exclusions. The final candidate must
pass fresh remote CI and comment checks before the authorized head-pinned merge.


### Queue/GC and admission latest-dev closeout

Live dev advanced to `ee1c1e7365c232a184e129bef1dec15afd85b24f` and then
`113e435ab0bff12e89ae1c9f06ec154f32f776bf` during qualification. On the first
rebase, **27 of 28 patches replay unchanged**. The one real overlap is durable
queue settlement: retain dev's `live_chain_owns_claim` implementation and its
exact live `current_entry_id`, which already implement the older hook patch's
reservation handling. No second flag is added. All upstream context-review
Resume/Retry/Discard propagation, trace-maintenance parking and GC freeze
behavior are retained. Eight relevant upstream owner files are byte-exact;
all newly changed runtime/controller/screen methods and the five relevant
coordinator settlement/review methods retain their upstream ASTs. Evidence:
`hooks-pr2946-ee1-range-diff.txt` and `hooks-pr2946-ee1-static-evidence.json`.
The second rebase replays **all 28 patches identically**; all five admission
implementation files are byte-identical to dev (`hooks-pr2946-113-range-diff.txt`).
ADR required: **no**. Existing ADR-126/163/197/198/210 govern; this adds no
storage, permission, runtime or UI contract.

The first combined targeted invocation records **170 passes, two failures**, then
is gracefully interrupted inside the existing 1,000-turn retention stress
case after **675.57s**. This is partial evidence, not a passing aggregate or a
stress-test verdict (`hooks-pr2946-ee1-queue-continuations.xml`). Its completed
queue/coordinator, two mounted queue UI files, complete hook-continuation file
and complete parked-maintenance file contribute **153 passes** (38/49/16/43/7).
The two dispatch-recovery failures reproduce individually on the branch and on
immutable upstream `ee1c1e` (`hooks-pr2946-ee1-recovery-isolated.xml` and
`hooks-pr2946-ee1-upstream-recovery.xml`): the shared submit double treats the
current `QueuedPrompt` argument as a string, and the actual ephemeral-send case
switches away from its collection-time admitted profile. Update only that test
helper and that case's existing `bootstrap_profile` marker. No production guard,
assertion, timeout or workflow is weakened. The complete repaired recovery file
passes **ten cases in 4.79s** after final import formatting
(`hooks-pr2946-113-recovery-final-imports.xml`). The durable retry/bypass,
single-entry reclaim and thinking-refusal controls pass **five cases in 5.92s**
in their separate admitted-profile invocation
(`hooks-pr2946-113-durable-reclaim.xml`). Do not reclassify the prior unrelated
recovery/pet exclusions or the earlier frozen 1,000-turn qualification as a
new full-suite pass.

On `113e435ab0` plus the fixture repair, fresh admission-reuse oracle,
hook-compaction and genuine v75 migration controls pass **62 cases in 30.86s**;
one expected unbound-profile edit has no record to mutate and is skipped
(`hooks-pr2946-113-admission-compaction.xml`). Full mounted hook review plus GC
freeze and affected boot guards pass **32 cases in 136.40s**, without skips,
with the same three headroom warnings (`hooks-pr2946-113-console-boot.xml`).
UI-ready remains **1032/1033**, pre-import **557/557**, and app import **680/686**;
no limits or snapshots change. All **eleven exact artifact checker commands**
pass (`hooks-pr2946-113-local-artifacts.json`); diagnostic inventory is
**631 owners / 1425 TASK-492 / 56 TASK-31551 / 7591 TASK-494 / 16 sinks** and
all **117** timestamp occurrences remain pinned. All **197 branch Python files**
parse and whitespace checks pass. The repaired test passes full Ruff and format;
the queue rebase adds no Ruff diagnostics to its inherited baseline (whole-file
legacy queue formatting still flags pre-existing ranges).

Published previous head `b23edf5385` ultimately passes its hosted required
artifact job, both fast lanes, all six native GGUF source/import jobs, native
wheel/structural qualification and UI latency. Its former runner queue is no
longer the blocker. The latest independently read dev remains `113e435ab0` before
publication. Qodo's persistent report still has nine resolved findings and one
independently disproved finding, all nine inline threads resolved; its separate
review object remains on the original head. Fresh final-head CI and comments
must qualify the new publication before the authorized normal head-pinned merge.
The user approved a ten-minute current-thread heartbeat to complete those checks,
fix actionable findings and merge after verification, then pause itself.


### Final Delete/Undo dev eba4305d83

The final pre-publication freshness check catches dev moving again, to
`eba4305d8389a2112c99ac19fa804e9abea394ba` (Delete/Undo PR #2941 /
TASK-33628.2). The qualified `113e435ab0` candidate is published as `c21e37983f`
with an exact previous-head lease; it is not merged or represented as including
the newly advanced base. Rebase all **29 commits** onto `eba4305d83`: every patch
replays identically (`hooks-pr2946-eba-range-diff.txt`). Eleven relevant upstream
Delete/Undo, recovered-media and stylesheet-owner files are byte-identical;
the DB subtree-restore AST is exact. Existing ADR-052/097/126/163/197/210 apply;
no new ADR or runtime behavior is introduced by this integration.

The actual mounted hook-review and Delete/Undo flows, modal stylesheet ownership
and affected boot guards pass **23 cases in 142.21s**, with no skips and one
existing headroom warning (`hooks-pr2946-eba-ui-boot.xml`). UI-ready remains
**1032/1033**. The persistence/media/semantic-inventory selection completes with
**112 passes and three failures in 693.42s**
(`hooks-pr2946-eba-delete-persistence.xml`): all **12 Delete/Undo persistence**
and **17 recovered-media** cases pass, as do **83 semantic-inventory controls**.
The three inventory failures are independently reproduced on immutable exact
upstream `eba4305d83`, not hidden or converted to xfails:

- `test_live_sql_mutation_sites_are_classified`: two unreviewed dynamic SQL
  executor sites in the existing staged credential reconstruction.
- `test_public_mutation_boundary_calls_are_classified`: five missing and three
  stale character/conversation/voice/importer route names.
- `test_repository_scan_results_are_cached_and_immutable`: the same credential
  dynamic-SQL assertion as the first failure.

Retain `hooks-pr2946-eba-upstream-semantic.xml` (the first two failures plus a
passing document-sync control) and `hooks-pr2946-eba-upstream-cached-scan.xml`
(the third failure). All six implicated source/scanner files are byte-identical
to dev; failure messages match the branch. These are explicit upstream inventory
exclusions, not a new hook regression or a clean aggregate pass. Earlier pet,
recovery and interrupted stress limitations remain unchanged.

All **eleven artifact commands** pass again on this latest-base source
(`hooks-pr2946-eba-local-artifacts.json`); diagnostic inventory reproduces at
**633 owners / 1427 TASK-492 / 56 TASK-31551 / 7593 TASK-494 / 16 sinks**. The
previously qualified recovery fixture, admission oracle and hook continuation
implementation remain unchanged. Qodo's persistent report refreshes on published
`c21e37983f` at **2026-10-02T15:16:47Z**, retaining nine resolved findings, one
independently disproved finding and no unresolved inline threads; its separate
review object still references the original head. The latest independently read
dev remains `eba4305d83`. Fresh checks and comments on the final published head
must pass before the authorized normal head-pinned merge; the approved ten-minute
heartbeat remains active to complete it and pause after confirmed merge.


### Inspector/keep-alive dev 6958e8dfa9 and review-Send gate

Independently read live dev advances to `6958e8dfa99a66680b1aec09fed65df0f1a91955`
(PR #2944). All **31 patches** replay identically, including the integration plan
recorded before this rebase (`hooks-pr2946-695-range-diff.txt`). Eleven relevant
upstream persistence, Inspector, keep-alive, worker-checker, census and fixture
owners are byte-identical. All six changed store/protocol methods and the
lifecycle error handler retain their qualified upstream ASTs. The branch's
existing three-line plugin shutdown remains separate from that handler.
Evidence: `hooks-pr2946-695-owner-evidence.json`. ADR required: **no**; existing
ADR-069/097/126/163/197/210 govern, with no new runtime or authority owner.

Initial targeted first-send/compaction/v75 invocation: **43 passed, 11 failed in
91.64s** (`hooks-pr2946-695-first-send.xml`). The complete first-send file on
immutable exact dev independently reproduces the same eleven assertion signatures
(after normalization of process-specific object addresses), with **18 passed,
11 failed in 48.02s** (`hooks-pr2946-695-upstream-first-send-v2.xml`); an isolated
branch node also fails. These controller tests reselect a per-test sandbox after
collection-time configuration admission, so they refuse with `Hooks unavailable`
before the intended persistence boundary. Keep the file on its admitted private
profile using the existing `bootstrap_profile` marker; the dedicated private
profile children and all transaction, rollback, cancellation and provider-entry
assertions remain intact. The full repaired file passes **29 cases in 34.81s**,
without skips (`hooks-pr2946-695-first-send-fixed.xml`). The unchanged compaction
and genuine v75 receipt files account for **25 passes** in the initial invocation.
No production guard, assertion or timeout is weakened.

Mounted hook review, actual project picker, full dead-pump recovery/teardown and
boot controls: **42 passed, three strict expected freezes in 340.31s**, with one
existing headroom warning (`hooks-pr2946-695-mounted-boot.xml`). The three freezes
are real upstream Send-button, Workbench and Enter review deadlocks, accepted
only when `FreezeObserved` is raised. They are **not** passing UI qualification
for those routes. Preserve these tests and their census rows until the
independently owned repair **PR #2945** lands; it is now rebased onto this dev.
Do not duplicate, cherry-pick or merge that owner's PR. **PR #2946 remains
unmerged until the repair is integrated and these routes pass as ordinary tests.**
The full W003 checker file passes **93 cases**, with two unchanged strict xfails
for split-function future recall (`hooks-pr2946-695-worker-checker.xml`).

All **eleven exact artifact commands** pass (`hooks-pr2946-695-local-artifacts.json`).
Diagnostic inventory reproduces at **634 owners / 1427 TASK-492 / 56 TASK-31551 /
7597 TASK-494 / 16 sinks**. All **198 branch-changed Python files** parse and
whitespace checks pass. The repaired fixture formats cleanly; its five inherited
Ruff diagnostics match immutable dev, with no added diagnostic. Budgets and
snapshots remain unchanged. Previous published `69e8226712` completes all
applicable hosted checks, including both fast lanes, required artifacts, all
native GGUF/wheel jobs and unsigned structural qualification; it does not include
this newly advanced base and is not merged. Fresh published-head CI, Qodo and
live-dev checks remain necessary, in addition to the concrete review-Send gate.
Earlier unrelated semantic-inventory, pet/recovery and interrupted stress
limitations remain explicitly retained. The approved heartbeat continues quietly
while the upstream repair and fresh checks are pending.
