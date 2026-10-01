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
