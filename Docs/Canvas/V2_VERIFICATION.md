# Canvas V2 Mermaid qualification

Task31941, ADR124. **Required local gates passed on 2026-09-08; the exact V2
profile is admitted. Independent review is pending.** The historical withdrawn
admission and SQLite investigation below remain unchanged evidence, not passing
results. This is local qualification, not hosted CI or merge approval.

## Current qualification — 2026-09-08

The separately reviewed SQLite prerequisite is complete; see
[acceptance closeout](../superpowers/reviews/2026-09-08-sqlite-acceptance-closeout.md).
Fresh candidate and admitted selections now pass against the unchanged immutable
`canvas-v2-mermaid-1` executable assets:

- Build: `5cdfdfcf09ed257bce900fa94472cc9ff79eb58506ec86e0a55ce0cd2d7e95d8`.
- Manifest: `17717bcab7c7bba4a28e0069354f6ecbf895d2ca58f4b8d1c0355b7726e2f466`.
- Admitted policy: `cd4f0cdd756732e686b05031ce12c6bd086473cc72ff2f9d58340d8528b40f15`.

Only the catalog's diagram default, policy identity, V2 execution flag and refusal
reason changed. V1 remains executable and the default without diagrams. No
manifest, library, worker, renderer, engine, quota, schema, archive format or
generated-code privilege changed. Unknown/revoked profiles remain source-only.

| Fresh stage | Result |
| --- | --- |
| Required five-file Chromium candidate selection | 176 passed, 2 optional browser skips, 1 warning, 475.87s |
| Complete Canvas candidate selection plus CI workflow contracts | 1383 passed, 2 optional browser skips, 2 warnings, 661.78s |
| Admission assertions against disabled catalog | 2 expected failures, 1 warning, 0.88s |
| Same assertions after exact catalog admission | 2 passed, 1 warning, 0.73s |
| Complete admitted selection plus CI workflow contracts | 1383 passed, 2 optional browser skips, 1 warning, 767.12s |
| Touched-file Ruff, formatter and whitespace checks | Passed |

Root followed every pytest invocation through process exit. Both integrated JUnit
records contain 1385 cases, zero errors/failures and only missing Firefox/WebKit
skips. Their collected node identities differ only by the two intentional
admission-test renames. No required Chromium or offline rebuild gate was skipped.
The admitted assertions load the real packaged snapshot and its verified assets,
not the test-only candidate snapshot. The RED failed on the disabled policy and
non-executable profile, not on setup or environment errors.

### Reproduction and retained evidence

Both integrated runs use the complete broader selection in the historical
Commands section below, plus `Tests/CI/test_github_actions_test_workflow.py`.
The first selection is the exact five-file browser command there. All use `-x`
for fail-stop, except the intentional two-node RED/GREEN commands, and unique
owned `--basetemp`, pytest cache and JUnit paths. The evidence root is
`/private/tmp/mermaid-qualification.lga1Y7`; retain it rather than reusing its
basetemp paths. Each phase has a full `*-invocation.json`, console `*.log` and
JUnit `*.xml`: `browser-candidate`, `canvas-candidate`, `admission-red`,
`admission-green`, and `canvas-admitted`. Browser output snapshots have the
matching `*-playwright` directory; earlier output is separately preserved as
`pre-qualification-playwright` and is not attributed to these runs.

Candidate runs started at clean commit `2ed6ce926f1b78a452b2cb2fd9cb66179996e0e4`.
The admitted source delta is `admission-diff.patch`, SHA-256
`89b67f053796d27d751fb8057cd3103f514c62603cc34617a44b02e304b50084`; root verified
that this exact diff remained unchanged through the admitted run. It contains
two admission-test updates, mechanical test-only lint cleanup, and the four
catalog policy fields. Complete command arguments and environment are in the
invocation receipts. The per-command environment uses:

- `TLDW_CANVAS_RUNTIME_ARCHIVE_DIR=/private/tmp/mermaid-qualification.lga1Y7/runtime-inputs`.
- `TLDW_CANVAS_MERMAID_INPUT_DIR=/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/canvas-mermaid-inputs-4i3rl_al`.
- `TLDW_CANVAS_CHROMIUM_EXECUTABLE=/Users/macbook-dev/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell`.
- `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/canvas-v1/packages/tldw_profile_core/src`.

The existing interpreter is Python 3.12.11; Node is 24.6.0, Chromium
151.0.7922.34, Playwright 1.58.0, Textual 8.2.8, aiohttp 3.13.5 and Ruff 0.16.6
on macOS 26.5.2 ARM64. Browser binary SHA-256 is
`7687bff7cb2db075f250e6d5848bbc8838cac3802ac3952a899c574f8eccab45`.
Repository pre-import pytest isolation supplies owned config/data; no ad hoc
application import, actual provider or user database is involved.

All eight declared Mermaid/Unicode inputs matched byte sizes and SHA-256. The
earlier QuickJS input cache was empty, so only its four already-pinned HTTPS
archives were restored into the new owned cache and authenticated with declared
SHA-512 SRI. No package installation, lifecycle script, dependency update or
runtime network permission was introduced. Both vendor builders regenerated
twice offline with exact checkout-byte equality. Each integrated run passed 39
runtime-assets cases, 34 Chatbook archive cases, 3 wheel/sdist cases and 19 CI
workflow contract cases. Hosted CI itself was not run in this continuation.

The candidate integration also emitted a compiler-test invalid-escape
SyntaxWarning; the admitted run emitted only the existing Requests dependency
compatibility warning. Neither is interpreted as a passing security/resource
refusal. Three Ruff findings in the touched asset-test file were reproduced
against the starting commit and corrected mechanically; final static checks
passed without new suppressions or rule waivers. No full repository sweep or
cross-engine qualification is claimed.

### Visual, resource and lifecycle checks

Root inspected all 16 wide/narrow candidate screenshots for branch/rejoin, notes,
both shipped guide examples, four diagrams, mixed HTML/diagrams, Unicode and the
near-limit chain. Tests separately exercise Count interactions, far-right scroll
reachability, first/last SVG visibility, inherited styles and explicit restyling.
A tall or horizontally scrolled screenshot alone does not establish complete
diagram visibility. Local Unicode glyph/wrapping observations do not establish
cross-font pixel parity or full bidi typography. The separate visual note is
`browser-candidate-visual-review.md` in the evidence root.

The eight standalone fresh-context load-to-ready samples ranged 79.745–103.937ms.
The near-limit chain took 103.937ms standalone, 85.352ms in candidate integration,
and 104.622ms in admitted integration. These are individual observed samples, not
percentiles, cold-machine startup, CPU-only timings or hard real-time guarantees.
Its 270-byte HTML has 16 nodes/15 edges, 152×1920 logical geometry, 63 SVG elements
and 410 startup patches. The admitted direct Node/QuickJS library+layout sample
was 65.892ms, with 165599 evaluated library bytes, 768076 guest bytes, 2318 work
units, 9244 output bytes and area 291840. Component heap is not complete guest DOM
memory or browser RSS. Mandatory browser engine quotas and worker termination
controls passed in all applicable selections without changed limits.

The final admitted run includes the actual-child confirmed unsent repair
(24.30s call) and separate parent/all-children same-origin restart/revocation
(68.09s call), alongside native/served create-update, two-browser isolation,
archive recovery and package closure. Their assertions retain the distinctions
described below: source acceptance versus preview success, explicit previous
revision versus silent fallback, and a confirmed composer draft versus sending.
Independent task and whole-branch review remain before Backlog completion.

## Historical qualification — 2026-09-07, admission withdrawn

At that checkpoint, **the release gate was blocked and V2 disabled.** The corrected
candidate selection passed, but the final admitted selection failed four tests.
Admission was rolled back; no release qualification is claimed. This record supplements the historical
[V1 verification](V1_VERIFICATION.md) and [Mermaid spike](V2_MERMAID_SPIKE.md).

## Identity and environment

Disabled candidate: `canvas-v2-mermaid-1`; manifest SHA-256
`17717bcab7c7bba4a28e0069354f6ecbf895d2ca58f4b8d1c0355b7726e2f466`.
Before first admission, the misleading `mermaid_candidate.qualification` field
was removed: the catalog alone owns execution policy. This metadata-only change
replaces candidate manifest `39291c03e34cd9fc9f1676dbd0a816ace68357d91b0f75a4fe958f326c762959`
without changing executable/library bytes. Build identity is now
`5cdfdfcf09ed257bce900fa94472cc9ff79eb58506ec86e0a55ce0cd2d7e95d8`.
Restored disabled policy identity is
`15430aedabd3179e8f0764611b8f6ed129f6b381a9469d5f20109f3b133d22d2`;
V2 is non-executable and the diagram default is null, with V1 unchanged.
The tested but withdrawn admission policy was
`cd4f0cdd756732e686b05031ce12c6bd086473cc72ff2f9d58340d8528b40f15`.
The V1 manifest/worker/renderer/engine remain byte-identical to the branch base.
Python 3.12.11, Node v26.0.0, Ruff 0.16.6, Chromium 151.0.7922.34,
Playwright 1.58.0, Textual 8.2.8, aiohttp 3.13.5;
`macOS-26.5.2-arm64-arm-64bit`. Tests use the repository's pre-import pytest
isolation, synthetic provider responses, owned SQLite/config, disposable TLS,
loopback listeners and browser contexts. No user database/provider/process or
installed user package is modified. No full repository suite is claimed.

## Commands and scope

All commands run in the `canvas-v1` worktree with `../../.venv/bin/python`.
Required browser/listener runs use approved sandbox escalation, not skips.
The final two mandatory selections are:

```sh
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Canvas/browser/test_canvas_zero_egress.py Tests/Canvas/browser/test_canvas_mermaid.py Tests/Canvas/browser/test_canvas_native_flow.py Tests/Canvas/browser/test_canvas_served_flow.py Tests/Canvas/browser/test_canvas_quota_probe.py
TLDW_CANVAS_MERMAID_INPUT_DIR=/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/canvas-mermaid-inputs-4i3rl_al TLDW_CANVAS_RUNTIME_ARCHIVE_DIR=/private/tmp/canvas-runtime-task-1.3-inputs ../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Canvas Tests/Chat/test_console_canvas_controller.py Tests/Agents/test_canvas_tool_provider.py Tests/Chat/test_console_message_actions.py Tests/Chatbooks/test_chatbook_canvas_round_trip.py Tests/Packaging/test_canvas_gateway_distribution.py Tests/Web_Server/test_canvas_control_spawn.py Tests/Web_Server/test_canvas_kill_switch.py
```

The browser-only command first qualifies the candidate. The broader command
includes that entire browser selection and runs before and after admission.
Recorded results:

| Selection | Result |
| --- | --- |
| Complete five-file browser candidate selection | 176 passed, 2 optional skips, 390.00s |
| Packaging + runtime assets, both offline inputs | 34 passed, 13.32s, no skips |
| CI workflow contracts | 19 passed, 0.53s |
| Useful eight fixtures + V1/V2/native adversarial corpus | 11 passed, 40.53s |
| Event publication/freshness controls | 12 passed, 2.38s |
| Source-only retirement + stale async recovery | 7 passed, 7.64s |
| Actual-child failed preview and confirmed unsent repair | 1 passed, 18.53s |
| Initial separate parent/all-child policy replacement | 1 passed, 50.90s; superseded by strengthened checks |
| Metadata/policy reproduction controls | 8 passed, 3.29s |
| First broad candidate selection | 9 failed, 1340 passed, 2 optional skips, 585.19s |
| Inherited integration correction group | 15 passed, 4.66s |
| Strengthened parent/all-child replacement | 1 passed, 49.38s |
| Forced owned-process-group cleanup | 1 passed, 1.58s |
| Complete corrected candidate selection | 1350 passed, 2 optional skips, 578.29s |
| Full package/source closure + runtime + CI contracts | 61 passed, 16.22s, no skips |
| Final admission assertions against disabled catalog | 2 failed, 0.57s (required RED) |
| Same assertions after exact catalog admission | 2 passed, 0.51s |
| Full admitted selection (admission subsequently withdrawn) | 4 failed, 1346 passed, 2 optional skips, 539.80s |
| Exact affected browser diagnostic selection | 2 failed, 6 passed, 48 deselected, 213.84s |
| Bounded native SQLite/faulthandler diagnostic | 1 failed, 3 passed, 52 deselected, 107.15s |
| Restored production-off profiles + offline runtime reproduction | 77 passed, no skips, 5.85s |

The final focused command was the two offline environment assignments above plus
`../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Canvas/test_profiles.py Tests/Canvas/test_runtime_assets.py`.
Its first un-escalated invocation passed 76 tests but could not bind the owned
redirect-refusal loopback server; approved escalation resolved that permission
failure. This focused GREEN does not supersede the failed release selection.

The four admitted-run failures were a snapshot-mismatch fixture that no longer
differed after admission (corrected to a fixed distinct test-only revoked policy),
a present-but-hidden Revision 2, a held publication read returning
`child_not_connected`, and a served adversarial navigation returning 503.
All previously fixed deterministic publication/recovery regressions and the
separate parent/all-child restart gate passed in that run.

Subsequent bounded diagnostics captured owned Python children exiting with
SIGBUS: macOS reported SQLite WAL recovery/frame lookup and `FS pagein error: 22
Invalid argument`. Faulthandler then captured the crashing Python operation in
Console trace-maintenance marking SQL, alongside concurrent workspace and
conversation reads. This is not a proven Mermaid, transport, fixture-cleanup,
or environment-only cause. Earlier untraced failures are not retroactively
attributed to it. No shared storage/security fix, WAL/mmap policy change, native
dependency change, or passing-rerun waiver was applied. A separately authorized
causal SQLite concurrency investigation is required before admission resumes.

The baseline RequestsDependencyWarning reports urllib3/chardet/charset_normalizer
version compatibility. Initial sandbox Chromium MachPort denial and denied
listener bind were environment failures, not product REDs or passing evidence.
Required Chromium failure behavior remains mandatory. Firefox and WebKit each
skip because the browser is not installed in this CI/worktree; Chromium is the
mandatory gate. These are cross-engine coverage gaps, not missing-cache skips.
The first broad run also emitted a fixture SyntaxWarning for an invalid escape;
the corrected full run emitted only the RequestsDependencyWarning. Neither
warning is represented as a failing security/resource result.

## Security, budgets, and usefulness

The canonical generated HTTP/WebSocket/navigation/popup/download/worker corpus
runs in both V1 and V2 against native-realm sentinels and an independent egress
listener, with positive benign render/interaction controls. Additional diagram
cases refuse directives (`unsupported-syntax`), markup (`unsupported-label`),
oversized graphemes (`label-limit`), dense DAGs (`edges-limit`), and hostile parser
lexemes (`unsupported-syntax`). Private handle isolation, prototype attacks,
multiple diagrams and failed startup are covered by the V2 browser tests.
Failed startup applies no partial diagram mutations or authored script effects.
A test-only infinite worker startup injection produces `worker-unresponsive`,
exactly one termination, no SVG and no post-start egress; it is watchdog testing,
not evidence that an arbitrary exception is an acceptable quota outcome.
The unchanged renderer backstops are 750ms for worker startup and250ms for
worker events, distinct from the guest's250ms/50ms execution interruption limits.

Shared ceilings remain 512 KiB HTML, 256 KiB total evaluated scripts (including
the library), 32 MiB guest heap, 512 KiB stack, 250ms startup, 50ms events,
100 pending jobs, 1800 DOM nodes, 900 CSS rules and 500 patches/operation.
Diagram/document ceilings respectively: input 8192/16384 bytes, nodes16/24,
edges24/32, participants6/8, messages16/24, notes8/12, labels4096/8192 bytes,
SVG elements250/400, output49152/65536 bytes, work10000/20000 units and
area4194304/8388608. Maximum four declarations, 512 bytes per label,
2048×4096 per diagram. No V1 cap or CSP/zero-egress restriction is raised.

Full-browser fixtures include six-node branch/rejoin, three participants with
all note placements, the two exact shipped authoring examples, four diagrams,
mixed HTML/flow/sequence plus a working Count button, Unicode/RTL/emoji/combining
labels, and a 16-node chain. Screenshots at 1600px and390px were inspected by
the root reviewer. Intrinsic geometry scrolls horizontally at narrow widths;
tests drive scrollLeft to its far edge, and scroll first/last diagrams into view.
The four-diagram screenshot alone is not evidence that all four are shown:
the DOM asserts four SVGs and exercises vertical reachability. Inherited40px
serif CSS does not override explicit16px monospace defaults; explicit22px text
restyling works without relayout. Unicode fallback glyphs/RTL wrapping were
visible locally; cross-font or cross-platform pixel parity is not claimed.

Near-limit full-browser measurement: 270 source bytes, 16 nodes/15 edges,
152×1920 logical geometry, 63 SVG elements, 410 startup patches, 85.12ms from
load invocation to ready in one fresh-context sample. This is not a percentile
or hard real-time guarantee. Separately, the Node/QuickJS library+layout probe
records 165599 evaluated library bytes, 768076 guest bytes, 54.95ms, 2318 work
units, 9244 scene-output bytes and area291840. Component guest memory is not
browser RSS or the complete virtual-DOM guest heap. The unchanged real-Chromium
quota probe separately exercises accepted/rejected engine allocations, stack,
startup/event interruption and exact500/501 patch boundary; it passed in the
five-file browser gate.

## Product path and lifetime distinctions

Mounted native/served tests exercise production gateway, authority, compiler and
browser routing in pytest. Actual TldwCli child tests additionally use real
Console provider finalization and composer state across AppService IPC; their
parent normally remains inside pytest. The dedicated restart gate instead runs
a separate owned parent and two real children, retains a live old load and a
pending receipt, stops them, checks all three old PIDs are gone, then replaces
the parent at the same origin with a fixed test-only revoked policy. The new
process reopens the same durable V2 revision as exact inert source/history and
explicitly creates a separate allowed V1 Canvas without rewriting the old rows.
Old shell/load and pending confirmation are refused. This is OS-process policy
replacement, not installation of a new distribution; packaging evidence is separate.
The controller creates a separate owned process session/group, verifies both
children belong to it, and uses bounded group cleanup only on timeout after
checking ownership. The forced-cleanup regression verifies both children and
parent disappear while preserving timeout as failure, not passing qualification.
Candidate/revoked wrappers exist only under Tests; no product environment or
archive can enable a profile.

The gate exposed a distinct new-Canvas event-publication race: trusted child
selection advanced while parent epoch stayed0 on old Canvas. The correction
reuses authoritative snapshot reconciliation only within the same captured
child/session/live shell before and after await. Mismatched events remain
discarded. Only proven advanced live epochs map to409; malformed, unchanged,
sibling/session, disconnect and child-rebind cases remain fail-closed. Another
deterministic regression verifies a valid new plan retires old source-only modal
and inert state. Neither correction forwards browser-selected authority or adds
generic retries. Earlier untraced failures are not retroactively explained by
these traces, and a passing rerun is not used as remediation.

The first broad candidate run also exposed stale scheduling/repository fake
owners without captured profiles, HTML block references without language, and
a served-startup fixture advertising obsolete protocol1. Fixtures now reflect
the current strict interfaces; production adds no fallback. A genuine inherited
eager compiler import in preparation was moved to first compilation, preserving
the unchanged fresh-subprocess startup guard. The new restart test's relative
renderer URL request was corrected to its owned origin. The focused correction
group and complete corrected candidate run passed. The later admitted run did
not pass and supersedes that result as the release decision.

Actual-child recovery saves the failed cycle revision, opens exact source,
explicitly views its prior revision, then confirms a source-free repair hint
into the actual unchanged composer. A hash/byte receipt and unchanged provider
call count prove the draft remains unsent. Native desktop application behavior
outside the existing mounted native/Console paths is not additionally claimed.

## Packaging, reproducibility, archives and CI

Wheel and sdist tests verify byte-exact V1/V2 manifests, catalog, engine,
worker/renderer, Mermaid JSON, both notice files, all authored build modules,
input inventory and shipped authoring guide. Wheel zip-import loads the full
verified snapshot/closure and guide without checkout package imports. Both
vendor builders regenerate twice into independent owned directories from verified
offline inputs, comparing every generated byte against checkout. Real browser
execution uses those checkout bytes with no generated egress; distribution
closure equality establishes the relationship, not a separate browser launched
from an installed wheel. Mermaid11.17.2/Jison0.4.18 and Unicode16.0.0 notices,
hash inventories and bounded build inputs remain verified; the full upstream
renderer/dependency graph is not bundled or exposed.
Rebuild policy tests retain unchanged admitted and revoked entries; changed
manifest, library, or notice inventory disables the candidate/default. The
builder has no runtime-enabling flag and preserves revocation rather than
silently re-admitting an exact profile.

Canvas exports use actual ChatbookCreator/ChatbookImporter format3.0 for single
and multiple conversations, preserving source/profile/history without executable
runtime assets or policy installation. Schema68/archive3.0 and sync exclusion
are unchanged; no legacy text/JSON Canvas exporter is introduced.
The existing broad non-UI core CI lane already collects Canvas browser tests;
it now installs mandatory Chromium after Playwright dependencies and before
pytest. Its contract test preserves collection and required-failure behavior;
unrelated lanes/sharding remain unchanged. Hosted CI itself is not run locally.

## TASK-31942 SQLite correction qualification — 2026-09-08

**Task 7 is delivered with concerns; the correction gate remains unqualified and
Canvas V2 remains disabled.**
Task 7 adds installed-wheel isolation and a UI-loop/thread boundary regression,
then exercises the exact storage and five actual-child selections. The tracked
test/docs change contains no runtime API, admission, budget, or production-code
change.

The wheel was built with the existing offline toolchain, installed with
`--no-index --no-deps` into a pytest-owned target, and its 15-file fixed-helper
closure matched the wheel and checkout byte-for-byte. The installed absolute
entry ran with `-I -S` from hostile cwd/`PYTHONPATH`: plain close, real prepare,
and fixed TTS initialization all completed with clean stderr. A supplemental
test-owned import/file audit records every loaded module name separately from
the product-file origin mapping. It admits only the fixed helper's approved
product-module closure and Python's standard-library top-level modules, thereby
excluding keyring/loguru/Textual and every provider SDK. It found no forbidden
import, resolved absolute open under trap roots, or poison sentinel access.
Controlled `ModuleType` names prove rejection of app, config, keyring, loguru,
Textual, two external provider SDKs, and provider-bearing LLM Calls, provider
catalog, Chat, Agents, and TTS namespaces without importing real credential or
provider code. Deleting an installed leaf produced source-free
`helper_unavailable`; hostile roots could not rescue it. The final audit-fix
packaging selection passed 19 tests with the existing dependency warning.
Independent task review identified the incomplete module trace; fix-only
re-review of `4af790d46..73692f21c` confirmed that finding addressed with no new
Critical/Important breakage. This is task-scoped review, not the still-pending
whole-correction review or a release qualification waiver.

The exact 27-file affected command and exact five-node Canvas command both
failed at collection because the shared virtual environment's
`tldw_profile_core` editable `.pth` targets an absent old worktree. This is a
stale environment install, not a new helper-wheel omission; the shared venv was
not repaired. The 26-file continuation completed with 1,483 passed, 5 skipped,
and 36 failed: three known strict-inventory deltas, eleven spawned cases blocked
at `multiprocessing.Event` creation by host semaphore ENOSPC, and 22 app/perf
subprocess nodes blocked by the absent package. No completed-continuation node
was unrun. A local-source diagnostic passed the omitted interop file (26/26),
but nested performance subprocesses deliberately replace `PYTHONPATH`, leaving
the ADR-097 972-module census and related import/payload ceilings unqualified.
No ceiling was changed.

With the same five Canvas node IDs and an explicitly labeled local-source path,
the required actual-child tests passed under approved owned-loopback/browser
execution: 5 passed in 137.13s. The initial sandbox bind denial is environment
evidence, not behavioral RED. Existing fixtures retained their child
fault/lifecycle captures; no disconnect or crash was treated as successful
refusal.

Five-sample benchmarks used identical current/baseline workloads in separate
owned roots and imported the pytest isolation bootstrap before every product
import. Current versus immutable `9bc73ffb3` medians were: cold app import
826.859/712.696ms, actual app UI-ready 8,157.564/8,111.325ms, and threaded TTS
repository open 142.452/6.428ms. The baseline predates the fixed-helper API, so
helper values are intentionally unavailable there. Current fixed prepare,
retained-child readiness, and repeated proof-recheck medians were respectively
43.392ms, 43.658ms, and 0.239ms. Test-owned 0.5ms sampling across operation and
cleanup boundaries observed repository high water of +9 FDs/2 live helpers and
fixed operations +3 FDs/1 live helper; these are sampled observations, not an
exact kernel-instantaneous FD maximum.

Local qualification used macOS arm64, CPython 3.12.11 and SQLite 3.49.1.
Python 3.11, Windows, and Linux were unavailable and are not claimed as passed.
An invalid first benchmark launch imported product configuration before owned
isolation and produced no metric; its bounded audit is preserved separately,
not promoted into qualification. The full command/result/failure accounting is
in the ignored Task 7 report. Whole-correction review, Backlog completion, and
the Task 8 candidate/admitted rerun remain separate controller decisions.

### Authorized environment repair and fresh checks — 2026-09-08

The user subsequently approved replacing only the stale `tldw_profile_core`
editable install with a locally built wheel. Committed source at `033c5a949`
was archived into an owned temporary directory and built with the existing
offline toolchain. Pip installed version 0.1.0 using `--no-index --no-deps`;
all 261 distribution names/versions remained unchanged. The installed package
and schema/fixture files match the archived source, and a fresh isolated import
resolves from `.venv` rather than a worktree. Old installation metadata and the
wheel are retained at `/private/tmp/tldw-core-repair.fNKBoL/`.

Without a `PYTHONPATH` workaround, the interop file and four performance files
returned **56 passed, 1 failed, 4 warnings in 33.57s**. The missing-package
failures are resolved; the UI-ready guard now reaches its assertion and measures
**979 modules against 972**. A fresh run of that unchanged guard on the immutable
pre-correction `9bc73ffb3` archive measured **973 against 972** (1 failed,
1 warning, 14.28s). Six newly loaded SQLite/proof modules account for the observed
correction delta, on top of the baseline's one-module breach. No limit or pinned
snapshot was changed. Boot-import weight passed at 641/660 modules; screen
pre-import passed at 500/500 modules and 366176/378740 LOC.

The **exact five Canvas nodes now pass without a source-path override**:
5 passed, 1 existing dependency warning, 134.49s. This supersedes their earlier
environment-blocked qualification, not the complete affected-selection gate.
Known semaphore/inventory failures and the measured startup-budget breach remain;
whole-correction review and V2 admission are still pending. The repair report and
raw Canvas/baseline logs are preserved with the Task 7 evidence.

## Authorized startup-budget repayment — 2026-09-08

TASK-31942 Task7b defers pure TTS profile-repository construction through the
existing app-owned first-use method. The configured path is still captured at
app construction, concurrent callers share one owner/open task, and close latches
even before first use. ADR-028 clarifies construction timing; ADR-097 budgets and
ADR-125 native SQLite admission/finalization remain unchanged.

The initial cut reached 963/972 at UI readiness but shifted the same 16 modules
onto screen preload, breaching 516/500. The required guard caught this. A scoped
profile-library import deferral now loads the genuine voice-bundle choice class
only at the existing user-decision helper; postponed annotations do not load the
portability/repository implementation. No service, repository, Canvas owner,
budget constant or snapshot was changed.

Implementation commits: `bd96a923c4`, `ae629ed080`. Targeted results:

- App ownership: 44 passed; an inherited shutdown-order fixture was reconciled
  with explicit recording hooks, without modifying production shutdown.
- Native helper/repository lifecycle and import closure: 246 passed, including
  the actual-app ordinary/abrupt exit gates with real first-use construction.
- Existing bundle UI selection: 59 passed, 104 deselected.
- Final performance/provenance/Canvas-startup selection: 50 passed, 5 warnings,
  65.92s. Import625/660; UI-ready963/972; preload499/500 modules,
  364325/378740 LOC and110163/123319 largest-route LOC.
- Controller check on committed `ae629ed080`: 5 passed, 3 warnings, 21.78s.
  This independently confirmed both UI/preload counts, real import/first-choice
  behavior, and ordinary idle/partial-setup native exit preservation.

The annotation-only follow-up had its own two-test passing check. Changed-span
static checks found no introduced findings; 474 inherited Ruff findings and four
pre-existing formatter-dirty files remain, so aggregate lint is not green.
Warnings include existing dependency/syntax issues, intentional budget-headroom
notices and the known host joblib serial fallback. Screen preload has only one
module of remaining headroom. No dependency/host repair or full suite was run.

Task-scoped independent review approved spec compliance and quality with no
Critical/Important findings. The startup-budget breach is
repaid, but known semaphore/inventory gaps and Canvas V2 admission remain
separate, incomplete gates. The subsequent whole-correction review and fix wave
are recorded below. TASK-31942 remains In Progress and
V2 remains disabled. Exact commands, RED/GREEN evidence and scoped static results
are preserved in this plan's `task-7b-report.md` and controller verification.

## Whole-correction review and bounded fix wave — 2026-09-08

The independent review of `9bc73ffb35..41f144ab90` inspected all 74 changed
files. It found no Critical issue, one Important incomplete correction and two
test Minors. Live-open cleanup could replace an original control-flow signal
when native close also failed. An isolated real-store/helper probe reproduced
the lost signal and then safely settled the retained owner. The replacement
pattern also existed at BASE; this is not a demonstrated new startup regression
or data-loss event. The two test defects were a borrowed-close assertion that
could not detect close and a UI-loop guard that skipped teardown on failure.

SQLite Task8 (not Canvas admission Task8), committed as `3b5031012c`, preserves
the selected exact signal and hands its complete live cleanup owner to the
repository after reservation settlement. Ordinary cleanup errors retain their
existing type. Retryable healthy proof and terminal proof loss retain distinct
classifications, native policy, SHARED lease and worker ownership. Guarded
exception-local metadata preserves earlier owners on sequentially reused signals
without stale adoption; settled wrapper references can remain for that exception's
lifetime, while live helpers remain subject to the existing four retained slots.
The two test guards now have explicit wrong-behavior sensitivity controls.

Verification on macOS arm64, CPython 3.12.11 / SQLite 3.49.1:

- Ten-file covering selection: **343 passed, 5 warnings, 260.93s**.
- Fresh final focused selection: **22 passed, 1 warning, 5.51s**, including the
  terminal-control parameter added after covering collection. Other changes
  during that run were static-only; no production runtime behavior changed.
- Controller committed smoke: **8 passed, 1 warning, 1.98s**, covering exact
  cancellation, hostile reuse, actual repository retries, both borrowed-close
  controls, combined UI-test failure cleanup and terminal control/proof loss.
- Budgets unchanged/pass: import 625/660, UI 963/972, preload 499/500 modules;
  preload 364325/378740 LOC and largest route 110163/123319 LOC.
- All six changed files pass formatting. BASE-mapped lint finds zero introduced
  diagnostics and 128 inherited findings; aggregate lint is not green. The
  controller's full correction-range whitespace check also passes.

Warnings remain the inherited Requests dependency mismatch, joblib's host
ENOSPC serial fallback and budget-headroom notices. These results are separate
runs, not counts summed into full affected-selection qualification. Exact
commands, RED failures (including discarded harness errors), cleanup accounting
and freshness limits are preserved in `task-8-report.md` and
`task-8-root-verification.md` in the existing plan's SDD directory.

The single independent fix-only re-review confirms **I1, M1 and M2 addressed**,
with no new Critical/Important issue identified. It records one limitation:
simultaneous reuse of the same exception instance by independent workers could
overwrite the current-owner slot. No such producer was identified in the
inspected runtime paths; concurrent same-object reuse was not part of the
approved sequential-reuse cases and is not claimed as supported. The controller
retains this as a documented limitation, not a demonstrated blocker or permission
for another automatic fix wave. A future shared-signal producer would require
an attempt-bound handoff and new coverage before using this interface that way.

Code-review completion does not close the broader qualification gaps. The
three known strict-inventory failures, eleven pre-body SemLock ENOSPC cases,
platform/optional coverage and final affected-selection/benchmark qualification
remain unresolved. No full suite, host cleanup, dependency repair, external PR
action or Canvas admission was performed in this wave. TASK-31942 remains
In Progress with all seven final acceptance criteria unchecked; V2 is disabled.

## Separately authorized inventory and host diagnosis — 2026-09-08

After the prior review-fix checkpoint `50a5700422`, the user approved repairs for
the three inventory failures and read-only diagnosis of the semaphore failure,
without deleting host resources. SQLite Task9 records this bounded continuation;
it does not reopen the completed Task8 review wave or authorize Canvas admission.

The exact three inventory nodes reproduced their failures: an unqualified legacy
Collections read-only connection, a trace-maintenance connection using another
module's owner ID, and the existing quiescent backup override missing from the
coarse call census. Commit `a6388ed5c4` fixes the two module-owned connections,
preserves legacy recovery's lexical no-follow/read-only source boundary and
transaction-setup cleanup, and qualifies the exact existing native backup
delegation with real reservation tests. C55/C56 are appended without reusing
retired IDs or adding backup authority. Existing ADR-029/113/125 apply.

Current targeted evidence on macOS arm64, CPython 3.12.11 / SQLite 3.49.1:

- New legacy RED: 5 failed, 2 passed; focused GREEN: 7 passed. Backup/inventory
  RED: 1 failed, 8 passed; GREEN: 9 passed. Unchanged backup behavior has real
  success/abort coverage plus five scanner-mutation controls.
- Original exact three inventory nodes: 3 passed, 1 warning, 39.16s.
- Seven-file covering selection: **426 passed, 2 skipped, 26 failed**, 75.11s.
  This is not all-green. All 26 reproduce with identical assertions on immutable
  BASE `50a5700422`: 26 failed, 1 warning, 4.95s. Twenty-two are core-owner tests
  with stale kwargs/parent-process assumptions or global path tripwires reached
  by helper startup; four compaction/admission cases return `vacuum_failed`.
  The import-verified BASE archive is preserved at
  `/private/tmp/task9-base-control.wN9FIV`. They remain qualification gaps.
- Final implementation smoke: 12 passed, 1 warning, 40.96s. Independent root
  committed smoke: 8 passed, 1 warning, 40.54s, covering all original failures
  and critical real privacy/cleanup/backup behavior.
- Fresh unchanged startup guards: 3 passed, 4 warnings, 12.70s. Counts remain
  625/660 at import, 963/972 at UI readiness, 499/500 on preload;
  364325/378740 total LOC and 110163/123319 largest-route LOC.
- Current and BASE each have the same 58 Ruff diagnostics and one inherited
  formatter-dirty console file; no introduced diagnostics. Whitespace checks pass.
  Aggregate lint/format are not green. Warnings are the inherited Requests
  version mismatch and intentional startup-headroom notices.

These are separate scoped runs, not a summed whole-suite result. Independent
task review approves spec compliance and quality with no Critical/Important
finding; the two Minors are the documented inherited warning and static debt,
not permission for shared-dependency repair or broad formatting. Exact commands,
RED/GREEN, BASE comparison and root
checks are preserved in this plan's `task-9-report.md` and
`task-9-root-verification.md`. No broadening into the 26 baseline failures was
performed.

The host diagnosis completed independently of Chatbook. An isolated stdlib
spawn-lock allocation failed with errno 28 both inside and outside the sandbox.
The exposed named-semaphore maximum is 10000; the observed 48 open handles do not
measure cached names or identify which process created the exhausted capacity.
Disk space was low but still had about 21 GiB available. No processes, semaphore
names, kernel limits, dependencies or user files were changed. Exact commands,
primary-source interpretation and attribution limits are in
`Docs/superpowers/reviews/2026-09-08-semaphore-allocation-diagnosis.md`.

The eleven spawned tests remain unqualified. Resume them only after the isolated
allocation control passes on a clean runner or after a user-coordinated host
restart. No restart or cleanup is authorized here. Platform/optional coverage
and final affected-selection/benchmark evidence remain outstanding; V2 stays
disabled and TASK-31942 stays In Progress.

Task9 and the read-only diagnosis close only the separately approved scope.
TASK-31942 AC8 (three inventory checks) and AC9 (host diagnosis without resource
mutation) are checked; the original seven final ACs remain unchecked. The 26
baseline failures, eleven host-blocked cases, platform/optional qualification and
final affected-selection/benchmark remain open. No full suite, cleanup, reboot,
dependency change, PR action or V2 admission was performed by this continuation.

## Approved baseline-test and maintenance repair — 2026-09-08

SQLite Task10 implements the separately approved diagnosis from
`Docs/superpowers/reviews/2026-09-08-sqlite-baseline-failure-diagnosis.md` in commit
`ad5c02e5a8`. Exactly four code/test files changed. The 22 core-owner failures were
stale observations of the helper boundary; production owner/privacy policies
remain unchanged. Dedicated trace-maintenance connections now install the same
real deterministic Canvas payload validator as ordinary connections and close an
acquired handle if setup fails. No semantic or deletion authority is added.
Existing ADR-029/097/121/125 apply; no new ADR or architecture is introduced.

Evidence on the existing macOS arm64 / CPython 3.12.11 / SQLite 3.49.1 runtime:

- Original owner tests: 22 failed before repair, then 22 passed. New compaction
  regressions: 6 failed before product edits, then 6 passed. All four originally
  failing compaction/admission nodes pass. Deliberate negative controls detect
  incorrect owner/path/helper/backup observations, omitted or permissive
  validators, added deletion authority and omitted setup cleanup.
- Final focused implementation selection: **115 passed, 1 warning, 22.69s**.
- Seven-file covering run: **479 passed, 2 skipped, 2 failed, 146.88s**. Both
  failures were gateway startup tests. Exact sandbox isolation reproduced them;
  an owned stdlib loopback bind failed with EPERM. The same two tests passed with
  the required loopback permission: **2 passed, 1 warning, 1.48s**. This does not
  rewrite the original covering run as all-green. Skips remain Windows-only.
- Independent root committed smoke: **31 passed, 1 warning, 9.30s**, covering
  all 26 original cases plus populated Canvas integrity, malformed payloads,
  denied authority and setup-close behavior.
- Fresh unchanged startup guards: **3 passed, 4 warnings, 16.22s**. Counts are
  boot625/660, UI963/972 and preload499/500 modules; 364325/378740 total LOC and
  110163/123319 largest-route LOC. Thresholds and snapshots are unchanged.
- Aggregate Ruff remains 599 diagnostics and three legacy files remain
  formatter-dirty. The first occurrence-preserving rule/message comparison could
  hide offsetting occurrences; the subsequent changed-range audit identified no
  new diagnostic and reproduced import-block findings from exact BASE blobs.
  Whitespace is clean. Aggregate lint/format are not green; Requests and budget
  warnings remain inherited or intentional, not permission for dependency edits.

Independent task-scoped review approves spec compliance and quality, with no
Critical or Important finding. One new Minor is deferred: wrap the setup-failure
test's database lifetime in `try/finally` so failed assertions also clean up its
test-owned registry. The other Minor is the disclosed inherited warning/static
debt. Root resolved the review's memory-bypass verification item by inspecting
the unchanged memory branch and its covering tests; this does not qualify the
explicitly skipped Windows or broader host/platform cases.

Exact commands, negative-control
results, baseline-attribution limits and committed checks are in this plan's
`task-10-report.md` and `task-10-root-verification.md`. Runs are separate evidence,
not summed whole-suite qualification. The diagnostic comparison archive and
earlier evidence remain preserved.

No full suite, host-resource cleanup/restart, dependency change, external PR
action or Canvas V2 admission was performed. The eleven host-blocked spawned
tests, platform/optional evidence and final affected-selection/benchmark gates
remain open. TASK-31942 stays In Progress and V2 stays disabled.

AC10 (helper-aware owner tests) and AC11 (physical compaction/admission repair)
are checked alongside the prior scoped AC8/9. The original seven final acceptance
criteria remain unchecked. No broader completion or admission is claimed.

## Final local qualification continuation — 2026-09-08

Task11 closes the reviewed setup-failure test teardown Minor in `55f74aa009`.
Real failure-path RED/GREEN, 20 covering passes and independent spec/quality
approval; root committed control and original test each pass. AC12 is checked.

At that checkpoint the37-file affected selection reports **1844passed, 5skipped,
11explicitly deselected, 9warnings**, with unchanged startup budgets. Separate
bundle consumer checks report59passes and the local schema parameter named
`live` reports1pass. These are separate targeted runs, not whole-suite evidence.

Fresh actual Canvas children report **4passed, 1failed**. The read-publication
case did not acknowledge the restored card action after reconnect. One unchanged
single-node rerun failed earlier at terminal first-byte readiness, leaving the
original boundary unresolved. Both lifecycle captures are preserved. Read-only
diagnosis suggests a readiness/action-delivery race but does not prove root
cause or attribute a SQLite crash. New bounded test diagnostics await approval;
no retries, timeout/assertion changes or production edits were used to mask it.

Five-sample checkout benchmarks completed: UI-ready median8814.697ms current
versus8525.682ms baseline; threaded repository open192.666ms versus13.214ms.
Both harness logs contain errors; timings are not pristine lifecycle evidence.
Final53-file static checks remain nonzero (1373Ruff diagnostics,12formatter-dirty
files); whitespace checks pass. No limits or dependencies changed.

Exact commands, results, warning/skip attribution, benchmark/resource limits and
diagnosis disposition: [final local qualification](../superpowers/reviews/2026-09-08-sqlite-final-local-qualification.md).
TASK-31942 remains In Progress and V2 disabled. Host semaphore, platform/optional,
nonzero static and fresh Canvas readiness gates remain explicit; no PR/external
action, host cleanup or full repository sweep occurred.

## Approved restored-card diagnostic spike — 2026-09-08

One source-free, test-only instrumented run reproduced the recovery failure
(`1 failed, 1 inherited warning, 59.36s`). Both F10 completion and F12 entry
46ms later observed zero Canvas cards and pending/coalesced UI sync. The
synthetic adapter's immediate lookup failed before real card dispatch; this run
does not demonstrate a native SQLite crash. Its first-byte observation was
4.883s after login, but the separate earlier startup failure remains unexplained.

All temporary test edits were reversed and both files verified byte-identical
to baseline. Exact patch, metadata and limits are preserved in the
[diagnostic report](../superpowers/reviews/2026-09-08-canvas-card-readiness-spike.md).
A retained harness readiness fix is not part of the spike. No gates are relabeled
passing, no new broad runs or host changes occurred, and V2 stays disabled.

## Approved restored-card harness correction — 2026-09-08

The separately approved test-only correction is committed in `72af5b63bd`.
The F12 adapter now waits for and captures the mounted current-session exact
card/button, then presses once within the existing receipt-wait budget. The
served-flow assertions and all production files remain unchanged.

Root verification: focused committed selection **10 passed, 1 inherited warning,
3.10s**; original actual-browser case **1 passed, 1 inherited warning, 49.52s**,
including restored pin/metadata/provider-count checks and owned cleanup.
Changed-file Ruff/format and whitespace pass. Independent scoped spec/quality
review approved with no Critical/Important findings; warning remains documented.
See the [correction evidence](../superpowers/reviews/2026-09-08-canvas-card-readiness-fix.md)
for exact commands and the browser reporting-wrapper limitation.

This closes the reproduced premature-card-action harness failure for the tested
workflow, not the separate historical startup failure or other host/platform/
static qualification gaps. TASK-31942 remains In Progress and Canvas V2 disabled.

## Approved shared startup deadline — 2026-09-08

Task13 (`235641b380`) replaces the initial implicit5s first-output and explicit45s
Composer waits with one user-approved shared45s deadline. Both conditions remain
required; only positive remaining time is passed, with no reset/retry. Later
workflow assertions, F12 behavior and all production budgets are unchanged.

Root actual-browser case: **1 passed, 1 inherited warning, 46.86s, exit0**.
Root committed focused file: **5 passed, 1 warning, 2.37s, exit0**, including
real DOM evidence. Independent scoped spec/quality approved without Critical/
Important findings. Ruff/new-file format/whitespace pass; existing served-flow
formatting debt is unchanged by a baseline/current edit-content comparison.
See [deadline correction evidence](../superpowers/reviews/2026-09-08-canvas-startup-deadline-fix.md).

This is an explicitly revised test contract, not a production speedup or proof
of the historical startup delay's cause. Prior evidence and remaining host,
platform/optional and aggregate static gates remain; task In Progress, V2 off.

## Static baseline attribution — 2026-09-08

The remaining static failures now have a paired source comparison at d599fa037b:
1373 current diagnostics versus 1411 baseline diagnostics under the same py312
target. Of the current diagnostics, 1360 match unchanged complete source spans;
the remaining 13 are ten modified previously failing import blocks, two updated
version guards and one relocated alias. The modified blocks are not blanket-
certified as correctly ordered. All 307 formatter edit groups in the twelve
previously failing files match baseline edit content and neighboring lines.

See [static attribution](../superpowers/reviews/2026-09-08-sqlite-static-attribution.md)
for the language-floor control, exact scope and limits. No fixes, suppression,
pytest/browser/benchmark replay or static-gate waiver; task In Progress, V2 off.

## Fresh macOS concurrency CI attempt — 2026-09-08

Task14's reviewed two-file workflow/test implementation is committed in
`13fe672c64`. Root committed contract checks: **15 passed in 1.12s**, no warnings;
scoped spec/quality review approved without findings. The user-approved push
created only the dedicated evidence branch, not a PR or main/dev update.

[Run34292597991](https://github.com/rmusser01/tldw_chatbook/actions/runs/34292597991)
failed during setup: GitHub has no macOS ARM64 Python3.12.11 package. The lock
control and all11product cases were unrun; no runtime metadata/JUnit/artifact
exists. Missing evidence was rejected, not accepted as a pass. A CI-only
3.12.10 pin is available in the official manifest and proposed, not implemented
or rerun. See the [full attempt evidence](../superpowers/reviews/2026-09-08-sqlite-macos-concurrency-evidence.md).

No local host/dependency change, local semaphore probe, broad run or release-gate
waiver. Original final ACs and AC15 remain unchecked; task In Progress, V2 off.

## Fresh macOS concurrency qualification passed — 2026-09-08

The user-approved CI-only Python3.12.10 amendment in `90e60aea42` changes just
the workflow pin and existing expectation. Test-first RED/GREEN and independent
fix-only review passed; root committed15contract tests pass in0.61s with no
warnings, and scoped Ruff/format/whitespace checks pass.

[Run34294412119](https://github.com/rmusser01/tldw_chatbook/actions/runs/34294412119)
on that exact commit passed: **11 product cases in11.61s, zero skips, failures,
errors, or pytest warnings**. The isolated lock allocation/acquisition/disposal
control passed. Runtime: **Python3.12.10 / SQLite3.49.1 / macOS15.7.9 / ARM64**.
Root independently checked all exact JUnit identities, complete four-file
artifact, metadata and control; raw bytes/hashes remain in the retained SDD
archive. Existing Node/action deprecation warnings are disclosed separately.

See the [full evidence](../superpowers/reviews/2026-09-08-sqlite-macos-concurrency-evidence.md).
Task14/AC15 are satisfied for this fresh runner. Earlier failed setup evidence
is preserved; no local host/Python changes, broad rerun, main/dev update or
PR/merge occurred. Other platform/optional/static gates remain open, original
final ACs remain unchecked, TASK-31942 stays In Progress and Canvas V2 disabled.

## SQLite correction closeout and Mermaid handback — 2026-09-08

TASK-31942 is now Done. The reviewed acceptance matrix reconciles the affected
storage/actual-child results, separately qualified eleven macOS concurrency
cases, final import-only coverage/unchanged startup budgets, and the explicitly
approved baseline-static-debt/platform limits without inventing one aggregate
all-green run. Historical failures and raw evidence above remain unchanged.

The narrow closeout review found one omitted requirement: actual Canvas-child
workflow timing against baseline. Task16 now supplies five samples per arm with
identical harness and synthetic workload. All ten invocations pass; median
36.834s current versus34.186s baseline (+2.648s/+7.75%). This measures the full
create/update/render/pin/reconnect/restore/cleanup call, not isolated SQL latency.
The initial missing-browser-path failure is preserved; the user explicitly
approved the corrected fresh series, which had no retries or discarded samples.
Each successful run retains the inherited Requests warning. Independent
fix-only review: ADDRESSED, spec compliance PASS, quality PASS, closeout READY,
with no new findings. No other completed suite, benchmark or broad review replay.

See the [acceptance closeout](../superpowers/reviews/2026-09-08-sqlite-acceptance-closeout.md)
and [paired workflow evidence](../superpowers/reviews/2026-09-08-sqlite-canvas-workflow-benchmark.md).
Existing ADR125 applies unchanged; no source/dependency/host or remote change in
this closeout. TASK-31941/Task8 must still run its separate candidate/admitted
qualification and review. V2 executable remains false and default profile null;
the earlier admitted1346-pass/4-fail result remains failed, not superseded by this
SQLite handback. Revalidate pinned input availability before reproducibility
runs; the earlier runtime archive cache was empty at preflight.
