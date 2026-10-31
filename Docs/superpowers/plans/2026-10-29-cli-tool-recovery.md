# CLI Tool Recovery Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for the primary work;
> dispatching-parallel-agents covers the independent Windows write repair.

**Goal:** Resolve the reproduced raw/virtual CLI failures with focused evidence.
**Architecture:** Reuse raw execution, closed workspace workers, and native
Windows file operations; preserve authority and existing executor bounds.
**Tech Stack:** Python 3.12+, existing stdlib/Pydantic/pytest/native file facade.
**Spec:** ../specs/2026-10-29-cli-tool-recovery-design.md

ADR required: yes.
ADR path: backlog/decisions/254-cli-search-outcomes-and-listing-compatibility.md.
Reason: document search outcome classification and optional listing wire field.

## Global constraints

Remove only the 4,000-character provider cap. Keep genuine errors bounded by
the existing failure guard. Preserve approval, confinement, exclusions and CAS.
No dependencies or full test sweep. Use worktree-local subprocess imports.

## Review focus

- Search-like compound commands must not hide another command's failure.
- Quoted patterns, executable paths and shell metacharacters retain semantics.
- Large stdout must not erase stderr or failure hints.
- Long listing must not disclose excluded entries or break remote workers.
- Native Windows writes must retain CAS, replacement and cleanup guarantees.

## Task 1: Raw shell results, search outcomes and cmd quoting (TASK-37630)

Files: Agents/raw_shell_tool_provider.py, Tools/raw_cli_executor.py, focused
Tests/Agents and Tests/Tools shell regressions. Add a small pure helper module
if needed to keep the existing large module within its current size ratchet.

- [x] Add failing tests for output beyond 4,000 chars including stderr; simple
  no-match grep/rg/findstr; real errors/compound controls; native quoted Python.
- [x] Remove the provider slice; pass exact command text to outcome projection;
  conservatively recognize empty searches without rewriting commands.
- [x] Preserve Windows cmd command text at subprocess launch; retain shell=False.
- [x] Run targeted shell/provider/runtime controls and scoped lint.

## Task 2: Virtual listing flags and grammar (TASK-37631)

Files: Tools/virtual_cli_impls.py, Agents/virtual_cli_provider.py,
Tools/local_tool_impls.py (listing region only), workspace_wire_decode.py,
workspace_tool_dispatch.py, associated parser/provider/wire/worker tests.

- [x] Add failing combined-flag, real long-output, usage and worker-parity tests.
- [x] Parse documented all/long aliases and forward long_format through the
  shared strict wire schema; preserve absent-field output byte compatibility.
- [x] Expose all supported command usage and include usage in parser errors.
- [x] Verify sensitive exclusions, confinement, local and bundled worker paths.

## Task 3: Windows pinned atomic writes (TASK-37632)

Files: Tools/local_tool_impls.py (write/read-target region only), a focused
native helper if necessary, focused atomic-write/real-worker tests.

- [x] Reproduce existing native creation/replace failure and add CAS controls.
- [x] Reuse native retained directory operations without changing POSIX/shipped
  stdlib import behavior; retain locking, exclusive temp, atomic swap and fsync.
- [x] Verify real fs_write/read and stale-write/refusal/cleanup controls.

## Task 4: Combined verification and review

- [x] Update architecture/user docs and task notes with exact targeted evidence.
- [x] Run the focused combined selection, changed-module static analysis and
  unchanged size/token/privacy/worker contract guards applicable to the patch.
- [x] Obtain one fresh review; resolve material findings with regression tests.
- [x] Check every AC, mark completed tasks Done through the CLI, and leave the
  isolated worktree attached with a concise outcome and any limitations.

## PR review follow-up (TASK-37630)

The user selected requesting-code-review in place of Qodo. The current-head
review reproduced PowerShell backslash handling that both rejected full-path
searches and concealed compound failures. Eight pure/native regressions failed
before preserving backslashes for PowerShell. Focused recovery checks passed33;
the full raw-provider/integration selection passed88. Existing ADR-254 applies.

## CI ferry follow-up (TASK-37630)

Python 3.14 CI timed out while the cancellation test inspected a worker Event
after termination. The test now observes real parent-side signaling. A local
retry exposed the same hazard in executor finalization; a bounded post-cleanup
signal guard failed before the fix. Finalization skips abort signaling after
worker death. All 46 native process controls passed, with 7 explicit POSIX skips.
Independent review accepted the repair and ran 10 additional focused controls.

Base/head runtime-source probes reproduced identical native module sets; a
background character read can race the Linux readiness flag. The final paydown
uses the existing Console lazy-export facade for ConsoleRenameSessionModal and
moves the session controller's remaining import into its Rename handler. Its
public export, original class identity and session source size remain intact.
The new cold package/session import regression runs in Perf Guard. Existing
ADR-097 applies; no limits, permissions or execution budgets increased.

Final verification: 7 budget/export/switcher/source-size checks passed, including
census 1031/1031, preimport 558/558 and first-send 93/94. Two current real SQLite
Rename success/changed-profile-refusal controls passed; 5 workflow-safety checks
passed. Independent review found no remaining findings in the final repair.

Deviation: the controller WatchlistsToolService import candidate passed boot but
transferred two modules into preimport (560/558), so it was reverted. The legacy
rename-persistence fixture was also left unchanged after its profile admission
and obsolete fake-owner route failed; current real SQLite controls provide the
Rename behavior evidence. No full suite was run and no new ADR was needed.

## Generated-worker size follow-up (TASK-37630, TASK-37632)

Fast Lane passed 3257 checks but found the generated worker at 5730 lines,
24 over its unchanged 5706 limit. The canonical write implementation now
shares complete descriptor reading, composes optional flags with equivalent
zero defaults, and joins duplicate successful-temp bookkeeping. The unchanged
builder regenerates exactly 5706 lines. No artifact compression, manual edits,
flag removal or budget increase is used.

The new real multi-chunk preview/CAS case passed before and after refactoring.
Native write/listing/bundled-loopback controls passed 43 cases, with another
20 write/edit/locking controls passing and one POSIX-mode skip. Size ceiling
and slack checks passed; five workflow safety checks passed; a bare Python
3.10.5 interpreter loaded the artifact and read 1048599 bytes correctly with
site packages disabled. The POSIX-only bundle test module cannot collect on
Windows because it imports fcntl; its full contracts are selected in the Linux
Tool Recovery matrix, along with early generated-size checks. Scoped lint has
only the existing undefined-re diagnostic. Fresh independent review found no
issues and independently repeated the size/routing checks and multi-chunk
preview/CAS control. Existing ADR-254 applies; no new ADR is needed.


## POSIX bundle import-guard follow-up (TASK-37632)

The full Linux bundle selection passed 194 checks but its import census rejected
`_write_file_os`'s Windows-only native backend. ADR-181 targets POSIX SSH hosts;
the import is unreachable there. The AST guard now recognizes only the true
body of the exact `os.name == "nt"` condition, without expanding the existing
function allowlist. Three POSIX-reachable condition mutations and an else-body
mutation still fail the guard. Executing the generated selector itself returns
the stdlib backend on POSIX and retains the loud import stub on Windows.

The sole fcntl test now imports fcntl inside its POSIX-marked function so the
portable bundle contracts can also collect on Windows. The original census
failure reproduced natively before the guard repair. Final targeted checks:
58 passed, one explicit POSIX skip; two size guards and CI routing passed;
bundle bytes still match the unchanged builder and lint adds no findings.
No production source, artifact, budget, workflow or permission changed.
ADR required: no; this verifies the existing ADR-181 platform boundary.


## Queue integration bundle follow-up (TASK-37632)

ADR required: no new ADR.
ADR path: backlog/decisions/281-primary-agent-multi-model-code-review.md and backlog/decisions/181-ssh-remote-workspace-bindings.md.
Reason: mechanically restore the existing host review/source ownership boundary without changing authority, public Git tools, worker closure, or generation policy.

The queue integration includes upstream host-only ReviewGitReader code inside the bundled Git tools module. Both Linux versions passed 200 recovery tests but failed committed-artifact parity. The canonical builder now emits 5934 lines against the unchanged 5706 ceiling.

1. Retain CI failure receipts and reproduce the parity failure; baseline real Git reader, byte preservation, refusal, object-format and exclusion contracts.
2. Move the unchanged review-only reader, entry and argv chunker into the existing Agents/code_review_source.py owner. Retain dynamic calls to the shared hardened Git runner and all validations. Simplify only the public text wrapper's one-use decoded-result locals while preserving truncation predicates.
3. Regenerate with the unchanged builder; verify byte parity, unchanged size guards, stdlib/platform contracts, real review capture, Git safety and native write controls. Correct the existing UTF-8/CRLF fixture to write its intended bytes explicitly on Windows.
4. Select the existing host review/source and complete Git safety tests in their own Linux recovery invocation, retain its JUnit evidence, and pin the routing. Native source fixtures require unavailable file-symlink privileges; Linux must exercise them unchanged. Obtain independent exact-head review, update task/evidence notes, safely push the scoped repair and re-arm after confirming the head and review threads. No full local test sweep.
