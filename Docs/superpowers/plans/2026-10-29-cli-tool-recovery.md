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
