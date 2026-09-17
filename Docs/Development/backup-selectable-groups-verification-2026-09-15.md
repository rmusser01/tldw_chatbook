# Selectable backup groups verification — 2026-09-15

TASK-32628 implements the approved selectable-group migration in
[PR #2642](https://github.com/rmusser01/tldw_chatbook/pull/2642).
Users can back up Everything or choose groups, then restore selected groups with
a verified safety copy while preserving unselected data. The implementation
remains Python across macOS, Linux and Windows.
Qualification completed on 2026-09-16.

## Tested source

The implementation is `5c5f5a6df7ea20bcfb486257e82480fc5691ff98`.
It includes `dev` at `657f70ffe7`, integrated by merge `17d8b5fc75`.
The original Windows run uses `5d448a1577`, which changes only CI checkout setup
and the task record. Fixture corrections and bounded diagnostic observations are
at `f5f15f6548`, followed by the credential-readiness correction at `39c4c286d8`.
Application and packaging sources remain identical to `5c5f5a6df7`.

## Native results

| Platform | Result | Evidence |
| --- | --- | --- |
| macOS arm64, Python 3.12.11 | 500 passed; zero failures, errors or skips; 827.51 seconds | JUnit and installed-wheel/source audit |
| Linux x86_64, Python 3.12.8 | 500 passed; zero failures, skips or unfinished cases; 901.84 seconds | Native authorized SSH host; bounded receipt, raw logs retained on host |
| Windows Server 2022, Python 3.12.10 | Final corrective run: 124 product cases and 59 native checks passed; zero failures, errors or skips; product duration 1,165.839 seconds | [Final run](https://github.com/rmusser01/tldw_chatbook/actions/runs/35070813386) |

The finite product selection is `_SELECTABLE_GROUP_TESTS` in
`Tests/Backup_Recovery/run_platform_product.py`. Windows also executes the native
filesystem and security-decoding selection. These are targeted backup checks,
not a whole-repository sweep.

Windows has passing evidence for all 500 distinct cases across the original full
run and the final 124-case corrective run. The latter includes all 28 cases that
failed or errored originally. Production and packaging sources are unchanged
between those runs. This is coverage across runs, not a claim that the latest
Windows job alone executed 500 cases.

## User journeys and preservation

- **First-time/default flow:** create a full plaintext backup through the actual
  F9 controls, restore separately, and open the restored profile with content
  readback. The encrypted Everything variant exercises Include Credentials,
  including a synthetic config secret and explicit acknowledgment of 17
  unavailable keyring records. It produces a verified partial archive; it does
  not prove a complete keyring-secret roundtrip.
- **Experienced/selective flow:** choose Conversations for plaintext and
  encrypted backup, restore separately, and open the restored data. Mounted
  selection controls also verify dependency disclosure and invalidation of a
  stale review after changing choices.
- **Replacement and recovery:** real service workflows replace Prompts or
  Settings in plaintext/encrypted cases, verify unselected bytes and identities,
  and exercise later rollback. Other native cases cover return to absence,
  restoring an empty Writing group, undo, Abort, and fresh-process Finish at
  interruption boundaries.
- **Refusal boundaries:** changed configuration, source files, SQLite companions,
  ownership, scope or dependencies cannot silently broaden a reviewed restore.
  Existing full archive parsing and default capture remain covered.

The four F9 journeys use a built wheel. Service replacement/rollback tests use
the exact verified source checkout and real native storage. The Open probe uses
the production launch environment and profile arguments with a bounded readback
child. UAT is agent-executed; it is not a human participant study. Earlier manual
keyboard journeys remain revision-qualified in the
[original UAT remediation report](backup-uat-remediation-2026-09-13.md).

## Verification and review

All 54 changed Python files in the original macOS qualification parse and match
the frozen implementation source at `5c5f5a6df7`.
The installed macOS package has 2,895 files; all 2,506 packaged Python files match
the wheel and exact Git source. Linux verifies all 16,912 source files and
membership before execution and confirms they remain unchanged afterward; all
2,506 packaged Python files match the manifest, wheel and install. The
[evidence receipt](backup-selectable-groups-verification-20260915.json) records
native run, source and package hashes.

Touched-scope Ruff reports 54 findings against 54 in the merged baseline, with
zero new findings. Bandit reports four unchanged baseline findings, zero new
findings and zero scanner errors across 25 changed production files. Generated
CSS, Mermaid, profile/diagnostic inventories, SQLite schema/index checks and
Windows-compatible task-name checks pass. The Windows checkout fix passes all
36 selected CI contract tests.

Independent reviews cover retained configuration, selected absence, root
ownership, projection preservation, latest-dev integration and the Linux
qualification driver. The final macOS results and installed-source identity
were independently audited.

The first Windows attempt stopped before tests because Git rejected an existing
long task filename. The backup job now enables Git long paths before checkout,
matching the repository's existing Windows setup pattern.

The next run completed its full selection but is not qualified. All 53 artifact
hashes and 16,912 exported source files verify, including 15,694 native CRLF
conversions; all 2,506 installed Python files match that export. The failures
include Windows test-fixture assumptions about symlinks, native identities,
ACLs, home isolation, TOML escaping and SQLite connection closure. Four history
recovery cases and one capture scope change required a native repeat with the
corrected comparisons and existing bounded diagnostic observers. Production
ownership and scope guards remain unchanged.

The fixture follow-up at `f5f15f6548` passes 160 cases on macOS (124 diagnostic
product cases plus 36 CI contracts) in 279.87 seconds, and all 124 diagnostic
cases on Linux in 299.60 seconds, with zero failures or skips. Linux verifies
16,914 source files and unchanged membership; all 2,506 packaged Python files
again match the manifest, wheel and install. The diagnostic selection contains
all 28 cases that failed or errored in the original Windows run. Its Windows
[native repeat](https://github.com/rmusser01/tldw_chatbook/actions/runs/35068720686)
passes 123/124 product cases and all 59 native checks, with zero skips or setup
errors. All 27 corrected fixture cases pass, including the four history recovery
journeys. All 37 artifact hashes, 16,914 native-export source files and 2,506
installed Python files verify. The nine changed fixture/runner Python files parse, Ruff
reports zero findings, and Bandit reports no new findings against its baseline.

The remaining encrypted-credential case records a legitimate scope transition:
startup history retention creates Agent Runs storage between review and capture.
The unchanged scope guard refuses publication and requires a new review. The
fixture now awaits that real startup retention pass before reviewing sources and
asserts that the reviewed Agent Runs owner is included. No timer, native guard or
production behavior is changed. The corrected installed credential journey and
11 native capture-admission cases pass on macOS (12 total, 85.64 seconds), and
the installed credential journey passes on Linux (92.95 seconds). Both installed
package audits verify all 2,506 Python files against the committed source.

Final Windows run `35070813386` at `39c4c286d8` passes all 124 product cases and
59 native checks. The credential journey completes backup, isolated restore and
Open/readback; its observer records no changed source scope. The expected first
credential-coverage refusal and explicit acknowledgment remain exercised. All 43
artifact hashes, 16,914 source files (including 15,696 native CRLF conversions)
and 2,506 installed Python files verify. All earlier Windows failures are
resolved. The final fixture correction passes Ruff and introduces no new Bandit
findings; its test module and embedded installed-app script parse.

## Boundaries

Configured Workflows storage is reported as unsupported; this migration does
not add a Workflows adapter. Admission still rejects uncovered or foreign roots;
the explicitly approved redundant-alias handling preserves raw registry and
profile history.

This report does not assert that unrelated PR checks pass. Two upstream
app-factory merge checks stop before their journeys with
`raw_source_selection_changed`; the Notes case reproduces on the exact premerge
revision. The targeted proper-runtime Notes and Workflows coverage checks pass.
The earlier unrelated Windows Models startup failure is outside this task.
The PR remains open against `dev` and is not merged.

Design: [approved spec](../superpowers/specs/2026-09-15-selectable-backup-groups-design.md).
Architecture: [ADR-126](../../backlog/decisions/126-complete-local-backup-and-recovery.md).
