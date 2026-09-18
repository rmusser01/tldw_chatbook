# TASK-32779 — Tool Profile export recovery

An existing destination, invalid archive name, or publication collision now
returns to filename selection with local recovery text and the previous
location/name. The reviewed immutable snapshot is retained. Existing files
are never replaced. Unsupported, failed and uncertain publication remain
terminal outcomes with distinct copy.

The real export journey also exposed a crash before filename selection:
the review read `payload.rules`, while `ToolProfilePayload` provides `tools`.
The review now reads the real contract and the existing policy-count fixture
constructs that type instead of inventing a matching mock attribute.

## Targeted evidence

86 distinct cases pass; no full suite or provider requests were run.

| Scope | Cases | Evidence |
| --- | --- | --- |
| Existing/invalid/racing destination, correction/cancel, exact snapshot reuse | 6 | [Workflow log](workflows.txt) |
| Unsupported, failed and uncertain outcomes do not reopen the picker | 3 | [Workflow log](workflows.txt) |
| Existing Settings Tool Profiles presentation/workflows | 30 | [Regression log](existing.txt) |
| Real service and publication boundaries | 47 | [Publication log](publication.txt) |

The six recovery cases use real capture and publication, including ZIP
manifest verification. Their sealed inventory retains real built-in/local
sources with an empty external-MCP reader. The three terminal-outcome cases
inject publication outcomes into the real mounted worker/modal journey;
they do not claim to exercise platform primitives. The publication suite
covers those lower-level boundaries. Capture checks initial profile authority;
publication writes the already reviewed snapshot, without a new policy review.

[Pre-fix failures](red-summary.txt) record both the real-payload crash and the
subsequent six missing-recovery failures. Earlier fixture construction failures
did not reach the feature and are excluded from regression evidence.

## Native visual qualification

The real app runs with its terminal driver, real Tool Pack inventory/service,
private profile, and owned tmux session. Each cell reviews an export, chooses
an existing archive, corrects its filename, verifies the resulting ZIP, and
cancels a second conflict without writing. No service/modal substitutions are
used. The native receipt also checks unchanged permission-store bytes.

| Theme / viewport | Review | Conflict recovery | Saved result |
| --- | --- | --- | --- |
| Dark 80×24 | [View](textual-dark-80x24-review.svg) | [View](textual-dark-80x24-recovery.svg) | [View](textual-dark-80x24-saved.svg) |
| Dark 170×48 | [View](textual-dark-170x48-review.svg) | [View](textual-dark-170x48-recovery.svg) | [View](textual-dark-170x48-saved.svg) |
| Light 80×24 | [View](textual-light-80x24-review.svg) | [View](textual-light-80x24-recovery.svg) | [View](textual-light-80x24-saved.svg) |
| Light 170×48 | [View](textual-light-170x48-review.svg) | [View](textual-light-170x48-recovery.svg) | [View](textual-light-170x48-saved.svg) |

Compact review content scrolls while its actions remain visible. The filename
input scrolls horizontally to its caret; local recovery text and Save/Cancel
remain visible in both themes. These captures qualify this bounded export
journey, not every file-picker or Tool Profiles management interaction.

[Native result](native-result.json), [capture hashes](capture-manifest.json),
and [lifecycle receipt](lifecycle.json) identify the final source and evidence.
Normal keyboard shutdown returned exit 0, left no live app process, released
the instance lock, preserved default-profile fingerprints, and left all 11
private databases healthy with no conversations/messages. The preceding run
also passed; its historical receipt predates only lambda formatting.

Scoped Ruff adds no diagnostics to existing baselines. New tests/native runner,
changed small files and the changed Settings method pass formatting. Backlog
uniqueness, Windows-compatible paths and diff checks pass. Independent review
found no introduced blocker and separately verified missing-parent correction.
See [verification](verification.json) for source hashes and counts.

ADR-107 applies. The [Tool Profiles review ledger](../../reports/2026-09-18-tool-profiles-review.md)
retains delayed import review, focus restoration and the remaining management
journeys. This task does not complete the broader component workstream.
