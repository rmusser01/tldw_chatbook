# Prompt More actions and deletion recovery — TASK-32750

The action review found three defects: a deletion error outside the visible
editor area, confirmed deletion silently ignored on the saved New prompt
route, and a pending resize callback reading the screen of a removed pane.
All three are repaired on `feat/component-pattern-library`.

## Changes and scope

- Single-item deletion failures reveal the existing status without rebuilding
  live fields or moving focus. Delete remains readable for retry.
- Single-item confirmation and mutation settlement accept the existing active
  editor predicate, covering both Browse and Create. Item/version, mutation
  token, and in-flight checks remain; bulk selection remains Browse-only.
- A deferred focus reveal ignores a pane that has been detached, even when
  Textual still reports it as mounted.

Existing ADRs 055, 060, 086 and 150 govern this repair. No new ADR, storage
contract, visual token, or stylesheet change is required.

## Automated evidence

`Tests/UI/test_library_prompt_action_journeys.py` uses real SQLite and the
production stylesheet harness. Its eleven cases cover:

- Six More actions controls, Tab order, painted labels and Escape-to-opener
  at 170×48 and 80×24 in dark and light themes.
- Detached Duplicate, Cancel, an injected storage failure, retry, Undo,
  restored rows/counts/content/version, and Dismiss at all four size/theme
  combinations. The saved source remains unchanged.
- Actual confirmed deletion and Undo after saving through New prompt, at
  both sizes.
- A captured resize callback invoked after removing its actual mounted pane.

Before repair, the failed-delete paint checks failed at both sizes, both
Create-route cases failed waiting for committed deletion, and the removed-pane
case raised `NoScreen`. A neighboring recovery test independently exposed the
same callback exception. Menu reachability already passed before changes.

Final command results are recorded in [verification.txt](verification.txt).
The neighboring selection covers existing stale-modal, selection-generation,
duplicate-call, conflict, partial-result and Undo guards. This is a targeted
review, not a full repository sweep.

The neighboring runs also exposed two test readiness gaps. The reader retry
test could end after the new Input appeared, before its sibling Select had
composed. The [diagnostic plugin](reader_compose_probe.py) delays that compose
by 200 ms and reproduced the same `SelectOverlay` failure with shutdown already
started ([trace](reader-mount-diagnosis.json)). Waiting for the Select to finish
mounting passes the same controlled interleave. No reader production code
changed for this failure.

The compact continuity check sampled a focused Advanced field while its
ancestor scroll was still moving: measured scroll positions 4.1 and 12.3 were
both short of the target 27 ([observations](focus-paint-diagnosis.json)). The
test now waits for the focused field to paint before retaining its original
content assertion. A generic pause and even animator-idle were insufficient
frame-readiness conditions. No forced scrolling or disabled animation is used.

[static.json](static.json) records no added Ruff diagnostics against
`38d06155a1`. The large controller and screen retain 16 and 205 baseline
diagnostics respectively; the reader test file retains one baseline import
diagnostic. Changed methods/tests pass formatting. The work pane, new tests,
continuity file and retained probes pass lint and formatting.

## Native evidence

The final private profile used `TldwCli` with `LinuxDriver`, exclusive profile
ownership, and all ten configured database paths plus data/config roots
contained in its disposable directory. Data originated from the previous
isolated resize profile. No personal profile or provider request was used.

[native_check.py](native_check.py) records the exercised journey. The retained
copy has import sorting, formatting and explanatory lint comments applied;
its assertions and actions match the executed probe. Setup and tmux process
ownership are external prerequisites, not performed by this script.

[result.json](result.json) records PID 1573 and two completed journeys:
170×48 dark and 80×24 light. Actual tmux resizing and measured app size agree.
Each journey checks all six controls, Escape, detached Duplicate, Cancel,
failure feedback and retained field identity, Undo to version 3, and Dismiss.
Only the first delete call per copy is intentionally failed; successful
delete/restore operations use the real local service.

The following exported screens were rendered and visually inspected:

| State | Wide | Compact |
| --- | --- | --- |
| Failure and retry | [170×48](failure-170.svg) | [80×24](failure-80.svg) |
| Undo / Dismiss receipt | [170×48](receipt-170.svg) | [80×24](receipt-80.svg) |

The error and recovery controls are readable at both sizes. The long compact
receipt title is clipped within its pane; Undo and Dismiss remain readable.

Normal Ctrl+Q at 80×24 returned from `app.run`, wrote a fresh exit code 0, and
returned to the observed `1205 zsh 80x24` pane before its owned session closed.
[persistence.json](persistence.json) was read from SQLite in read-only mode
after exit: original ID 7 remains live at version 1; copies 8 and 9 remain
deleted at version 4 after Undo and the second deletion. All three retain
their expected body; the live count is 7.

[native-log-review.json](native-log-review.json) excludes copied historical
log entries and records the fresh run's existing Console sidebar, optional
audio, terminal capability and worker-status notices. Successful Prompt
verification does not qualify clean app startup or an external service.

The earlier run-001 exposed the Create-route defect and failed its assertions.
It required an additional quit gesture before returning exit 1. Its shutdown
is not counted as the final normal-exit result.

## Remaining work

This pass verifies reachability of Export, Copy Markdown, Collections and
History; their complete action flows and Use in Console still need the next
Prompt review. Skills, Collections, ingestion and other destinations remain
in the broader audit. Integration into `dev` is pending.
