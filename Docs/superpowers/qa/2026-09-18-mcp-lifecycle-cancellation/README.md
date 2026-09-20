# MCP lifecycle cancellation — TASK-32880 (formerly TASK-32829)

## Current integration

PR2713 resumes on current dev with the later recovery and inspector changes
preserved. [Current evidence, conflict choices and twelve-capture gallery](current-dev/README.md)
record 96 targeted passes and fresh native qualification. TASK-32880 remains
In Progress pending rebased-head CI/review; owner visual approval is recorded.
The original task number below is historical; the landed task keeps that number.

## Historical saved evidence

Cancelling a connection now keeps the profile busy until the original operation
and its cleanup finish. Repeated Cancel cannot interrupt cleanup, and a delayed
Cancel belongs to the operation originally displayed rather than a later retry.
The UI says “Cancelling…”, then publishes the actual terminal outcome. Service
and redraw work are created lazily so cancellation before startup leaks neither
coroutine. Admission also stays reserved through final readiness collection.

Base: merged dev `149acda36be8939fe8cd5e589bf77d13462257e7`.
Branch: `codex/mcp-lifecycle-cancellation-review`. Saved as [draft PR #2713](https://github.com/rmusser01/tldw_chatbook/pull/2713).
Independent of the implementations in draft PR2711 and PR2712.
ADR required: no; existing ADR-161 governs the component contract. Transport,
timeout, permission and persistence contracts are unchanged; no CSS/token changes.

## Verification

- **35 distinct passing targeted cases**: [21 lifecycle UI cases](targeted-ui.txt)
  plus [one native-scheduling regression](targeted-native-scheduling.txt)
  and [13 token/generated-CSS checks](targeted-css.txt). The [exact inventory](test-results.json)
  includes eleven new regressions: held cleanup, independent profiles, repeated Cancel,
  queued retired buttons/messages across same/different server attempts,
  cancellation before startup, actual success/error after suppressed cancellation,
  held final collection, and operation identity across awaited detail rendering.
- [Initial probe](initial-probe.json) reproduced early marker removal, premature
  completion feedback, duplicate attempts and the old wrapper erasing later ownership.
  [Eight baseline failures](red-cancellation.txt) and the separate
  [render-boundary failure](red-render-boundary.txt) pin the repaired behavior.
- [Preflight](preflight.txt): all seven derived-artifact guards passed.
  [Scoped static receipt](static.json): no introduced Ruff diagnostics against dev
  (78 existing diagnostics across these large modules); all 25 changed production
  ranges formatted. New tests and the native runner pass Ruff lint and format.
- [Independent review](independent-review.md): two queue/render identity findings
  fixed with regressions; final review found no remaining blockers.
- The seven service-neighbor checks produced four configuration-bootstrap failures
  (`RecoveryRequired: raw_source_selection_changed`) and three passes, reproduced
  unchanged on fresh dev: [current run](service-neighbors-initial.txt),
  [baseline run](service-neighbors-baseline.txt). They do not qualify service outcomes
  and are not counted among the 35 passes. No full repository sweep was run.

## Native evidence and visual review

The [archived runner source](https://github.com/rmusser01/tldw_chatbook/blob/2518d40bf77c67f53bf2989d20541bb05b266d9e/Docs/superpowers/qa/2026-09-18-mcp-lifecycle-cancellation/native_check.py)
drove real TldwCli, LinuxDriver and TTY streams in a
fresh private profile. Only the client's connection call was replaced: its first
attempt waits, catches cancellation and holds cleanup; retry raises a controlled
connection error. The real UI, local profile/governance path, control-plane timeout
wrapper, attempt persistence and final readiness projection remain active.

The old executable copy is retired. Its pinned source preserves the historical
runner hash without advertising an obsolete CLI. Use the supported
[current runner](current-dev/native_check.py), which validates paths and tmux
identifiers before startup and refuses an occupied fixture profile ID.

[Final result](native-result.json) records four passing dark/light cells at 80×24
and 170×48. All twelve SVG captures were rendered and visually inspected; terminal
text is beside each. Cancel, disabled Cancelling and the subsequent Refresh tools
retry action are fully visible. The compact detail toolbar still has the known
clipping repaired separately by PR2712; this independent branch does not qualify
that toolbar as a whole.

The [lifecycle receipt](native-lifecycle.json) confirms exit 0, returned app,
absent PID, reacquired instance lock, ten healthy private databases, zero
conversations/messages, unchanged default config/UI/policy fingerprints, no
error/faulthandler output and matching final production hashes. The final runner
hash is recorded in the result. [Export hashes](export-manifest.json) preserve
provenance through whitespace-only artifact normalization.

| Theme / size | Working | Cleanup pending | Failed retry remains actionable |
| --- | --- | --- | --- |
| textual-dark / 80x24 | [Cancel](textual-dark-80x24-working.svg) | [Cancelling](textual-dark-80x24-cancelling.svg) | [Refresh tools](textual-dark-80x24-retry-available.svg) |
| textual-dark / 170x48 | [Cancel](textual-dark-170x48-working.svg) | [Cancelling](textual-dark-170x48-cancelling.svg) | [Refresh tools](textual-dark-170x48-retry-available.svg) |
| textual-light / 80x24 | [Cancel](textual-light-80x24-working.svg) | [Cancelling](textual-light-80x24-cancelling.svg) | [Refresh tools](textual-light-80x24-retry-available.svg) |
| textual-light / 170x48 | [Cancel](textual-light-170x48-working.svg) | [Cancelling](textual-light-170x48-cancelling.svg) | [Refresh tools](textual-light-170x48-retry-available.svg) |

The [initial native attempt](native-initial-retry-selector.json) stopped because
the harness expected Connect after a recorded failure; the real UI correctly
renders Refresh tools. The runner now accepts the appropriate lifecycle action.
The [next run](native-before-format.json) passed all four cells; final evidence
was repeated after a production formatting-only change to match exact sources.

This qualifies cancellation ownership, progress and post-settlement retry with
controlled client delays/failure. It does not qualify a successful external
server connection, connected tool execution, every recovery path, or the entire
MCP destination. Current-head CI, accumulated remote review and owner visual
approval remain separate merge gates.

## Real connection follow-up — native worker scheduling

A real stdio fixture exposed another admission boundary: Textual `App.run()`
installs `asyncio.eager_task_factory`, while `run_test()` normally does not. On
unchanged dev, a cached Refresh completed synchronously before its worker was
registered, leaving a SUCCESS worker permanently busy. The existing PR2713
observer already handles this ordering. A [failing dev regression](red-native-scheduling.txt)
now [passes](targeted-native-scheduling.txt), including immediate retry, without
additional production changes. The total is 35 distinct passing cases.

The [native wire comparison](native-wire-comparison.json) completed connection,
Refresh, disconnect, reconnect and final disconnect on the saved implementation;
[shutdown and persistence checks](native-wire-comparison-lifecycle.json) passed.
The fixture processes were reaped (`returncode=-15` from normal client terminate).
This probe also exposed a separate open bug: connected Refresh reports success
but returns its cached catalog; only reconnect discovers the changed tool. That
service repair belongs to the next bounded review, not this cancellation PR.
