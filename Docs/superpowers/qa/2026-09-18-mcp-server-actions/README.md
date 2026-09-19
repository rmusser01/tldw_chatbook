# MCP server action ownership — TASK-32825

Queued server actions now belong to their displayed profile. Replacing or hiding
the detail view retires its controls synchronously. An accepted Confirm retains
the original server key across awaited toolbar rebuilding, so selecting Beta
cannot redirect an Alpha deletion. Deferred Keep focus uses the same live-control
check. A two-column compact toolbar keeps all actions fully visible.

Base: merged dev `149acda36be8939fe8cd5e589bf77d13462257e7`.
Branch: `codex/mcp-server-action-ownership`. Saved PR: pending creation.
This branch is independent of draft PR2711's inspector implementation.
ADR required: no; existing ADR-161 governs the component contract. No storage,
service admission, execution authority or token value changed.

## Verification

- **109 distinct passing targeted cases**: 94 UI/server/governance checks,
  14 Workbench lifecycle neighbors, then 19 final CSS/layout checks (18 repeated
  cases and one additional selector-ratchet case). See [exact case inventory](test-results.json),
  [UI](targeted-ui.txt), [lifecycle](targeted-lifecycle.txt) and [final CSS](targeted-final-css.txt).
- The 17 new cases cover all seven queued toolbar verbs, the pre-removal await,
  accepted confirmation, hidden form, hidden Servers mode while disarming waits,
  and actual full-app toolbar paint/focus at four size/theme combinations.
- [Preflight](preflight.txt): all seven derived-artifact guards passed. After
  the final selector rekey, final CSS checks reverified generated output,
  token governance, bytes and the unchanged 274 selector-candidate ceiling.
- [Scoped static receipt](static.json): no introduced Ruff diagnostics versus
  dev (four existing diagnostics); all ten changed production ranges formatted.
  New tests/native runner pass Ruff lint and formatting.
- [Independent read-only review](independent-review.md): hidden-mode finding
  addressed; no remaining blockers. No full repository test sweep was run.

## Native evidence and visual review

[Final runner](native_check.py) drove real TldwCli with LinuxDriver and TTY streams
in a fresh private profile. [Result/source hashes](native-result.json) and
[independent lifecycle receipt](native-lifecycle.json) record four passing cells,
exit 0, returned app, absent PID, released instance lock, ten healthy databases,
zero conversations/messages, unchanged default config/UI/policy fingerprints,
no error or faulthandler output, and matching eight production source hashes.
The final runner hash is recorded in the result. All twelve captures were rendered
and visually inspected. Native terminal text is beside each SVG. The
[export manifest](export-manifest.json) records raw and normalized hashes.

| Theme / size | Safe confirmation | Retired Connect ignored | Alpha removed, Beta preserved |
| --- | --- | --- | --- |
| textual-dark / 80x24 | [Keep focused](textual-dark-80x24-safe-confirmation.svg) | [No attempt](textual-dark-80x24-retired-action-ignored.svg) | [Beta remains](textual-dark-80x24-beta-preserved.svg) |
| textual-dark / 170x48 | [Keep focused](textual-dark-170x48-safe-confirmation.svg) | [No attempt](textual-dark-170x48-retired-action-ignored.svg) | [Beta remains](textual-dark-170x48-beta-preserved.svg) |
| textual-light / 80x24 | [Keep focused](textual-light-80x24-safe-confirmation.svg) | [No attempt](textual-light-80x24-retired-action-ignored.svg) | [Beta remains](textual-light-80x24-beta-preserved.svg) |
| textual-light / 170x48 | [Keep focused](textual-light-170x48-safe-confirmation.svg) | [No attempt](textual-light-170x48-retired-action-ignored.svg) | [Beta remains](textual-light-170x48-beta-preserved.svg) |

Each cell verifies Escape keeps both real private profiles, holds an Alpha Connect
message across Beta selection and observes no lifecycle attempt, then holds an
accepted Alpha Confirm across presentation rebuilding and Beta selection. The
actual private service/store removes only Alpha; Beta remains selected and saved.
Final cleanup deletes the remaining fixture through the same real service.
Profiles use `/usr/bin/false`, but the command is never executed. No external
connection, connected tool execution or recovery journey is qualified. The
connected toolbar's geometry is covered by controlled full-app test fixtures.

## Failures retained and bounds

- [Initial probe](initial-probe.json) dispatches both delayed Connect and an
  accepted Alpha deletion to Beta on unchanged source.
- [Ten action regressions](red-actions.txt), [three hidden-mode regressions](red-hidden-mode.txt)
  and [two compact layout failures](red-layout.txt) reproduce before their repairs.
- [First native run](native-initial-clipping.json) failed on clipped compact
  Delete; its [lifecycle](native-initial-lifecycle.json) still confirms clean
  app return and release. Qualification exit 1 is not counted as success.
- [Intermediate native run](native-before-selector-rekey.json) passed all four
  cells with [clean lifecycle](native-before-selector-rekey-lifecycle.json).
  Final native evidence repeats the journeys after narrowing CSS selectors.
- PR2711's remote selector-budget failure exposed the same bare-Button pattern
  in this new toolbar rule. The [local child failure](red-selector-child.txt)
  records 275/274. Existing action classes now scope the rule; the limit is unchanged.
- Initial `/private/tmp/32825-red` profile setup errors and
  `/private/tmp/32825-layout-red` splash readiness timeouts were harness failures,
  not behavioral red tests. The interrupted `/private/tmp/32825-hidden-red`
  teardown was replaced by the clean failure evidence above.

Remaining: source transitions, full add/edit/connection/reconnection flows,
connected runtime failures, tool execution/recovery, Audit and remaining permission
journeys. PR2711 owns selected-tool inspector refresh; its remote CI and visual
approval gates remain separate. This receipt does not complete the MCP destination
or the broader design-system feature review.
