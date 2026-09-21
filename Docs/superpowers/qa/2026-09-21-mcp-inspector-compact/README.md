# Compact MCP Test Tool inspector — 2026-09-21

TASK-32882 restores the existing Test Tool form at 80×24. The inspector uses
Textual's scrolling container; its form measures its full height, and action
labels wrap within the available width. After resize, the currently focused
child is revealed without restoring old focus. No tokens, tool authority,
permission rules or execution behavior change. Existing ADR-150/161 apply;
no new ADR is required.

PR2769's mechanical size/startup integration repair merged as `85e7153158`.
This layout branch rebased onto it without conflicts; captured runtime/CSS
source hashes remain identical. The final width token substitution leaves
generated CSS byte-identical, and [17 focused keyboard/governance cases pass](token-final.txt).
[All four keyboard journeys pass again on merged dev](merged-dev.txt). All 16
final native images are pixel-identical to the inspected captures.
The owner approved this layout on 2026-09-21. Current-head CI/review and
current-dev checks still gate merge.
Audit and the remaining MCP/component review scope remain open.

## Verification

- [Four real-app keyboard journeys](keyboard.txt): scalar and raw JSON forms
  in both themes; Tab/Shift+Tab, editing, resize through 170×48, 100×30 and
  80×24, retained field identity/value, fully painted permission copy, approved
  Run and Close. Positive compositor size, full clipping and center hit targets
  are checked. Permission copy is reachable with the inspector's Up scrolling.
- [372 neighboring cases](neighbors.txt): inspector, schema form, catalog
  refresh, design-token and CSS bundle governance, CSS selector performance.
- [134 native admission/fixture checks](native-admission.txt) and
  [all nine artifact guards](preflight.txt).
- [No introduced Ruff findings](static.json), [14 changed ranges formatted](format.json),
  and the new test file passes Ruff/format. No full-suite sweep was run.
- Independent review found no blockers in production, keyboard tests or the
  reused native runner. The callback reads current focus after refresh and
  checks attachment, visibility and active screen before scrolling.

[Initial reproduction](baseline.txt) exposed invisible argument controls.
Follow-up checks found default button widths and panel height clipped controls.
[Raw-editor resize reproduction](resize-baseline.txt) exposed a separate
focused-but-invisible TextArea; the resize callback repairs it. Initial harness
attempts used unsupported TextArea selection keys and skipped an already-focused
Close button; those were corrected before the four final passing cases.

The shared native CLI also depended on `validate_username`, removed by upstream
PR2741. Its exact documented ASCII identifier contract is now in the shared
input-validation module; a strict Pydantic adapter rejects coercion and uses
absolute string anchors. Profile/path admission is unchanged. The first launch stopped
at that import before starting the app; all 134 admission checks now pass.

## Native evidence

The [existing real-stdio runner](../2026-09-18-mcp-tool-errors/current-dev/native_check.py)
now types arguments and uses Tab/Up/Enter, checking fully painted wrapped action
labels and permission text. Historical tool-error receipts retain their original
runner hashes; this qualification records the updated runner separately.

Fresh [compact result](compact-native-result.json) and [wide result](wide-native-result.json)
each cover dark/light failure then recovery over the same live connection.
Each private profile records four real wire invocations and four audit rows.
Launch receipts: [compact](compact-native-launch.json), [wide](wide-native-launch.json).
Both native runs matched all 14 captured source hashes and the runner hash at
qualification. The later review repair changes only the shared validation
module among those captured sources; all pre-existing executable AST is
unchanged there. See [review follow-up](review-followup/README.md).

[Compact cleanup](compact-native-lifecycle.json) and [wide cleanup](wide-native-lifecycle.json)
verify app/fixture exit, lock release, ten healthy databases, zero conversations
or messages, unchanged user defaults and unrelated sentinels, no logged errors,
empty fault traces and zero network attempts. The native runs qualify the scalar
real-server workflow; the separate keyboard tests cover raw JSON and resizing.

Export/log trailing whitespace is normalized without changing rendered content.
Sixteen representative captures are retained below; each run also generated
intermediate/retry argument and permission captures in its private evidence root.

| Theme / terminal | Arguments | Permission and Run | Failure | Recovery |
| --- | --- | --- | --- | --- |
| Dark / 80×24 | [View](textual-dark-80x24-failed-arguments.svg) | [View](textual-dark-80x24-failed-permission.svg) | [View](textual-dark-80x24-failed.svg) | [View](textual-dark-80x24-recovered.svg) |
| Light / 80×24 | [View](textual-light-80x24-failed-arguments.svg) | [View](textual-light-80x24-failed-permission.svg) | [View](textual-light-80x24-failed.svg) | [View](textual-light-80x24-recovered.svg) |
| Dark / 170×48 | [View](textual-dark-170x48-failed-arguments.svg) | [View](textual-dark-170x48-failed-permission.svg) | [View](textual-dark-170x48-failed.svg) | [View](textual-dark-170x48-recovered.svg) |
| Light / 170×48 | [View](textual-light-170x48-failed-arguments.svg) | [View](textual-light-170x48-failed-permission.svg) | [View](textual-light-170x48-failed.svg) | [View](textual-light-170x48-recovered.svg) |
