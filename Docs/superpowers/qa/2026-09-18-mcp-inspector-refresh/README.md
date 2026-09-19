# TASK-32823 — selected MCP tool refresh

Resumed from merged `dev` (`149acda36b`, PR #2707) on
`codex/mcp-inspector-refresh-review`, reusing the preserved investigation.
Saved as [draft PR #2711](https://github.com/rmusser01/tldw_chatbook/pull/2711) against `dev`.
This bounded change reconciles selected-tool details with the refreshed catalog.
An unchanged definition/context keeps its mounted form, argument draft, preview
and focus. A changed definition retires the form and preview with visible
reopening guidance; removal clears the selection without choosing another tool.

Pending preparation is bound to an opaque form identity cleared synchronously
before teardown. The worker generation advances before refusal paths, preventing
an old mint from publishing into a refused replacement form. Newer selection and
focus take precedence over delayed refresh. The inspector now scrolls, tool
buttons fit its content width, and the form uses natural height. Focus reveal
runs after content/viewport layout and cancels obsolete scroll motion.

ADR required: no. Existing ADR-161/170 apply. Service admission, permission
persistence and execution authority are unchanged.

## Evidence

- [Preview and ownership run](preview-targeted.txt): 42 targeted cases pass,
  including all eleven refresh/race cases then present and 31 adjacent prepared
  test, cancellation, retry, profile-lease and stale-panel cases.
- [Final inspector and governance run](final-ui.txt): 320 cases pass, covering
  the complete inspector test module, all thirteen new refresh/visibility cases,
  design-token governance and generated-bundle reproduction.
- [Final CSS budgets](final-budget.txt): boot byte budget and class-level CSS
  ratchet both pass without allowance changes. Across these three runs,
  [353 distinct cases pass](test-results.json); overlapping cases count once.
  The 42-case ownership run precedes the final layout-only repairs; the final
  320-case run rechecks refresh ownership and visibility after those repairs.
- [Preflight](preflight.txt): all seven derived-artifact checks passed before
  the final compact button/form/focus repairs. The final runs above recheck the
  affected generated CSS, tokens and budgets on the final production sources.
- The original four-case [red](red-results.json)/[green](green-results.json)
  prototype evidence is historical. Exported logs normalize trailing whitespace. [Held mint/teardown and focus](race-red.txt),
  [refused reopen](refused-reopen-red.txt), [compact overflow](overflow-red.txt)
  and [replacement-focus paint](refresh-focus-paint-red.txt) each failed before
  its repair. Intermediate failures are retained rather than counted as passes.
- [Static comparison](static.json): no introduced Ruff diagnostics; the two
  production files retain 13 and 65 existing findings. The new test/runner files
  pass lint and formatting. [Changed production ranges](format-ranges.txt) pass
  formatting; unrelated baseline formatting is preserved.
- [Independent review](review.md) found no remaining blocker after the two
  additional ownership/focus findings were reproduced and corrected.

## Native matrix

The final run used real `TldwCli`, `LinuxDriver`, TTY output streams and a fresh
private profile, PID 60183. A controlled projection changes one actual built-in
`chat_with_llm` catalog definition; production selection, keyboard activation,
form composition and the real permission-preview service remain in use.
No tool was executed and no external server/provider was contacted by the journey.
All twelve captures below were rendered and inspected.

| Theme / viewport | Retained argument draft | Changed details and return focus | Retained raw JSON draft |
| --- | --- | --- | --- |
| Dark 80×24 | [Capture](native/textual-dark-80x24-retained-draft.svg) | [Capture](native/textual-dark-80x24-changed-details.svg) | [Capture](native/textual-dark-80x24-raw-draft.svg) |
| Light 80×24 | [Capture](native/textual-light-80x24-retained-draft.svg) | [Capture](native/textual-light-80x24-changed-details.svg) | [Capture](native/textual-light-80x24-raw-draft.svg) |
| Dark 170×48 | [Capture](native/textual-dark-170x48-retained-draft.svg) | [Capture](native/textual-dark-170x48-changed-details.svg) | [Capture](native/textual-dark-170x48-raw-draft.svg) |
| Light 170×48 | [Capture](native/textual-light-170x48-retained-draft.svg) | [Capture](native/textual-light-170x48-changed-details.svg) | [Capture](native/textual-light-170x48-raw-draft.svg) |

[Journey result](native-result.json), [exact runner](native_check.py) and
[lifecycle receipt](native-lifecycle.json) record four successful cells,
unchanged permission profiles, process absence, exit 0, released instance lock,
healthy private databases, zero conversations/messages, unchanged default
config/UI/policy fingerprints and eight matching production-source hashes.
[Capture manifest](capture-manifest.json) preserves raw and exported hashes;
exports normalize trailing whitespace only.

Native runs 001 and 002 failed qualification and exited cleanly. The first
exposed missing vertical scroll; the second exposed offscreen focus after the
form's height shrank. Their result/lifecycle JSON files are retained. Mounted
regressions also exposed narrow button clipping and fractional form height.
Only run 003 qualifies the final native matrix.

## Remaining scope

This does not qualify remote-server connection lifecycles, disconnected/stale
server execution, argument validation, actual tool execution, diagnostic empty
state routing or every permission/Advanced/Audit journey. These remain separate
bounded reviews in the [MCP ledger](../../reports/2026-09-18-mcp-review.md).
No full repository test suite was requested or run. Follow-up PR CI and owner
visual approval remain separate from the local evidence here.
