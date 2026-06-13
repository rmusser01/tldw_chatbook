# Notes Screen — Senior UX/HCI Review Fixes (2026-06-11)

A design review of the rebuilt Notes screen (Notes / Sync / Templates) against the
Console and Settings screens it must match. Structural parity was already in place
(rail headers, collapse handles, `#6f7782` frames, docked action bar); the findings
below are about honesty of controls, information design, and copy. Every finding was
verified in code before fixing, and every fix verified in the live app
(textual-serve + scripted browser, isolated env, seeded DB).

## Findings and fixes

| # | Severity | Finding | Fix |
|---|----------|---------|-----|
| 1 | Critical (honesty) | Auto-sync switch had **no handler anywhere** — toggling silently did nothing | Wired: persists `notes.auto_sync`, restores at mount, runs the configured folder sync every 5 min while the app is open, announces enable/disable + each run in the activity log, skips quietly when busy/misconfigured |
| 2 | High (honesty) | Conflict option "Ask me each time" never asks — engine `ASK` records the conflict and skips the note | Relabeled "Skip and record for review"; each recorded conflict now listed in the activity log (capped at 20 + overflow line). Real resolution dialog tracked separately |
| 3 | Medium | Details pane showed machine timestamps (`2026-06-11 13:25:09.785000+00:00`) | `_format_timestamp`: local time, `YYYY-MM-DD HH:MM`, raw-string fallback |
| 4 | Medium | Action bar rendered the collided string "Ready Words: 10" | "Ready \| 4 words" — pipe separator matching the header status row, pluralized noun |
| 5 | Medium | Every template entry ended in "template" ("Empty note template", "Template for meeting notes") | `template_display_label()` strips the noise in both the navigator Select and the Templates list; sorted by display name |
| 6 | Medium | Export buttons cryptic ("Markdown / Text / Copy MD / Copy Text") — file vs clipboard unclear | Two labeled rows: "Export to file: [.md file] [.txt file]" and "Copy to clipboard: [Markdown] [Plain text]" |
| 7 | Medium | Keywords box had no format/save affordance | Hint line "Comma-separated · saved with the note" |
| 8 | Medium | Header status row claimed "saved" in Sync/Templates modes | Mode-aware: "Sync \| last sync: never", "Templates \| 9 templates" |
| 9 | Medium | "Server (0)" / "Workspaces (0)" sections rendered nothing beneath them (lists never populated at mount) | Navigator `on_mount` seeds both lists so the one-line empty hints render |
| 10 | Low | Sync activity empty state not actionable | Added "Choose a folder and press Sync Now." |
| 11 | Low | Create buttons ragged (three auto-width stacked buttons) | Full-width form stack |
| 12 | Low | Enter on a template did nothing beyond what highlight already did | Two-step activation: first Enter/click arms with an explicit hint, second creates the note and returns to Notes mode (single-click create would punish browsing) |
| 13 | Low | Footer binding said "Edit Note" for an action that focuses the editor | "Focus Editor" |

Deliberately left alone for consistency: underlined mode-chip active style (matches
Library), Sync Now in the docked bar (matches Console's composer bar), scope-aware
inspector title.

## Evidence (live captures, 2050×1240, fontsize 12)

- `review-fixes-notes.png` — sort row with ↓ Newest, empty-section hints, full-width
  create stack, "Empty note" label, humanized timestamps, labeled export/copy rows,
  keywords hint, "Ready | 4 words" bar, "e Focus Editor" footer.
- `review-fixes-sync.png` — "Sync | last sync: never" status row, actionable empty
  state, "Skip and record for review" present in the conflict options.
- `review-fixes-templates.png` — clean sorted names, "Preview — Bug report" header,
  placeholder hint, armed-state hint after first activation, "Templates | 9 templates".
- `review-fixes-autosync-on.png` — switch on, activity line "Auto-sync on: syncs
  every 5 minutes while the app is open."

## Tests

7 new tests in `Tests/UI/test_notes_workbench_layout.py` (timestamp formatter,
template label helper, mode-aware status row, auto-sync timer + persistence,
conflict option label honesty, empty-section hints, two-step template create).
Full battery: `test_notes_workbench_layout.py` (28), `test_notes_screen.py`,
`test_screen_navigation.py`, `test_ux_audit_smoke.py` — all green except the
pre-existing chatbooks `UnresolvedVariableError` baseline failure.
