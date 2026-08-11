---
id: TASK-4010
title: Settings Appearance calls a method that does not exist
status: Done
assignee: []
created_date: '2026-08-09 11:45'
labels:
  - settings
  - crash
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the Library residue batch (task-3223's diagnosis pass, 2026-08-09, dev `4d0232358`):
`SettingsScreen._appearance_bool_label` is called at NINE sites in
`tldw_chatbook/UI/Screens/settings_screen.py` (compose-time rows at 12526/12536/12546, button
handlers at 14200/14213/14229, refresh sites at 17793/17799/17805 — reduce_motion, ascii_glyphs,
smooth_scrolling) but is defined NOWHERE in the tree (`grep -rn "_appearance_bool_label"` returns
only the call sites). Composing the Appearance section therefore raises `AttributeError`.

Four tests in `Tests/UI/test_settings_footer_hints.py` fail ambiently on dev with exactly this
error. Likely a lost hunk from a recent Settings arc (the method's name suggests a
"On/Off"-style label builder for the three boolean appearance toggles).

Fix: restore/implement the label builder (match whatever contract the nine call sites and the
failing tests expect — they encode the intended strings), and get the four ambient tests green.
Verify Settings ▸ Appearance live.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings ▸ Appearance composes and its three boolean toggles render/flip without AttributeError, verified live
- [x] #2 The four ambient failures in Tests/UI/test_settings_footer_hints.py pass on dev
- [x] #3 A regression test (or the repaired ones) pins the label contract the nine call sites share
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Already fixed before this task was picked up — closed on evidence, no work done.**

Re-verified at dev `642567627` (the standing re-verify-first rule): `_appearance_bool_label` now
HAS a definition (`UI/Screens/settings_screen.py:5318`) with a Google-style docstring, alongside
its nine call sites. Fixed by another session in commit `6c97ce15c`
("fix: restore _appearance_bool_label + repair stale Checkbox assertion (task-13156)").

`Tests/UI/test_settings_footer_hints.py` → **9 passed** (the four failures this task was filed for
are gone).

Third time this programme that re-verifying before implementing avoided redundant work
(cf. LIB-03, half of LIB-09, and task-4020's self-refutation).
<!-- SECTION:NOTES:END -->
