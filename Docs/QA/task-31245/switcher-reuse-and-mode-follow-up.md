# Session switcher native-report follow-up — 2026-09-07

Separate follow-up to merged PR #2469, based on dev
`29442a160bb69cf2183a3aa0d1e68c7f8990e3bf`. This does not reopen the
broader release qualification or the deferred TASK-31966 GC investigation.

## Report and corrections

The user reported duplicate Console tabs after reopening a conversation and
could not identify the selected switcher mode. The screenshot establishes the
duplicate tabs, not which mode caused them. Controlled tests reproduced
History-to-History and Character-to-History duplicates; the other two mode
combinations already reused the exact runtime.

History now opts into exact persisted-ID reuse after saved-record validation.
It preserves authority checks and missing-record refusal; IDs beginning with
`native:` are not mistaken for runtime IDs. Other cold-resume callers retain
their intentional fresh-session behavior. Warm activation uses the existing
guarded presentation path, including refreshing the current retrieval scope.
Existing duplicate tabs are not automatically closed or consolidated.

The existing border title now names `[Active]`, `[History]`, or
`[Character chats]`, including explicit History widening. The selected mode
uses the app's selected styling, independently of focus. The title uses primary
text color: visual inspection caught that inheriting the border's dark color
made the first implementation unreadable. No rows, shortcuts, or CSS budget
were added. Impeccable's label-before-color guidance informed this correction.

## Verification

- Initial duplicate/mode regression: 6 failed, 4 passed; review regressions for
  deleted records and prefixed IDs failed before their corrections.
- Real registry scope regression: cached one source remained stale after the
  inactive target's workspace changed to two; then passed after refresh repair.
- Covering run after activation repairs: **48 passed**, one expected CSS
  headroom warning, 81.47 seconds. Covers the new reuse tests, installed
  activation, trust, geometry, History widening/loading, and CSS budget.
- After the final title-color correction: the two mode tests pass at **52×20
  and 120×50**, checking actual compositor text and at least 4.5:1 title contrast
  with search or an inactive mode button focused. Geometry and CSS rerun:
  **12 passed**, one expected headroom warning. Boot CSS is **803,784/804,000
  bytes**, 216 bytes remaining; no cap raise.
- No added Ruff diagnostics against the base; changed focused files formatted,
  whitespace clean, and all six preflight derived-artifact guards pass.
- Covering resource observer: regular descriptors plateau at seven with no
  remaining SQLite file handles; only first-use FIFO/socket additions. Tests
  used the existing isolated compatible qualification dependency overlay, not
  a shared-environment dependency change.
- Independent static review: identity, deletion, and scope findings addressed;
  no remaining scoped findings. Only targeted tests were run, not a full sweep.

Reproduce the main functional and visual regressions with:

```sh
python -m pytest Tests/UI/test_console_switcher_reuse_and_modes.py -q
python -m pytest Tests/UI/test_console_character_activation_presentation.py Tests/UI/test_console_session_switcher_trust.py Tests/UI/test_console_character_switcher_geometry.py Tests/Performance/test_boot_css_byte_budget.py -q
```

## Remaining manual handoff

The user's existing synthetic native app and its checkout were left unchanged;
the fixes were built in a separate worktree. Headless Textual evidence is not
new native-terminal acceptance. On a restarted fixed build, verify:

1. Ctrl+K and F3 visibly identify all three modes, even after focus moves.
2. Resume a saved conversation from History, switch away, then resume it again:
   the same tab is selected and the tab count does not grow.
3. Repeat across History and Character chats; exact transcript and composer
   focus must match. Existing duplicates from the older build may remain.

Task-31245 stays In Progress. Existing native, Windows, participant, and deferred
performance qualifications in the main QA report are not waived. Existing
ADR-120, ADR-031, and ADR-097 apply; no new architectural decision is required.
