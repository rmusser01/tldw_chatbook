# Console Rail IA Phase 1 — QA evidence (2026-07-02)

Branch: claude/console-rail-ia-phase1 (base 1156ece1, head c85dba6c + fixes)
Captured from textual-serve (real app CSS) in headless Chrome, isolated HOME
(/private/tmp/tldw-qa-home-rail-ia), fresh first-run state.

- console-rail-ia-fresh-2026-07-02.png — first run: four sections (Session /
  Context / Model / Details) in order, Details collapsed (+), Model shows the
  two compact lines, active conversation row carries the "now" age label.
- console-rail-ia-details-expanded-2026-07-02.png — Details toggled open (-):
  Storage / Sync / File tools / Server handoff / Handoff / ACP rows demoted
  into the disclosure.
- console-rail-ia-two-tabs-age-labels-2026-07-02.png — second native session
  via New tab: rail lists "> Chat 2 … active session - now" and
  "Chat 1 … open session - now".

## Real-provider send evidence (second capture session, same day)

Captured from textual-serve at http://127.0.0.1:9063 (real app CSS) in
headless Chromium, isolated profile /private/tmp/tldw-console-rail-ia-cdp-20260702
(HOME + XDG_DATA_HOME + XDG_CONFIG_HOME), live llama.cpp server at
http://127.0.0.1:9099 (gemma-4-26B gguf) — real streamed assistant responses,
no fixtures or seeded transcripts.

- console-rail-ia-real-send-auto-titles-age-labels-2026-07-02.png — two sends
  in two native sessions: tabs auto-titled "Explain how" / "Draft a haiku"
  from the first accepted user message; the transcript shows the real streamed
  haiku; Chats rows show "> Draft a haiku abo… active session - now",
  "Explain how tides… open session - now" and a persisted
  "Summarize why the… saved workspace - 16m" (relative age buckets live:
  now / 16m, later 49m / 1h as the clock advanced). DB check: persisted
  conversation titles are the 30-char truncated auto-titles.
- console-rail-ia-details-persisted-relaunch-resume-2026-07-02.png — full
  relaunch (fresh app process), resumed the saved "Draft a haiku about
  morning…" conversation from its rail row: Details renders EXPANDED from the
  conversation-scoped stored preference
  [console.rail_state."console_rail_state:workspace-default:<conversation-uuid>"]
  with details_open = true.

## Defect found and fixed during this QA (commit 86230bb4)

The first relaunch-persistence attempt failed honestly: details_open=true was
stored under the conversation-scoped key, but the resumed Console always
rendered Details collapsed. Root cause: section open flags were only seeded at
compose time; _sync_console_rail_visibility never re-applied them when the
preference scope switched at runtime (resume). Fixed by applying all four
section flags in the rail sync (fix(console): re-apply rail section
preferences on runtime scope changes) with pilot test
test_console_rail_section_sync_applies_stored_scope_preferences. The
relaunch-resume capture above was taken after the fix.

Known scoping limitation (documented deviation from the plan's capture-4
wording): rail section preferences are scoped per workspace+conversation
(session-scoped fallback for unsaved chats), matching the pre-existing
left/right rail persistence design. A section toggled on a fresh UNSAVED chat
resets to defaults after a relaunch because the new session gets a new scope
key; persistence across relaunches applies to saved conversations on resume,
as captured. If global section persistence is wanted instead, that is a
follow-up design decision.

Test verification (2026-07-02): affected set = 327 passed, 2 failed, both
pre-existing baseline (confirmed failing at branch base 1156ece1):
test_console_session_surface_uses_flex_height_not_full_percent_height,
test_console_browser_selecting_duplicate_membership_row_ignores_other_workspace_open_session.
After the 86230bb4 fix, five-suite Console UI gate = 299 passed, 2 failed
(same two baseline failures); Tests/UI/test_console_persistent_rails.py =
30 passed.
