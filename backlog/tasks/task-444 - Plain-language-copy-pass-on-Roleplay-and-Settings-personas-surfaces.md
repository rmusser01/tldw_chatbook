---
id: TASK-444
title: Plain-language copy pass on Roleplay and Settings personas surfaces
status: Done
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-24 08:32'
labels:
  - roleplay
  - settings
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). User-facing surfaces leak internal governance/architecture language: "Dictionaries (embedded copies)" / "World Books (embedded copies)" with no explanation of the embedded-copy semantics, "Authority: Local", the status line "Source: Local | Attachments: Console", and Settings > Domain Defaults > Personas showing "State: Read-only contract" / "destination ownership must be implemented before mutation". Rewrite for users; move contract language to docs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Embedded-copy semantics are explained in place or renamed to something self-evident
- [x] #2 Settings personas category describes what the page is for in user language
- [x] #3 Status-line terms are user-meaningful or removed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Grep-verify every copy site named in the task brief plus any not yet listed (embedded-copy headers, Authority row, status-line, Settings > Domain Defaults > Personas) fresh against the current tree, and locate every test that asserts the old strings.
2. Rewrite the two character-attachment panel headers ("World Books (embedded copies)" / "Dictionaries (embedded copies)") to self-evident copy - a rename only (no new Static row), since both panels are height-capped (world books max-height:4) and cannot absorb an extra explainer line without risking layout breakage.
3. Remove the "Authority: Local" row entirely from the Inspector pane (id, compose Static, show_selection kwarg, clear_selection reset) since it is always "Local" at every call site and carries no user-decision value - update all 9 screen call sites and all screen updateaffected tests.
4. Drop "Source: Local | Attachments: Console" from the status row, keeping only the count (the useful part per the review); update the stale docstring comment that referenced the old de-dup rationale.
5. Rewrite Settings > Domain Defaults > Personas in plain language: category summary description, the PERSONAS SettingsDomainCategoryContract (source_of_truth/rows/follow_up, plus a doubled "Follow-up: Follow-up:" prefix bug), and the three GENERIC (shared-by-all-domain-categories) renderer strings that render on the Personas page: the state banner, the detail-pane "Settings mode"/"Writes allowed" rows and section header, and the Scope Inspector's guidance tuple - verified via grep that none of these are pinned by exact-string test assertions.
6. Update every test file touched by the rename/removal (personas inspector pane tests, personas workbench status-row test) and run the full affected test suites plus an app-import smoke check.
7. Update the task file to Done with AC boxes checked and an Implementation Notes before/after table.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Grep-verified every copy site fresh (more existed than the brief listed: two GENERIC Settings domain-category renderers - `_category_state_banner_text`, `_render_domain_category_detail`, `_inspector_guidance` - all shared by Artifacts/Skills/Schedules/Watchlists/Workflows/MCP/ACP too, plus a pre-existing doubled "Follow-up: Follow-up:" prefix bug in the Personas contract's own follow_up string). None of the changed strings were pinned by exact-value test assertions in the shared renderers (verified via grep before editing), so those edits also incidentally declutter the other read-only domain-category Settings pages.

Removed (not reworded) the "Authority: Local" row entirely - it was hardcoded to "Local" at all 9 call sites in personas_screen.py and carried no user-decision value, matching the review's own "removal is the likely right call." Removed the `authority` kwarg from `PersonasInspectorPane.show_selection()` and the reset in `clear_selection()`; updated all 9 screen call sites, all 10 test call sites, and 2 assertions in Tests/UI/test_personas_inspector_pane.py.

Kept both attachment-panel header renames as rename-only (no new explainer Static row): both panels are height-capped in personas_screen.py's DEFAULT_CSS (world books max-height:4, dictionaries max-height:12, comment explicitly says the combined budget is "close to the ceiling already" at the 100x30 floor), so an added line risked breaking that tuned layout.

Before -> after (compact):

| Surface | Before | After |
|---|---|---|
| World Books header | "World Books (embedded copies)" | "World Books (copied into this character)" |
| Dictionaries header | "Dictionaries (embedded copies)" | "Dictionaries (copied into this character)" |
| Inspector authority row | "Authority: Local" (always) | removed (row + `authority` kwarg deleted) |
| Status row | "Characters: N \| Source: Local \| Attachments: Console" | "Characters: N" (count-only, same for Personas/Mode rows) |
| Settings category description | "Character/persona discovery and Console attach defaults." | "Character and persona browsing, plus how they attach to Console chats." |
| Settings contract source_of_truth | "Character/persona scope service" / "Personas destination runtime handoff" | "Your saved characters and personas" / "Whatever's currently open in Roleplay" |
| Settings contract rows | "Runtime selection: Personas owns character/profile selection and Console attach payloads" / "Settings role: future defaults may choose discovery/display preferences, not active persona runtime" | "What Roleplay controls: Picking a character or persona, and sending it to Console" / "What Settings might add later: Browsing or display preferences - never which persona is active" |
| Settings contract follow_up | "Follow-up: add persona display/default controls after Personas exposes a persisted category source." (doubled "Follow-up: Follow-up:" when rendered) | "Add persona display/browsing preferences once Roleplay can hand Settings a saved preference to edit." |
| State banner (all domain categories) | "State: Read-only contract \| X owns workflow actions and setup." | "State: View only \| Manage this in X." |
| Detail-pane section header | "Domain ownership contract" | "How this page works" |
| Detail-pane "Settings mode" | "read-only defaults/status contract" | "View only - shows current defaults and status" |
| Detail-pane "Writes allowed" | "No - destination ownership must be implemented before mutation" | "No - change this in X instead" |
| Scope Inspector guidance | "none yet - this category is an ownership/status contract" / "open X for workflow actions and setup" / "X remains the runtime owner; Settings cannot mutate it yet" | "none yet - nothing on this page is editable" / "go to X to make changes" / "X owns this; Settings only shows it" |

Verification: Tests/UI/test_personas_inspector_pane.py (20 passed), Tests/UI/test_personas_workbench.py (193 passed), Tests/UI/test_personas_character_world_books.py + test_personas_character_dictionaries.py + test_personas_character_world_books_screen.py (18 passed), Tests/UI/test_settings_configuration_hub.py (246 passed, 1 pre-existing unrelated flake - test_theme_category_opens_without_crashing, confirmed pre-existing by running it standalone on both the pre-change and post-change tree, passing both times), `python -c "import tldw_chatbook.app"` clean.

Modified files: tldw_chatbook/Widgets/Persona_Widgets/personas_character_world_books.py, personas_character_dictionaries.py, personas_inspector_pane.py; tldw_chatbook/UI/Screens/personas_screen.py; tldw_chatbook/UI/Screens/settings_screen.py; Tests/UI/test_personas_inspector_pane.py; Tests/UI/test_personas_workbench.py.
<!-- SECTION:NOTES:END -->
