---
id: TASK-435
title: Resolve the Personas vs Roleplay naming split
status: Done
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-23 16:36'
labels:
  - roleplay
  - ux
  - navigation
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). The nav destination is "Personas", the screen header says "Roleplay", and a mode INSIDE the screen is also called "Personas" (who you are) - three overlapping meanings. A new user hunting "character chat" must guess that the Personas tab is the RP hub, and "Personas" means something different at each level. Decide one public name for the destination (e.g. Roleplay) and reserve "Personas" for the user-identity mode.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Nav label, screen header, and in-screen mode names are mutually consistent and each name has exactly one meaning
- [x] #2 Legacy aliases still route correctly
<!-- AC:END -->

## Implementation Notes

Gave the Roleplay/Personas nav destination one public name — **"RP&CD"** (compact rail label) / **"Roleplay & Chat Dictionaries"** (full label) — and reserved **"Personas"** exclusively for the in-screen user-identity mode. Pure copy/label change; no internal id/route/class/file renames.

**Surfaces renamed (AC#1):**
- `UI/Navigation/shell_destinations.py` — the `personas` `ShellDestination`: `label` "Personas"→"RP&CD", new `full_label="Roleplay & Chat Dictionaries"` (so `accessible_label`/palette/header render the full form).
- `Constants.py` — `TAB_DISPLAY_LABELS[TAB_CCP]` and `[TAB_PERSONAS]` → "RP&CD".
- `UI/Screens/personas_screen.py` — header title → "Roleplay & Chat Dictionaries" at **both** sites (`compose_content:584` and the live-refresh `_update_title:1702`).
- `app.py` — both tab tooltips (`TAB_PERSONAS`, `TAB_CCP`) renamed and the retired word "prompts" dropped; plus 4 navigation toast strings ("Opened Personas…" → "Opened Roleplay & Chat Dictionaries…").
- `Widgets/Chat_Widgets/chat_session.py` — Console first-run "context lanes" hint "Personas"→"RP&CD".
- `MODE_LABELS["personas"] = "Personas"` deliberately **unchanged** — now the single meaning of "Personas".

**Legacy routing (AC#2):** added only `"roleplay"` to `legacy_routes`; `resolve_shell_route("personas")` and `("roleplay")` both resolve to destination `personas` (verified), and `"personas"` stays palette-searchable via `destination_id`/`primary_route`.

**Testing:** RED→GREEN on the label/palette suite (`test_shell_destinations`, `test_command_palette_shell_routes`, `test_command_palette_providers` — 81 passed); stale nav-label/header assertions updated in `test_screen_navigation`, `test_master_shell_navigation`, `test_unified_shell_phase6_first_time_replay`, `test_personas_workbench`. Task review: spec ✅, quality Approved.

**Deferred:** the Settings domain-ownership audit still names the destination "Personas" in copy that overlaps the Settings-screen category taxonomy → follow-up **TASK-496** (deliberate copy pass rather than a guess).
