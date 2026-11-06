---
id: TASK-36521
title: 'User Guide: the Ctrl+6 row, the nav-bar screenshots, the Domain Defaults count and a duplicated approval paragraph describe retired layouts'
status: Done
assignee:
- '@claude'
created_date: 2026-10-24 00:26
updated_date: '2026-11-06 00:03'
labels:
- follow-up
- docs
priority: low
references:
- Docs/User_Guide/index.md
- Docs/User_Guide/settings.md
- Docs/User_Guide/console/agent-runs-and-tools.md
- Docs/User_Guide/images
dependencies:
- TASK-36387
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36387's review (PR #3226, 2026-10-23), checked on dev 5b5920b8e2 and at PR #3226's head:
- **The Ctrl+6 row.** `index.md:40` lists "Ctrl+6 | [Artifacts](artifacts.md) | Generated outputs…" as a screen. ADR-172 retired the top-level Artifacts destination. Ctrl+6 is now a compatibility shortcut into Library ▸ Artifacts (`ARTIFACTS_COMPATIBILITY_SHORTCUT`, shell_destinations.py:212), and `artifacts.md` already says so. PR #3226 rewrote the neighbouring Ctrl+9 row in that form.
- **Screenshots.** All 11 User Guide SVGs that show the nav bar show labels the app no longer has: ⌃9 MCP, ⌃4 or ⌃6 Artifacts, F7 Lab, F8 Logs and F9 Settings. The shipped bar is ⌃1-⌃5, ⌃7, ⌃8, ⌃0, then F2 Lab, F3 Logs, F4 Settings, F5 Research and F7 Meetings (`SHELL_DESTINATION_SHORTCUTS`). The files: `images/console/{action-row,attachment-staged,overview}.svg`, `images/home/{details,idle,overview}.svg`, `images/library/prompts-editor.svg`, `images/settings/{console-draft,overview,rag}.svg`, `images/watchlists/console-workflow-complete.svg`.
- **The Domain Defaults count.** The Layout tour (`settings.md:69`) says the toggle reads "Domain Defaults ▸ (10)" and shows "ten rows". The group has 12 categories (`DOMAIN_SETTINGS_CATEGORY_IDS`; Video Gen, Dreams and Guardian joined later), and the label counts them.
- **A duplicated paragraph.** In `console/agent-runs-and-tools.md:551-565`, a paragraph that says **Always** covers "MCP tools *and* local workspace tools" is followed by one that says **Always allow** is for "MCP tools only". Between them is a stray fragment, "install and run-script confirm cards follow the same rule).", which repeats the end of a sentence at :486. The fragment and the second paragraph came in with 6a6242ce3a, a rebased feature commit.

Not in scope: the badge table and the category map in settings.md are TASK-35013 AC#7 and TASK-35016 AC#12. TASK-35016 (To Do) also replaces the Domain Defaults group; if it lands first, the Layout tour follows its group labels instead. TASK-36519 rewrites the approval-card lines this guide quotes in the same agent-runs-and-tools.md section, so whichever lands second rebases onto the other.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 index.md's Ctrl+6 row says Ctrl+6 opens Library ▸ Artifacts and that Artifacts is not a screen of its own
- [x] #2 No User Guide image shows a nav-bar label the app no longer has; each listed image is recaptured from the current app at a full-screen size, or removed with its reference
- [x] #3 settings.md's Layout tour names the collapsed settings group and the count exactly as the app shows them
- [x] #4 agent-runs-and-tools.md says once which tools Always covers, with no contradicting paragraph and no sentence fragment
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. index.md: rewrite the Ctrl+6 row in the Ctrl+9 row's form (Library ▸ Artifacts, not a screen of its own).
2. Survey every User Guide SVG's painted nav row (not only the 11 listed): the older "1 Home … 4 Artifacts … 9 MCP" captures carry retired labels too.
3. Recapture each image whose documented state a fresh scratch profile can reach through the real UI: `App.run_test(size=(235, 52))` + `save_screenshot` in a python run through run.sh, non-loopback sockets refused in-process, a loopback fake llama.cpp server for demo replies. Strip the cdnjs font URLs (the guide's capture recipe).
4. Remove the images whose state cannot be reached honestly, with their references; delete unreferenced images.
5. settings.md (and rag.md, which quotes the same label): the Domain Defaults toggle count as the app paints it.
6. agent-runs-and-tools.md: one paragraph about which tools Always covers; delete the fragment and the contradicting duplicate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- **AC#1** `index.md`: the Ctrl+6 row reads "[Library ▸ Artifacts](artifacts.md) | All artifacts, Chatbooks, and Reports. Not a screen of its own: Ctrl+6 opens this Library section, and it has no nav label.", the same form as the Ctrl+9 row.
- **AC#2** A survey of every User Guide SVG's painted nav row found 25 with retired labels, not 11: fourteen older captures use the "1 Home … 4 Artifacts … 9 MCP … More: Ctrl+P" form. AC#2's first clause covers them all, so all 25 were handled.
  - Recaptured 18 from the current app at 235x52. Each run was a python script through run.sh: a real `TldwCli` in `App.run_test(size=(235, 52))`, then `save_screenshot`. The script reset the scratch profile on each run and set `[first_run] setup_completed = true` there. Non-loopback `socket.connect` was refused in-process. A loopback fake llama.cpp server returned one canned demo reply. Each state was reached through the real UI: keys, clicks, a pasted path, typed slash commands.
    - Console: overview, action-row (a selected assistant message), attachment-staged (a pasted PNG path; `[model_capabilities.models]` marks the demo model vision-capable), rewind-modal (`/rewind`, #1 selected), tabs-coachmark (Ctrl+T after the first send).
    - Home: idle, details (Details expanded), and overview (a real failed import of a corrupt PDF through Library ▸ Import media, shown under Needs Attention).
    - Settings: overview, console-draft (one edited Console Behavior field: "State: Draft — save with s · 1 unsaved · revert with r | …"), rag.
    - Roleplay: overview, character-card, character-editor (Default Assistant), dictionary-entries and lore-entries (created through + New, entries added, preview run).
    - Library: import-media, search-rag (a search over the sent conversation).
    - The cdnjs font URLs were stripped (`_template.md` step 5). Every remaining nav row now equals the shipped bar exactly: ⌃1-⌃5, ⌃7, ⌃8, ⌃0, F2 Lab, F3 Logs, F4 Settings, F5 Research, F7 Meetings, F8 Plugins, F9 Kanban.
  - Removed with their references: `library/file-notes.svg` (it needs a linked folder through the folder picker) and `watchlists/console-workflow-complete.svg` (it needs a completed Watchlists workflow run), together with that image's caption.
  - Deleted five images no page referenced: `library/{media-viewer,notes-editor,overview,prompts-editor,skills-list}.svg`.
  - `approval-card.svg` and `context-modal.svg` are harness captures with no nav bar, and are unchanged.
- **AC#3** `settings.md`'s Layout tour reads "Domain Defaults ▸ (12)" and "twelve rows", as the recaptured `settings/overview.svg` paints it. `settings/rag.md` quoted the same "(10)" label and was corrected too. The group name is unchanged, since TASK-35016 has not landed.
- **AC#4** `console/agent-runs-and-tools.md`: the "**Always** applies to MCP tools *and* local workspace tools" paragraph is now the only statement of what Always covers. Deleted the "**Always allow** (MCP tools only)" paragraph, the stray "install and run-script confirm cards follow the same rule)." fragment, and two more remnants of the same rebased commit (6a6242ce3a) in that section: a truncated duplicate "Bulk controls" bullet with the retired-label decision bullet ("Approve once / Approve for session / Always allow"), and the stray "flags reads that could exfiltrate…" fragment.
- Checks: `Tests/Docs`, `Tests/MCP/test_mcp_documentation_contract.py` and three more doc-reading files: 8 failed and 368 passed, the same 8 failing with dev's content (README/media-and-conversations contract pins, RecoveryRequired, a Python-version pin). All SVGs parse as XML.
- The capture scripts stayed in the scratchpad; they are not a repo tool.
- **Review round 1.**
  - `home/idle.svg` was recaptured after making a note (Library ▸ New note ▸ Blank note, Save) and then a conversation (a Console send) through the UI. It shows "Resume last conversation", "Conversations: 1 · Notes: 1" and the Resume button. `home.md` now says an empty Library shows "Import Library sources", and lists the **Start a conversation** button (`_home_idle_canvas_controls`).
  - Prose that a recaptured image contradicted is corrected from the code:
    - `console.md`: the Sessions section is gone (TASK-23199), with its table row and its 15-row limit, and the pinned **Terminal** button is added. The rail controls are **Context ◂** / **▸ Inspect** on open rails and **Context ▸** / **◂ Inspect** on collapsed handles (`console_rail_handle.py`, `left_rail.py`, `right_rail.py`). The control bar gains **⚓**, the hook-permissions button, in the layout tour and the control table. The header badge lists the states that take the place of the readiness word (`console_workbench_state.py`).
    - `chat-basics.md` and `sessions-tabs-workspaces.md`: the same handle names.
    - `attachments-images-voice.md`: **Attach file** now lives in the composer Menu, the "Attach +" relabel is gone (`console_composer_bar.py`), and the composer buttons are **Send** and **Dictate**.
    - `roleplay-chat-dictionaries.md`: the item count is in the purpose line (`roleplay_frame_state.purpose_line`) and the Status row is gone. The centre pane opens on the first item (F-031). The Inspector actions are **Chat now** / **Send to Console draft**, and the readiness copy is the current set.
    - `characters-and-personas.md`: the card has Edit and Conversations (N) buttons, the docked panels read ▸ Dictionaries (N) / ▸ World Books (N), and the action names are updated.
    - `chat-dictionaries.md` and `lore-books.md`: the Inspector copy.
    - `settings.md`: the **Plugins** button row.
    - `settings/rag.md`: the field is **Agent Library access**. The `Tests/Docs/test_console_library_controls_docs.py` pin for rag.md is updated to match.
    - `import-and-export.md`: the target line only shows with a server, and the path field is labelled.
<!-- SECTION:NOTES:END -->
