---
id: TASK-41526
title: "User Guide pages lack Media select mode, the Prompts and Skills Import row, slow Rendered view, Change Review dialog and Tool Packs"
status: Done
assignee: ['@claude']
updated_date: 2026-11-19 16:10
created_date: 2026-11-17 05:27
labels:
- follow-up
- docs
priority: low
references:
- Docs/User_Guide
dependencies:
- TASK-41101
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the docs sweep (TASK-41101, round 15). Facts with no owning page: Media select mode "Select items to enable." (TASK-32045); the Prompts and Skills `Import...` row caret (TASK-21116); the slow Rendered view on very large documents (TASK-22660); the Change Review commit and push dialog (mid-merge refusal TASK-19703, push-URL disclosure TASK-19701); and Tool Packs (Settings, Tools and MCP), which no User Guide page mentions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each fact has a place in its owning page, matching the current copy
- [x] #2 No dated or `Verified against` text is added
- [x] #3 A page's change is checked against the running screen in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. For each fact, read the code that produces it and pick its owning page: Media select mode and the slow Rendered view in `library/media-and-conversations.md`; the Import row in `library/prompts.md` and `library/skills.md`; the Change Review commit and push dialogs in `console/agent-runs-and-tools.md` (its Change review section, where the old "Git actions" section was lost); Tool Packs in `settings.md` (the Tool Profiles category) with a pointer from `mcp.md` (Permissions mode's profile selector).
2. Write each in the page's own style, quoting on-screen copy from the code, with no dated or stamp text.
3. Check each against the running screen headlessly (run_test through run.sh): the existing mounted tests that pin the copy, plus throwaway probes (deleted) where no test shows it.
4. Tests/Docs, preflight, task-file check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Each fact now has a place on its owning page, written from the code at c176f553b3 and checked headlessly against the mounted screen (run_test through run.sh; the brief rules out the live TUI). No date or stamp text was added: `git diff -- Docs | grep '^+'` finds no ISO date, "Verified", "(Was" or "superseded".

**Where each fact went, with the code behind it.**
- **Media select mode**, `library/media-and-conversations.md`, a new bullet under "The shared select / export grammar". Media's labels are short words: Export/Review/Delete at `Widgets/Library/library_media_canvas.py:898`, `:906`, `:939`, disabled with "○" while `selected_count == 0` (`:867`). The rows are Clear/Export/Review (`:1440`), the reason line (`:1481`), Analyze (`:1501`) and Delete (`:1520`). The line is `LIBRARY_BULK_ACTIONS_NO_SELECTION_REASON = "Select items to enable."` (`Library/library_shell_state.py:96`). It is visible only with nothing checked, rows present and no failed load (`:1488-1495`), and it uses `visibility`, so it keeps its height. Analyze (`:925`) uses the same count gate, so it also reads "○ Analyze" with nothing checked. An unready analysis provider outranks the count: the label stays "○ Analyze" at compose (`:930-933`) and through the in-place flip (`UI/Library_Modules/canvas_sync.py:528-529`), with its own reason line (`library_media_canvas.py:1513`).
- **Prompts and Skills Import row**, `library/prompts.md` ("Importing prompts") and `library/skills.md` ("Importing"). Opening parks the caret in the path field; Cancel returns focus to the opener. Prompts: `UI/Library_Modules/library_prompts_controller.py:1748` and `:1770`; the opener label is "Import…" (`Widgets/Library/library_prompts_canvas.py:735`). Skills: `UI/Screens/library_screen.py:22732` and `:22761`; the opener label is "Import skill…" (`Widgets/Library/library_skills_canvas.py:1284`).
- **Slow Rendered view**, a Quirks bullet in `library/media-and-conversations.md` that points at TASK-22660 (still To Do). Rendered mounts Textual's `Markdown` on the whole content (`Widgets/Library/library_media_content.py:224`, `:375`); Raw is the virtualized `VirtualizedRawContent` (`:352`). A document defaults to Rendered whenever a heading, fence or table appears in its sniff window, whatever its size (`Library/library_media_viewer_state.py:200`, `:242`).
- **Change Review commit and push dialogs**, a new "Commit and push from the working tree" subsection under "Change review" in `console/agent-runs-and-tools.md`. This page's "Git actions in change review" section, which had both facts, was dropped by the squash 6a6242ce3a (2026-09-12); the replacement is shorter and was re-checked line by line:
  - Mode label `CURRENT_MODE_LABEL` (`UI/Screens/change_review_screen.py:79`). Gate `get_cli_setting("change_review", "git_actions", True)` (`Workspaces/change_review_provider.py:211`). A subfolder is refused (`Workspaces/git_workspace.py:346`).
  - Commit: run-active refusal before the dialog (`change_review_screen.py:1749`, `:99`); dialog header and fields (`:3667`, `:3689`, `:3693`); warnings (`:329`). The mid-merge refusal comes from the engine's `in-progress-check` at confirm (`git_workspace.py:779`, `:939`) and lands as `Commit failed at …` (`change_review_screen.py:2026`). Success copy is at `:2028`.
  - Push: always confirms; only a single-repository PR skips the dialog (`:2207`). Title `Push from …` (`:3856`); summary lines (`:3902`, `:3906`, `:3910`); every push URL (`:3956`). Never forced (`git_workspace.py:1170`). PR hosts (`:1053`); "push the branch first" (`change_review_screen.py:131`).
- **Tool Packs**:
  - `settings.md`: a "Data & Privacy — Tool Profiles" section, plus its row in the category map. The category sits in the Data & Privacy group (`UI/Screens/settings_screen.py:6658`) with "Immediate actions". Panel copy: `Widgets/Settings_Widgets/tool_profiles_panel.py:369-493`. Import, export and bind reviews: `tool_pack_import_review.py:144`, `:175`, `:354-404`, `:550-585`, `:733-783`. Flows: `settings_screen.py:6208` (Inspect), `:6111` (Exported Tool Pack), `:6468-6471` (Remove), `:6513` (Edit deep link), `:6545`, `:6573` (Bind). Removal rules: `Tool_Packs/service.py:510`.
  - `mcp.md` Permissions mode: a paragraph on the **Tool policy profile** selector (`UI/MCP_Modules/mcp_permissions_mode.py:876`, `:1204`; hint `:89`; origin `ws-` → workspace-managed at `mcp_workbench.py:2236`), linking to the new Settings section.

**Checked against the running screen.** All runs went through run.sh at load 30-70 on this host. The throwaway probes lived in `Tests/UI/test_zz_r17dc_probe.py` and were deleted before the commit; a copy is in the scratchpad (`fu/r17dc/probe_r17dc.py`).
- Media, 235x52 painted cells, nothing checked: `Clear     ○ Export     ○ Review`, then `Select items to enable.`, `○ Analyze` and `○ Delete`. After checking row 0, the first row reads `Clear       Export       Review` and the reason line paints nothing. The pin `test_library_media_render_fixes.py::test_select_mode_bulk_reason_is_painted_with_nothing_selected[wide|narrow]` passed.
- Import row: `test_library_per_click_recompose_t21116.py::test_prompts_import_open_and_cancel_are_canvas_scoped` and `::test_skills_import_open_and_cancel_are_canvas_scoped` passed. They assert focus on the path field on open and on the opener after Cancel.
- Rendered: probe mounting `LibraryMediaContentBody` on generated Markdown. Rendered took 1.11 s at 102 KB (752 blocks) and 11.34 s at 1.02 MB (7,484 blocks); Raw took 0.06 s and 0.07 s. The machine was heavily loaded, so the page gives no figure ("several seconds or more").
- Change Review:
  - Probe: a repository in `git merge --no-commit` state. The commit dialog opened; on confirm the only notice was `('Commit failed at in-progress-check: finish or abort the merge/rebase/cherry-pick first', 'error')`, and nothing was committed.
  - Probe: the push dialog with an upstream and a `pushurl` painted `Push from <repo>` and `main ↑1 → origin/main  ·  pushes to origin (ssh://git@pushtarget.invalid/r.git)`, with buttons Push and Cancel.
  - `Tests/UI/test_change_review_commit_ui.py` and `test_change_review_push_ui.py` (warnings, run-active refusal, pushurl/pushInsteadOf/multi-URL, never forced) and `Tests/Workspaces/test_git_workspace_commit.py` all passed.
- Tool Packs: `Tests/UI/test_settings_tool_profiles.py` gave 1 failed (below), the rest passed. `Tests/UI/test_mcp_permissions_handoff.py` passed; it drives Settings ▸ Tool Profiles ▸ Edit into Permissions with the imported profile chosen.
- Combined covering run (`-n 4`): 2 failed, 119 passed. Neither failure is caused here; this commit changes no code.
  - `test_library_crit8_keyboard.py::test_tab_from_a_library_canvas_never_reaches_the_nav_bar` died on the local `RecoveryRequired` trap. Under a temporary `@pytest.mark.bootstrap_profile` marker (reverted) it passed.
  - `test_settings_tool_profiles.py::test_removal_confirmation_does_not_interpolate_the_profile_id` failed 3 of 3 alone (`assert '' == 'Remove cancelled'`). TASK-36387's notes already record it as red with dev's source, and it is on no PR lane. Reported as a proposed follow-up.

`Tests/Docs`: 44 passed (the stamp contract test included).

Lesson: added to `backlog/docs/lessons-backlog-hygiene.md`. A "no page covers X" finding can be a section a squash deleted. 6a6242ce3a removed 16 headings from this page. Two more of them are still missing, both listed as proposed follow-ups:
- "Reviewing or undoing a turn's changed files" (the card's Undo All);
- "Queued progress from sub-agents" (`report_to_supervisor`).

**Fix round 1 (review r17dc-rev1, five Minor findings).**
- The Media bullet now names Analyze. It reads "○ Analyze" with nothing checked, and with no analysis provider it stays off with its own reason line (code cited above).
- The lesson's heading count is now 16, not 15, and it names the third lost section. Its `comm` command, re-run at 1f1fdba3b4, prints 16 removed headings and three still absent.
- `settings.md`'s Tool Profiles intro notes that Deny is shown as **Off** in Permissions mode.
- `mcp.md`'s Tool policy profile paragraph now sits after the precedence paragraph, so the matrix introduction runs straight into the states list.
- The settings/rag.md history fix is in TASK-41527.

Files: `Docs/User_Guide/library/media-and-conversations.md`, `library/prompts.md`, `library/skills.md`, `console/agent-runs-and-tools.md`, `settings.md`, `mcp.md`; `backlog/docs/lessons-backlog-hygiene.md`.
<!-- SECTION:NOTES:END -->
