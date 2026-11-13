---
id: TASK-40332
title: "The Plugins User Guide page is a stub that explains none of the install, authority or restore flows"
status: Done
assignee: ['@claude']
updated_date: '2026-11-13 03:17'
created_date: 2026-11-12 19:36
labels:
- follow-up
- docs
- plugins
priority: low
references:
- Docs/User_Guide/plugins.md
- Docs/User_Guide/index.md
- tldw_chatbook/UI/Screens/plugins_screen.py
- Docs/Plugins/operations.md
- scripts/capture_user_guide.py
dependencies:
- TASK-40132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (docs stream). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** F8 Plugins is a top-level destination with install, authority and restore behaviour, but `Docs/User_Guide/plugins.md:1-26` is marked as a stub: orientation, button names and a 4-row keys table. It does not link `Docs/Plugins/operations.md`, which already covers configure, disable/uninstall, recovery and "After restoring a backup". TASK-32695 owns `Docs/Plugins/*` and never names this page.

**What the screen does** (`plugins_screen.py:52-166`, `Plugins_Modules/{browse_state,details,operations,review}.py`): a Scope line; Installed / Browse / Marketplaces views (keys 1/2/3 switch views inside Plugins; they are not destination keys); Repository... (Git URL or absolute local directory, Git ref default HEAD); Import... (Cursor or Codex); Add source... (Detect, Agent Plugins, Cursor, Claude `marketplace.json`); Preview -> Review install...; a paged list; detail actions (Review preview..., Preview entry, Refresh source, README, Manage...); Authority / History / Candidates / Reset authority; and since TASK-40132 a restore puts plugins back and keeps them off until authority recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 plugins.md follows _template.md and covers each view, the three entry forms through Review install..., the detail actions, the authority and history controls, and the off-until-authority-recovery state after a restore
- [x] #2 It includes one 235x52 capture made with scripts/capture_user_guide.py and links Docs/Plugins/operations.md
- [x] #3 The stub marker is gone from the page and from the index row, and the 1/2/3 keys are described as view keys within Plugins
<!-- AC:END -->

## Implementation Plan

1. Re-read the Plugins screen, its modules and modals at base 26b2e677dd (views, entry forms, detail actions, Manage…, Authority/History/Candidates/Reset authority, the TASK-40132 restore gate); probe the Review install… path for each entry form under a private-profile pytest run (temporary probe, not committed).
2. Add a `plugins` target to `scripts/capture_user_guide.py` that adds a demo local marketplace through Add source…, refreshes it, and previews its entry from Browse; write `Docs/User_Guide/images/plugins/browse-preview.svg` at 235×52.
3. Rewrite `Docs/User_Guide/plugins.md` to the template's sections (no "Verified against" footer): each view, the three entry forms through Review install…, detail actions, Manage…, the authority and history controls, and an "After restoring a backup" section of its own; link `Docs/Plugins/operations.md`; describe 1/2/3 as view keys inside Plugins.
4. Drop the stub marker from the page and the 🚧 from the index rows; update the template's driver-target list; run the Docs tests and the driver through the wrapper; commit.

## Implementation Notes

`Docs/User_Guide/plugins.md` is now a full page in the template's sections (no "Verified against" footer, per CLAUDE.md): layout tour with a capture, the three views, the three entry forms through Review install…, the Manage plugin actions, Authority/History/Candidates/Reset authority, an "After restoring a backup" section of its own (for TASK-40308/40309 to extend), common tasks, keys, related docs, quirks. The index row lost its 🚧; the page has no stub banner.

**Evidence at base 26b2e677dd** (line numbers are this tree's):
- Screen: `UI/Screens/plugins_screen.py:26-32` (1/2/3 view bindings, `/`, Escape), `:52-165` (labels, form placeholders, catalog choices), `:188-201` (narrow < 100 columns, Back to list), `:20` (25 per page), `:289-293` (scope line), `:318-329` (metadata-unavailable status), `:401-414` (empty states), `:453-499` (detail-action labels, Review preview…/Review install… enablement), `:615-658` (Authority… postures and messages), `:693-707` (form titles), `:751-831` (preview/import/add-source outcomes and messages). Details text: `Plugins_Modules/details.py`. Keys typed into an Input do not reach screen bindings (Textual Input consumes printable keys).
- Dialogs: `Widgets/plugin_review_modal.py:355-365` (unlock mismatch message), `:368-438` (setup/unlock, 8-character minimum, confirm field), `:441-503` (Reset plugin authority), `:506-641` (Plugin operations / Persisted candidates), `:644-764` (Manage plugin: interpretation, action list, buttons), `:1596-1603` (Console hint). Reset/setup: `Plugins/coordinator.py:107-117` (`bootstrap` refuses while installations, data roots or the data anchor exist), `:122-139` (`reset` keeps them).
- Restore: `Plugins/service.py:301-321` (unlock runs recovery and approves the restored generation), `:332-338` (`recover`), `Plugins/admission.py:520-521` (`plugin_restore_review_required` refuses every run until then); wording follows `Docs/Plugins/operations.md:97-140` (TASK-40132).
- URL forms: `Plugins/git_source.py:13` (https/ssh/local), `:56-78` (`github:owner/repo`). Import candidates: `Plugins/imports.py:121-187`. Marker needs a secure keyring: `Plugins/authority_store.py:141-147`, `Plugins/service.py:210`.
- Probe (temporary private-profile pytest, deleted, not committed): a Repository… preview of a local plain folder, Review install…, Review action, Confirm reviewed action produced the quoted "Preview captured…", "Preview retained…", "Review ready…", reminders and "Submitted…" text at 235×52. The same probe for **Import…** then **Review install…** raised `AttributeError: 'ImportPreview' object has no attribute 'inspection'` in the review modal's load worker (`plugin_review_modal.py:823`), and worker failures exit the app (`app_lifecycle.py:891-904` defers them to Textual's default). The page warns about it; proposed as a follow-up.

**Capture.** New `plugins` target in `scripts/capture_user_guide.py` (`_plugins`, `:367`; listed in `TARGETS` and the module docstring; `_template.md`'s target list updated): F8, wait for the first load, export. Run through the wrapper: `run.sh … python scripts/capture_user_guide.py plugins` -> `wrote Docs/User_Guide/images/plugins/overview.svg` (viewBox 2885 × 1318.8 = 235×52, no cdnjs). The driver's profile uses the null keyring, so plugin authority cannot be set up there; the capture therefore shows the fresh-profile state ("Plugin metadata unavailable…"), and the page says so. An attempt to drive Add source… in that profile failed with "Operation unavailable…", which is the second quirk on the page.

**Tests.** No new test (the criteria ask for none); `Tests/Docs/` 32 passed (image links, F-key lists, index nav table); local links of plugins.md/index.md/_template.md checked with `missing_local_links`: none missing.

**Review round 2** (coordinator, 2026-11-13):
- Discard draft: on an install review it also releases the previewed copy (`plugin_review_modal.py:1609-1621` -> `Plugins/source_service.py:1276-1300` `_discard_candidate`), while Review preview… stays enabled because `browse.preview` is untouched (`plugins_screen.py:497-499`). The page now says so and says to preview again.
- Common task "Add a marketplace" step 4 now links step 4 of "Install a plugin from a repository or a folder" (it pointed at itself).
- Cancel operation / Cancel affected work: enabled from `runtime_status["cancel_operation"]` (`plugin_review_modal.py:1564-1566`), which `Plugins/service.py:459-483` sets for a revision drain in phase reserved/waiting and for a saved-data operation in phase waiting. Drains are reserved only by `coordinator.review_selection` (`:279`, Save component selection) and `review_revision` (`:321`; Check for update via `source_service.py:963/983`, Rollback via `coordinator.py:352`); saved-data operations come from `data_cleanup._begin_data_operation` (`:1113`), used by Delete exact saved roots and Resume pending saved-data cleanup. `RevisionDrain.cancel` withdraws the proposal before committing, `cancel_work` cancels the runs the drain waits for (`Plugins/revisions.py:148-170`). The sentence now names exactly those cases.
