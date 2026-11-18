---
id: TASK-41757
title: "The media controller still takes 43 LibraryScreen method callbacks for shell, review-set, Notes and workspace behaviour"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- architecture
priority: medium
references:
- tldw_chatbook/UI/Library_Modules/library_media_controller.py
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/Architecture/test_library_media_wiring.py
- Tests/Architecture/test_screen_size_ratchet.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36952.3.8 AC #3 promises this task, "filed when the media series lands". Coordinator ruling R4 read the parent's AC #2 ("no screen reference or screen-method callbacks for media behaviour") as met once the controller took a typed `LibraryMediaHost` and no constructor argument called a media-named screen method. The shell, review-set, Notes and workspace callbacks were ruled not media behaviour and left for this task. R4 named its own cost if wrong: the owner may want the controller to hold no screen object at all.

At dev 908e9840aa (2026-11-18), after the series merged, `LibraryScreen.__init__` builds the controller at `library_screen.py:3048` as `LibraryMediaController(self, ...)` with 73 keyword arguments, every one a lambda. 43 of them are method callbacks of the form `lambda *a, **k: self._<name>(*a, **k)`:
- **26 shell callbacks,** among them list-entry focus arming and disarming, `focus_library_control`, `focus_library_list_entry`, `focus_library_list_entry_if_current`, `focus_library_rail_action`, `select_library_rail_row`, `acknowledge_library_destination_change`, `library_entry_route_key`, `schedule_library_entry_reconcile`, the choice strips, the stage-pick landing, the narrow-stage return, the open-item surface, the Export canvas, the ingest regions, footer shortcuts, the reader-layout refresh, `run_library_service_call`, `safe_text`, `source_record_id` and `library_source_total`.
- **11 review-set callbacks:** `active_review_set_banner`, `cancel_pending_review_set_resume`, `notify_review_set`, `review_dismiss_receipt_name`, `review_dismiss_undo_worker`, `review_selected_worker`, `review_set_picker_worker`, `review_set_service`, `review_these_worker`, `toggle_reviewed_unguarded` and `walk_active_review_set`.
- **4 sibling reader-layout syncs:** Conversations, Notes, Prompts and Skills.
- **2 workspace-depth callbacks:** `library_workspace_depth_state` and `invalidate_library_workspace_depth_state`.

The other 30 arguments are accessors and setters for single screen fields, among them the Notes ones (`notes_controller_accessor`, `notes_state_accessor`, and the programmatic-focus and restoring-focus pairs). They are not this task's subject: the unbound-call follow-up filed beside this one takes them (its AC #3). Each callback keeps the controller bound to the screen's private methods, and each is one of the lambdas `_LIBRARY_INIT_LAMBDAS` (397 at dev) counts. Related: TASK-36952, the work stream, whose AC #1 asks for the media controller to be constructed without the screen and tested directly; TASK-41051 applies the same rule to services the app wires.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No LibraryMediaController constructor argument calls a LibraryScreen method: each of the 43 callbacks counted at dev 908e9840aa is reached through a typed service the controller receives, or is no longer needed
- [ ] #2 A guard in Tests/Architecture/test_library_media_wiring.py fails if a constructor argument that calls a screen method is added back
- [ ] #3 The notes say whether LibraryScreen itself implements any of the new services, with the owner's ruling if it does (R4's cost if wrong)
- [ ] #4 _LIBRARY_INIT_LAMBDAS is below 397 and re-pinned to its measurement, and LibraryScreen's _BUDGETS row does not rise
- [ ] #5 Every test that reaches a converted callback passes, with any retarget limited to the receiver
<!-- AC:END -->
