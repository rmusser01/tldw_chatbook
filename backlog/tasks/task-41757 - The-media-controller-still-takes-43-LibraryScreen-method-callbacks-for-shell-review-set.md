---
id: TASK-41757
title: "The media controller still takes 43 LibraryScreen method callbacks for shell, review-set, Notes and workspace behaviour"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 16:14
updated_date: '2026-11-22 21:45'
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
- [x] #1 No LibraryMediaController constructor argument calls a LibraryScreen method: each of the 43 callbacks counted at dev 908e9840aa is reached through a typed service the controller receives, or is no longer needed
- [x] #2 A guard in Tests/Architecture/test_library_media_wiring.py fails if a constructor argument that calls a screen method is added back
- [x] #3 The notes say whether LibraryScreen itself implements any of the new services, with the owner's ruling if it does (R4's cost if wrong)
- [x] #4 _LIBRARY_INIT_LAMBDAS is below 397 and re-pinned to its measurement, and LibraryScreen's _BUDGETS row does not rise
- [x] #5 Every test that reaches a converted callback passes, with any retarget limited to the receiver
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Owner ruling 2026-11-22: the screen implements protocols; no separate service objects.

1. In `library_media_controller.py`, declare four typed `Protocol`s with public members, grouped as the ruling suggests: shell/focus (26), review-set (11), sibling reader-layout sync (4), workspace depth (2). `LibraryMediaHost` extends all four, so the controller's existing `screen: LibraryMediaHost` handle carries them; its own body stays the 19 framework services.
2. The controller drops its 43 callback keyword parameters and their `_fn` attributes. Each of its 43 binding properties keeps its name (the moved bodies stay byte-for-byte) and returns the host's public member instead of a stored callback.
3. `LibraryScreen` implements the protocols structurally: one block of class-level public names, each the existing method under its protocol name (no new methods, so the `_BUDGETS` method count holds). The 43 lambdas leave the `LibraryMediaController(...)` call. Full renames were measured first; if their blast radius (callers outside the screen, dual-receiver dispatchers, tests) is out of proportion, keep the private names for their other callers and record why, per the coordinator's size-budget allowance.
4. Guards in `Tests/Architecture/test_library_media_wiring.py`: no constructor keyword is a lambda/partial/bound method calling a screen method (AC #2, shown red by re-adding one); `LibraryScreen` implements every service-protocol member with a matching parameter list; the typed-host reach check accepts the service members.
5. Re-pin `_LIBRARY_INIT_LAMBDAS`, `LibraryScreen`'s `_BUDGETS` row and the controller's module-size row to measurements, with dated comments.
6. A/B the covering tests (every test file naming a converted callback, plus the media/review-set suites) at base and head; retarget only receivers (rename-only patches) where a test patched a private screen name the controller now reaches publicly. Run `Tests/Architecture` serially, preflight, ruff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Code commit `9d1b0d1c6e` on base `8a55555798` (dev, 2026-11-22). Evidence and scripts: `.worktrees/_arch-sdd/scratchpad/p4ms/`.

**What changed.** The 43 screen-method callbacks are gone from `LibraryMediaController(...)`. `LibraryMediaHost` now extends four typed `Protocol`s in `library_media_controller.py`: `LibraryShellServices` (26 members), `LibraryReviewSetServices` (11), `LibraryReaderLayoutSync` (4) and `LibraryWorkspaceDepthServices` (2). The controller calls them on the host it already holds. Each of its 43 group-(a) binding properties keeps its private name, so the moved bodies stay byte-for-byte, and now returns `self._screen.<public member>` instead of a stored callable. None of the 43 was dropped as "no longer needed": `_MEDIA_CONTROLLER_BOUND_NAMES`, the AST census of names the moved bodies use, still lists all 43, so each one became a protocol member.

**AC #3: `LibraryScreen` implements all four new protocols** (structurally; it does not subclass them). Owner ruling, 2026-11-22: "**The screen implements protocols.** Extend the existing `LibraryMediaHost` pattern: replace the 43 callbacks with a few typed `Protocol`s ... that `LibraryScreen` implements, and have the controller call them on the host object it already receives. The controller still holds the screen object; that is accepted. Do not build separate service objects." That is R4's named cost: the controller keeps a screen object, by the owner's decision.

**Public names.** Two members were already public: `action_library_narrow_stage_return` and `request_library_reader_layout_refresh`. The other 41 are one block of class-level names at the end of `LibraryScreen`, each bound to the existing private method (`select_library_rail_row = _select_library_rail_row`). They are not wrappers, and the private names stay.
- **Why not a full rename.** At base, the 41 private names have 765 references in about 130 test files (`_select_library_rail_row` alone has 202 in 43 files) and about 250 in source outside the screen. Several are resolved by shared dispatchers on either the screen or a forwarding controller:
  - `canvas_sync._sync_library_canvas` calls `screen._sync_library_{notes,prompts,skills}_reader_layout_from_shell()` and probes `getattr(_screen, "_focus_library_list_entry_if_current", None)`;
  - `library_browse_route_swap` calls `screen._library_entry_route_key()`;
  - `library_notes_sync_attention` probes `getattr(host, "_run_library_service_call")`.

  If the screen's method were renamed, those `getattr`-default and `except Exception` paths would silently skip the screen leg, unless every forwarding controller also gained the public name.
- **Budget.** The coordinator's method-names ruling allows keeping the private name when the size budget allows it. It does: the `_BUDGETS` row falls 32828 -> 32753 lines, and the method count holds at 1288, because the aliases are assignments, not methods.
- **Patches.** Because the public names are bound when the class body runs, a patch of the private name does not reach the media controller. The block's comment says so.
- **Follow-up.** Proposed: move the remaining callers to the public names and drop the private ones.

**Name collision found on the first head run.** As a protocol member, `register_footer_shortcuts` shadowed `BaseAppScreen.register_footer_shortcuts(*, source, shortcuts)`: every Library mount raised `TypeError`, and 785 covering tests went red. The member is now `register_library_footer_shortcuts`. The conformance test checks every class in `LibraryScreen.__mro__[1:]`; it had checked only Textual's `Screen`. The lesson is recorded in `lessons-testing-evidence.md`.

**AC #2.** The guard is `test_no_media_controller_argument_calls_a_screen_method` in `Tests/Architecture/test_library_media_wiring.py`.
- It fails on a constructor keyword whose value is a lambda calling `self.<name>(...)`, a `partial` of `self.<method>`, or a bound `self.<method>`. Methods include the alias names. The 30 accessor and setter lambdas pass.
- Red at base, naming all 43 (`red-guard-base.log`).
- Negative controls at head, each reverted (`negctl.txt`): a lambda to `self._safe_text`, `functools.partial(self._safe_text)`, a bound `self.select_library_rail_row`, and a lambda to the public `self.focus_library_control`. Each went red naming its keyword.

Two more tests guard the protocols:
- **`test_library_screen_implements_the_media_service_protocols`** pins the 43 members by protocol. It checks that each resolves on `LibraryScreen` with the same descriptor kind and parameter list (names, kinds, defaults) and shadows nothing inherited. It is red at base. Negative controls: dropping the `safe_text` alias ("missing on LibraryScreen"), a changed protocol default ("signature differs"), and the `register_footer_shortcuts` name ("shadows ['BaseAppScreen']").
- **`test_media_controller_takes_a_typed_host_instead_of_the_screen`** now accepts the service members as host reads, and still goes red on a planted `self._screen._safe_text`.

**AC #4.**
- `_LIBRARY_INIT_LAMBDAS`: 397 -> 354 (-43).
- `_LIBRARY_UNBOUND_TEST_CALLS`: unchanged at 511, since no call on the class or binding was added or removed.
- `LibraryScreen` `_BUDGETS` row: 32828/1288 -> 32753/1288.
- `library_media_controller.py` module-size row: 11085 -> 11136 (+51: the protocols +125, the host's bases +1, the parameters and stored callables -89, prose +14), with a dated comment.

**AC #5: retargets.** Each is on the same screen object with the same arguments and assertions. Each was red at head before the change and green at base, and is green after it:
1. `Tests/UI/test_library_media_render_fixes.py::test_dismiss_receipt_paints_undo_at_the_items_pane_width`: `screen._review_dismiss_receipt_name = ...` -> `screen.review_dismiss_receipt_name = ...`.
2. `Tests/UI/test_library_rail_switch_keeps_reader_and_note.py::test_controller_hosted_media_sync_re_requests_an_armed_list_entry_focus`: `monkeypatch.setattr(screen, "_focus_library_list_entry_if_current", ...)` -> `"focus_library_list_entry_if_current"`.
3. `Tests/UI/test_library_review_round_t21116.py::test_routine_snapshot_midswap_still_runs_the_projection_follow_up`: `original_arm = screen._arm_library_list_entry_focus` and `screen._arm_library_list_entry_focus = spy_arm` -> the public name (2 lines).
4. `Tests/UI/test_library_shell.py::test_library_resize_geometry_rejects_transient_without_polling`: the one retarget that is more than a rename.
   - The test is parametrized over receivers. The media param's receiver (the controller) reaches the allocation check through the public member, but the screen's own syncs call the private name.
   - So the patched attribute follows the receiver (`"library_adaptive_reader_allocation_is_current" if media else "_library_adaptive_reader_allocation_is_current"`), with the same object, the same Mock and the same assertion.

**Test evidence** (all through `run.sh`, xdist `-n 4` unless noted).
- **Covering set:** the 133 test files outside `Tests/Architecture` that name a converted callback or the media controller.
  - Head before the collision fix: 2557 failed and 13 errors (785 the `TypeError` above; most of the rest the local RecoveryRequired).
  - Head after the fix, on those 2570 ids: 1145 failed and 4 errors.
  - Base on those 1149: 1144 failed too. Of the 5 that failed only at head, 4 are retargeted above, and `test_artifacts_dreams_rows.py::test_enabled_dreams_with_no_rows_renders_none_yet` passed on its rerun.
- **The 1144 fail-at-both ids under a temporary `@private_profile_test` marker** (`tempmark2.py`; files restored and diff verified): head 110 failed and 4 errors; base 111 failed and 4 errors.
  - The 7 ids that differ each flip on both sides across two reruns per side (`tm-rerun.log`).
  - The common failures' messages match, apart from log-order noise.
  - `test_programmatic_media_row_restore_keeps_reader_until_user_keys` fails 2 of 3 runs on each side.
- **`Tests/Architecture`, serial:** 1494 passed and 5 skipped (base 1492 plus the 2 new tests).
- **Static checks:** `preflight.sh` (Python 3.12) passed; `ruff check --select F` is clean on the changed files, with no new B/ASYNC/PLE findings per file; `Tests/lint/ruff_gate.py` passes.

**Files:**
- `tldw_chatbook/UI/Library_Modules/library_media_controller.py`
- `tldw_chatbook/UI/Screens/library_screen.py`
- `Tests/Architecture/test_library_media_wiring.py`
- `Tests/Architecture/test_screen_size_ratchet.py`
- `Tests/Architecture/test_library_modules_size_ratchet.py`
- the four retargeted `Tests/UI` files above
- `backlog/docs/lessons-testing-evidence.md`

No User Guide change: behaviour is unchanged.
<!-- SECTION:NOTES:END -->
