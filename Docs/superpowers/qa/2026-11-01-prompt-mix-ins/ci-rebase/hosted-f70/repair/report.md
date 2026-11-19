# f70 UI19 Library readiness: bounded causal diagnosis

No repository source, tests, Git state, thresholds, wait predicates, deadlines, or authority flags were changed. One native explicit-system-event probe demonstrated a lost snapshot publication callback. It reproduces the hosted failure's exact generation/error/DOM metadata; it does not establish the historical hosted event ordering by timestamp.

## Actual observations

- Hosted run41552416723 attempt1, UI19 job125652587027, head `f70eb635f5991d25b7cd5d32d789406d837be32b` (synthetic merge3987574232 into dev2dcf038cb8). `mixins-f70-ui19.log` lines623–625 fails the original healthy-source-DOM wait before note typing: loadedtrue, no lookup/canvas error, rendered0/state1, entrydirtytrue, Notes rail row present/parentactive, current LibraryScreen. This is a publication failure despite a healthy completed source snapshot.
- Hosted log lines1572–1683 also contain an actual onboarding-worker path: `_gather_library_onboarding_evidence` → `_apply_library_onboarding_evidence` → `_sync_library_rail_lifecycle_presentation` → `_sync_library_landing_lifecycle_presentation` → `_sync_library_canvas` → `LibraryLandingCanvas.sync_state`, which raises NoMatches querying `#library-canvas-landing` inside the queryable landing parent. This supplies a concrete native missing-child event, beyond the prior f095 diagnostics.
- Root's unwrapped exact native node passes:1passed30.81s, `/private/tmp/mixins-f70-ui19-local.log`. That observation remains distinct from the hosted failure and from the deliberately scheduled probe.
- Explicit event probe:1failed43.05s, original readiness AssertionError with exactly the hosted diagnostic metadata. The probe did not reach or change the note-link behavior assertions.

## Demonstrated ownership chain

`_apply_local_source_snapshot` publishes source generation1, marks entry dirty, and queues one current reconcile. The landing reconcile calls `_sync_library_canvas(..., then=partial(_finish_library_entry_canvas_sync, ...), allow_screen_fallback=False)`. That exact finish callback lives on the retained landing canvas until native recompose finishes.

Textual's installed `Widget.recompose` holds `async with self.batch()` across its real child removal and `mount_all`. The diagnostic wraps only `LibraryLandingCanvas.mount_all` entry. At this actual mount boundary it observes a current original finish callback, current pending generation, attached/mounted/queryable parent, and zero children. It invokes the original `_sync_library_landing_lifecycle_presentation` once, without modifying state or replacing any handler/callback, then returns the original `mount_all` result unchanged.

The onboarding sync requests no new widget-set shape. `LibraryLandingCanvas.sync_state` immediately patches nonexistent children and raises the same NoMatches as hosted. `_sync_library_canvas`'s failure path unconditionally clears the canvas's callback slot, even though this sync installed no finish callback. `owner-events.jsonl` records original finish callback identity4842228608 before the event, None immediately after; generation0/1, dirtytrue and matching pending remain unchanged. Native remount continues, but no finish remains to call `_complete_library_entry_reconcile`. The unchanged thirty-second healthy-DOM wait then fails. This demonstrates callback loss rather than an outstanding source worker or pure profile-selection failure.

The first command exited before app imports because its newly named bootstrap directory did not yet exist. Creating that private directory corrected setup; only one actual native probe ran. Probe code is diagnostic instrumentation and is not an unwrapped production-source identity claim.

## Scope and proposed smallest repair

`source-proof.json` compares nine complete AST owner functions against dev2dcf038cb8: every owner is unchanged on f70. The current canvas_sync addition is solely the Media focus-restoration helper; the other Media argument is used only on Media entry. The failed journey is still on the startup landing before Notes entry. The demonstrated failure is inherited behavior, not evidence that the new Media path caused Notes startup.

Recommended owner-local repair for review, not implemented here: in `LibraryLandingCanvas.sync_state`, extend the existing widget-set-change recompose condition with `self._recompose_required or self.lock.is_locked`. The method already adopts the latest state first. When native recompose is pending or owns the lock, request its next recompose instead of querying transient children. Existing `PostRecomposeCallback.recompose` detects `_recompose_required`, retains the original finish, and fires it after the final native tree is mounted. This adds lifecycle checks without changing generic callback replacement/focus semantics, route ownership, retries or budgets.

The independent read-only reviewer identified the pending-window sibling: a structural sync adopts new state and sets `_recompose_required` before native recompose acquires the lock. A second sync for that same new widget set would otherwise compare equal and patch the old children. Both pending and running ownership must be covered; a lock-only change is incomplete.

Before adoption, independently review the actual Textual RLock/batch ownership and verify a focused native regression at the real removal/mount boundary. The regression should retain the real snapshot finish, change actual landing copy/state through the original onboarding seam, observe final DOM plus generation settlement, and include normal unlocked in-place sync and stale route/generation behavior. A plain rerun pass cannot establish this repair.

## Exact actual probe command

```sh
PYTHONPATH=/private/tmp/mixins-f70-ui19-diagnosis TLDW_TEST_CONFIG_ROOT=/private/tmp/mixins-f70-ui19-diagnosis/profile /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p owner_probe Tests/UI/test_library_note_links_picker_and_preview.py::test_typing_two_brackets_opens_a_keyboard_picker_that_inserts_a_real_link --basetemp=/private/tmp/mixins-f70-ui19-diagnosis/pytest-tmp --junitxml=/private/tmp/mixins-f70-ui19-diagnosis/owner-probe.xml
```

Output is `owner-probe.log`; metadata is `owner-events.jsonl`; exact instrumentation is `owner_probe.py`. The hosted source log is `/private/tmp/mixins-f70-ui19.log`, with head/job/origin/SHA-256 in `/private/tmp/mixins-f70-ui19-origin.json`.

No broad tests, baseline app runs, live providers, additional profiles, generic callback changes or production fixes were attempted. All probe processes have stopped. Historical hosted ordering remains inferred from matching native path/symptom, not proven by the log alone.
