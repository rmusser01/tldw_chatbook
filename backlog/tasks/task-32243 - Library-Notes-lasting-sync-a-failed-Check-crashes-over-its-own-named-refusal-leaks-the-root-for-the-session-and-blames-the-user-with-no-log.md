---
id: TASK-32243
title: >-
  Library Notes lasting sync: a failed Check crashes over its own named refusal,
  leaks the root for the session, and blames the user with no log
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 15:06'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - sync
  - p0
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PROVEN, three parts, from a bisect run in parallel with the critique. "Keep a folder synced" is one of the three worlds the canvas advertises and the one the guide gives the most space to; both assessors hit this on their first real attempt and neither could ever retry (C caps 73/75/76/77 on two different folders; D caps 72/73/74, reproduced).

(1) On any non-OWNER admission, `notes_sync_runtime.py:1598` `_ensure_lease` calls `_publish(..., persist=True)` -> `notes_device_state_store.update_root_status` (:1005), which raises `NotesDeviceStateError: The requested sync root does not exist` for a root setup that was never persisted. The intended `RuntimeError("root_lease_unavailable")` at :1771 is therefore never reached: the honest, named failure is destroyed by a secondary crash.

(2) Each failed Check leaks a `_root_paths` entry (:1769-1774 -- the pop is skipped when `_ensure_lease` raises), so the next attempt on the same folder is rejected with `lasting_root_overlap`. The folder is permanently un-admissible for the rest of the session, invisibly, even after the user fixes the real cause. There is no way back but a restart.

(3) `library_notes_sync_controller.py:806-814` wraps the call in a bare `except Exception:` and substitutes the fixed string "Check failed. Review the folder and settings, then try again." -- no reason, wrong advice, unretryable. The module contains zero `logger` references, so nothing is written anywhere.

Incidence caveat, stated plainly: the trigger in this harness is environment-specific. The fixture vault sat inside the profile's config directory (`private_path_overlap`) and everything under `/private/tmp` carries group `wheel` != egid, tripping `notes_sync_filesystem.py:191-192` `owner_group == os.getegid()` -> `unsupported_metadata` for all 60 files -> `root_discovery_incomplete` (filed separately as task-32244). A realistic vault under `$HOME` is admitted in 0.19 s. So the *frequency* of hitting an inadmissible folder is unknown; the three failure-handling defects above are not conditional on it. The 33.4 s both assessors would otherwise have reported is host contention -- there is no timeout on this path and failure normally arrives in 0.1-1.3 s.

Provenance: NOT a wave regression. Identical traceback at `c4a7b1911f`, and every failure site is original to the feature commits (`_ensure_lease`/`_publish` -> `0dae7bb8b3`; `notes_sync_coordinator.private_path_overlap` -> `eea5244886`; `notes_sync_filesystem` group check -> `d2f5901467`). Pre-existing since 2026-08-21, newly found. Nothing pins any of it: `grep root_lease_unavailable\|root_discovery_incomplete Tests/` returns 0 hits and `grep "Check failed" Tests/` hits only Watchlists.

Smallest fix, in one paragraph: make `_ensure_lease` fail with its own named error instead of crashing inside `_publish` on an unpersisted root; pop `_root_paths` in a `finally` so a failed Check leaves no residue and the same folder can be retried; catch the named admission states in the controller and render each with its own next action -- the screen already owns exactly that grammar four rows above the button, where a disabled radio reads "Unavailable - server sync-folder capability not installed"; and log the exception with the folder and the state name.

Everything downstream -- Activate, the conflict path, Manage sync folders -- is unreachable while this stands, so the guide documents a chapter no one can enter (task-32269).

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A lasting-sync Check that cannot admit a folder reports the runtime's own reason and a next action at the control, not a fixed 'Review the folder and settings' string
- [ ] #2 `_ensure_lease` surfaces its named failure (`root_lease_unavailable`) rather than being masked by a `NotesDeviceStateError` raised from `_publish` on a root that was never persisted
- [ ] #3 A failed Check leaves no `_root_paths` residue: retrying the same folder in the same session reaches the same admission decision a fresh session would reach
- [ ] #4 Every admission failure on this path is logged with the folder and the state name
- [ ] #5 Covered by a test that a lease failure surfaces `root_lease_unavailable` rather than `NotesDeviceStateError`
- [ ] #6 Covered by a test that a second Check on the same folder is not rejected with `lasting_root_overlap` because of the first
- [ ] #7 Covered by a test that the controller renders a distinct named reason per admission state
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce headlessly: review_setup() against a vault inside the profile dir and one under /private/tmp; capture the NotesDeviceStateError traceback and the leaked _root_paths entry.
2. Write RED tests: (a) rejected admission on the setup path raises a reason-carrying error, not NotesDeviceStateError; (b) a failed Check leaves _root_paths empty and the folder is admissible on retry; (c) the controller renders distinct copy per reason and logs a metadata-only warning.
3. Fix: _ensure_lease(..., persist: bool) with persist=False on the unpersisted setup/activation paths; release the setup authority (which pops _root_paths) on any failure, not only on observation failure; carry admission.reason_code on the raised error; reason-mapped controller copy plus a path-free logger.warning; keep the bare except.
4. Re-pin the diagnostic inventory, run the touched suites against a dev baseline.
5. Live verification at 235x52: refusal copy on an in-profile vault, then a successful Check on a $HOME vault in the same session.
<!-- SECTION:PLAN:END -->
