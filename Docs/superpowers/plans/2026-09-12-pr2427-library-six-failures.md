# PR 2427 Library regression reconciliation

> **For agentic workers:** Use superpowers:subagent-driven-development, with
> separate spec and quality reviews before integrated verification.

**Goal:** Repair the six recorded Library failures without weakening their
geometry, focus, scroll, identity or authority contracts.

**Architecture:** Keep the accepted TASK-32346/32360 presentation and all existing
runtime owners. First correct five test-side contracts; trace the sixth browse
receipt failure before proposing any runtime change. No new owner or cap increase.

**Tech Stack:** Python, pytest, Textual Pilot, native macOS FD observation.

ADR required: no for test-only reconciliation.
ADR path: existing ADR-031; TASK-32346, TASK-32360 and TASK-32175.
Reason: preserve accepted interface behavior and observe settled framework state.
Any demonstrated runtime repair must be specified after its cause is established.

## Baseline and ownership

- Frozen published checkpoint: `f37e9eb2b4037547905df273ca9673d6e7d0d9c9`.
- Work only in `.worktrees/pr2427-review-recovery`; root alone writes Git.
- Preserve the unrelated untracked reader-paydown plan without opening it.
- Baseline RED: `/private/tmp/pr2427-dev934-original-regressions.G1FiDJ/pytest.log`
  records 38 passed / 6 failed and zero final SQLite/instance locks.
- The unchanged isolated receipt case fails scroll 0 versus 5 at
  `/private/tmp/pr2427-notes-return-diagnosis.Tmfyz3`; its separate 13 SQLite /
  one instance-lock finding remains open despite the combined run's clean end.

## Task 1: Five test-side corrections

Only implementation file: `Tests/UI/test_library_shell.py`.

- [x] In `_assert_task8_compact_chrome` (the 60x20 helper), replace the obsolete
  authority-line noun expectation with the exact current owner: source-strip
  `#library-notes-source-database` Button label `Library notes`, selected,
  displayed, nonzero region, contained in the visible one-row strip and screen;
  `#library-notes-source-files` label `Folder files`, not selected, also contained.
  Require the narrow authority line not to start `Library notes · `, retaining
  its exact two rows and all existing loading/Next and total geometry assertions.
  Do not alter the separate >=64-column authority-prefix contract.
- [x] In `test_library_note_footer_covers_navigator_create_sync_and_exit`, expect
  the exact narrow rail-input context:
  `esc focus rail |  typing in field |  after esc: / focus search | F6 next pane`.
  Preserve all prior navigation, focus, ancillary and non-typing assertions.
- [x] In `test_notes_footer_registration_preserves_media_route_typing_hints`,
  expect the exact wide context tuple:
  `(("", "typing in field"), ("esc", "focus rail"),
  ("", "after esc: / focus search · s select"), ("F6", "next pane"))`.
  Keep route source, field attachment/identity and non-typing assertions.
- [x] In `test_library_note_pilot_delete_pending_locks_and_cancel_restores_context`,
  await `pilot.wait_for_scheduled_animations()` after the focused/visible Delete
  predicate and before `origin_scroll` capture; await it again after cancel/focus
  settlement and before the unchanged exact scroll comparison. Never round,
  approximate, force the scroll or remove focus/mutation assertions.
- [x] Verify the five failing parameters, all parameters of the shared compact
  row-allocation tests, the existing >=64 prefix control, and native FD output.
  Use exact selections with fresh `mktemp` report directories; retain baseline
  failure evidence. Run affected-file fatal Ruff and whitespace checks.
- [x] Obtain independent spec review, then quality review; fix actual findings.

## Task 2: Diagnose browse-return loss and isolated fixture retention

Task 1 follow-up, accepted TASK-32360 contract: its recorded 33-cell browse
actions do not fit the 32-cell narrow pane, so Select occupies a separate
one-row `#library-notes-browse-actions-overflow`. Add that exact row to only
the normal and filtered-empty expected maps and change their flexible content
height from six to five. Preserve every remaining geometry assertion and rerun
the same 16-node native cohort. No production behavior or ADR changes.

- [x] Trace `_restore_library_notes_browse_return_receipt`, targeted sync,
  width-dependent recomposition, focus restoration and scroll maxima using
  temporary observation only. Preserve the initial scroll5, captured receipt5,
  note identity and independent rail scroll assertions.
- [x] Establish whether the failing observation is unsettled layout or a later
  runtime write. Do not infer the cause from one clamp or insert a blind wait.
- [x] Attribute the isolated app resources to their actual builder and reuse
  established exact-owner fixtures if applicable; no shared/global cleanup.
  Fresh native RED `/private/tmp/pr2427-resize-approved-red.yCI1mj` reproduces
  13 SQLite handles plus one instance lock, all attributed to this test's
  factory products (workspace/subscriptions/evals/collections, including a
  second collections thread handle). Import the existing
  `close_owned_console_resources`, `close_owned_console_test_apps` and
  `close_owned_console_workers` fixtures only in the file-notes workspace
  module. All 20 local factory calls already cross its `_build_test_app` seam;
  do not alter test bodies, shared fixtures or production lifetime. Verify the
  isolated failing receipt's terminal resource snapshot separately from its
  behavioral failure, then run existing cleanup fault controls and the complete
  affected file once runtime is frozen. No new ADR for fixture-only reuse.
- [x] Record a bounded repair and its controls only after the cause is proven.

## Task 3: Integrated qualification and checkpoint

- [ ] Freeze all changed source and rerun the original six plus relevant adverse
  controls; then complete affected files, partitioned only by an independently
  checked exact-once collection if needed. No full-repository sweep.
- [ ] Record behavior and terminal native resources separately. Review before
  commit/push; publish with a fresh exact remote lease if a rebase is involved.
- [ ] Keep 15 size guards, earlier resource/preload/CSS gates, new dev integration
  and final PR review/CI open unless independently repaired and verified.

## Implementation notes and current evidence (2026-09-12)

Task 1 changes only `Tests/UI/test_library_shell.py`: accepted presentation
contracts and public animation settlement, retaining every geometry, identity,
focus and exact-scroll assertion. Sequential independent spec and quality
reviews approved the slice. Fatal Ruff and `git diff --check` pass.

Native targeted run `/private/tmp/pr2427-library-five-final.JzOESi/pytest.log`:
**14 passed / 2 failed**, 32.86 seconds, 16 collected nodes. All five originally
failing Shell nodes pass, as does the separate 100-column authority-prefix
control. All shared compact-helper callers were exercised. The two newly
exposed failures retain their original assertions: normal list and
filtered-empty region expect six rows but receive five. Their cause remains
under separate investigation; complete-file qualification is not claimed.
The final native inventory has six descriptors and no SQLite or instance lock.

Follow-up verified: TASK-32360's documented Select overflow row accounts for
the missing content row. Only the two expected maps now pin that extra row
and five-row content. Independent spec and quality reviews approved the delta.
The identical fresh native cohort at
`/private/tmp/pr2427-library-five-post-overflow.3XSb0t/pytest.log` passes
**16/16 in 33.10 seconds**, with no final SQLite or instance-lock descriptors.
Fatal Ruff and whitespace pass. This supersedes the two geometry failures
above, not the remaining runtime receipt or isolated-resource findings.

The remaining browse-return failure is a runtime regression. Temporary trace
`/private/tmp/pr2427-notes-scroll-trace.0IQ6nq/trace-animations.jsonl` establishes:
the receipt retains `(0, 5)`, the after-layout callback restores and records
five with max-scroll 21, then `LibraryNotesCanvas.on_resize` changes its measured
width from 38 to 68 while `pane_width == 0`. A toolbar-shape flip queues a direct
recompose with no post-recompose callback. Its replacement list starts at zero
and focus drops to the shared grip. Public animation settlement does not repair
this. No runtime code or receipt assertion was changed for the diagnosis.

User-approved bounded runtime repair (2026-09-12): make the Notes
geometry-triggered recompose participate in the existing Notes semantic-focus
and exact-scroll restoration boundary. Capture before detachment and restore
through the existing canvas post-recompose callback and controller, retaining
newer explicit callbacks and user-intent guards. No new scheduler retries,
generic protocol or Media ADR-104 expansion. A same-ID focus-only restore does
not retain exact scroll and is insufficient. Keep the original failing receipt
test and add adverse controls for newer focus/action intent and unchanged
toolbar decisions; qualify the complete file-notes workspace, Notes wave-list
and crit10 Notes detail files once source is frozen. This is a repair of the
existing Notes contract; no new ADR is proposed.

The isolated 13-SQLite/one-instance-lock finding remains open. The file-notes
workspace module has 20 real-app calls through its local `_build_test_app`, but
does not import the existing exact-owner app fixtures. This is an ownership
lead, not verified cleanup: no fixture change or clean isolated rerun yet.

Approved-turn cleanup follow-up: the seven fixture-import lines leave all
non-import test code unchanged. The isolated control at
`/private/tmp/pr2427-notes-owned-resources.jjT0q9` still fails the original
scroll assertion (zero versus five) but ends with zero SQLite/instance-lock
handles, six descriptors total. Both independent review stages approve this
exact-owner change. The two complete existing cleanup-control files pass
32/32 in 4.82 seconds with zero SQLite/locks at
`/private/tmp/pr2427-notes-cleanup-controls.CFc9z3`. This supersedes the isolated
retention finding only; complete file-notes qualification remains pending.

## Task 2 implementation: approved resize preservation

ADR required: no.
ADR path: N/A; preserve existing Notes restoration and TASK-32175/32360 contracts.
Reason: wire a missing automatic-recompose entry into existing behavior; no
new owner, stored state, service protocol or Media ADR-104 extension.

Files: `tldw_chatbook/Widgets/Library/library_notes_canvas.py` owns the resize
trigger; `tldw_chatbook/UI/Library_Modules/library_notes_controller.py` already
owns semantic restoration and guards. Thread its existing optional restore guard
through immediate focus and the already deferred scroll callback; preserve
existing callers' behavior with the default guard. Add focused
controls to `Tests/UI/test_library_notes_wave_list.py`; retain the existing
`Tests/UI/test_library_file_notes_workspace.py` receipt case unchanged.
If the Screen's existing `_restore_library_notes_after_targeted_sync` delegator
is used, update only its signature/forwarding in
`tldw_chatbook/UI/Screens/library_screen.py` for the optional guard. No new
Screen-owned behavior or size-cap change is authorized.

- [x] Reproduce the original receipt case once with the unchanged native runner.
  Expected RED is exact final list offset zero versus five, not setup failure.
- [x] Add mounted resize controls before implementation: shape-changing resize
  retains semantic row focus and exact nonzero offset; no shape change does
  not rebuild; an already queued explicit callback survives; a later callback
  supersedes the automatic restore; newer user focus outside the canvas or a
  new route is never pulled back. Specifically change user intent after the
  immediate focus restore but before the deferred scroll callback and assert
  that the newer offset/focus wins. Use existing harnesses and real callback
  dispatch, not mock-only call-count evidence.
- [x] In the actual shape-change branch only, capture the current Notes
  identity before detachment if a live Notes owner can provide it. Do not
  overwrite `has_pending_recompose_callback`. Queue a one-shot restore through
  `queue_after_recompose`, calling the existing Notes targeted-sync restoration
  only while the captured owner/route and current-user intent remain eligible.
  A newer explicitly queued callback replaces this naturally. Keep standalone
  canvas harnesses functional, with no new fake Screen API requirement. Preserve
  all existing resize width/shape gates and toolbar layout decisions. Reuse the
  existing user-intent guard at both immediate and deferred applications (the
  latter currently omits it); do not add sleeps, scheduler turns,
  polling, general protocols or global fallback restoration.
- [x] Run the new controls and original receipt case; require exact focus/scroll
  checks to pass. Keep native resource findings separate from behavior.
- [x] Obtain independent spec review then quality review of the actual diff.
- [x] Freeze implementation and qualify the complete wave-list, crit10 details
  and file-notes workspace files plus `Tests/UI/test_library_canvas_sync_defects.py`
  (existing default-guard caller behavior) using the unchanged native observer, exact
  selected paths and fresh report directories. Run fatal Ruff and whitespace.
  Complete Shell qualification remains a separate large affected-file gate.
- [ ] Record results and reviewed changes in TASK-31932, this plan and the
  existing automation. Do not merge or claim whole-PR qualification.

### Reviewed runtime implementation and focused qualification

The canvas captures the existing Notes recompose identity only on a toolbar
shape change with live descendant focus and no pending explicit callback.
It queues the existing targeted restore with the captured three-generation
guard. The controller passes that optional guard to both immediate focus and
deferred exact-scroll restoration; the Screen only forwards the optional
argument. Existing callers retain the default behavior. No CSS, cap, new
state owner, polling or scheduler retry was added.

The first test controls were replaced during review with mounted Library tree
controls. To establish RED evidence for those final tests, temporary mutations
removed only the automatic callback branch and then only the deferred guard
argument. The first fails because focus falls to the adaptive grip
(`/private/tmp/pr2427-mutation-autoqueue-red.X4b5bG`); the second fails because
old offset seven overwrites the user's newer offset four
(`/private/tmp/pr2427-mutation-deferredguard-red.0gptT1`). Both mutations were
restored, with all three runtime SHA-256 values matching the reviewed originals.
These are diagnostic failures, not native cleanup qualification.

Final native focused qualification at
`/private/tmp/pr2427-quality2-native-final.gak71T/pytest.log` passes **8/8 in
13.49 seconds**, including the unchanged original browse-return case. The final
inventory contains six descriptors and zero SQLite/instance-lock handles.
Independent spec and final quality reviews approve the actual implementation
and exact outside-focus/deferred-scroll assertions. Fatal scoped Ruff and
whitespace checks pass. The complete Notes wiring file passes **15/15** and all
seven preflight checks pass at `/private/tmp/pr2427-resize-architecture.JHGkcv`.

The four complete affected Notes files pass **233/233 in 622.33 seconds** on
frozen source at `/private/tmp/pr2427-resize-complete-files.vgDKHL`. The process
exited zero; its final inventory contains seven descriptors and zero
SQLite/instance-lock handles. The three warnings concern installed deprecated
dependency APIs. Complete Shell qualification is now running separately at
`/private/tmp/pr2427-resize-shell-complete.CAd9yu`; it and the previously recorded
merge gates remain open. This is not a whole-PR success claim.

Independent read-only coverage inventory confirms the four-file cohort plus a
complete `Tests/UI/test_library_shell.py` run covers all original 44 executed
regressions and the prior 16-node controls. The original 40 selectors expand
to six Workspace nodes and 38 Shell nodes (the backlinks selector has five
parameters). The prior controls split into one Crit10Details node and 15 Shell
nodes. No original selector is omitted; this avoids an unnecessary separate
44-case replay after both complete-file runs qualify the same frozen source.
