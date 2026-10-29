# Post-autosave Note reopen investigation

Read-only investigation, 2026-10-29 UTC. No product, permanent test, Backlog,
or Git changes. App-test slot released to root after the third native run.

## Finding

The reopen helper can press a newly mounted Navigation row while another
response from the same fresh tree visit still owes a canvas recompose. The
press is queued on the Button, then native recomposition retires its parent
before the message bubbles. No note load begins; the editor-arm wait therefore
cannot succeed. This is a lost row action, not a completed open superseded by
refresh, an autosave persistence failure, or a bracket-picker failure.

The controlled native probe reproduced the exact CI line245 failure after the
original picker insertion and real autosave assertions had passed. The original
Linux CI log did not record row identity/message delivery, so its historical
event ordering cannot be proved directly from that log. The source gap, a
natural successful run exposing the pending publication at the press, and an
ordered native reproduction establish the mechanism on the failing path.

## Source trace

- `Tests/UI/library_note_links_support.py:144–166`: `_open_notes` presses the
  Notes rail, then waits for any tree note row; `_open_note` subsequently finds
  its target by title, presses it, and retains `_armed_editor` unchanged.
- `Tests/UI/test_library_shell.py:3880`: the generic selector helper pauses
  once, but can return the original match when the post-pause query is empty.
  The note helper's subsequent target-row wait repairs absence, but does not
  wait for all publications from the new visit.
- `UI/Library_Modules/library_notes_controller.py:5573`:
  `_try_switch_retained_library_notes_route` closes the prior session, starts
  a fresh tree visit on browse reentry, and synchronizes the canvases.
- `UI/Screens/library_screen.py:17789`, `17917`, `18170`: the fresh visit
  requests root/open-branch slices independently; each applied response can
  synchronize the Notes canvas. Placements can paint the target before the
  folders response arrives.
- `Widgets/Library/library_notes_canvas.py:2001`: list `sync_state` queues
  `refresh(recompose=True)`; installed Textual's native `Widget.recompose`
  removes and remounts its children.
- Installed Textual `Button.press` posts `Button.Pressed` on the Button.
  `MessagePump._on_message` bubbles only while `is_parent_active` and
  `is_attached` hold. Attachment alone does not protect a queued action.

## Native evidence

All runs used the approved `run-targeted.py`, native project Python, separately
owned 0700 profile/TEMP roots, the original 180s limit, and the exact original
test selector. No timeout, privacy guard, ACL, fixture, service, or getter was
replaced. Observation/probe code and receipts are under
`.test-tmp/textual-37266-37268/ci-note-open/`.

1. **`original-trace-01`: 1 failed, XML119.748s. Unqualified for reopen.**
   The unchanged journey failed earlier at line139: initial editor arming
   preceded title focus, keys navigated away, and title lookup raised
   `NoMatches`. No reopen occurred. The sandbox's earlier private-directory
   creation refusal did not launch pytest and is also not mechanism evidence.
2. **`titleprep-trace-01`: 1 passed, XML99.143s.** A scratch wrapper added only
   an actual Pilot click on the initial title control and waited for that
   control's focus before the existing title typing. Every original picker,
   Escape/latch, body, caret, focus, persistence, navigation, backlink and OS
   URL assertion passed. At first `_open_notes` return, rows had become empty
   and both fresh branches were loading. The subsequent original row wait
   found Navigation. Its Button was attached/nonpruning at press, with
   placements loaded but folders still loading. The press won this ordering
   and opened the target. The later folder response still recomposed Items.
3. **`held-delivery-trace-01`: 1 failed, XML125.810s, exact line245 timeout.**
   Same initial title-click preparation. After the actual folder service had
   returned its real answer, its publication was held until the original
   Navigation Button's queued `Pressed` reached its own message pump. Delivery
   of that message was then held until native recomposition retired the
   parent. Both gates had a five-second ceiling; the actual message hold was
   47ms, and both holds released inside the App. The real SQLite/services and
   original row action remained intact.

The third run's first-reopen timeline, relative to helper entry:

| Time | Receipt |
| --- | --- |
| +3.234s | Target Button `0x1c25ca0acc0`, note `394d946a-f3ec-47cd-a3df-4fd29a341b40`, attached=True, pruning=False; folders loading=True, placements loading=False |
| +3.250s | Original `post_message(Button.Pressed)` accepted=True; delivery held on that Button |
| +3.266s | Actual folder answer released to the original page reducer |
| +3.281s | Native Notes Items/Work recomposition starts, both branches now loaded |
| +3.297s | Original message resumes: attached=True, pruning=True, parent_active=False; it cannot bubble to the screen |
| +3.438s | Native Items recompose finishes; ten replacement rows mounted |
| +33.250s | Original arm wait expires: view=list, snapshot=None, selected empty, armed=False; no `_begin_library_note_load` occurred |

`evidence-receipt.json` contains actual XML/trace digests and per-run source
hashes captured inside pytest. `titleprep.timeline.json` and
`holdpress.timeline.json` are concise native row/snapshot/DOM receipts; the
full traces retain all observations. XML SHA256:

- Original: `2693528859109e7627a1e3d9493e8c74dfaaf48bca2dbe48feceb8f8afcd2104`
- Title preparation: `d5ea0dc14ad6ab6687e43758df40bd90f93343b0d9241401000bb0256cbf15a5`
- Ordered delivery: `b99c5da11194fe0481a2d888676cbbca41bebed0c4906f81427920334cf1c70f`

## Minimal bounded repair proposal

Repair this control's setup barrier in `library_note_links_support`, preserving
the single original row press and every existing outcome assertion:

1. Capture the prior tree topology epoch before pressing Notes. Wait for the
   new browse visit to be acknowledged (new epoch, list view, prior note
   session cleared), and for its expected root/open-branch reads to settle.
2. Also await **actual DOM publication** of that settled snapshot. Both
   retained `LibraryNotesCanvas` instances (`#library-notes-canvas` and
   `#library-note-work-pane`) must be attached/nonpruning with pending
   `_recompose_required=False` and existing `recompose_in_flight=False`.
   `RecomposeCaptureGuard` already keeps the latter true across native child
   teardown and replacement mount; no product flag is needed. Require the
   ordered live row `(placement_id, note_id)` identities to match note rows
   from the current screen tree projection, with live nonpruning ancestors.
   Recheck that same epoch/snapshot and reacquire the same target row across a
   Pilot layout/message turn. A nonbusy flag alone is insufficient: the
   ordered trace has both branches nonbusy while old rows are being pruned,
   then zero rows, before ten replacements mount. Keep this loop inside the
   existing 30-second bound rather than giving each phase a fresh budget.
3. Reacquire and press the matching target row once. Keep `_armed_editor` and
   its note-id/body/armed requirements unchanged. Keep all current bounds;
   avoid global worker waits, fixed sleeps, retries, direct note-load calls,
   and assertion deletion.
4. Pin the setup barrier with the actual async folder publication held after
   its real service result: the helper must remain pending when placements
   have already mounted, then finish only after the held answer is released
   and the current DOM rows publish. Release the hold in `finally` inside the
   App. The old helper must fail this pending/publication oracle. Verify the
   original full link journey afterward; a passing snapshot-only helper test
   cannot replace its existing link oracles.

The qualified native probes also need this tiny, explicit preparation in
`_blank_note_with_body_focused`, after its existing `_new_blank_note` and before
its existing `_type(VERIFIER)`:

```python
await pilot.click("#library-note-title")
await _until(
    pilot,
    lambda: screen.focused is screen.query_one("#library-note-title"),
    "the title to take focus",
)
```

That is the exact preparation used by both the successful and held probes.
Retain the existing title equality, body click/focus and all later assertions.
It is a native user gesture with an observed precondition, not a focus/getter
substitute. The initial title-focus issue in run1 remains unqualified for CI
line245; this proposal does not change or claim to fix product new-note focus.
The final actual owned-source GREEN must include legitimate title preparation
and the full original link journey, not only the helper-publication regression.
This investigation has not implemented or green-qualified the proposed barrier.

## Ownership and ADR

The existing link-control owner is **TASK-34000.11**. Its body/persistence/
navigation/backlink assertions must remain.

The real product obligation to perform an action once or visibly refuse it
during Notes recomposition is already owned by **TASK-34000.85** (fresh
all-ref/local task catalog, exact structured entry read). That task records
attached Move-note Button presses dropped by recomposition, plus filter-key
loss. This helper barrier does not fix or close that broader product task.
Do not create a duplicate owner or claim the user-click race is fixed.

ADR required: **no** for this test setup synchronization.
ADR path: existing **`backlog/decisions/027-portable-database-note-session-coordinator.md`**.
Reason: the existing host/session/persistence boundary is retained. No new
service, schema, lifecycle, authority, or durable UX decision is proposed.
