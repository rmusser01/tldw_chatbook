# Console prompt queue design

Date: 2026-08-09
Status: approved design, pending implementation
ADR: [ADR-046: Visible bounded Console prompt queue](../../../backlog/decisions/046-visible-bounded-console-prompt-queue.md)

## 1. Goal

Let a Console user queue follow-up messages while an accepted agent turn is
still running. A session may hold up to ten text-only prompts. They run in FIFO
order as independent conversational turns, so each assistant response becomes
context for the next queued user message.

The feature must remain visible, controllable, session-scoped, and safe under
parallel tabs, background approvals, failures, Stop, branching, navigation,
quit, voice input, and narrow terminal geometry.

## 2. Current system and amendment scope

Console already has the required foundation:

- `ConsoleRunState` and worker groups are per session.
- One run is allowed per session under a configurable global cap.
- Session drafts and pending attachments survive tab switching.
- Background runs retain their owning session and workspace.
- The accepted-send hook separates a confirmed turn from a blocked optimistic
  echo.

Today, Send is disabled while the target session has an active run and the
controller refuses a second same-session send. The parallel-agent design also
states that there is no hidden queue and names run queues as out of scope.

This design amends that decision only for a visible same-session prompt queue
behind an already accepted turn. It does not queue an idle tab that is blocked
only by the global cap. That case keeps the existing explicit refusal.

## 3. Locked product decisions

1. Queued messages run as separate sequential turns, never one combined prompt.
2. Each session has its own FIFO queue with a hard maximum of ten entries.
3. Users can edit, reorder, remove, pause, resume, retry, skip, or clear queued
   work before it starts.
4. A draining queue retains the session's current global agent slot.
5. Failure or Stop pauses the queue. Nothing cascades automatically through
   missing or deliberately interrupted context.
6. Failed turns retry first by default, with an explicit skip option.
7. Stopped turns use `Resume next` as the primary action and offer a separate
   `Retry stopped turn` action.
8. The queue survives Console tab and workspace switching but not leaving
   Console or quitting the app. Destructive exit requires confirmation.
9. Version one queues canonical text only. Attachments and staged Library/RAG
   evidence are excluded.
10. The primary queue surface is a compact shelf directly above the expanded
    composer.
11. While an accepted turn runs, Send becomes Queue and Enter enqueues.
12. No new global shortcut or configuration key is introduced.

## 4. User experience

### 4.1 Action lifecycle

The composer action has three relevant labels:

- `Preparing...` while the current send is still validating and has not crossed
  the accepted-turn boundary. Queueing is unavailable.
- `Queue` after the turn is accepted, or whenever that session already has
  queue-owned future work, including a paused queue or a pre-accept claim.
- `Send` when the session has no active turn and no paused queue controlling
  order.

When a queue is paused, new text appends to the queue. It never bypasses older
entries. The user must resume, retry, skip, reorder, or clear deliberately.

Recognized slash commands are never queued. Parsing still occurs before the
normal-text decision, and every command keeps its own existing safety gate.
Unknown commands retain the existing two-Enter confirmation before they may be
treated as literal text. `$skill` messages remain ordinary text sends and are
revalidated for trust at dispatch time.

### 4.2 Queue shelf

Once at least one future entry exists, one row appears above the expanded
composer:

```text
Queue 3/10 · Draining · Next: "Review the tests..."    Manage  Pause
```

The shelf always carries readable text; color only reinforces state. It shows
one state-dependent primary action plus `Manage`:

| State | Primary action | Shelf copy |
| --- | --- | --- |
| Draining | Pause | `Queue 3/10 · Draining` |
| Pause requested | Keep draining | `Queue 3/10 · Pauses after this turn` |
| Manually paused | Resume | `Queue paused · 3 waiting` |
| Failed turn | Retry | `Queue paused · Turn failed · 3 waiting` |
| Stopped turn | Resume next | `Queue paused · Turn stopped · 3 waiting` |
| Dispatch refused | Try again | `Queue paused · Next message could not start` |
| Branch changed | Review | `Conversation changed · review 3 queued` |

Secondary recovery actions live in the manager so the shelf remains usable at
80 columns. Clear is never a shelf action.

At narrow widths, the next-message preview disappears before the count, state,
or actions. Preview text is markup-escaped, normalized to one line, and
truncated by terminal-cell width. Prompt content never appears in background
toasts or fleet labels.

When the composer is collapsed, the existing privacy contract wins: only count
and status remain, such as `Queue 3/10 · Draining`. Draft text, prompt previews,
and attachment names stay hidden.

### 4.3 Queue manager

`Manage` opens a bounded, scrollable modal with at most ten numbered one-line
previews. Normal buttons and Tab navigation expose:

- Edit.
- Move up.
- Move down.
- Remove with the repository's guarded destructive-action convention.
- Retry a stopped turn.
- Skip a failed turn and resume.
- Clear all with confirmation.
- Close with Esc.

The active or already accepted prompt is not a future queue entry and cannot be
edited from the manager. A claimed prompt that is still crossing validation is
shown as `Starting...` and locked against edit, reorder, and removal until it is
accepted or returned to the queue.

Edits pass through the same text and command classification as admission. An
edit that would turn an entry into a recognized slash command is refused
intact; commands are never smuggled into the automatic drain through the
manager.

The manager consumes live revisioned snapshots. Selection follows stable entry
IDs. If the selected entry starts or is removed, selection moves to the next
surviving entry. An edit or reorder callback revalidates its entry ID and queue
revision. A stale edit never changes the active turn; it retains the edited text
and may offer to add it as a new entry if capacity remains.

### 4.4 Refusals and feedback

Queue acceptance clears the exact captured draft and updates the shelf without
an additional toast. Refusal preserves the exact composer stash, including the
canonical content behind collapsed paste tokens.

Required visible refusals include:

- `Queue full (10/10). Manage or remove an item.`
- `Attachments cannot be queued yet. Remove the attachment or wait for this turn to finish.`
- Equivalent explicit copy when staged Library/RAG evidence is present.
- The existing global-cap refusal when resuming a queue cannot reacquire a slot.

Voice and programmatic dispatch use a screen-facing outcome with `sent`,
`queued`, or `refused`. Hands-free acknowledgement for a queued send names the
state, for example `Queued, 3 waiting.`

### 4.5 Background visibility

The full shelf belongs to the viewed session. Background tab and conversation
rows expose a content-free count:

- `Queue 3` where space permits.
- `Q3` in compact tab chrome, documented in F1 help and the marker legend.

A draining queue remains Running between its turns. Intermediate completions do
not stamp Finished, clear the transcript timer, or emit completion toasts. One
final completion signal fires when the queue empties. A failure or Stop produces
one paused/failure signal.

## 5. State model

### 5.1 Queue entry

`QueuedPrompt` is immutable and contains:

- Stable entry ID.
- Exact canonical text.
- Owning session ID or ownership through its registry bucket.
- Insertion order.
- The queue's expected branch epoch when admitted.

It contains no attachment, evidence bundle, provider selection, secret, widget,
task, or persistence identifier.

### 5.2 Per-session queue state

`PromptQueueState` contains:

- Ordered immutable entries.
- At most one claimed entry.
- Monotonic queue revision.
- Mode: `draining`, `pause_after_turn`, or `paused`.
- Pause reason: manual, failed, stopped, context changed, or dispatch refused.
- Expected branch epoch.
- Whether the session reserves an agent slot.

The hard limit counts `queued + claimed`. Once a claimed entry is accepted as a
real user turn, it leaves queue accounting and normal message ownership begins.
Visible capacity counts use the same rule; a pre-accept claim therefore still
occupies one of the ten positions even though its row is labeled `Starting...`.

### 5.3 Accepted-turn and submission origin

Queueing is enabled by an explicit per-session accepted-turn flag, not by a
broad run-status guess. The flag is set at the same boundary that currently
fires the accepted-send hook and cleared at terminal teardown.

Every submission carries:

- Session ID.
- Origin: manual or queued.
- Draft text.
- Queue entry ID for queued origin.
- Composer stash only for manual origin.

Only an accepted manual submission may consume or clear its own captured
composer stash. An automatic queued submission never touches the live composer
or another session's draft.

### 5.4 Branch epoch

`ConsoleChatStore` owns `branch_epoch(session_id)` because it owns the message
tree and active leaf. The epoch changes only when active lineage changes:

- Rewind or direct active-leaf selection.
- Sibling or variant navigation that selects another branch.
- Edit-and-resend or regeneration that creates/selects a branch.
- Deletion that changes the active path.

Ordinary linear message appends, streaming updates, status changes, persistence,
feedback, and display overlays do not change it.

A queue-authorized failed retry or stopped regeneration may legitimately change
the epoch. On successful recovery the coordinator adopts the resulting epoch
before draining continues. Any unrelated mismatch pauses for review.

## 6. Architecture and ownership

### 6.1 Pure queue registry

`tldw_chatbook/Chat/console_prompt_queue.py` owns pure models and a per-session
registry. It implements FIFO admission, validation results, the limit, claims,
revision checks, edits, moves, removals, pause/resume transitions, reservations,
session removal, and shutdown. It has no Textual, provider, persistence, or
widget dependency.

### 6.2 Controller and coordinator

`ConsoleChatController` owns queue behavior through the registry and a focused
controller-side coordinator. The coordinator interprets terminal results and
owns claim, submit, retry, skip, reservation, and shutdown decisions.

The controller exposes one per-session activity projection used by all fleet
consumers. It answers:

- Occupies an agent slot.
- Has a live accepted turn.
- Needs approval.
- Has queued prompts.
- Queue is paused.
- Terminal outcome is eligible for a marker or notification.

Busy count, global-cap enforcement, run markers, fleet summary, transcript
polling, completion notifications, and navigation warnings derive from this
projection rather than duplicating queue awareness.

### 6.3 Per-session provider selection

Background dispatch resolves a `ConsoleProviderSelection` for the queue's owning
session ID. The resolver reads that session's stored settings, workspace
binding, and current provider configuration. It never reuses the viewed tab's
controller projection.

Provider, model, system prompt, workspace, RAG defaults, tools, and other
ordinary settings are resolved at dispatch time. The queue stores only text.
Pinned and one-shot prefill keep their existing session-level next-send
semantics.

### 6.4 Screen dispatch

`UI/Console_Modules/message.py` extends the existing visible-send dispatcher. It
returns the screen-facing `sent | queued | refused` result, starts one exact
per-session Textual worker, and awaits `controller.run_prompt_chain(...)`. It
does not interpret queue phases or terminal statuses itself.

### 6.5 Queue region and modal

`UI/Console_Modules/prompt_queue.py` owns shelf pixels, responsive presentation,
focus restoration, and typed UI intents. The manager is a focused modal in
`Widgets/Console/console_prompt_queue_modal.py`.

Both render immutable snapshots. Unchanged queue revisions produce no update or
recompose during the Console's 0.2-second active-run poll.

Construction and named late-binding callbacks live in
`UI/Console_Modules/wiring.py`. Queue work does not increase the
`chat_screen.py` size or method ratchet.

### 6.6 Application quit

`TldwCli.action_quit()` remains a thin non-blocking dispatcher into one
exclusive confirmation worker. A `_quit_in_progress` guard absorbs repeated
requests. The worker consults a generic asynchronous screen pre-quit seam,
then sets the app shutdown flag and runs existing cleanup only after approval.

Choosing Stay or encountering a confirmation error fails closed, clears the
guard, and preserves the mounted queue manager, edited text, runs, and queues.

## 7. Send and drain flow

### 7.1 Enqueue

1. Capture the target session synchronously.
2. Parse recognized slash commands before normal-text routing.
3. Require either an accepted live turn or an existing nonempty queue that
   already controls the session's next-turn order.
4. Capture the exact draft transaction.
5. Reuse normal text validation.
6. Refuse intact when an attachment or staged evidence rider is present.
7. Atomically enforce `queued + claimed < 10`.
8. Append an immutable entry and advance the queue revision.
9. Clear the captured manual draft only after admission succeeds.
10. Return `queued` with the new waiting count.

When the queue is already paused, new text appends without starting or bypassing
older work.

Admission and the chain's final queue-empty transition use one atomic
controller decision. If admission wins, the chain observes the new entry before
it can release its reservation. If the terminal transition wins, no entry is
created and the unchanged draft is rerouted once through normal manual Send
against the now-idle session. This prevents both stranded prompts and duplicate
turns at the response boundary.

### 7.2 Drain

The original manual send and every automatic queued turn run inside one
per-session chain:

1. Await the current real turn.
2. Inspect its accepted and terminal outcome.
3. On successful completion, honor `pause_after_turn` first.
4. Verify the store branch epoch.
5. Atomically claim the FIFO head.
6. Resolve the owning session's current provider and workspace selection.
7. Submit with queued origin.
8. At the accepted boundary, settle the claim and let normal message
   persistence own the user turn.
9. Repeat until the queue empties or pauses.

No user input event can claim the released slot between queued turns. The
reservation remains visible to global cap and fleet derivation for the whole
chain.

### 7.3 Refusal and exception boundary

If a queued submission is refused before acceptance, its claim returns to the
head and the queue pauses with the refusal recovery copy. If an exception occurs
after acceptance, the entry is already a real user turn and is not requeued;
normal failed-assistant handling pauses the remaining queue.

Unexpected coordinator errors fail closed. Unclaimed entries remain in order,
the reservation is released, and visible recovery copy is surfaced without
logging prompt content.

## 8. Pause, failure, Stop, and recovery

### 8.1 Manual pause

`Pause` changes the mode to `pause_after_turn`. The current response continues.
The shelf action becomes `Keep draining`, which cancels that request. At
successful terminal completion, a still-pending pause releases the slot and
leaves all future entries untouched.

### 8.2 Failure

A failed accepted turn pauses the remaining queue. `Retry` uses the existing
failed-assistant retry path. Success adopts any authorized context change and
continues. `Skip & resume` leaves the failed turn visible and advances without
deleting or rewriting it.

### 8.3 Stop

Stop cancels only the current session's live response and immediately pauses
its queue. `Resume next` accepts the stopped partial as history and dispatches
the next prompt after reacquiring a slot. `Retry stopped turn` uses the existing
regeneration path, retaining the partial stopped response as a sibling. Success
adopts the new branch epoch and continues.

### 8.4 Resume after reservation release

A paused queue holds no global agent slot. Resume, retry, and retry-stopped must
reacquire one. If the cap is full, the operation refuses visibly and leaves the
queue paused. It never registers a hidden global waiter.

### 8.5 Approval waits

An approval or skill confirmation is still part of the current accepted turn.
It does not pause the queue by itself. The shelf names the waiting approval,
and the user may edit, reorder, remove, or request pause while the turn waits.
Normal approval denial may still let the agent finish; only the final turn
outcome decides whether draining continues.

## 9. Session, navigation, and shutdown lifecycle

Queues are keyed by session ID. Switching tabs or workspaces changes only the
rendered snapshot. Closing a session removes only that session's queue after a
confirmation that includes its exact unsent count.

Navigation and quit guards count:

- Live runs.
- Sessions containing unsent queue work.
- Total unsent queued prompts, including a pre-accept claim.

The dialog uses those separate counts and does not imply that a paused queue is
a running agent.

Shutdown first marks the controller as shutting down. Claim, resume, retry, and
drain entry points then become no-ops or explicit refusals. Only after that guard
is active may shutdown cancel runs, deny approval waits, release reservations,
and clear claims and queues. Terminal transitions caused by cancellation cannot
start another queued turn.

Queue state never enters `TaskResumeState`, the native Console screen snapshot,
database rows, prompt history, diagnostics, or logs. After a queued entry is
accepted as a real turn, its user message and prompt-history entry follow normal
persistence rules.

## 10. Hands-free and other input paths

All visible and programmatic composer sends route through the same dispatcher.
Keyboard, mouse, voice, and hands-free callers receive the same typed outcome.

A queue reservation owns next-turn priority. Hands-free may enqueue user-authored
text, but it cannot independently start another automatic turn while the queue
chain retains a reservation. Exactly one next turn may begin after a response.

Image generation and other recognized slash commands remain outside the prompt
queue. Their existing per-command in-flight and safety rules stay authoritative.

## 11. Accessibility and terminal geometry

- Queue state is always text-labeled; color is supplementary.
- Focus uses the existing semantic focus treatment and never changes geometry.
- Disabled queue actions retain readable explanatory copy.
- The shelf spends one row at normal supported heights.
- Responsive reduction removes preview content before count, state, or actions.
- The collapsed composer never reveals prompt content.
- The manager uses standard buttons, Tab/Shift+Tab, Enter, and Esc. No screen or
  widget binds terminal-convention keys or shadows global bindings.
- F1 help documents `Queue N`, compact `QN`, pause, retry, and resume semantics.

Mounted verification must assert neighboring control geometry, not only shelf
text or display flags. At 80x24, 100x30, and 160x40, Send/Queue, Stop, Dictate,
disabled-reason copy, transcript jump controls, and the queue shelf remain
inside the viewport.

## 12. Verification plan

### 12.1 Pure state tests

- FIFO order and a hard maximum of ten per session.
- `queued + claimed` capacity accounting.
- Edit, move, remove, clear, pause, resume, and keep-draining transitions.
- Stable entry IDs and stale revision rejection.
- Session isolation and exact session removal.
- Branch-epoch behavior: linear append stable, lineage mutation increments.
- Shutdown prevents all new claims.

### 12.2 Controller and joined async tests

- Three queued prompts create three ordered user/assistant pairs.
- A drain retains exactly one global slot until empty or paused.
- Other sessions remain independent.
- Background dispatch uses its owning session's provider, model, system prompt,
  and workspace roots while a different session is viewed.
- Activity projection drives busy count, markers, fleet summary, polling, cap,
  notifications, and navigation consistently.
- Intermediate completions create no Finished marker or completion toast.
- Approval waits retain queue state.
- Failed, stopped, branch-changed, dispatch-refused, and unexpected-exception
  paths pause without duplication or loss.
- Failed retry, skip, stopped regeneration, and resume-next start exactly one
  following turn.
- A paused queue refused by the global cap remains paused.
- Shutdown cannot dispatch after cancellation wakes a terminal transition.
- Hands-free and queue completion cannot start duplicate turns.

### 12.3 Mounted UI tests

- `Preparing...`, `Queue`, and `Send` appear only at their correct boundaries.
- Enter, button, voice, and programmatic paths report honest outcomes.
- Full-queue and non-text-rider refusals restore exact composer transactions.
- Shelf count, state, primary action, preview escaping, and compact labels.
- Collapsed composer privacy.
- Live manager snapshots, stable selection, and stale-edit recovery.
- Background labels remain session-correct.
- Geometry assertions at 80x24, 100x30, and 160x40.
- Revision-gated rendering performs no update on an unchanged 0.2-second tick.

### 12.4 Application lifecycle tests

- Stay preserves live runs, queues, manager state, and edited text.
- Leave and quit proceed only after confirmation.
- Warnings report exact live-run, queued-session, and queued-prompt counts.
- Repeated quit requests produce one dialog and one cleanup pass.
- Confirmation failure fails closed.
- Unsent queue text is absent from snapshots, persistence, history, and logs.
- Accepted queued turns persist normally.

### 12.5 Mutation checks

At minimum, deliberately break and confirm a red test for:

- The ten-entry limit.
- Origin-aware composer clearing.
- Stale revision rejection.
- Branch epoch checking.
- Shutdown claim suppression.
- Owning-session provider selection.
- Intermediate completion notification suppression.

### 12.6 Live verification

Use the repository's absolute virtual-environment Python and an isolated scratch
profile. Set `TLDW_TEST_MODE`, `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`,
`TLDW_CONFIG_PATH`, and the scratch config's `[paths] data_dir` before importing
or launching the app.

With one available local provider, create two Console sessions with distinct
system prompts and workspace roots:

1. Start both agents.
2. Queue three prompts in one session.
3. Edit and reorder them while the first turn runs.
4. Park an approval.
5. Switch sessions and verify background count and workspace isolation.
6. Approve and observe sequential draining.
7. Pause after a turn, resume, Stop, retry the stopped turn, then resume next.
8. Confirm one final completion notification.
9. Exercise session-close, leave, and quit warnings.
10. Inspect compositor output at 80x24 and a normal terminal size.

Provider-selection isolation across different provider/model combinations is
proved deterministically in automated tests rather than requiring multiple paid
credentials for the live smoke.

## 13. Documentation and compatibility

Implementation must update:

- `Docs/User_Guide/console/chat-basics.md`.
- `Docs/User_Guide/console/agent-runs-and-tools.md`.
- Collapsed-composer behavior and screenshots where the queue changes visible
  states.
- F1 help and fleet-marker legend.
- Leave and quit behavior.
- The parallel-agent design's explicit amendment note.

The design introduces no database migration, persistent queue format,
configuration key, or global shortcut. Existing `ConsoleSubmitResult` remains
compatible; the new immediate `sent | queued | refused` outcome belongs to the
screen dispatcher.

## 14. Non-goals

- Persistent or restart-resumable queues.
- Global-cap waiting queues.
- Cross-session or cross-workspace queue movement.
- Attachments, staged evidence, or arbitrary message riders.
- Combined or batched prompts.
- Per-entry provider or settings snapshots.
- Per-workspace queue caps.
- Scheduling times, delays, recurrence, or external automation.
- Automatically continuing after failure or Stop.

## 15. ADR check

ADR required: yes

ADR path: `backlog/decisions/046-visible-bounded-console-prompt-queue.md`

Reason: the feature changes long-lived Console send behavior, controller/screen
interfaces, application quit coordination, transient state ownership, and the
previously approved no-queue policy. ADR-046 records the bounded visible
same-session exception and its lifecycle limits.
