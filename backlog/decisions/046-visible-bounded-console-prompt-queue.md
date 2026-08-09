# ADR-046: Visible bounded Console prompt queue

Status: Accepted
Date: 2026-08-09
Related Spec: [Console Prompt Queue Design](../../Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md)
Supersedes: N/A

## Decision

Chatbook will support a visible, per-session prompt queue in Console. While an
accepted agent turn is active, the user may queue up to ten text-only prompts.
The prompts run as separate FIFO turns, each starting only after the previous
assistant turn completes successfully.

The first entry requires an already accepted turn. Once a queue exists, users
may append while it is draining or paused; older entries always retain priority.
Admission and the coordinator's terminal queue-empty transition are atomic, so
a boundary-racing draft is either observed by the drain or rerouted unchanged as
a normal manual send, never stranded or duplicated.

The queue is owned by the Console chat controller through a focused, pure queue
registry. Textual widgets render immutable snapshots and emit typed intents;
they do not own scheduling or mutate queue state directly. A controller-side
coordinator owns claiming, submission, pause, retry, slot reservation, terminal
state, and shutdown behavior. The screen starts one per-session worker and
awaits that coordinator.

Queue transitions are synchronous and confined to the Textual event-loop
thread. Worker-thread callbacks marshal before queue access. Revision-checked
registry methods are the atomicity boundary; widgets never acquire their own
locks or mutate the queue directly.

One draining session retains its existing global agent slot across queued
turns. The reservation ends when the queue empties or pauses. A paused queue
must explicitly reacquire a slot on resume or retry; if the global cap is full,
the existing cap refusal remains visible and the queue stays paused. There is
no automatic global-cap waiting queue.

The queue is process-memory-only and scoped to the mounted Console screen. It
survives session, tab, and workspace switching inside that screen, but it is not
serialized, persisted, restored after leaving Console, or replayed after an app
restart. Closing a queued session, leaving Console, or quitting the app requires
a count-aware confirmation before unsent prompts are discarded. This guarantee
applies to user-initiated in-app exits; forced process termination cannot offer
interactive confirmation.

Confirmed session close marks that session's queue chain closing before it
signals Stop or cancels the active stream. Terminal callbacks caused by close
cannot claim another prompt. One combined close confirmation reports transcript,
live-turn, and queued-prompt impact instead of stacking dialogs.

Queueing begins only after the active turn crosses the existing accepted-send
boundary. During provider and skill validation the action remains unavailable
and reads `Preparing...`. Once accepted, the normal `Send` action becomes
`Queue`; Enter and the button both enqueue the exact canonical text draft.
Successful queue admission, not an attempted enqueue, clears the captured draft.

Queued entries contain text only. Drafts carrying staged attachments or staged
Library/RAG evidence are refused intact. Recognized slash commands are never
queued; each command keeps its existing command-specific execution or refusal
rules. Large pasted text and `$skill` messages remain valid text prompts.
Attachments and manually staged evidence are rechecked before automatic
submission, so a rider added after admission pauses the queue and remains
unconsumed. Session-targeted Auto-RAG may still run for a queued text turn when
enabled, but it is generated for that owning session at dispatch and cannot
consume the screen's resident staged-evidence slot.

The queue pauses after a failed, stopped, context-invalidated, or pre-accept
refused turn. Failed turns retry their existing assistant response before
draining continues, with an explicit skip option. A stopped turn offers
`Resume next` as the primary action and a separate `Retry stopped turn` action;
the retry uses the existing regeneration path and keeps the stopped partial in
history. A user may also request `Pause after this turn` without stopping the
current response.

Conversation safety is based on a dedicated conversation-context epoch owned by
`ConsoleChatStore`, the authority for provider-relevant history, summary
boundary, active leaf, and message tree. Linear user/assistant appends do not
change that epoch. Active-path content edits, selected textual variants,
summary changes, rewind, sibling selection, active-path delete,
edit-and-resend, and branch creation do. A mismatch pauses the queue for review
rather than silently sending prompts against changed context.

The chain records the context epoch of the active turn's committed provider
payload even while the queue is empty. A later first entry inherits that
baseline, so an edit made during the response cannot be masked by queueing only
after the edit.

Resuming after an unrelated context mismatch requires an explicit
`Use current context & resume` confirmation that revalidates both queue revision
and context epoch before adopting the new baseline. Other recovery actions do
not silently adopt unrelated edits.

Background queue turns resolve one immutable turn execution context containing
provider, model, capabilities, system prompt, generation settings, workspace,
and other per-turn values for their owning session ID, never from the currently
viewed tab. The same snapshot is threaded through payload construction and
stream execution so a tab switch or settings change cannot produce a mixed
turn. Intermediate queued completions do not emit finished markers or
completion toasts. The session remains visibly running until the whole drain
finishes or pauses.

The immutable execution context stabilizes configuration only. Credentials,
tool approvals, skill trust, and other authority are revalidated through their
existing runtime seams and are never retained as queue state.

The existing no-argument `on_submission_accepted` callback remains a
manual-origin compatibility seam. Queued acceptance uses a separate
content-free coordinator event and cannot clear the visible composer or its
undo history.

Whenever queue-owned future work exists, the coordinator is the sole next-turn
authority. Existing transcript Retry/Regenerate recovery for the failed or
stopped turn delegates into it; unrelated Continue, Regenerate, and Edit &
resend actions cannot bypass older queued prompts.

## Context

Console already supports one active run per session and multiple sessions in
parallel under a global cap. A user can continue typing while a long agent turn
runs, but Send is disabled and a second same-session send is refused. Users who
know their next steps must wait at the keyboard and manually submit each one.

The parallel-agent design deliberately rejected a hidden queue because an idle
tab blocked only by the global cap should not fire later without visible user
control. This feature does not reverse that global-cap decision. It adds a
different, visible contract for follow-up prompts in the same session whose
accepted turn already owns an agent slot.

Several existing boundaries make an implicit implementation unsafe:

- The accepted-send callback is currently origin-agnostic and can clear the
  visible composer. An automatic queued submission must never erase a new draft
  the user is typing.
- Provider execution inputs are projected from the viewed session and read by
  multiple payload helpers. A background queued turn must resolve one immutable
  owning-session execution context and thread it through the whole turn to
  prevent cross-tab or mid-validation leakage.
- Run completion currently stamps background markers and toasts per turn. A
  ten-prompt drain must not produce ten completion notifications or flicker
  between finished and running.
- Navigation guards currently count live runs only, and the app quit action does
  not consult Console. Paused queues can contain unsent private text with no
  active run.
- The chat store already owns provider-relevant history, summary, and branching.
  A parallel queue-specific view of conversation context would drift from that
  authority.

These are long-lived Console UX, application lifecycle, cross-module interface,
and state-ownership decisions, so a canonical ADR is required.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep refusing same-session sends | Forces users to babysit long turns and does not meet the requested workflow. |
| Combine queued text into one next user message | Loses the requested turn-by-turn assistant context and makes ordering semantics ambiguous. |
| Own the queue in `ChatScreen` | Couples scheduling to mounted widgets and makes recomposition, background sessions, and teardown races harder to reason about. |
| Store queues in `ConsoleChatStore` session records | Overstates durability and mixes transient scheduling with conversation and message storage. |
| Persist queues in the database | Creates restart replay, privacy, migration, and stale-context problems not required for the feature. |
| Queue idle tabs behind the global cap | Reintroduces the hidden delayed-send behavior the parallel-agent design explicitly rejected. |
| Release the agent slot between every queued turn | Allows another tab to race into the gap and violates the approved immediate-drain behavior. |
| Queue attachments and staged RAG evidence | Requires durable ownership and lifecycle rules for binary and source-authority payloads; text-only is the approved first version. |
| Auto-continue after failure or Stop | Can cascade prompts through missing or deliberately interrupted context. |

## Consequences

### Benefits

- Users can plan up to ten follow-up turns without waiting for each response.
- Every queued prompt remains visible, editable, reorderable, removable, and
  pausable before it starts.
- Separate-turn semantics preserve the conversation context users expect.
- Per-session ownership keeps parallel workspaces and providers isolated.
- Explicit pause and recovery states prevent silent cascading after errors.
- Process-memory scope avoids migrations, restart replay, and durable storage of
  unsent private prompts.

### Accepted Trade-offs

- A ten-message drain may occupy one configured agent slot for a substantial
  time.
- Queues are lost after confirmed Console exit or in-app quit, and cannot be
  protected from forced process termination.
- Attachments and staged evidence cannot be queued in the first version.
- The app needs a generic asynchronous pre-quit confirmation seam and
  reentrancy guard.
- The chat store needs a conversation-context epoch distinct from its broad
  payload revision.
- Existing fleet and run-state derivations must become queue-aware through one
  controller activity projection.

### Verification Consequences

The feature requires pure queue-state tests, joined controller tests, mounted
Textual tests, application leave/quit tests, and isolated live verification.
The most important guards must be mutation-checked: the limit, origin-aware
composer clearing, stale-revision rejection, shutdown suppression,
owning-session immutable turn-context selection, context-epoch coverage, and
intermediate notification suppression.

Unsent queue state must be absent from database persistence, prompt history,
screen snapshots, and logs. Once an entry is accepted as a real turn, normal
message persistence and prompt history apply.

## Links

- [Console Prompt Queue Design](../../Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md)
- [Parallel Agents Across Workspaces Design](../../Docs/superpowers/specs/2026-07-26-parallel-agents-across-workspaces-design.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
- [ADR-031: TUI Keybinding and Footer-Hint Conventions](031-tui-keybinding-and-footer-hint-conventions.md)
- [ADR-033: Application Session State Ownership](033-application-session-state-ownership.md)
