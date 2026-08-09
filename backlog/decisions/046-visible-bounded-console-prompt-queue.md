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

One draining session retains its existing global agent slot across queued
turns. The reservation ends when the queue empties or pauses. A paused queue
must explicitly reacquire a slot on resume or retry; if the global cap is full,
the existing cap refusal remains visible and the queue stays paused. There is
no automatic global-cap waiting queue.

The queue is process-memory-only and scoped to the mounted Console screen. It
survives session, tab, and workspace switching inside that screen, but it is not
serialized, persisted, restored after leaving Console, or replayed after an app
restart. Closing a queued session, leaving Console, or quitting the app requires
a count-aware confirmation before unsent prompts are discarded.

Queueing begins only after the active turn crosses the existing accepted-send
boundary. During provider and skill validation the action remains unavailable
and reads `Preparing...`. Once accepted, the normal `Send` action becomes
`Queue`; Enter and the button both enqueue the exact canonical text draft.
Queue acceptance, not an attempted enqueue, clears the captured draft.

Queued entries contain text only. Drafts carrying staged attachments or staged
Library/RAG evidence are refused intact. Recognized slash commands are never
queued; each command keeps its existing command-specific execution or refusal
rules. Large pasted text and `$skill` messages remain valid text prompts.

The queue pauses after a failed, stopped, context-invalidated, or pre-accept
refused turn. Failed turns retry their existing assistant response before
draining continues, with an explicit skip option. A stopped turn offers
`Resume next` as the primary action and a separate `Retry stopped turn` action;
the retry uses the existing regeneration path and keeps the stopped partial in
history. A user may also request `Pause after this turn` without stopping the
current response.

Conversation lineage safety is based on a dedicated branch epoch owned by
`ConsoleChatStore`, the authority for the active leaf and message tree. Linear
user/assistant appends do not change that epoch. Rewind, sibling selection,
delete, edit-and-resend, and branch creation do. A mismatch pauses the queue for
review rather than silently sending prompts against a different branch.

Background queue turns resolve provider, model, system prompt, workspace, and
other per-turn settings for their owning session ID, never from the currently
viewed tab. Intermediate queued completions do not emit finished markers or
completion toasts. The session remains visibly running until the whole drain
finishes or pauses.

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
- Provider selection is projected from the viewed session. A background queued
  turn must resolve settings by owning session to prevent cross-tab leakage.
- Run completion currently stamps background markers and toasts per turn. A
  ten-prompt drain must not produce ten completion notifications or flicker
  between finished and running.
- Navigation guards currently count live runs only, and the app quit action does
  not consult Console. Paused queues can contain unsent private text with no
  active run.
- The chat store already owns branching. A parallel queue-specific view of
  lineage would drift from that authority.

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
- Queues are lost after confirmed Console exit or app quit.
- Attachments and staged evidence cannot be queued in the first version.
- The app needs a generic asynchronous pre-quit confirmation seam and
  reentrancy guard.
- The chat store needs a branch epoch distinct from its broad payload revision.
- Existing fleet and run-state derivations must become queue-aware through one
  controller activity projection.

### Verification Consequences

The feature requires pure queue-state tests, joined controller tests, mounted
Textual tests, application leave/quit tests, and isolated live verification.
The most important guards must be mutation-checked: the limit, origin-aware
composer clearing, stale-revision rejection, shutdown suppression,
owning-session provider selection, and intermediate notification suppression.

Unsent queue state must be absent from database persistence, prompt history,
screen snapshots, and logs. Once an entry is accepted as a real turn, normal
message persistence and prompt history apply.

## Links

- [Console Prompt Queue Design](../../Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md)
- [Parallel Agents Across Workspaces Design](../../Docs/superpowers/specs/2026-07-26-parallel-agents-across-workspaces-design.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
- [ADR-031: TUI Keybinding and Footer-Hint Conventions](031-tui-keybinding-and-footer-hint-conventions.md)
- [ADR-033: Application Session State Ownership](033-application-session-state-ownership.md)
