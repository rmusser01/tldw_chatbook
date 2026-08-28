# ADR-094: Own accepted Console turns above screen navigation

Status: Accepted
Date: 2026-08-27
Related Task: TASK-22514
Supersedes: the screen-scoped ordinary-turn policy recorded by TASK-1143 and
TASK-15860 acceptance criterion 2

## Decision

Every accepted Console turn is owned by the app-level `ConsoleRuntime`, not by
`ChatScreen` or a Textual worker attached to that screen. Navigation detaches the
Console view and stops view-owned resources without cancelling ordinary chat, agent
execution, turn-launched tools/media, delegated work, queues, or pending human
decisions.

The screen transfers an immutable, send-time request to the runtime before clearing
the composer. The runtime registers custody before task scheduling and owns the task
through final cleanup. The existing `ConsoleChatController` and `ConsoleChatStore`
remain the only run-state, queue, cap, cancellation, streaming, and transcript
authorities; the runtime registry tracks task lifetime only and is not a parallel state
machine.

Every screen-derived input needed by the turn is frozen at handoff or represented by an
app-owned service. A surviving turn never consults a widget, dead screen callback, or a
newly mounted screen's settings. Heavy request preparation remains asynchronous inside
runtime custody.

Tool approval, skill-install confirmation, and skill-script confirmation are retained
independently of the view. Their decision clocks count only while the owning card is
answerable. A hidden round emits one app-wide notice and waits without approving,
denying, or consuming decision-active time. Returning to Console re-derives the active
session's ordered pending decisions.

Every user-visible terminal transcript commit writes a local-only
`CONSOLE_UNSEEN` conversation mark in the same ChaChaNotes transaction. A mounted view
clears that mark only after the owning session synchronizes the terminal row. Shell
navigation displays a boolean attention glyph while any terminal mark or pending
decision exists. The mark never enters conversation sync, export, prompt payloads, or
remote APIs. `AgentRunsDB` and shell state are projections, not participants in a
cross-database transaction.

Navigation is not a cancellation boundary. Stop cancels its selected turn/chain;
session close fences and cancels that session; confirmed app quit revision-pins the
active set, fences admission, and disposes the runtime. Cancellation and completion use
one atomic terminalization gate. Bounded cleanup and generation fences prevent an
uncooperative provider or thread from hanging shutdown or committing late output.

`ChatScreen` remains a fresh, disposable view. Detachment is guaranteed even when
other unmount cleanup fails, and attachment generations prevent an outgoing screen
from clearing a successor's hooks. Realtime microphone/audio resources, open modals,
animations, screen timers, and unsent composition remain view-scoped.

Application exit remains the final runtime boundary. This decision makes no promise
that turns continue after a crash or restart; existing durable dispatch recovery keeps
exclusive ownership of restart repair.

## Context

TASK-15860 moved the Console store, provider gateway, agent bridge, and controller to
an app-owned runtime so supervisor wake turns could run with no Console mounted. It
intentionally preserved a prior rule: leaving Console cancelled ordinary user streams
and denied parked decisions. `ChatScreen` still dispatches ordinary turns through a
screen-owned Textual worker, and runtime `leave_console()` still invokes per-visit
turn cancellation.

That rule makes navigation unsafe for the product's primary agent workflow. Opening
Settings, Library, or another destination can interrupt the main response even though
the store and controller already outlive the view. Existing headless-wake work also
proved that caching or retaining a removed screen is not a safe solution: off-screen
sync workers can query a torn-down DOM and terminate the TUI.

The surviving-runtime seam already separates app-owned domain objects from view hooks.
Extending task ownership into that runtime is therefore smaller and more coherent than
adding a second job system or changing the navigation stack model.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Shield the screen worker with `asyncio.shield()` | The underlying task might survive, but draft recovery, callbacks, finalization, and cleanup would still belong to a dead screen. Cancellation bugs would move rather than disappear. |
| Cache or hide `ChatScreen` | Conflicts with the fresh-screen invariant and recreates the measured dead-DOM/self-rearming-worker crash class. It also keeps timers and heavy DOM state resident. |
| Add a separate background-job service | Duplicates controller queues, run state, caps, persistence, and cancellation. The existing app-owned runtime is already the correct lifetime boundary. |
| Persist raw turn requests for restart continuation | Expands privacy, authority, side-effect replay, and recovery scope far beyond navigation. Existing restart recovery has a separate owner. |
| Show global approval dialogs on the current screen | Interrupts unrelated work and spreads Console-specific security UI across every destination. App-wide notice plus Console-local decision is sufficient. |
| Keep wall-clock approval deadlines running headlessly | Denies decisions while the user has no answerable control, contradicting the notify-and-wait promise. |
| Clear attention on Console mount | Loses unrelated or background-session outcomes before the user sees them. Positive per-session terminal acknowledgement is deterministic. |
| Use one mark for fleet and ordinary outcomes | Supervisor wake delivery legitimately clears `FLEET_UNSEEN`; sharing it could erase an unrelated user-turn result. |
| Make transcript and `AgentRunsDB` terminal state one transaction | They are separate SQLite owners. An outer transaction cannot roll back another database's commit. ChaChaNotes owns user-visible terminal state and attention. |

## Consequences

- The ordinary send path changes from a screen-owned await to an app-runtime custody
  handoff. Existing controller behavior remains authoritative after handoff.
- Screen hook responsibilities split into frozen turn inputs, app-owned runtime
  services, and disposable repaint callbacks. No surviving task may retain a screen.
- `leave_console` can no longer mean both view detach and ordinary-turn cancellation.
  Navigation uses detach semantics; Stop/session close/app disposal own cancellation.
- Approval timing needs a decision-active clock and a unified view projection for all
  three human-decision types.
- `ConversationLocalMarksService` gains a distinct `CONSOLE_UNSEEN` allowed type. The
  existing table is sufficient; no schema migration is expected.
- Terminal transcript finalization and `CONSOLE_UNSEEN` insertion share one ChaChaNotes
  transaction. Visible views acknowledge afterward, so races may briefly retain an
  extra marker but never silently lose one.
- The navigation bar and overflow menu gain a boolean, non-color-only attention state.
  Detailed counts and statuses remain inside Console.
- The old navigate-away loss confirmation, teardown notice, and screen-scoped User
  Guide language are removed. Quit confirmation remains revision-pinned.
- Runtime cancellation waits become bounded and late-write fenced. Underlying
  non-cancellable threads may finish, but cannot mutate terminal state after fencing.
- Tests must cover real navigation, viewless approvals, multi-session isolation,
  restart-durable local marks, privacy, dead-view collection, narrow layouts, and an
  isolated live streamed response.
- The feature does not broaden process lifetime. Confirmed quit and forced process
  death still end runtime execution.

## Links

- [Approved design spec](../../Docs/superpowers/specs/2026-08-27-console-turns-survive-navigation-design.md)
- [TASK-22514](../tasks/task-22514%20-%20Console-turns-survive-screen-navigation.md)
- [TASK-15860: app-owned headless Console runtime](../tasks/task-15860%20-%20Headless-wake-fire-the-supervisor-auto-wake-with-no-Console-screen-mounted.md)
- [TASK-1143: former screen-scoped fleet warning](../tasks/task-1143%20-%20Screen-navigation-silently-kills-the-agent-fleet.md)
- [ADR-085: Console activity receipts and switcher ownership](085-console-activity-receipts-and-switcher-ownership.md)
