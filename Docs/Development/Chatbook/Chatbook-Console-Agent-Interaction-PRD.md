# Console Agent Interaction — Product Requirements Document (PRD)

**Features:** (A) the agent can ask the user multiple-choice questions mid-run; (B) a persistent, always-visible task list.
**Reference behaviour:** Claude Code's `AskUserQuestion` and `TodoWrite`, verified against the shipped binaries rather than recalled.
**Status:** requirements for review. Date 2026-09-03.

## Summary

Today a Console agent can act but cannot ask. When it hits a fork it genuinely cannot resolve — which provider, which of three files, which of two readings of an ambiguous request — it guesses, or it stops and writes prose the user may not be watching for. And while it works, its plan is invisible: the task list it already keeps is rendered as a text block that scrolls out of view within a few turns.

This PRD specifies two features that close those gaps the way Claude Code does:

- **A — `ask_user`.** The agent pauses, presents one to four multiple-choice questions in a card, the user answers (or types, or walks away), and the run continues with the answers.
- **B — Persistent task list.** The agent's task list lives in a pinned panel that updates in place as work progresses, always visible without scrolling.

Both build on machinery that already ships. The task-list backend is complete; only the surface is missing. The blocking ask-and-wait round trip exists for tool approvals; questions ride it.

## Background & Repo Context

**Already built — do not rebuild.**

- `Agents/session_todo_store.py` (`SessionTodoStore`) and the four agent tools `todo_create` / `todo_update` / `todo_get` / `todo_list` in `Agents/local_tool_provider.py`. Stable IDs, compare-and-swap versioning, a 50-item cap, one-`in_progress` invariant, snapshot persistence across in-app navigation. Governed by ADR-032; hardened in TASK-13216. The only UI is `format_todo_marker` (`Chat/console_agent_bridge.py`), a display-only `☰ Tasks (n in progress):` block appended to the transcript on every mutation.
- The approval round trip: `ConsoleChatController.request_mcp_approvals` blocks an agent worker thread, marshals a card to the UI, polls at 1s until the user decides or a deadline passes, parks the card with a badge and toast when its session is not the one being viewed, mounts it on visit, and fails it closed on run cancel. Owner-agnostic despite the name. PR #1836 (task-15661) made this machinery safe for a second concurrent round *of the same kind* in the same session — queued rounds now wait FIFO instead of overwriting each other. Questions become a fourth kind with the same guarantee among themselves; they do not share a queue with approvals.

**Reference implementations.**

| | Claude Code | Codex |
|---|---|---|
| Ask tool | `AskUserQuestion`: 1–4 questions, 2–4 options each, header ≤12 chars, per-option description, `multiSelect`, an automatic free-text "Other" | `experimental_request_user_input`, config-gated under `[tools]`, off by default |
| Where it renders | inline in the conversation | bottom pane, above the composer |
| Timeout | "Question auto-continue timeout" setting, including `never` | live `auto-resolves in …` countdown on the card |
| Unanswered questions | — | `Submit with N unanswered` allowed |
| Task list | `TodoWrite`: `content` / `status` (pending, in_progress, completed) / `activeForm`; one in-progress at a time; rendered as a checklist that updates in place | `plan_update` events |

Neither queues questions deep; both handle depth by batching several questions into one call.

## Goals

1. An agent can ask the user a question and get an answer without the user leaving the Console, and the run continues with that answer.
2. The user can always see what the agent has done, is doing, and plans to do, without scrolling.
3. Both features reuse the card slot and behaviours users already know from approvals — same place, same park/badge/toast semantics, same cancel behaviour. No second interaction model.

## Non-Goals

- A unified interrupt surface refactor, away-notifications, or live-progress indicators. Those are separate follow-ons; nothing here depends on them.
- Free-form prompting beyond the per-question "Other" escape hatch.
- Editing tasks from the panel. The agent owns the list, as in Claude Code; the panel is read-only in v1.
- Task lists that survive an app restart. Session-scoped is a deliberate, documented decision (TASK-13216 AC #6).
- Claude Code's `preview` option (side-by-side mockup comparison). No TUI analogue worth building.

## Users & Use Cases

- **U1 — the fork.** The agent must choose between three plausible interpretations of "clean up the imports." It asks; the user picks in two keystrokes; the run continues. Today it guesses, and is wrong a third of the time.
- **U2 — the glance.** Twenty minutes into a long run, the user looks at the Console to see where things stand. The panel shows four done, one in progress ("Writing the migration"), two pending. Today they scroll back looking for the last `☰` block.
- **U3 — the other tab.** The user is in the Library when a background session's agent asks. The rail badge lights, a toast fires, and when they return to that session the question is waiting. Existing park semantics; nothing new to learn.
- **U4 — the walk-away.** The user goes to lunch mid-question. By default the question simply waits — the run is paused, not hung, and its tool-call clock pauses with it (ADR-067) — and they answer when they are back. A user who would rather the run press on sets a timeout; after it the run continues without the answer and tells the model so.
- **U5 — the cancel.** The user stops the run while a question is up. The question fails closed; the card clears.
- **U6 — the fleet.** A sub-agent asks. The card says which one.

## Assumptions & Dependencies

- ADR-067 (accepted 2026-08-15): blocking human prompts wait **indefinitely by default** (`0` = no deadline), and the per-tool-call clock (`RunBudget.max_tool_call_seconds`, 300s at defaults) **pauses while a human decision is pending** via `use_human_input_wait`. A question is a human prompt and inherits both.
- PR #1836's round-keyed FIFO is **per kind**: approvals, skill-install, and skill-script each have their own registry, payload map, and card, so two different kinds can already be mounted side by side. Questions get their own kind with the same semantics. Ordering *across* kinds is not provided today and is not claimed here (A10).
- The Console composer stays live during approvals today (`_console_send_blocked_reason` gates only on readiness, attachments, and empty RAG launches). A8 preserves that.

## Feature A — `ask_user`

### Functional Requirements

- **A1 — Tool shape.** `ask_user(questions)` with 1–4 questions. Each: `question` (≤500 chars), `header` (≤12, the chip), `multiSelect` (bool), `options` (2–4 × `label` ≤100, `description` ≤300). Validation is strict — types, UTF-8, no blanks, control characters flattened for render — and a rejected call returns a tool error the model can act on. Same discipline as `SessionTodoStore`, for the same reason: model-controlled text goes straight to a renderer.
- **A2 — "Other" is always present.** Injected by the card per question, never declared in the schema. The model cannot suppress the user's escape hatch.
- **A3 — Rendering.** In the existing task-card slot above the transcript, where approvals appear — outside the scroll region, so always visible. One section per question: header chip, question text, options with descriptions, the "Other" input. Bounded height with internal scroll; four questions of four described options must not push the transcript off screen.
- **A4 — Keyboard.** `1`–`4` selects within the focused question, `Tab` moves between questions, `Enter` submits, `Esc` returns focus to the composer without dismissing (existing binding). The card never steals focus from someone mid-sentence; the approvals chip's existing focus action covers it.
- **A5 — Partial submission.** Submit is allowed with unanswered questions; they return `unanswered: true`. Codex allows this; Claude Code's "Other" makes it moot. Forcing every answer is stricter than both.
- **A6 — Result.** `{"answered": true, "answers": [{"question", "selected": [labels], "other_text", "unanswered"}]}` or `{"answered": false, "reason": "timeout" | "cancelled" | "busy"}`.
- **A7 — Timeout is a user setting; off by default.** `[console] ask_user_timeout_seconds`. **`0` — the default — disables the timeout:** the question waits until it is answered, or the run is stopped or cancelled. This follows ADR-067, which made every other blocking human prompt wait indefinitely by default, and it is safe because the per-call tool clock pauses for the whole wait. A positive value opts into **auto-continue**: on expiry the run proceeds with `answered: false` — the model decides what to do without the answer; the round never fails the run — and the card shows the remaining time using the approval card's existing deadline copy ("Auto-continues in m:ss"). The same key is also the user's off switch for anyone who set a timeout and regrets it. Claude Code's `never` is therefore the default here, not an aspiration.
- **A8 — Composer stays usable; typing answers.** Matching approvals, the composer is never locked. A message sent while a question is mounted answers it: the text becomes `other_text` for every unanswered question, the round resolves, and the message is not also sent as a turn. Two exceptions: slash commands dispatch normally and leave the question pending; a send with staged attachments or RAG evidence goes out as a normal turn and leaves the question pending — carrying staged context into a tool result is meaningless, and discarding it silently destroys work.
- **A9 — One question at a time.** One live question round per session. A second `ask_user` — from a sibling sub-agent, or a second call in the same turn — returns `busy` immediately with instruction not to retry; two consecutive `busy` results in one run become a hard refusal so a retry loop cannot drain the turn budget. Depth is expressed by batching up to four questions per call, which is how both references do it.
- **A10 — Background, cancel, headless.** A question for a session not being viewed parks with the badge and toast and mounts on visit. Run cancel or revocation resolves it `cancelled`. In a run with no UI, the tool is not registered at all — the same posture as the `todo_*` tools and Codex's refusal in exec mode.
- **A11 — Sub-agents may ask.** The card names the asking agent.
- **A12 — Gate defaults on.** `[tools] ask_user_enabled` exists and defaults **on** — a deliberate exception to the repo's off-by-default gates. Every other gate is off because the tool touches data, disk, or network; this one touches only the user's attention, and a tool whose purpose is to initiate contact cannot be discovered while invisible. It is exempt from the Allow/Ask/Off permission layer: raising an approval card to ask whether the agent may ask is two interruptions for one.
- **A13 — Restraint is part of the tool.** The tool description spends most of its words on when *not* to ask: only for decisions genuinely the user's to make, never for a choice with a conventional default or a fact discoverable from the code. With the gate on, the default behaviour is what every user gets.
- **A14 — Transcript record.** On resolve, one marker line per question with the answer, `(unanswered)`, `(timed out)`, or `(cancelled)`, in the same style as task markers. Written on resolve only; a question pending when the app is killed leaves nothing.

### Non-Functional Requirements

- No new locks, no new blocking on the UI thread; the round rides the existing machinery.
- A question and an approval in the same session are both decidable, each in its own card. Neither evicts, blocks, or reorders the other; cross-kind ordering is a follow-on, not a requirement here.
- Every bound in A1 is enforced before anything reaches a widget.

## Feature B — Persistent task list

### Functional Requirements

- **B1 — Placement.** A pinned panel above the transcript, outside the scroll region, alongside the task-card slot. Visible whenever the session's list is non-empty; hidden when it is empty. It never appears on its own for a session that has no tasks.
- **B2 — Content.** A header line — `Tasks · 3 of 7 done · Writing the migration` — then one line per task with its status glyph (`[ ]` pending, `[~]` in progress, `[x]` done) and label. The in-progress task shows its `activeForm` and is visually distinguished. Labels are one line each, truncated (the existing 200-char sanitisation applies).
- **B3 — Live updates.** Updates in place on every `todo_create` / `todo_update`, through the same change callback that writes the transcript marker today. The transcript marker stays — it is the after-the-fact record in scrollback; the panel is the live view.
- **B4 — Scope.** Session-scoped, like the store: switching sessions shows that session's list, and the list survives in-app navigation exactly as the store's snapshot already does. It does not survive restart (TASK-13216 AC #6, deliberate). Sub-agent tasks appear — the store is shared by design.
- **B5 — Collapse.** One click or keybinding toggles between the full list and the header line alone, so a long list does not consume the transcript. Expanded, the panel has a maximum height with internal scroll. The collapsed state is remembered per session for the life of the app.
- **B6 — Read-only.** No editing from the panel in v1. Completed tasks stay listed until the agent removes them.

### Non-Functional Requirements

- Zero cost when there are no tasks: the panel is not mounted, not just hidden.
- Rendering a 50-task list (the store's cap) must not block the UI thread perceptibly.

## UX Requirements (both features)

- Existing approval, skill-install, and skill-script behaviour is byte-identical afterwards. That is the parity oracle for any implementation: the existing interrupt test battery passes unchanged.
- Neither surface steals keyboard focus from the composer.
- Both render in light and dark, on the Console's existing tokens.
- Console's User Guide page is updated with both features (CLAUDE.md's UI-change rule).

## Acceptance Criteria

**A — ask_user**

- [ ] AC-A1 An agent call with 2 questions × 3 options renders a card with both questions; selecting an option in each and pressing Enter returns both labels to the model in the A6 shape.
- [ ] AC-A2 A call with 5 questions, or an option list of 1, or a 13-char header, is rejected with an actionable tool error and renders nothing.
- [ ] AC-A3 Every question shows an "Other" input regardless of the call's options; text entered there returns as `other_text`.
- [ ] AC-A4 Submitting with one question unanswered returns that question with `unanswered: true` and the others answered.
- [ ] AC-A5 With the default `ask_user_timeout_seconds = 0` and `max_tool_call_seconds` lowered to 2, an unanswered question is still mounted and the run still alive after 5s; the card shows no countdown.
- [ ] AC-A5b With `ask_user_timeout_seconds = 2`, an unanswered question resolves `{"answered": false, "reason": "timeout"}` within 3s and the run continues; the card showed a countdown before expiring.
- [ ] AC-A6 Typing a message and sending while a question is mounted resolves it with the text as `other_text` and does not dispatch a user turn; a `/` command sent in the same state dispatches normally and leaves the question mounted; a send with a staged attachment dispatches normally and leaves the question mounted.
- [ ] AC-A7 A second `ask_user` while one is live returns `busy` in under 100ms; a third consecutive one in the same run is refused.
- [ ] AC-A8 A question raised for a background session lights the rail badge, fires one toast, and mounts when that session is visited; answering it there resolves the waiting run.
- [ ] AC-A9 Stopping the run with a question mounted clears the card and the tool returns `cancelled`.
- [ ] AC-A10 A question and an approval raised for the same session are both decidable, each in its own card; resolving either leaves the other mounted and answerable.
- [ ] AC-A11 With `[tools] ask_user_enabled = false` the tool is absent from the catalog; in a headless run it is absent regardless of the gate.
- [ ] AC-A12 After resolve, the transcript carries one marker line per question with its outcome.
- [ ] AC-A13 A 4-question × 4-option card leaves at least half the transcript's previous height visible.

**B — task list**

- [ ] AC-B1 A session with no tasks shows no panel; after `todo_create` the panel appears above the transcript with the task and a correct header count.
- [ ] AC-B2 `todo_update` to `in_progress` moves the highlight and shows the task's `activeForm` in the header; `completed` changes its glyph to `[x]` and the count.
- [ ] AC-B3 Switching to a session with a different list shows that list; switching back shows the original, unchanged.
- [ ] AC-B4 Collapsing hides every task line and leaves the header; the state persists across a session switch and back.
- [ ] AC-B5 A 50-task list renders with internal scroll and the transcript remains usable.
- [ ] AC-B6 The existing transcript `☰ Tasks` marker still appears on each mutation.
- [ ] AC-B7 The existing interrupt battery passes unchanged.

## Testing Plan

- **Unit** — A1 bounds (every limit, both sides); A6 result shapes for all four outcomes; B2 header/glyph rendering from a snapshot.
- **Concurrency** — question + approval same session (AC-A10); double `ask_user` → `busy` and the refusal ceiling (AC-A7); typed-answer resolve does not double-send (AC-A6). Written against the existing concurrency-suite harness (`FakeApp` / `_wait_until`).
- **UI geometry** — AC-A13 and AC-B5 under the bundled CSS, following the approval card's existing geometry tests.
- **Parity** — the existing interrupt battery, unchanged, is a gate on every PR.
- **Headless** — AC-A11.

## Implementation Plan & Milestones

Smallest-first. Each milestone is one PR and ships value on its own.

| # | Delivers | Why this order |
|---|---|---|
| **M1** | Feature B — the task panel (B1–B6) | The backend is done; this is a surface over an existing store. Cheapest real win, and it retires the "where did my task list go" complaint on day one. |
| **M2** | Feature A core — tool, validation, card, timeout, `busy`, park/cancel/headless (A1–A7, A9–A14) | The feature itself. Rides the existing round machinery; no typed-answer yet. |
| **M3** | A8 — typed-answer interception, plus User Guide | The one piece that touches the send path, isolated so it can be reviewed on its own. |

## Rollout

- M1 and M2 ship behind nothing new: the task panel appears when there are tasks, and `ask_user` is gated on by default per A12. A user who dislikes being asked flips `[tools] ask_user_enabled = false`.
- Announce in the release notes with the A13 restraint framing, so expectations match: the agent asks when it is genuinely stuck, not routinely.

## Open Questions

1. **Sub-agent tasks in the panel.** The store is shared across the fleet by design, so a child's tasks appear alongside the primary agent's. Is per-agent attribution in the panel wanted, or is one merged list right?
2. **Keep the transcript task marker** once the panel exists (B3 says yes). Is the scrollback record worth the transcript noise?

## Risks & Mitigations

- **The model over-asks.** Mitigation: A13's restraint text, the A9 `busy` ceiling, and measuring asks-per-run after release.
- **Unsubmitted selections lost.** Today every task-card update rebuilds all cards, so an unrelated round resolving while a question is half-answered would discard the user's toggles. Mitigation: the question card ignores a re-push whose round id matches the one it is showing (guard resets on hide). Known; scoped to the card.
- **Silent queue.** With A9, a queued question on the *active* session has no card and no toast until the current round resolves. Acceptable for v1 (the alternative is the pre-#1836 behaviour, where it was destroyed outright); worth a small "1 question waiting" chip later.
- **Stale requirements.** This repository merges roughly 200 commits a day. Every implementation PR re-establishes its test baseline at its own merge base rather than trusting a recorded count.

## File Touchpoints (planned)

- `Agents/local_tool_provider.py` — `ask_user` spec, registered when a UI callback is supplied (the `todo_*` pattern).
- `Chat/console_chat_controller.py` — `request_user_questions`, riding the round machinery from PR #1836; the todo change callback gains a panel sink.
- `Widgets/Chat_Widgets/chat_question_card.py` (new) — the card; `Widgets/Console/console_task_panel.py` (new) — the panel.
- `Widgets/Console/console_session_surface.py` — mounts the panel; `Widgets/Chat_Widgets/chat_task_cards.py` — routes question payloads to the card.
- `UI/Screens/chat_screen.py` — `TaskResumeState.pending_question`, the `QuestionAnswered` handler, the A8 send-path check.
- `Agents/builtin_tool_gate.py` — the `ask_user` gate entry (hand-listed, like `web_deep_search`).
- `Docs/User_Guide/console/agent-runs-and-tools.md`.

## Success Signals

This app has no telemetry pipeline, and this PRD does not add one. The signals below are limited to what existing stores already record; anything else is stated as unmeasured rather than promised.

- **A — derivable from `AgentRuns_DB` today**, because every `ask_user` call and its result is persisted as a tool step: asks per run (watch for inflation), share of asks that resolved `answered: true` versus `timeout`/`cancelled`, and `busy` rate (target: near zero). Time-to-answer is the gap between the tool-call and tool-result step timestamps. One read-only query, no new instrumentation.
- **B — not measurable without new instrumentation.** Panel expand/collapse has no sink and none is added. If a "did the panel help" signal is wanted, that is a separate, explicitly scoped follow-on.

## References

- Claude Code `AskUserQuestion` tool contract; `TodoWrite` semantics.
- Codex `tui/src/bottom_pane/request_user_input/`, `experimental_request_user_input`.
- ADR-067 (indefinite human approval waits with a pausable per-call clock) — the basis of A7.
- ADR-032 (local agent tool permission boundary); TASK-13216 (`SessionTodoStore`); task-15661 / PR #1836 (round-keyed FIFO payloads, per kind).
- `Docs/superpowers/specs/2026-08-19-console-user-interaction-design.md` — the broader program design this PRD is carved from; its follow-on sub-projects are out of scope here.
