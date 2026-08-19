# Console user-interaction and support features — program design

Date: 2026-08-19
Status: design approved; sub-project A specified, sub-project C requires its own design pass
Related: ADR-032 (local agent tool permission boundary), TASK-13216, TASK-910, task-581

## 1. Problem

The Console agent runtime can act but cannot *ask*. An agent that hits a genuine
fork — which provider, which file, which of three interpretations of an ambiguous
request — has no way to put that question to the user. It must guess, or stop and
emit prose the user may not be watching for.

Separately, the agent can already track a task list, but the user effectively
cannot see it: the list renders as a plain-text block appended to the transcript
on every mutation, which scrolls away within a few turns.

This program adds the agent↔user interaction surface the Console lacks, and
unifies the four ad-hoc "agent needs you" cards that grew up around approvals.

## 2. Ground truth — what already exists

Roughly half of this is built. Establishing that first, because the naive reading
of the request ("build a question tool and a task list") duplicates working code.

### Task list — backend complete, no UI

- `Agents/session_todo_store.py` — `SessionTodoStore`: stable IDs, compare-and-swap
  versioning, 50-item cap, one-`in_progress` invariant, snapshot export/import.
- `Agents/local_tool_provider.py` registers `todo_create` / `todo_update` /
  `todo_get` / `todo_list`, governed by ADR-032. `todo_write` was replaced in
  TASK-13216.
- The only UI is `format_todo_marker` (`Chat/console_agent_bridge.py:733`) —
  a display-only `☰ Tasks (n in progress):` block appended to the transcript.
  Not interactive, not pinned, not re-derived after restart.
- Per TASK-13216 AC#6, session-scoped-not-durable is a deliberate decision, not
  an oversight. This program does not change it.

### Agent asks the user — nothing, but the round-trip machinery exists

There is no ask/elicitation seam. There is a complete blocking round-trip:

- `ConsoleChatController.request_mcp_approvals` (`console_chat_controller.py:3810`)
  is **owner-agnostic** despite its name — MCP tools and built-in agent tools
  (`server_key="agent:builtin"`) both ride it.
- Worker thread blocks → `call_from_thread` pushes into
  `TaskResumeState.pending_approval` → `ChatTaskCards.sync_state` →
  `ChatApprovalCard.set_batch` → `ApprovalDecided` → `resolve_pending_approval`
  releases the thread.
- With round-ID bookkeeping, park-if-background-session, fleet badge + toast,
  1s deadline polling, and cancel-revocation.

### Bounds that constrain any design here

- `RunBudget.max_tool_call_seconds = 300.0` (`Agents/agent_models.py`) caps a
  single tool call. Exceeding it means the wrapper reports "timed out" for a call
  still live on an abandoned thread.
- MCP approval timeout defaults to 120s, polled at `_MCP_APPROVAL_POLL_SECONDS = 1.0`.
- `ChatApprovalCard` must stay synchronous end-to-end — `sync_state` cannot `await`.

### Attention and progress — nothing

No bell, no `notify`, no desktop notification anywhere in the app. The only
attention affordance is the Console rail badge (`"1 appr"`), visible only on the
Console with the rail open. Live progress amounts to one `"Working."` string in
`console_status_chips.py:581`; step markers land only after a step completes.

## 3. The defect this program inherits

`_parked_approval_payloads` is keyed by **`session_id`**, not `round_id`
(`console_chat_controller.py:4019`). The round *registry* is round-keyed and
tracks siblings independently, but the retained payload — the sole source
`switch_session` re-derives the mounted card from — is one slot per session with
last-armed-wins semantics.

Consequence: two same-session rounds mean the second overwrites the first. The
first round's card vanishes and it hangs undecidable until its own timeout fails
it closed.

A two-part TOCTOU guard sits on top of this invariant
(`console_chat_controller.py:4187-4247`), accumulated over three fix rounds — the
original Qodo TOCTOU, then a fix-round-3 stranded-card regression from dropping
the older-sibling check. That guard prevents a resolving round from *clearing* a
sibling's card. It does not give an older sibling a turn on screen.

**The same shape exists three times.** `_parked_skill_install_payloads`
(`:1620`) and the skill-script equivalent are also session-keyed, each with its
own re-derive path. TASK-910 and task-581 already converted the *event
registries* of those two bridges from single global slots to per-round
registries, for exactly this reason — but left the payload/mount half unfixed.

So the re-key is a twice-landed pattern applied to the half that was skipped, not
novel risk. `Tests/UI/test_skill_install_concurrent_confirms.py` already contains
the same-session concurrency tests to mirror, along with a reusable
`FakeApp`/`_wait_until` harness.

## 4. Program decomposition

| # | Sub-project | Delivers |
|---|---|---|
| **PR0** | Parked-payload re-key | Round-keyed retained payloads with per-session FIFO across all three bridges and their re-derive paths. Pure defect fix, no user-visible change. |

**PR0 is three parallel edits, not one abstraction.** Each bridge owns an
independent lock — `_approval_state_lock` (`:1331`),
`_pending_skill_install_lock` (`:1613`), `_pending_skill_script_lock` (`:1641`) —
so the re-key cannot share a critical section across them. Each bridge has its
own retained-payload map (`:4019`, `:1620`, `:1645`) and its own three re-derive
call sites (`:3265`/`:3440`/`:3509` for install, `:3266`/`:3441`/`:3510` for
script, and the approvals equivalents). The edit is the same shape three times
under three separate locks; attempting to unify the locking is a different and
much riskier change that belongs to C, if anywhere.
| **C** | Unified interrupt surface | One host and one card model for approvals, skill-install, skill-script, resume, and questions. |
| **A** | `ask_user` | The tool, the question card as C's first renderer, typed-answer interception. |
| **B** | Persistent task list | A pinned surface over the existing `SessionTodoStore`. |
| **D** | Attention when away | Bell / notification / cross-screen badge when a run blocks on the user. |
| **E** | Live progress | What the agent is doing now, elapsed time, visible cancel during long tool calls. |

**Order: PR0 → C → A → B → D → E.**

C moved ahead of A on evidence (§6): both reference implementations converged on
a single unified interrupt surface rather than per-kind bespoke cards. Building
A's card as the fifth bespoke card and rewriting it in C is waste.

**C is not designed yet.** It was sequenced third when A was brainstormed, and
moving it forward means it needs its own design cycle before implementation. A's
design below is written against a card host that C must define.

## 5. Sub-project A — `ask_user`

### 5.1 Tool contract

Registered in `Agents/local_tool_provider.py` beside `todo_*`, conditional on the
controller supplying an ask callback. Absent in headless runs; absent from the
MCP Hub. Same ADR-032 posture the `todo_*` tools hold.

```
ask_user(questions: [
  { question: str,        # <= 500 chars
    header: str,          # <= 12 chars, the UI chip
    multiSelect: bool,
    options: [ {label: str <= 100, description: str <= 300} ]   # 2-4
  }
])  # 1-4 questions

-> {"answered": true,
    "answers": [{"question": ..., "selected": [...], "other_text": str|null,
                 "unanswered": bool}]}
-> {"answered": false, "reason": "timeout" | "cancelled" | "busy"}
```

"Other" free-text is injected by the card, never declared in the schema — the
model cannot suppress the user's escape hatch. Validation matches
`SessionTodoStore` strictness: strict types, UTF-8, no blanks, control characters
flattened for render. The payload is model-controlled text going straight to a
renderer.

**Depth 1, fail fast.** One live question round per session. A second `ask_user`
— from a sibling fleet child or a second call in the same model turn — returns
`{"answered": false, "reason": "busy"}` immediately. Depth is expressed by
batching up to four questions into one call, not by queueing calls. This matches
both reference implementations (§6) and avoids blocking a worker thread for the
full budget only to report that nothing was ever shown.

Approval and skill-confirm rounds do **not** fail fast — an agent cannot proceed
without them, so they use PR0's FIFO. Only questions bounce.

**A bounce must not become a retry loop.** A model that receives `busy` and
immediately retries burns turns against `max_model_turns` (30) with no progress.
The `busy` result therefore carries explicit instruction not to retry — proceed
without the answer, or ask again in a later turn — and the run counts consecutive
bounces, degrading to a hard refusal after the second so a determined retry loop
terminates rather than consuming the turn budget.

**Gating.** `[tools] ask_user_enabled`, **default ON**. This is a deliberate
deviation from the repo convention that gates default off, and from Codex, which
ships its equivalent experimental and off by default. The argument: every other
gate defaults off because the tool touches user data, disk, or network.
`ask_user` touches only the user's attention, and a tool whose entire purpose is
to initiate contact cannot be discovered while invisible. Claude Code ships its
equivalent always-on. As a `LocalToolSpec` rather than a `_GATEABLE_BUILTINS`
row, it needs a hand-added entry in `all_tool_gates()` — the `web_deep_search`
precedent.

**Restraint guidance is part of the deliverable.** A model handed a new tool
uses it; because the gate defaults ON, every user gets whatever the default
behavior is. Both reference implementations spend most of their tool description
on *when not to ask* — reserving it for decisions genuinely the user's to make,
not for choices with a conventional default or facts discoverable from the code.
This text ships as the tool description and is registered in the internal-prompts
registry alongside the other agent prompts, not hardcoded at the call site.

**Permission gate: exempt.** No `risk_tags`, never floored to `ask`. Raising an
approval card to ask whether the agent may ask a question delivers two
interruptions for one question.

**Sub-agents may ask.** Any agent in the fleet can call `ask_user`; the card
attributes the question to the asking agent. `run_context.current_run_id()` is
already available inside local tool handlers; mapping a run id to a child's
display label goes through the fleet coordinator.

### 5.2 Round plumbing

Extract the round lifecycle currently inlined in `request_mcp_approvals` — mint
round id, register, mount-or-park, `event.wait(1.0)` to deadline, resolve, tear
down — into a shared `_run_pending_round(...)`. `request_mcp_approvals` keeps its
name and behavior; `request_user_questions(questions, *, session_id)` is a
sibling calling the same helper with a different payload and resolver.

- `_pending_approval_rounds` entries gain `kind: "approval" | "question"`. The
  map name stays, for the same reason `request_mcp_approvals` keeps its legacy
  name; renaming belongs to C.
- `TaskResumeState` gains `pending_question`, routed to the question renderer.
- Budget: **240s from request**, one deadline covering everything, under the 300s
  ceiling with slack for 1s poll granularity and marshalling. Semantics are
  *auto-continue*, matching Claude Code's naming: on expiry the agent proceeds
  with `answered: false` rather than failing the run.
- Because the ceiling is hard, we cannot offer Claude Code's `never` option. A
  question that must not expire is not expressible here.

### 5.3 Card and interaction

Rendered through C's host. Built on Textual's own `RadioSet` / `SelectionList`
per `multiSelect` rather than hand-rolled selection state, and synchronous
end-to-end, reusing `ChatApprovalCard`'s collision-safe remount pattern.
(Verified against the pinned Textual 8.2.8: `RadioSet`, `RadioButton`,
`SelectionList`, `Checkbox`, and `Input` are all available.)

Per question: header chip, question text, options with descriptions, and an
always-present "Other" `Input`.

- **Live countdown.** The card renders remaining time, as Codex does
  (`auto-resolves in ...`). A silent deadline is user-hostile: a card waiting for
  you looks identical to one about to expire.
- **Partial submission allowed.** Unanswered questions submit as
  `unanswered: true` rather than blocking the submit button. Both reference
  implementations permit this; Codex renders `Submit with N unanswered`.
- **No focus stealing.** The approvals chip exposes an explicit
  `Enter -> review_approval` action to focus its card, which means these cards
  deliberately do not grab focus. A question card that auto-focused would eat
  keystrokes from a user mid-sentence. The chip must therefore also cover
  question rounds so keyboard-only users can reach the card.
- **Height must be bounded.** The approval card is `height: auto` at any row
  count (`css/tldw_cli_modular.tcss:9433`, `:9448`) in a non-scrolling region
  above the transcript, with live-repro'd `fr-inside-flex` ballooning scars at
  `:9408` and `:9466`. Approval batches are small, so it never bit. Four
  questions with four described options each plus four "Other" inputs is 30-40
  rows and would consume the transcript. This card needs an explicit
  `max-height` and internal `overflow-y: auto` — discipline no sibling card has —
  and a geometry test alongside the existing
  `test_batch_row_widgets_have_nonzero_geometry_and_do_not_overlap_under_bundled_css`.
- **Esc is already bound.** `Binding("escape", "focus_console_composer_home")`
  (`UI/Screens/chat_screen.py:1782`) already returns focus to the composer. No
  new binding.

### 5.4 Typed-answer interception

The composer stays usable while a question card is up — matching approvals, which
do not block sends (`_console_send_blocked_reason`, `chat_screen.py:17141`, gates
only on empty RAG launches, readiness, and attachments). Locking the composer
would be new, divergent behavior, and this codebase already carries a scar from
an accidental composer lockout (`chat_screen.py:14294`).

Instead: if the active session has a **mounted** question round, a send resolves
that round — returning any selections already made plus the typed text — and
appends the message to the transcript so the exchange reads naturally. It is not
also dispatched as a user turn. Only a mounted card intercepts; a bounced or
background round never touches the composer.

Two cases the interception must handle explicitly:

- **Slash commands are never swallowed.** Console has a command grammar and
  popup surface; a command dispatches normally and leaves the round pending.
- **Staged context refuses interception.** If attachments or staged RAG evidence
  are present when the user sends, the send dispatches as a normal user turn and
  the question round stays pending. Carrying staged context into a tool result is
  meaningless, and silently discarding it would destroy work the user staged
  deliberately. The user answers the question via the card.

### 5.5 Edges

| Case | Result |
|---|---|
| Deadline expires while mounted | `{"answered": false, "reason": "timeout"}` — auto-continue |
| Second concurrent ask | `{"answered": false, "reason": "busy"}`, returned immediately |
| Run cancelled / revoked | `{"answered": false, "reason": "cancelled"}`. Note `_revoke_run_approvals` stamps `"deny"` per name today; a question round needs its own resolution, not a verdict string |
| Session switch | Card re-derives from PR0's round-keyed payload map |
| App restart with question pending | Round is in-memory and lost, as approvals are today. The transcript marker is written **on resolve only**, so a question pending at kill time leaves nothing in scrollback |

The resolve marker follows `format_todo_marker`'s conventions
(`console_agent_bridge.py:733`): display-only, appended in-memory with
`persist=False`, rendered markup-off, embedded newlines and terminal controls
flattened, each line truncated. One header line naming the asking agent, then one
line per question with its answer — or `(unanswered)`, `(timed out)`,
`(cancelled)` as applicable.
| Headless run | Tool never registered — no UI callback. Matches Codex, which refuses `request_user_input` in exec mode outright |

### 5.6 Phasing

1. **PR0** — re-key, all three bridges, mirroring the existing concurrency suite.
2. **C** — unified interrupt surface (needs its own design first).
3. **A.1** — tool, schema, gating, round plumbing.
4. **A.2** — question card on C's host: countdown, partial submit, bounded height.
5. **A.3** — typed-answer interception.

## 6. Reference implementations

Established by string analysis of the shipped `claude` and `codex` binaries on
this machine. Strong evidence for feature existence and UI copy; weaker for exact
control flow.

|  | Claude Code | Codex |
|---|---|---|
| Tool | `AskUserQuestion`, 1-4 questions per call | `experimental_request_user_input`, config-gated under `[tools]` in `ConfigToml` |
| Placement | inline in transcript flow | bottom pane — `tui/src/bottom_pane/request_user_input/{mod,render}.rs` |
| Timeout | `askUserQuestionTimeout`, a policy/settings value labelled "Question auto-continue timeout", with a `never` option | live `auto-resolves in ...` countdown |
| Partial answers | — | `Submit with N unanswered`, per-field `Answer required` |
| Navigation | — | `<-/-> to navigate fields`, `ctrl+p / ctrl+n change field`, `Question X/Y`, `option X/Y`, `esc to cancel`, `to interrupt` |
| Concurrency | — | single bottom-pane view host (`bottom_pane/mod.rs`); `mcp_server_elicitation.rs` and `request_user_input/` share that one slot |
| Headless | — | refused: "request_user_input is not supported in exec mode for thread" |
| Unified surface | — | one `tui/src/app/background_requests.rs` serves exec approval, file-change approval, permissions approval, MCP elicitation, and request_user_input |

Conclusions drawn: neither system holds a deep queue (both use one display slot
plus multi-question batching); timeout semantics are auto-continue, not failure;
a visible countdown and partial submission are both established practice; and
both converged on a unified interrupt surface, which is why C moved ahead of A.

Where we knowingly diverge: our card sits in a top banner, which neither
reference does, because in this codebase that slot already exists with working
mount, sync, park, and visibility plumbing.

## 7. Testing

1. **Mirror `Tests/UI/test_skill_install_concurrent_confirms.py`** for PR0 across
   all three bridges, reusing its `FakeApp`/`_wait_until` harness.
2. Concurrent same-session rounds in every pairing — approval+approval,
   approval+question, install+script — each decidable in FIFO order, none
   stranded.
3. Schema-bound validation at `SessionTodoStore` strictness.
4. Every resolution path: answered, partial, timeout, busy, cancelled.
5. Typed-answer interception resolves the round, does not double-send, and does
   not swallow slash commands.
6. Card geometry under bundled CSS: bounded height, internal scroll, no overlap
   at four questions with four options each.
7. Headless: tool absent from the catalog.
8. A repeated `ask_user` against a live round bounces, and a persistent retry
   loop terminates at the bounce ceiling rather than draining `max_model_turns`.
9. PR0 regression coverage runs per bridge, under each bridge's own lock — a
   passing approvals suite is not evidence for skill-install or skill-script.

## 8. Decisions log

| Decision | Choice | Note |
|---|---|---|
| Program scope | Full interaction family, five sub-projects | |
| Tool shape | Full AskUserQuestion parity, minus `preview` | |
| Placement | Existing `ChatTaskCards` slot | Diverges from both references |
| Composer | Stays usable; typing answers the mounted question | Approvals do not lock the composer either |
| Sub-agents | May ask | With depth-1 fail-fast |
| Queue depth | Fail fast beyond 1 | Matches both references |
| Round plumbing | Re-key parked payloads, all three bridges, own PR first | Fuller fix over entry-point serialization |
| Gate default | ON | Deliberate deviation from repo convention and from Codex |
| Sequencing | PR0 -> C -> A -> B -> D -> E | C pulled forward on reference evidence |

## 9. Open items

- **C needs its own design cycle** before implementation, now that it precedes A.
- Backlog tasks are not yet filed. IDs must be assigned after a fresh sweep of
  all worktrees and branches — this repo has had six ID collisions.
