# Bound Console conversation history by tokens before dispatch (TASK-322)

**Date**: 2026-07-22
**Status**: Approved design, pending implementation plan
**Base**: origin/dev @ `c67f161b6`
**Backlog**: TASK-322 (depends on 320/321 — see Dependency)

## Why

The native Console send path sends the entire conversation to the provider on
every turn with no token cap — `_provider_message_payloads` only budgets
*images* (`max_history_images`), never text tokens. Once a conversation
exceeds the model window, cloud providers reject the request with a 400 (the
conversation becomes un-continuable), local servers silently front-truncate
(dropping the prepended system prompt first), and token cost grows without
bound. This bounds the dispatched history by real tokens against the model
window, dropping oldest turns while always preserving the system prompt and
the current turn, and tells the user when it trimmed.

## Decisions (user-approved)

1. **Build on the existing `token_counter` seam now.** `get_model_token_limit`
   and `count_tokens_messages` already exist; 320 (refresh the window table)
   and 321 (real tokenizer for non-OpenAI) improve the numbers behind that
   same seam later, with no change to this task's trimming logic.
2. **Notify via a display-only inline transcript note**, once per trimming
   send: "Earlier messages were trimmed to fit the model's context window
   (N dropped)."
3. **Budget = window − response reservation − safety margin**, where
   `margin = max(512, 2% of window)`.

## Architecture

A pure, unit-testable module does the counting + trimming; the controller
wires it in at the single dispatch choke point and appends the note.

### 1. `tldw_chatbook/Chat/console_history_budget.py` (new, pure)

```python
@dataclass(frozen=True)
class BoundResult:
    messages: list[dict]      # trimmed payload, ready to dispatch
    dropped_count: int        # number of history messages removed

def count_console_message_tokens(message: dict, model: str) -> int: ...
def count_console_messages_tokens(messages: list[dict], model: str) -> int: ...
def bound_messages_to_window(
    messages: list[dict],
    *,
    model: str,
    provider: str,
    response_reservation: int,
    per_image_tokens: int = 1024,
    window: int | None = None,           # override; None -> token_counter lookup
    count_fn: Callable[[list[dict], str], int] | None = None,  # injectable counter
) -> BoundResult: ...
```

**`count_fn` is injectable (Finding — tiktoken absence).** tiktoken is not
installed in the dev/test venv, so the real counter falls back to word-split
estimation (the estimate task-321 will replace). `bound_messages_to_window`
defaults `count_fn` to `count_console_messages_tokens`, but accepts an
override so unit tests pass a **deterministic** counter (e.g. one token per
word, or a fixed count per message) and verify the trim *logic* — drop-oldest,
system/current-turn preservation, boundary — independent of whether tiktoken
is present. Tests never assert exact real-tokenizer numbers.

**Token counting adapter (Finding 1 — required).** `count_tokens_messages`
in `token_counter.py` assumes `content` is a string and crashes on the
Console's multimodal payloads (`content` is a *list* of `{type:text}` /
`{type:image_url}` parts for vision turns). `count_console_message_tokens`
handles both shapes:
- `content` is a `str` → count via `count_tokens_tiktoken(content, model)`
  plus the per-message role/format overhead `count_tokens_messages` applies.
- `content` is a `list` → sum `count_tokens_tiktoken` over the `text` parts,
  plus `per_image_tokens` for each `image_url`/image part (image count is
  already bounded by `max_history_images`, so this stays finite). The default
  `per_image_tokens = 1024` is a deliberately conservative flat estimate;
  refining it per provider is out of scope (a natural follow-up alongside
  321).
`count_console_messages_tokens` sums the per-message counts plus the small
fixed reply-priming base that `count_tokens_messages` adds, so its numbers
line up with the existing counter for text-only conversations.

**Trimming (`bound_messages_to_window`).**
- `window = window if window is not None else get_model_token_limit(model,
  provider)`; `budget = window − response_reservation − max(512, window // 50)`.
  **Unknown-model caveat (Finding):** `get_model_token_limit` returns a
  conservative `4096` for models it doesn't recognize (after prefix + provider
  defaults), which would *over-trim* local/custom models that actually have a
  large window (common with llama.cpp under a custom model name). The optional
  `window` override exists precisely so a configured context limit can take
  precedence — this is the seam task-325 (wire the dead `chat_context_limit`
  key) feeds later; until then, the controller passes `window=None` and the
  conservative default applies. Documented as a known limitation, not a
  blocker: over-trimming degrades gracefully (shorter history) and never
  400s, and task-320 refreshing the table plus task-325 wiring the override
  both improve it transparently.
- **Always keep**: (a) the leading system message(s) — the contiguous
  `role == "system"` prefix; (b) the **current turn** — everything from the
  last `role == "user"` message to the end (Finding 4: keyed off the last
  user message, not the last message, so the regenerate/synthetic-user sites
  that can end on an assistant message still preserve the whole live turn).
- **Middle history** (between the system prefix and the current turn) is
  grouped into whole **turns**: each `user` message plus the assistant
  messages that follow it (a leading assistant with no preceding user is its
  own unit). Drop turns **oldest-first** until
  `count_console_messages_tokens(kept, model) ≤ budget`. Whole-turn dropping
  keeps valid user/assistant alternation and never orphans a tool_result from
  its tool_call (AC#2). *Note: `_provider_message_payloads` already filters
  `tool`/`system` rows out of the dispatched payload, so no tool pair is ever
  present to split; the whole-turn rule keeps this correct if that changes.*
- **Degenerate case**: if the system prefix + current turn alone already
  exceed `budget`, keep them anyway (the current turn cannot be dropped) and
  return them — we've applied the maximum safe trim; the provider/local server
  handles any residual overflow as before, but the common case (a long history
  behind a short current turn) is fully bounded.
- Returns `BoundResult(kept, dropped_count)` where `dropped_count` is the
  number of removed message dicts.

This module imports only `token_counter` — the seam 320/321 sharpen.

### 2. Wiring in `console_chat_controller.py`

- **Single choke point.** All four send sites `return await
  self._stream_assistant_response(...)`, whose first action is the
  agent-vs-direct branch (`if self._agent_runtime_enabled and
  self._agent_bridge: return await self._run_agent_reply(...)`), and **both
  branches receive `provider_messages`** — the agent branch immediately copies
  it (`agent_messages = list(provider_messages)`), so a trim applied before the
  branch flows into the agent path too. Trim at the **top of
  `_stream_assistant_response`, before that branch**, so the direct provider
  path and the agent/tool path are both bounded at dispatch (AC#1, AC#4):
  ```python
  bound = bound_messages_to_window(
      provider_messages,
      model=self.model or self.configured_model or "",
      provider=self.provider,
      response_reservation=self.max_tokens or DEFAULT_RESPONSE_RESERVATION,  # 1024
  )
  provider_messages = bound.messages
  if bound.dropped_count:
      self._append_history_trimmed_note(<session>, bound.dropped_count)
  ```
- **`response_reservation`** = `self.max_tokens` when set, else
  `DEFAULT_RESPONSE_RESERVATION = 1024` (Finding 5).

### 3. Trim note (AC#3)

`_append_history_trimmed_note(session_id, dropped)` appends a **display-only**
`ConsoleMessageRole.SYSTEM` row (mirroring `_append_failure_system_row`) with
"Earlier messages were trimmed to fit the model's context window (N dropped)."
The `session_id` is resolved inside `_stream_assistant_response` from its
`assistant_message_id` parameter via `self.store.session_id_for_message(
assistant_message_id)` — the same accessor `_run_agent_reply` already uses.
`_provider_message_payloads` filters `SYSTEM` out of the dispatched payload,
so the note renders in the transcript but is never resent (and never itself
counts toward a future budget). Shown once per trimming send.

**Placement (Finding — store is append-only).** `ConsoleChatStore.append_message`
is append-only (the `position` field it carries is for image attachment slots,
not message ordering — there is no insert-at-position). Since the trim runs at
the single choke point *after* the assistant placeholder was appended by the
send site, the trim note renders **with the current exchange** — immediately
after the streaming reply row for that turn. This is the shipped behavior.
Rendering the note strictly *above* the reply would require a new store
insert-before capability (or trimming+noting at all four send sites before the
placeholder is created); both are heavier than the value and are left as a
possible follow-up. The trim/count correctness does not depend on placement.

**Frequency.** The note is appended on every send that actually drops history.
In a long, continuously-trimming conversation that means one note per turn.
This is honest (each send did trim) and low-cost (a dim display-only row), and
is kept as-is; collapsing repeats or a single per-session marker is a possible
later refinement if it proves noisy.

## Testing (AC#5)

- **Unit (`console_history_budget`)** — the trim-logic tests inject a
  deterministic `count_fn` (e.g. one token per whitespace word, fixed cost per
  image part) so they are exact and tiktoken-independent: text-only
  fits-under-budget (no drop); exactly-at-boundary; over-budget drops oldest
  whole turns; system prefix always preserved; current turn (last user +
  trailing assistant) always preserved; leading-assistant edge unit;
  degenerate system+turn-over-budget (kept, dropped_count for the middle
  only); `window` override takes precedence over the lookup; a resume/
  regenerate shape ending on an assistant message preserves the live turn.
- **Counting-adapter unit** (real `count_console_messages_tokens`, no injected
  `count_fn`): a multimodal `content` list is counted without error and
  includes the per-image cost; a string-content message matches the existing
  `count_tokens_messages` for the same text (parity, so 320/321 flow through).
- **Controller**: a send whose history exceeds budget dispatches the trimmed
  list AND appends exactly one SYSTEM trim note; a send that fits dispatches
  the full list and appends no note; the note is filtered from the next send's
  provider payload.

## Out of scope

- Refreshing the window table (320) or the non-OpenAI tokenizer (321) — this
  consumes their seam; both improve accuracy transparently later.
- Per-provider `per_image_tokens` calibration (flat 1024 estimate here).
- The deprecated enhanced/legacy chat path (being replaced by the Console).
- Server-side/local front-truncation behavior (only the dispatched payload is
  bounded here).
- **Intra-agent-loop growth.** This bounds the conversation history at the
  *initial* dispatch (direct path, and the messages entering the agent/tool
  loop). The agent loop's own multi-iteration growth — tool-call/result
  messages accumulating across iterations within a single turn — is bounded by
  **task-326 (agent RunBudget)**, not here. 322 bounds history-into-the-loop;
  326 bounds the-loop-itself. A very long tool-calling loop could still
  approach the window on a later iteration until 326 lands; that is an
  intentional, documented boundary.
