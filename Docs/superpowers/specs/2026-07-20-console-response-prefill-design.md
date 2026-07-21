# Console Response Prefill — Design

**Date:** 2026-07-20
**Status:** Approved design, pending implementation plan
**Surface:** Native Console chat only. The legacy Chat window is deprecated and gets nothing.

## 1. Overview

Let the user supply the opening of the assistant's reply. The prefill text is sent to the
provider as a trailing `{"role": "assistant", "content": <prefill>}` message (Anthropic-style
prefill; OpenAI-compatible endpoints accept the same shape), and the model continues from it.
In the transcript, the reply visibly grows out of the prefill; the saved message is one
ordinary assistant message containing prefill + continuation, which is exactly what later
turns should see as history.

**Use cases:** steering format/structure (one-shot), keeping roleplay/character voice
(pinned), and steering past hedging/preamble.

### Non-goals (v1)

- Legacy Chat window support (deprecated).
- Multi-line prefill editor (real-world prefills are short; single-line command is enough).
- SillyTavern-style "start the assistant message in the transcript and complete it".
- Prefill threaded through the agent runtime/tool loop — a prefilled send bypasses the agent
  loop for that turn instead (§2); engine-level support is a possible follow-up.
- Reworking `continue_from_message` semantics (it keeps its synthetic user instruction).
- Engine-native continuation flags (vLLM `continue_final_message`, DeepSeek `prefix: true`
  beta) — candidate follow-up, not v1.
- Recording "this response was prefilled with X" per message (no message-metadata column
  exists; overloading `feedback` is a hack we refuse).

## 2. User-facing behavior

### Command grammar

Registered as `prefill` in `default_console_registry()`
(`tldw_chatbook/Chat/console_command_grammar.py:162`), handled in `chat_screen.py` following
the `_console_command_apply_system` pattern (including clearing the composer draft on any
handled outcome).

| Input | Effect |
|---|---|
| `/prefill <text>` | Arm one-shot prefill for the next normal send. |
| `/prefill pin <text>` | Pin prefill for every submit/retry/regenerate this session until cleared. |
| `/prefill clear` | Clear both one-shot and pinned. |
| `/prefill` | Inline status line: current one-shot and pinned values (or "none"). |

Parse rules: `clear` matches only when it is the **entire** args string; `pin` matches only as
the **first token with trailing text**. A one-shot whose text literally starts with `pin ` or
equals `clear` therefore cannot be expressed — documented limitation, not guarded against.

Validation at arm time: text is right-trimmed (Anthropic rejects trailing whitespace in a
trailing assistant turn; applied uniformly so display and payload never diverge), empty after
trim is an inline error, length capped at 4,000 characters with an inline error.

### Lifecycle

- **One-shot** applies to the next `submit_draft` send only. Precedence: if both one-shot and
  pinned are set, one-shot wins for that send; pinned resumes afterward.
- **Consumption rule:** a one-shot is consumed only when a send that used it terminates
  `complete` or `stopped`. It stays armed through `_block()` outcomes (provider not ready,
  skill refusal, policy) and through `failed` sends — so a retry of the failed send
  reproduces the original intent.
- **Pinned** applies to `submit_draft`, `retry_message`, and `regenerate_message`.
  `continue_from_message` never gets prefill (it keeps today's synthetic user-instruction
  semantics). A one-shot never applies to retry/regenerate.
- Both states are **per-session**. Pinned additionally persists per-conversation (§4) and is
  loaded when a conversation binds to the session. Switching sessions or conversations never
  leaks armed state across.
- **Agent loop bypass:** there are no "agent sessions" — the agent runtime is a
  controller-wide gate (`[console].agent_runtime`, default on) checked at the top of
  `_stream_assistant_response`, and under default config it handles every Console send. A
  send where prefill applies takes the **direct provider stream for that turn**, skipping
  the agent loop: true prefill semantics, but native tool-calling/MCP do not run on that
  turn. Skill substitution and chat dictionaries are unaffected (they run before this
  split). Sends without an applicable prefill route exactly as today. The arm confirmation
  copy states that tool calling is disabled on prefilled sends.

### Feedback

- Inline transcript system messages via `_append_native_console_system_message`, with user
  text markup-escaped: "Prefill armed for next send: '<preview>'" / "Prefill pinned:
  '<preview>'" / "Prefill cleared." Confirmation copy notes that the continuation glues
  directly to the last character (consequence of right-trimming).
- The **"What's in play"** inspector block shows both states, labeled "next send only" vs
  "pinned", alongside dictionaries.
- No new composer chrome in v1.

## 3. Architecture and data flow

### State

Following where existing per-session state lives:

- **Pinned** → a new field on `ConsoleSessionSettings` (`console_session_settings.py:145`,
  frozen dataclass, updated via `replace(...)` + `replace_session_settings`) — the same home
  as the per-session system prompt. This means it rides the existing settings flows for
  free: screen-state serialization across app restarts, and the conversation-resume seam
  (§4).
- **One-shot** → transient per-session state on `ConsoleChatSession`
  (`console_chat_store.py:124`), like `draft` and `pending_attachments`, with store
  accessors keyed by session id. Never serialized.

### Single seam: `_stream_assistant_response`

`_stream_assistant_response` (`tldw_chatbook/Chat/console_chat_controller.py:1817`) gains a
`prefill: str | None` parameter and owns both halves of the feature. When `prefill` is set,
the agent-runtime gate at the top of the function is skipped for that turn (§2 bypass) and
the send proceeds down the direct provider-stream path unconditionally.

1. **Payload:** appends `{"role": "assistant", "content": prefill}` as the final element of
   `provider_messages`. The payload always ends in a user turn when it arrives here —
   `submit_draft` builds it ending in the fresh user message, and retry/regenerate run
   `_ensure_user_continuation_instruction` — so appending yields exactly one trailing
   assistant turn and alternation stays valid.
2. **Display seed:** pushes the prefill as the *first stream chunk* via
   `append_stream_chunk`. The store's existing buffer machinery
   (`console_chat_store.py:607-630`) then treats it as ordinary streamed content — zero store
   changes. The seed is injected outside the provider-chunk loop and must not set the
   `emitted_content` flag — a provider that streams nothing still terminates in today's
   "Provider stream ended without content" failed state (§6). The injection point is
   mode-specific:
   - **Normal send:** immediately after streaming starts on the empty placeholder.
   - **Regenerate:** immediately after `begin_variant_stream` (which wipes content/buffer).
     Stop-during-regenerate then discards the prefill along with the partial variant and
     restores the base — correct behavior for free.
   - **Retry:** at the existing lazy `prepare_message_retry` point, before the first real
     chunk. This preserves the current guarantee that a retry producing zero tokens leaves
     the original failed content untouched. (A failed message rejects chunks until prepare
     runs, so eager seeding is not an option.)

Call sites (`submit_draft`, `retry_message`, `regenerate_message`) only compute which prefill
applies — one-shot-or-pinned for submit, pinned-only for retry/regenerate — and pass it.
`continue_from_message` passes `None`.

### Ordering relative to existing transforms

Prefill is resolved and appended **after** `_apply_skill_substitution` and
`_apply_chat_dictionaries` — neither transform ever rewrites prefill text — and after
`_ensure_user_continuation_instruction`, so the synthetic user-continue can never land after
the prefill turn.

## 4. Persistence

- **Pinned:** stored in the existing `conversations.metadata` JSON column (schema v20 — **no
  migration**) under key `pinned_response_prefill` (string). No generic metadata helper
  exists — the dictionary-attach helpers are key-specific — so this adds a small analogous
  helper following the `_write_active_dictionaries` pattern
  (`local_chat_dictionary_service.py:794-808`): read-modify-write that re-parses the current
  metadata, sets only its own key, and writes the whole dict back via
  `db.update_conversation` under optimistic lock (that column is shared with
  `active_dictionaries` — sibling keys must be preserved; `ConflictError` propagates). Reads
  are guarded like `_active_dictionaries` (`json.loads(record.get("metadata") or "{}")` in
  try/except with dict coercion — the known `json.loads(None)` crash class). Write-through
  on pin/clear when a persisted conversation exists; if a session with a pinned prefill is
  persisted later (`persist_session_if_needed`), the pinned value is flushed then. Ephemeral
  sessions keep it in memory only.
- **Load seam:** when a saved conversation is resumed,
  `_console_session_settings_for_resume` (`chat_screen.py:3225`) already receives the full
  conversation record (`SELECT *` includes `metadata`) and today reads only
  `system_prompt`; it additionally parses `pinned_response_prefill` into the session
  settings there. App-start restore needs nothing new — settings already ride screen-state
  serialization.
- **One-shot:** memory only, never persisted.
- **The final message:** a plain assistant message (prefill + continuation), persisted by the
  existing `mark_message_complete` / `finalize_variant_stream` flush. No marker of the
  prefill's extent is recorded (non-goal).

## 5. Provider compatibility

- **Anthropic** (`LLM_API_Calls.py:874`): trailing assistant text passes through verbatim;
  the API guarantees literal continuation. Trailing-whitespace rejection handled by
  right-trimming at arm time.
- **OpenAI-compatible** (OpenAI, Groq, Mistral, OpenRouter, Moonshot, local llama.cpp /
  Ollama / vLLM / etc.): trailing assistant message passes through unmodified
  (`LLM_API_Calls.py:521`). The API model is a *new* assistant message, so the model usually
  continues but may restart or repeat — user docs state this honestly rather than pretending
  prefill is uniform across providers.
- **Verification items** (during implementation, not design unknowns to resolve here):
  - Gemini (`chat_with_google`, role remap to `model` + alternation enforcement) — confirm a
    trailing model turn is accepted.
  - OpenAI Responses-API branch (`use_responses_api`, `payload["input"]`) — confirm shape.
- **Incompatible-provider guard:** if a provider is verified to reject a trailing assistant
  turn, the send skips the prefill in **both** the payload and the display seed and posts an
  inline warning — the transcript must never show text the provider did not receive. v1
  ships the guard mechanism with an empty (or verification-determined) provider list.

## 6. Error handling and edge cases

| Case | Behavior |
|---|---|
| Provider streams zero tokens | Existing "Provider stream ended without content" semantics: message marked `failed`, showing the prefill text; one-shot retained; retry available. |
| Stream fails before any chunk (normal send) | Message shows prefill, status `failed`; one-shot not consumed; retry re-applies. |
| Stop mid-stream (normal send) | Existing stopped semantics: partial content including prefill kept; one-shot consumed. |
| Stop mid-regenerate | Existing base-restore semantics: prefill discarded with the partial variant. |
| Retry produces zero tokens | Original failed content untouched (lazy-prepare preserved). |
| Tool-call turns | Cannot occur on a prefilled send: the agent-loop bypass (§2) means the direct stream handles it, and only the agent bridge ever calls `reset_stream_content`. |
| Empty/whitespace-only prefill text | Inline error; nothing armed. |
| > 4,000 chars | Inline error; nothing armed. |

## 7. Testing

- **Grammar unit tests:** `pin`/`clear`/bare/plain-text parse forms, including the
  `clear`-exact-match and `pin`-requires-trailing-text rules.
- **Controller unit tests:** one-shot consumed on complete/stopped, retained on
  blocked/failed; one-shot precedence over pinned; site matrix (submit/retry/regenerate get
  pinned, continue never, retry/regenerate never get one-shot); payload ends in exactly one
  assistant turn even when the continuation-instruction logic ran; skill/dictionary
  transforms leave prefill untouched; right-trim and length-cap validation; **agent-loop
  bypass**: with a (fake) bridge present and agent runtime enabled, a prefilled send takes
  the direct provider path while an unprefilled send still routes to the bridge. (Note: the
  in-memory test harness gets `bridge=None` from `_ensure_console_agent_bridge`, so bypass
  tests must inject a fake bridge explicitly — the default harness only ever exercises the
  direct path.)
- **Real-store integration tests** (unmocked, in-memory SQLite, real `ConsoleChatStore`, fake
  provider gateway capturing the outbound payload — per the project rule that new protocol
  params get real-implementation integration tests): seeded placeholder streams to
  prefill+continuation and persists; regenerate variant = prefill+continuation with correct
  base restore on stop; retry zero-token leaves failed content; pinned metadata round-trips
  across a conversation reload and preserves sibling metadata keys.
- **Live smoke during verification:** Anthropic (native prefill honored) and local
  llama-server; Gemini/Responses-API as the §5 verification items.

## 8. Verification results (2026-07-20, live smoke on llama-server :9099 + remote API checks)

- **llama.cpp: SUPPORTED, with a verification-determined fix.** llama-server returns HTTP 400
  ("Assistant response prefill is incompatible with enable_thinking.") when the loaded
  template's thinking mode is on. Fix shipped in this branch: `build_llamacpp_chat_payload`
  adds `chat_template_kwargs: {"enable_thinking": false}` whenever the payload ends in an
  assistant turn (the wire-level signature of a prefilled send). Verified live: the model
  literally continues the prefill. This replaces §5's skip+warn guard for llama.cpp — the
  provider is compatible, not incompatible. Rationale: thinking-first generation is
  incoherent with a forced response opening (same logic as the agent-loop bypass).
- **Resume-path bug found and fixed:** `normalize_conversation_row` whitelisted away the
  `metadata` column, so pinned prefill never restored on conversation resume. Fixed with a
  raw passthrough; verified live across a full app restart + rail resume.
- **All lifecycle legs verified live in the running TUI:** one-shot arm → literal
  continuation → consumed on complete; retained through blocked sends (unreachable-provider)
  and failed sends (HTTP 400), with the seed visible on the failed message; pin → voice on
  subsequent sends; merge-safe metadata write AND clear confirmed in SQLite; `/prefill pin`
  (no text) usage error; status lines correct at every stage.
- **Anthropic:** the trailing-assistant payload passes the app's handler unmodified and was
  POSTed to api.anthropic.com (401 — the configured key is invalid, so literal continuation
  could not be confirmed live; Anthropic documents prefill as supported).
- **OpenAI (Chat Completions):** accepted the trailing-assistant payload via the app's
  handler; response was influenced-but-not-literal continuation, exactly as §5 documents.
- **NOT verified (no key / not configured):** Gemini (`chat_with_google` role-remap) and the
  OpenAI Responses-API branch. These remain open §5 verification items; no guard entries
  were added for them.

## 9. Follow-up candidates (explicitly not v1)

- Engine-native continuation flags (vLLM `continue_final_message`, DeepSeek `prefix: true`).
- Multi-line prefill editor modal.
- Per-message record of prefill extent (needs a message-metadata column).
- Composer chrome (chip/toggle) if slash-command discoverability proves insufficient.
