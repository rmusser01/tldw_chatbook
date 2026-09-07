# Console Conversation Inspector — exchange capture + unified cost/context review

**Date:** 2026-08-18
**Status:** Draft for owner review
**Supersedes (UI):** `ConsoleCostModal` (cost chip) and `ConsoleContextModal` (Ctrl+Shift+P) as separate surfaces.

## Why

Clicking the Console token/cost chip today opens a numbers-only breakdown
(`ConsoleCostModal`): per-message token buckets and dollars, no content. The
full next-send payload viewer (`ConsoleContextModal` — system prompt, messages,
tools, staged sources, raw JSON) exists but hides behind Ctrl+Shift+P, and
neither surface can show what was *actually* sent on past turns: the stored
transcript is not what went over the wire (system-prompt edits, tool-set
changes, and transforms happen between turns), and nothing captures requests at
send time.

Goal: clicking the token/cost counter opens one review surface where the user
can see, for every turn, **exactly** what was sent and received — full content
of each piece, system and tool prompts included — alongside what it cost.

## Owner decisions (2026-08-18)

1. **Scope:** capture exact historical payloads per turn, not just next-send.
2. **Granularity:** every LLM call in a turn (each tool-loop iteration),
   request AND response sides.
3. **Size policy:** persist text verbatim; stub binary/base64 content as
   `[<mime>, <size>, sha256:<prefix>]`. No text cap.
4. **UI:** one unified Conversation Inspector replacing both existing modals;
   cost chip, Ctrl+Shift+P, and the command-palette entry all open it.
5. **Capture gate:** on by default with a config kill-switch; captures live and
   die with their conversation; ephemeral sessions keep captures in-memory
   only.
6. **Abandoned regenerations:** keep their captures, marked `abandoned`, rather
   than dropping them the way usage does.

## Non-goals (v1)

- Auxiliary one-shot calls (`complete_auxiliary`: title generation,
  summarization, prompt-improve). Real spend, not conversation turns. A tagged
  aux capture view is a possible follow-up.
- Byte-literal provider wire JSON. Capture happens at the provider-adapter
  boundary (`chat_api_call` kwargs); each `chat_with_<provider>` builds its own
  HTTP framing internally, and prompt-caching `cache_control` markers injected
  inside adapters will not appear. The one exception: the llama.cpp gateway
  branch builds its own wire payload, so its capture IS wire-literal. The UI
  states this caveat.
- Cross-capture content dedup. Captures grow ~O(n²) per session (each call
  re-sends history); per-blob zlib compression (see Persistence) keeps this
  manageable. Content-addressed dedup only if real sessions prove pain —
  stability over cleverness.
- Retention/pruning UI ("clear captured payloads"). Follow-up if needed.

## Architecture

### Capture seam

`ConsoleProviderGateway.stream_chat` is the single choke point both send paths
go through: the direct path (`console_chat_controller.py`,
`_run_direct_provider_reply`) and the default agent path
(`console_agent_bridge.py`, one call per tool-loop iteration). The gateway
already marks per-call boundaries for usage on the per-run
`ConsoleProviderStreamSignals` object, which the controller creates
unconditionally at dispatch (final-review F1 of the cost ticker: signals must
exist for every run, not just citation-repair runs).

Capture rides that same carrier:

- `ConsoleProviderStreamSignals` grows `record_exchange_request(...)` /
  `record_exchange_response_delta(...)` / `close_exchange(status)` mirroring
  its `record_usage_payload`/`close_usage_call` API.
- `stream_chat` builds the request capture from its `_chat_api_kwargs` output
  immediately before `chat_api_call`, tees the response side as it decodes
  stream items (content deltas, tool calls via the existing
  `_ToolCallAccumulator`, usage payloads), and closes the exchange at the call
  boundary. Mid-stream teeing must not alter what the transcript receives
  (the cost-ticker PR3 finisher caught a mid-stream-animation violation; same
  discipline here).
- The llama.cpp branch (`stream_llamacpp_chat` / `complete_llamacpp_chat`)
  bypasses `chat_api_call` and gets its own capture hook recording the literal
  payload it builds. (Its usage gap was cosmetic — local is $0 — but a capture
  gap would not be: local-model users still want to see what is sent.)

Capture is derived from the dispatched `resolution` and the per-run signals
object, never from the controller's mutable `self.provider`/`self.model`
(fleet-session race, cost-ticker PR3 lesson).

### New pure module: `Chat/console_exchange_capture.py`

Dataclasses and pure builders; no I/O.

```
@dataclass(frozen=True)
class ExchangeCapture:
    run_tag: str          # UUID minted per dispatch, alongside the signals object
    seq: int              # 0-based call index within the run
    created_at: str       # ISO timestamp
    provider: str         # resolution.provider
    model: str
    endpoint: str | None  # resolution base URL when known
    request: dict         # allowlisted, stubbed — see Sanitization
    response: dict        # accumulated content, tool_calls, finish info, raw usage payloads
    status: str           # "complete" | "stopped" | "error"
    usage_json: str | None        # THIS call's normalized usage, as
                                   # ProviderUsage.to_json() (shipped as a
                                   # JSON string, not a live ProviderUsage
                                   # instance, so the dataclass stays
                                   # trivially blob/JSON round-trippable)
    omitted_keys: tuple[str, ...] # names of request kwargs dropped by the allowlist
```

`response` additionally carries two boolean markers, set at capture time
rather than post-hoc: `synthetic_fallback` (this call's content is locally
synthesized UI copy — `NO_PROVIDER_CONTENT_COPY`/`UNSUPPORTED_PROVIDER_
RESPONSE_COPY` — not provider output, because the provider returned nothing
usable) and, only on the oversize-truncation path below, `truncated`
(alongside a matching marker in `request`) — `status` itself is never
overloaded to signal either condition; it always reflects the real
complete/stopped/error outcome.

Storing per-call normalized usage is what lets the inspector price individual
calls through the existing pricing catalog — finer than today's per-message
rows, and independent of the known 9-of-11-providers streaming-usage gap
(capture records what we send and what content arrives regardless of whether a
usage chunk ever does; cost columns simply show "unpriced" where usage is
absent).

### Sanitization: allowlist by construction

The request capture enumerates known content-bearing kwargs from
`_chat_api_kwargs`: `system_message`, `messages_payload`, `tools`, `model`,
`api_endpoint`, `api_base_url`, `streaming`, and each sampling/reasoning param
(`temp`, `topp`, `maxp`, `topk`, `minp`, `max_tokens`, `seed`,
`presence_penalty`, `frequency_penalty`, `reasoning_effort`,
`reasoning_summary`, `verbosity`, `thinking_effort`, `thinking_budget_tokens`,
`prompt_caching`).

Any key NOT on the list — today `api_key`, tomorrow whatever ships next — is
dropped by construction and its *name* recorded in `omitted_keys`, so a future
secret-bearing kwarg is safe by default and the viewer still sees what was
withheld. This is the repo's hard-won rule: redaction of provider payloads must
be allowlist-shaped, never denylist (the cost-ticker denylist failed twice on
one file in one session).

Binary/base64 content inside message rows (image blocks, file attachments)
becomes an honest stub: `[image/png, 2.1 MB, sha256:ab12…]` — the hash lets a
user verify identity without megabytes of duplicate base64 per call. Stubbing
is deterministic (same bytes → same stub). Serialization uses `default=str`
defensively; a row that still fails to serialize is replaced by an error
marker, never propagated.

### Lifecycle: store, flush, stop, regenerate, ephemeral

- The controller mints `run_tag` when it creates the signals object and, as
  each exchange closes, attaches the capture to `ConsoleChatStore` keyed by
  `assistant_message_id` (already assigned before dispatch) — in memory only at
  that point.
- **Flush timing:** captures persist at exactly the points usage persists —
  the terminal mark, or the late-attach-against-terminal branch
  (`set_message_usage`'s stop-path inversion: `stop_active_run` finalizes the
  message before cancelling the stream task, so late attaches must flush
  themselves). Piggybacking on usage's schedule inherits both the
  parent-message-durability guard (a not-yet-persisted row cannot take an FK
  child) and the stop-path handling. No DB writes on the event-loop thread
  between "send" and first token; a hard crash mid-call loses that one
  in-flight capture (accepted).
- A stopped call's capture keeps its request plus whatever partial response
  streamed, `status="stopped"`.
- **Regeneration:** a stopped/failed regenerate restores the previous answer
  and adds the message to `_variant_restored_message_ids`; usage attaches are
  dropped there, but capture attaches are KEPT with `status` unchanged and the
  run marked abandoned (owner decision 6) — the traffic really happened. A
  successful regenerate accumulates: both generations' captures remain,
  grouped by `run_tag`, latest current. Nothing ever overwrites an earlier
  run's captures.
- **Ephemeral sessions:** captures live in the in-memory store and are never
  flushed, matching message persistence policy exactly.
- **Idempotency:** persistence is an upsert on unique
  `(message_id, run_tag, seq)`, so a double flush (call-close + terminal, or
  retry) cannot duplicate rows.

### Persistence: ChaChaNotes `message_exchanges`

New table, local-only (no `sync_log` triggers — the `usage_json` precedent:
usage-only flushes are version-neutral and never enqueue sync rows; captures
follow the same contract via a dedicated
`append_message_exchanges_local(...)` write path). No FTS.

```
CREATE TABLE message_exchanges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    run_tag TEXT NOT NULL,
    seq INTEGER NOT NULL,
    status TEXT NOT NULL,
    abandoned INTEGER NOT NULL DEFAULT 0,
    capture_blob BLOB NOT NULL,       -- zlib-compressed JSON of ExchangeCapture
    created_at TEXT NOT NULL,
    UNIQUE(message_id, run_tag, seq)
);
CREATE INDEX idx_message_exchanges_message ON message_exchanges(message_id);
```

- **Compression:** each capture JSON is zlib-compressed at rest. Repetitive
  prompt text compresses 5–10×, which is what keeps "no text cap" sane against
  ~O(n²) session growth (a 100-turn text session sums to tens of MB
  uncompressed).
- **Backstop:** a single blob exceeding 16 MB post-compression (pathological)
  truncates with a structured `truncated: true` marker inside the (now
  stubbed) `request`/`response` dicts, rather than failing the flush --
  `status` is left as the call's real outcome, never overwritten.
- **Schema version:** assumed v32 → v33 at time of writing; the guess was
  stale before implementation started (this repo has had version collisions
  land mid-flight, exactly as anticipated here) — shipped as **v40 → v41**
  (`_CURRENT_SCHEMA_VERSION = 41`, migration file
  `chachanotes_v40_to_v41_message_exchanges.sql`), with the sibling
  migration tests that hard-assert the constant updated to match.
- Soft-delete visibility follows the parent join (a soft-deleted conversation's
  captures are unreachable through normal reads); hard deletes cascade.
- Reads are lazy: the inspector loads one turn's captures on expand, on a
  worker, and decompresses off the event loop.

### Config

`[console] exchange_capture = true` (default on). Read through the resolved
settings layer — NOT raw TOML top-level: the live loader nests raw TOML under
`COMPREHENSIVE_CONFIG_RAW`, and a top-level read is a silent no-op (cost-ticker
PR2's Qodo F4 was exactly this bug). A test asserts the kill-switch works
against live-shaped config. When off, the gateway builds no capture objects at
all; turns recorded while off render as "no capture recorded" (below).

## UI: the Conversation Inspector

One `ModalScreen` replacing `ConsoleCostModal` and `ConsoleContextModal`.
Opened by: the cost chip (`ConsoleCostChipPressed`), Ctrl+Shift+P
(`action_view_chat_context` repointed), and the command-palette entry
(`console_command_provider.py`). Three tabs (`TabbedContent`, the context
modal's existing idiom):

1. **Costs** — today's per-message rows and totals (`build_cost_rows` /
   `build_cost_rows_totals`, unchanged as ground truth), each row expandable
   into its turn's captured calls with per-call cost priced from each capture's
   normalized usage. Where per-call sums and the message's `usage_json`
   disagree (rounding, unpriced calls), both are shown; nothing reconciles
   silently. **Not shipped** (M6, final review): this line originally planned
   "selecting a row jumps to that turn in the Exchange tab" as a cross-tab
   navigation affordance. It was never implemented and there is no
   `console-inspector-cost-row` → `console-inspector-exchange-turn` wiring
   anywhere in `console_conversation_inspector.py` — the drill-in that
   shipped instead mounts each call's cost line directly in place, inside the
   Costs row's own Collapsible (`_load_turn_captures`), so the user never
   leaves the Costs tab to see it. Deferred; no follow-up task filed as of
   this writing since the in-place drill-in already answers the same "what
   did this turn cost" question the jump was meant to serve.
2. **Exchange** — per turn, per call: collapsible sections for System prompt,
   Messages, Tools (schemas), Response, Tool calls/results, Sampling params,
   plus the `omitted_keys` manifest line and status badges
   (`stopped`/`error`/`abandoned regeneration`). Reuses the
   Collapsible+read-only-TextArea idiom from `ConsoleContextModal`.
   Per-piece token counts are *estimates* labeled as such, computed through the
   same estimator seam the composer/budget uses (estimator parity — the
   inspector must never disagree with the composer about the same text),
   shown alongside the provider's *reported* buckets which remain authoritative.
   Turns with no captures (pre-feature conversations, kill-switch off, capture
   failure) render an explicit "no capture recorded for this turn" row with the
   reason when known — never a blank.
3. **Next Send** — the current `ConsoleContextModal` content carried over
   intact: `build_context_snapshot` factory, Raw JSON toggle, Copy JSON,
   Save to File, the ephemeral save-block (`blocked_reason("save-context")`),
   the in-progress warning, Refresh, and the empty-state compaction (LY-13).
   Existing context-modal tests migrate to this tab.

Per-call Copy/Save affordances on the Exchange tab follow the same ephemeral
save-block policy. **Performance:** collapsible bodies are lazy-mounted on
first expand — a 50-call agent turn must not mount hundreds of TextAreas up
front (input-latency audit discipline).

## Error handling

Capture is best-effort and must never break a send: builder exceptions are
logged (via the safe-summary path, never raw payloads — `loguru` sinks leak)
and the exchange is recorded as a failure marker; nothing propagates into the
stream or the store's message lifecycle. Flush failures log and retry at the
next flush point; the inspector shows "capture failed" for affected calls.

## Testing

- **Sanitization:** property test — an unknown/never-seen kwarg key never
  appears in a persisted capture (the same test shape that caught the last
  denylist regression); stub determinism; `default=str` fallback; compression
  round-trip.
- **Store lifecycle:** stop-path late flush persists partial captures against a
  terminal message; regen-restore KEEPS captures marked abandoned (contrast
  with usage's drop, both pinned); successful-regen accumulation by `run_tag`;
  ephemeral sessions never flush; idempotent double-flush.
- **Gateway:** one capture per tool-loop call on the agent path (the DEFAULT
  path — trace it end-to-end, not the path the plan names); direct path;
  llama.cpp branch; kill-switch off builds nothing; mid-stream tee leaves
  transcript output byte-identical.
- **Migration:** v41 schema test + sibling version-constant updates.
- **Config:** kill-switch honored against live-shaped (nested) config.
- **UI:** three tabs render; per-turn cost drill-in (in place on the Costs
  tab — see M6 above, not a cross-tab jump); no-capture rows;
  estimate-vs-reported labeling; lazy mount on expand; Next Send tab behavior
  pins migrated from the context-modal tests.
- **Live verify:** one real-provider session (repo-root key) exercising a
  multi-call agent turn with a tool, a stop mid-stream, and a regenerate;
  confirm captured system/tool prompts match what the adapter was given and
  the abandoned run is visible.

## Risks and open follow-ups

- **DB growth** in long heavy sessions even compressed; revisit dedup or a
  pruning affordance only on evidence.
- **Adapter-boundary caveat** (stated in UI): provider-internal framing and
  injected `cache_control` markers are not visible; llama.cpp branch excepted.
- **Retired surfaces:** `ConsoleContextModal` and `ConsoleCostModal` files are
  absorbed; their tests migrate rather than delete — behavior pins carry over.
- **Aux-call capture** and a "clear captures" maintenance affordance are
  candidate follow-ups, filed at implementation time if wanted.
