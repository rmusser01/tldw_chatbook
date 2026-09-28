# Quit-time Session Usage Summary — Design

- **Date:** 2026-09-22
- **Status:** Draft (pending review)
- **Source:** GitHub issue #365 "Add optional quit-time session usage summary"
- **Branch/task:** backlog task to be created at plan time

## Context and Problem

Quitting Chatbook currently goes straight from confirmed quit into shutdown
cleanup. The app already reports usage *during* a session (the Console cost
chip, per-conversation cost snapshots) but nothing answers "what did I do this
session?" at the end. Issue #365 asks for an optional, config-gated summary
screen shown after quit is confirmed and before the app exits: total tokens
plus elapsed time, auto-dismissing after a few seconds, skippable with a
keypress, and never able to block shutdown.

Two facts about the current codebase shape this design:

1. **The issue's implementation notes are stale.** `chat_token_events.py`
   and `app.current_token_count` no longer exist (retired by the Console
   cost-chip work). There is no session-wide token total anywhere today —
   the Console tracks per-conversation costs, and legacy/Evals paths log
   usage without accumulating it. A new session accumulator is required.
2. **Usage capture is scattered.** `ProviderUsage`
   (`Chat/provider_usage.py`) is the single normalizer for provider usage
   payloads, but payloads reach it (or get logged and dropped) at many
   sites: nine duplicated non-streaming log blocks in
   `LLM_Calls/LLM_API_Calls.py`, the Console gateway's SSE capture,
   Anthropic's stream accumulator, the realtime session, the agent
   service, and Library RAG answers. Streaming paths outside the Console
   gateway mostly discard usage entirely.

## Goals

- After a confirmed in-app quit (`Ctrl+Q` → quit flow), if enabled, show a
  brief modal: total session tokens and elapsed session time.
- Full token coverage: every completed LLM response contributes exact
  provider-reported usage when the payload contains it, else a char-based
  estimate. Covers Console, legacy Enhanced Chat, Evals, research,
  sub-agent fleets, voice/realtime.
- Auto-dismiss after a configurable duration; any keypress skips; shutdown
  can never hang on the summary.
- Graceful degradation at every level; missing data shows a neutral line,
  never an error.
- Default **off**; documented and configurable via `config.toml` and the
  settings screen.

## Non-goals (v1)

- Persistent historical analytics, cost estimates, per-conversation or
  per-workspace breakdowns, exporting the summary (per issue #365).
- Embedding tokens. RAG ingestion and eval embedding steps burn large
  invisible counts; including them would swamp the "how much did I do"
  signal. The ledger API supports adding an embeddings bucket later.
- Idle-excluded "active" time. Elapsed time is wall clock since launch.
- Idle-tracking, per-provider exactness fixes beyond what falls out of the
  shared helper (e.g. adding `stream_options.include_usage` to
  OpenAI-compatible streaming requests is a plan-time option, not a
  requirement).

## Design Overview

Three additive components:

1. **`Chat/session_usage.py` — `SessionUsageLedger`**: a process-wide,
   thread-safe accumulator. No Textual dependencies; importable from
   worker threads and non-UI services.
2. **Tap points**: a small `record_*` call at each site where a provider
   response's usage is parsed — the system boundary. Nine duplicated
   usage-log blocks in `LLM_API_Calls.py` are consolidated into one shared
   helper that logs (existing histograms) *and* records.
3. **`Widgets/session_summary_dialog.py` — `SessionSummaryDialog`**: a
   `ModalScreen[None]` shown by the quit worker after approved-quit
   cleanup, before `App.exit()`.

```
provider response parsed ──► SessionUsageLedger (exact | estimate)
        (LLM_API_Calls, gateway,          │
         agent_service, library RAG,      │ snapshot (counters only)
         realtime)                        ▼
quit confirmed ──► cleanup runs ──► SessionSummaryDialog ──► App.exit()
                                        (set_timer auto-close,
                                         any-key skip, hard cap)
```

## SessionUsageLedger

```python
# Chat/session_usage.py
class SessionUsageLedger:
    def record_provider_usage(
        self, *, provider: str, model: str,
        usage_payload: Any,                       # raw provider payload (may be None/malformed)
        fallback_texts: tuple[str | None, str | None] = (None, None),  # (prompt, response) for estimation
    ) -> None: ...
    def record_exact(self, usage: ProviderUsage, *, provider: str, model: str) -> None: ...
    def snapshot(self) -> SessionUsageSnapshot: ...

@dataclass(frozen=True)
class SessionUsageSnapshot:
    exact_tokens: int
    estimated_tokens: int
    calls: int            # debug/test only; not displayed
```

Rules:

- **Thread-safe** (`threading.Lock`); LLM calls run in workers and
  `asyncio.to_thread` contexts. Counters only — memory is O(1).
- **Never raises.** Accounting must not be able to break a provider call
  (mirrors the `agent_service.py:1591` rule). `ProviderUsage.
  from_provider_payload` already degrades malformed payloads to `None`
  rather than fabricating zeros.
- **Estimate fallback:** when the payload carries no usable usage and
  `fallback_texts` were supplied, estimate via `estimate_tokens`
  (`Chat/usage_recorder.py`, ~4 chars/token) — reuse, don't duplicate.
  Exact always wins when present.
- **Module-level singleton** (`session_usage` accessor) plus
  `reset_for_tests()` — console-internal services can record without App
  access. `Chat/__init__.py` is light and `LLM_API_Calls` already imports
  from `tldw_chatbook.Chat.*`, so no import cycle.
- Display formatting reuses `format_token_count`
  (`Chat/cost_display.py:84`) for consistency with the Console cost chip.

## Tap Points and the Boundary Rule

**Rule: record exactly once, where the HTTP response's usage is parsed.**
Sites that merely *re-read* usage downstream (e.g. the Console cost tracker
recomputing from transcript rows) are not taps. Each provider response
must be counted exactly once.

Known boundary taps (plan phase re-verifies with a
`from_provider_payload` call-site audit and classifies each as *tap* or
*downstream*):

| Site | Kind | Notes |
| --- | --- | --- |
| `LLM_API_Calls.py` ×9 non-streaming blocks | exact, else estimate | Consolidate each duplicated log block into one shared `record_response_usage(...)` helper (logs existing histograms + records). Estimate via prompt/response texts already in scope. |
| `LLM_API_Calls.py` Anthropic SSE accumulator (≈1834–1994) | exact | Usage already accumulated at stream end. |
| `LLM_API_Calls.py` OpenAI Responses stream (`completed_usage` ≈331) | exact | Injected into final chunk; tap where consumed. |
| Other streaming loops in `LLM_API_Calls.py` | estimate | Most discard usage today (no `include_usage` requested). The shared helper's estimate fallback records them from prompt/response texts. Adding `stream_options.include_usage` + final-chunk capture is an optional plan-time exactness win. |
| `Chat/console_provider_gateway.py` `record_usage_payload` (≈6997) | exact | Console + its streams. |
| `Agents/agent_service.py` (≈1594) | exact | Sub-agent fleets do their own HTTP — parsed from their own `resp`, not gateway data (verified: no double count). |
| `Library/library_rag_answer_service.py` (≈654) | exact | Own HTTP response. |
| `LLM_Calls/realtime/openai_session.py` | exact | Voice/realtime. |
| `Chat/console_chat_controller.py`, voice gateways, `UI/Console_Modules/realtime.py` | audit | Classify at plan time; tap only if they parse their own responses rather than consuming gateway signals. |

`Chat/usage_recorder.py` (`UsageTokenRecorder`) is **not** a tap: its
scopes wrap calls that go through the provider functions above, so its
estimates would double-count. It keeps serving the research budget ledger
unchanged.

Embeddings (`get_openai_embeddings`) are not tapped (non-goal).

## SessionSummaryDialog

- `ModalScreen[None]` in `Widgets/session_summary_dialog.py`, patterned on
  `ConfirmationDialog` (structure) and `SplashScreen` (timed close).
- Constructed with `(snapshot, started_monotonic, duration_seconds)` — it
  reads only the passed-in data; no recompute during shutdown.
- Content (copy per issue, minimal):

  ```
  Session summary
  42,318 tokens
  1h 12m session
  ```

  - Token line: `exact + estimated` via `format_token_count`; when
    `estimated_tokens > 0`, append `· includes estimates`.
  - Zero calls: `No usage recorded this session` in place of the token
    line.
  - Elapsed: wall clock from `app._startup_start_time`
    (`app.py:7664`, `time.perf_counter()`), formatted `Xh Ym session`.
  - One dim hint line: `press any key to exit` (implemented action, per
    ADR-031 footer-hint honesty).
- **Timed close:** `on_mount` → `self.set_timer(duration, self._close)` —
  strong reference kept on the screen; `_close` is idempotent and stops
  the timer on every close path (lessons: `set_timer(0.0)` never fires;
  unreferenced timers get GC'd). Duration is a constructor param — the
  config clamp (1–30 s) happens at load, tests may pass short durations.
- **Skip:** `on_key` → any key closes (splash precedent; covers the
  issue's Enter/Esc/q). No `BINDINGS` are added; `ctrl+q` is not rebound
  (ADR-031: it is app-global). No `SafeModalDismissMixin` — there is no
  cancel semantics; dismissal always proceeds to exit.
- Styling: `DEFAULT_CSS` with design tokens (`$ds-*` per ADR-150),
  consistent with other small dialogs; no new stylesheet file, no hex
  literals.

## Quit-Flow Integration

Slot: inside `_run_approved_quit_cleanup` (`app.py:19214–19232`), **after**
`_run_blocking_quit_persistence` completes, **before** `self.exit()`:

```python
if summary_enabled:
    dialog = SessionSummaryDialog(
        snapshot=session_usage().snapshot(),
        started_monotonic=self._startup_start_time,
        duration_seconds=summary_duration,
    )
    try:
        await asyncio.wait_for(
            self.push_screen_wait(dialog), timeout=summary_duration + 2.0
        )
    except asyncio.TimeoutError:
        pass  # hard cap: exit regardless
self.exit()
```

Safety properties:

- **Cleanup never blocked:** persistence, audio teardown, and timer stops
  have already run when the dialog is pushed; the `asyncio.wait_for` hard
  cap bounds the worst case even if dismissal never fires.
- **Watchdog unaffected:** `arm_exit_watchdog` arms at `on_unmount`
  (`Utils/app_shutdown.py:378`) — after `App.exit()` — so the summary
  delay never consumes shutdown grace.
- **Disabled (default):** the branch is skipped and the flow is
  byte-for-byte today's.
- **Scope:** only the confirmed interactive quit path. SIGTERM/SIGINT and
  watchdog hard-exits bypass the summary by design.
- `push_screen_wait` from the quit worker is the established idiom (the
  quit-confirmation dialog at `app.py:19174` does the same).

## Configuration and Settings

New TOML section in `CONFIG_TOML_CONTENT` (`config.py`, near
`[splash_screen]`), with template comments documenting defaults (satisfies
the "default behavior is documented" AC):

```toml
[session_summary]
enabled = false          # show the quit-time session summary
duration_seconds = 3     # auto-dismiss delay, clamped to 1..30
```

- Read via `get_cli_setting("session_summary", ...)` at quit time; invalid
  or missing values fall back to defaults; duration clamped 1–30.
- Settings screen: one small instant-apply group (Checkbox "Show session
  summary on quit" + integer Input for duration), following the
  `[model_catalog]` group pattern (`settings_screen.py:16155–16169`,
  persist worker via `save_settings_to_cli_config`). Placement in an
  existing appearance/behavior category at plan time.

## Error Handling / Degradation Ladder

1. Exact provider usage present → exact total.
2. No usage in payload → estimate from texts (`includes estimates` marker).
3. Nothing recorded all session → `No usage recorded this session`.
4. Ledger/record path raises internally → swallowed (never breaks a call);
   worst case the summary shows (3).
5. Dialog fails to dismiss → `wait_for` hard cap exits anyway.
6. Config garbage → documented defaults.

## Testing

- **Ledger unit tests** (`Tests/Chat/`): exact accumulation, estimate
  fallback, exact-wins-over-estimate, malformed payloads never raise,
  concurrent `record` from threads, `reset_for_tests` isolation.
- **Shared helper tests:** the nine consolidated sites produce identical
  histograms as before plus a ledger record; estimate path when usage
  absent.
- **Config defaults** (`tomllib.loads(CONFIG_TOML_CONTENT)` pattern,
  `Tests/test_config_model_catalog_defaults.py`): section present, values,
  clamp behavior.
- **Dialog pilot tests** (`Tests/UI/`, harness per
  `test_prompt_delete_confirmation_modal.py`): copy visible via
  `export_screenshot` SVG assertions (render evidence per lessons), any-key
  skip dismisses, auto-dismiss **fires** (assert the close callback ran —
  timer-fire evidence, not timer-armed), short duration param.
- **Quit-path harness tests** (bind real methods without full app mount,
  per `Tests/UI/test_app_quit_guard.py`): enabled path pushes dialog then
  exits; disabled path never pushes; hard-cap test with a stub screen that
  never dismisses still exits; `CancelledError` preserved as durable
  failure (quit-path lesson).
- Follow repo guidance: targeted runs only; no full sweep unless requested.

## ADR Check

```text
ADR required: no
ADR path: N/A
Reason: additive feature — ephemeral process-lifetime counters, no schema,
sync, security, or persistence change; no existing interface is altered.
The one policy decision future contributors must honor (the boundary tap
rule: record once at response parse, downstream re-reads are not taps) is
recorded in this spec's Tap Points section and will be cited in the task.
```

## Alternatives Considered

- **Recompute at quit from conversation cost snapshots** — rejected:
  conversation totals include pre-session messages, several surfaces keep
  no records, and it recomputes during the shutdown window.
- **Textual event bus (revive `chat_token_events.py`-style App messages)**
  — rejected: provider calls run in threads/workers; every site would need
  `call_from_thread` plumbing for data a plain thread-safe object carries
  fine.

## References

- Issue #365 (source of ACs; this design satisfies each).
- ADR-031 (keybinding/footer-hint conventions), ADR-150 (design tokens).
- `backlog/docs/lessons-textual.md` (set_timer(0.0); GC'd timers;
  exclusive workers cancel), `lessons-testing-evidence.md` (timer-fire and
  screenshot evidence; CancelledError durability; mount production
  hierarchy).
- Patterns: `Widgets/confirmation_dialog.py`, `Widgets/splash_screen.py`,
  settings `[model_catalog]` group, `Tests/UI/test_app_quit_guard.py`.
