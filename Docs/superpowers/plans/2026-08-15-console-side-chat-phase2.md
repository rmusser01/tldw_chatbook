# Console Side Chat — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ephemeral side-chat modal answering questions about selected transcript text — `More Details` (auto-send template) and `Ask in Side Chat` (freeform) from the selection menu — using a user-configurable model + prompt template, streaming, nothing persisted.

**Architecture:** A headless service (`Chat/console_side_chat.py`) resolves the side-chat model (falling back to the session model) and streams via `ConsoleProviderGateway.stream_chat` — the same persistence-free seam the prompt-improvement flow uses (proven precedent: it bypasses Console history and persistence entirely). `ConsoleSideChatModal` (ModalScreen + `SafeModalDismissMixin`) owns the ephemeral message list, runs a non-exclusive worker in group `console-side-chat`, and never touches the chat store. Menu/transcript handlers reuse the exact Phase-1 quote plumbing (`cap_quote` → message → screen handler → push modal).

**Tech Stack:** Textual 8.2.8 ModalScreen + workers; existing `ConsoleProviderGateway`; pytest with fake gateway via `console_provider_gateway_factory`.

**Spec:** `Docs/superpowers/specs/2026-08-14-console-selection-annotations-design.md` §2 (Menu Actions, Settings, Side-chat execution isolation), §4 (error handling), §5 (testing), §7 phase 2.

```text
ADR required: no
ADR path: N/A (implements existing ADR-068 + spec §2; no schema, no new cross-module interface beyond the side-chat service seam, which follows the prompt-improvement precedent)
Reason: direct implementation of the approved design; ADR-068 already records the system decision.
```

## Global Constraints

- Side chat is ephemeral: NO chat-store session, NO persistence, NO transcript interaction — modal-local message list only; `gateway.stream_chat`/`complete_auxiliary` are the only provider seams (their docstrings guarantee persistence-free).
- Worker: `run_worker(..., exclusive=False, group="console-side-chat")` — must never cancel or block `console-run-{session_id}` workers (spec §2 isolation).
- Model resolution: `[console] sidechat_model` when set; else fall back to the current session's `turn_context.provider_selection`. Template default: `"Give me more details about: {selection}"`; `{selection}` is the only placeholder.
- Selection quote capped by `cap_quote` (4000) before entering the modal; modal reply buffer keeps the tail, capped at `SIDE_CHAT_BUFFER_CAP = 100_000` chars.
- Failures surface inline in the modal with Retry; modal always escapable (task-16211 contract via `SafeModalDismissMixin`); cancellation shows "Cancelling…" then resolves.
- The new modal MUST be added to the dismissal inventory in `Tests/UI/test_console_modal_dismissal.py` (AST-based completeness test at ~line 1005 enforces it).
- No API keys logged; provider errors go through `safe_provider_error_copy` semantics (the gateway already does this).
- Baselines on this branch: `test_console_native_transcript.py` 3 pre-existing failures, `test_console_native_chat_flow.py` 1, markdown-widget 4 — no new failures allowed.

---

### Task 1: Config keys

**Files:**
- Modify: `tldw_chatbook/config.py` (template block ~2685-2701, loader coercion block ~1278-1327)
- Test: `Tests/test_config_console_defaults.py` (extend)

**Interfaces:**
- Produces: config keys `[console] sidechat_model` (str, default `""` = session model fallback) and `[console] sidechat_prompt_template` (str, default `"Give me more details about: {selection}"`), readable via `get_cli_setting("console", "sidechat_model", "")` etc.

- [ ] **Step 1: Write failing tests** — default values present in `CONFIG_TOML_CONTENT` template, coercion round-trips through the loader (string keys survive as strings; missing keys fall back to defaults). Mirror the existing `stack_collapsed_rail_labels` tests at the top of the file (~lines 79-136) but for string coercion.

```python
def test_console_sidechat_model_default_is_empty_string():
    from tldw_chatbook.config import get_cli_setting
    assert get_cli_setting("console", "sidechat_model", "") == ""

def test_console_sidechat_prompt_template_default():
    from tldw_chatbook.config import get_cli_setting
    assert (
        get_cli_setting("console", "sidechat_prompt_template", "")
        == "Give me more details about: {selection}"
    )

def test_console_sidechat_keys_survive_loader_coercion(tmp_path, monkeypatch):
    # point the config dir at tmp_path, write a config.toml with custom values,
    # load, assert both come back as the custom strings (follow the existing
    # loader tests' tmp_path/monkeypatch pattern in this file)
    ...
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/pytest Tests/test_config_console_defaults.py -k sidechat -v` → FAIL (keys absent).
- [ ] **Step 3: Implement** — add both keys to the `[console]` template block with a comment (`# Ephemeral side chat (selection menu) — empty model = session model`), and string coercion entries in the loader block (follow neighboring `coerce_*` lines; strings need presence-validation only).
- [ ] **Step 4: Verify pass** + full file green.
- [ ] **Step 5: Commit** — `feat(console): sidechat_model and sidechat_prompt_template config keys`

### Task 2: Settings surface

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (Console Behavior category)
- Test: `Tests/UI/test_settings_console_side_chat.py` (new, modeled on `Tests/UI/test_settings_console_rail_labels.py`)

**Interfaces:**
- Consumes: Task 1 keys.
- Produces: both settings visible/editable/savable in the canonical settings screen under Console Behavior.

- [ ] **Step 1: Write failing tests** (mirror `test_settings_console_rail_labels.py` structure): model + template render with loaded values; staging a change doesn't mutate runtime config until save; exact save payload via monkeypatched `SettingsConfigAdapter` (assert `console.sidechat_model` / `console.sidechat_prompt_template` in sections); failed save keeps draft; revert restores.
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement** — follow the `stack_collapsed_rail_labels` trail exactly: add both keys to `CONSOLE_BEHAVIOR_CONSOLE_KEYS` (~626) and `CONSOLE_BEHAVIOR_SAVE_ORDER` (~695); ownership records (~3663); compose widgets — `Input` for the model (placeholder "empty = current session model") and `Input` (or TextArea if multi-line is idiomatic) for the template with help Static mentioning `{selection}` (~12354); loaded-value getter + map (~3825/~3983); staging via `_stage_console_default_value` (~6120); change handler (~17995); focused-field inspector copy (~6277); save rides `_save_console_behavior_values` (~20996).
- [ ] **Step 4: Verify pass** + `test_settings_console_rail_labels.py` still green (no category regressions).
- [ ] **Step 5: Commit** — `feat(console): side chat settings in Console Behavior category`

### Task 3: Side-chat service (headless)

**Files:**
- Create: `tldw_chatbook/Chat/console_side_chat.py`
- Test: `Tests/Chat/test_console_side_chat_service.py`

**Interfaces:**
- Consumes: `ConsoleProviderGateway` (`resolve_for_send(selection) -> resolution`, `stream_chat(resolution, messages) -> async iterator of chunks`), `ConsoleProviderSelection` (`provider`, `explicit_model`, `streaming`), `cap_quote`.
- Produces:

```python
DEFAULT_SIDE_CHAT_PROMPT_TEMPLATE = "Give me more details about: {selection}"
SIDE_CHAT_BUFFER_CAP = 100_000

@dataclass(frozen=True)
class SideChatOutcome:
    text: str            # joined reply, tail-capped at SIDE_CHAT_BUFFER_CAP
    provider: str
    model: str
    status: str          # "complete" | "cancelled" | "provider_error"
    error: str = ""      # safe copy only, when status == "provider_error"

class ConsoleSideChatService:
    def __init__(self, gateway) -> None: ...          # gateway: ConsoleProviderGatewayProtocol-shaped
    def render_prompt(self, template: str, selection: str) -> str:
        # template.format-safe: only {selection} substituted; missing placeholder appends
        # the selection on a new line; other placeholders left literal (use replace, not str.format)
    async def run(
        self,
        *,
        selection_quote: str,          # already cap_quote()d by the caller
        prompt: str,                   # fully-rendered user prompt
        provider_selection: ConsoleProviderSelection | None,  # session fallback when config model empty
        sidechat_model: str = "",      # raw config value
    ) -> AsyncIterator[SideChatEvent]: # "chunk" (delta str) / "done" (SideChatOutcome) events
```

(The exact event shape may be `async for delta, done_outcome` or a small event dataclass — implementer's choice, but cancellation must surface as `status="cancelled"`, provider errors as `status="provider_error"` with safe copy, never raise through the modal boundary except `asyncio.CancelledError` semantics.)

- [ ] **Step 1: Write failing tests** with a FakeGateway (constructor-injected): chunked stream joins to full text; provider error → provider_error outcome with safe copy and no raise; cancellation (cancel the consuming task mid-stream) → cancelled outcome; `render_prompt`: `{selection}` substituted, missing placeholder appends selection, literal other braces untouched, empty template → default template; buffer tail-cap truncates a huge reply; model resolution: `sidechat_model="x"` wins over session selection; empty → session selection used; explicit streaming selection built with `explicit_model` set.
- [ ] **Step 2: Verify fail** — module not found.
- [ ] **Step 3: Implement** — mirror `prompt_improvement_service.py`'s shape (validation → single provider call → typed outcome; no store imports). Model resolution mirrors `prompts.py:482-489` identity pattern: build `ConsoleProviderSelection(provider=..., explicit_model=sidechat_model or None, streaming=True)`.
- [ ] **Step 4: Verify pass.**
- [ ] **Step 5: Commit** — `feat(console): headless side-chat service over provider gateway`

### Task 4: ConsoleSideChatModal

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_side_chat_modal.py`
- Modify: `Tests/UI/test_console_modal_dismissal.py` (inventory + contract rows)
- Test: `Tests/UI/test_console_side_chat_modal.py` (new)

**Interfaces:**
- Consumes: `ConsoleSideChatService.run` (Task 3), `SafeModalDismissMixin` (`Widgets/modal_dismissal.py`), prompt-improvement choreography (`console_prompts_modal.py:1050-1157, 1695-1703`).
- Produces:

```python
class ConsoleSideChatModal(SafeModalDismissMixin, ModalScreen[None]):
    def __init__(self, *, service, provider_selection, sidechat_model: str,
                 quote: str, auto_send_prompt: str | None) -> None: ...
    # auto_send_prompt: rendered prompt for More Details (service started on mount)
    # None → Ask in Side Chat mode: quote shown read-only, prompt TextArea empty, Send button
    # API: push from ChatScreen; dismiss(result=None); Escape/backdrop ≡ Cancel per mixin
```

- [ ] **Step 1: Write failing tests**: More Details mode auto-sends on mount and streams chunks into the reply area (fake service/gateway); Ask mode waits for Send with typed prompt; Stop cancels (status "Cancelling…", then cancelled outcome shown, Retry visible); provider error shows inline + Retry works; Escape mid-stream cancels the worker and dismisses; reply buffer capped; quote displayed read-only; provider·model summary line shows resolved identity. All with a fake service injected — no live LLM.
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement** — skeleton from `ConsoleEditMessageModal` (`SAFE_MODAL_CONTENT = "#console-side-chat-modal"`, `escape → request_safe_cancel`); worker `run_worker(self._run(...), exclusive=False, group="console-side-chat")` with monotonic request-id stale guard; Cancel/Retry/Close buttons per prompts-modal pattern; `on_unmount` cancels worker; reply `Static` updated via `call_from_thread`-safe pattern used by prompts modal (worker is a thread worker? NO — `stream_chat` is async, use an async worker; follow `console_prompts_modal`'s async worker + `post_message`/direct-update pattern). Add inventory + contract rows to `test_console_modal_dismissal.py` (import + `_Task2ModalContract` entry; the AST test enforces completeness).
- [ ] **Step 4: Verify pass** incl. the full dismissal suite.
- [ ] **Step 5: Commit** — `feat(console): ephemeral side-chat modal with streaming, cancel, retry`

### Task 5: Menu entries + transcript + screen wiring

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_selection_menu.py` (messages + buttons + handlers)
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (two `@on` handlers mirroring `_selection_add_to_chat` ~3501-3525)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (handler beside `_console_selection_quote_requested` ~19788)
- Test: extend `Tests/UI/test_console_selection_menu.py`, `Tests/UI/test_console_selection_end_to_end.py`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: `ConsoleSelectionMenu.MoreDetails` / `.AskInSideChat` messages; `ConsoleSideChatRequested(quote: str, mode: str)` message from transcript ("more-details" | "ask"); ChatScreen handler resolves config + gateway (`_ensure_console_provider_gateway` ~5239), renders the template for more-details mode, and pushes `ConsoleSideChatModal`.

- [ ] **Step 1: Write failing tests**: menu shows all three stacked options in order (Add to chat / More Details / Ask in Side Chat); pressing More Details clears selection+menu and the screen receives `ConsoleSideChatRequested(mode="more-details")` with capped quote (app-level capture harness); Ask mode likewise with `mode="ask"`; screen handler pushes exactly one modal with rendered template for more-details (fake service seam) and empty-prompt ask mode; end-to-end: drag → More Details → modal mounted with the quote visible.
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement** per interfaces; template rendering lives in the screen handler via the service's `render_prompt`; provider fallback from the active session's `turn_context.provider_selection`.
- [ ] **Step 4: Verify pass** + all selection suites + baselines (3/1/4) unchanged.
- [ ] **Step 5: Commit** — `feat(console): More Details and Ask in Side Chat from selection menu`

### Task 6: Wrap-up

**Files:**
- Modify: backlog task (create via CLI: "Console side chat phase 2"), ADR-068 (one-line consequence noting phase 2 landed), `Docs/superpowers/specs/2026-08-14-console-selection-annotations-design.md` (unchanged — spec already covers).

- [ ] **Step 1:** Full selection + side-chat + dismissal + settings suites green; `uvx ruff check` on branch-owned files; baselines 3/1/4 unchanged.
- [ ] **Step 2:** Backlog task with honest ACs checked + Implementation Notes; ADR-068 consequence line; do NOT mark Done (live spike pending like phase 1).
- [ ] **Step 3:** Commit — `feat(console): side chat wrap-up docs`.

## Self-Review Notes

- Spec §2 coverage: entries+modal (T4/T5), auto-vs-freeform (T5), settings (T1/T2), isolation (T3 gateway-only + non-exclusive worker group, enforced in T3/T4 tests), cap (T1 quote cap reuse + T3 buffer cap), error handling inline+retry (T3/T4), nothing persisted (T3/T4 tests assert no store interaction).
- The modal-dismissal inventory constraint (would otherwise fail an AST test) is explicitly in T4.
- Deviations allowed with documentation: exact modal layout, event-shape details, whether the settings widget is Input vs TextArea.
