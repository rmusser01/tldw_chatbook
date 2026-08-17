---
id: TASK-17167
title: Console side chat phase 2
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-15 17:27'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ephemeral side chat modal for selected transcript text (More Details / Ask in Side Chat)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 More Details auto-sends rendered template
- [x] #2 Ask in Side Chat freeform
- [x] #3 sidechat model/template settings saved via Console Behavior
- [x] #4 Streaming with stop/retry
- [x] #5 Nothing persisted
- [x] #6 Tests green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Per Docs/superpowers/plans/2026-08-15-console-side-chat-phase2.md (tasks 1-6: config keys → settings surface → headless service → modal → menu/transcript/screen wiring → wrap-up)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADR: backlog/decisions/068-console-text-selection-and-annotations.md (phase-2 consequence line added). Plan: Docs/superpowers/plans/2026-08-15-console-side-chat-phase2.md. Spec: Docs/superpowers/specs/2026-08-14-console-selection-annotations-design.md §2/§4/§5/§7.

Phase 2 implemented across commits f87a111c9..41eb62744 (+ this wrap-up) on feat/console-side-chat:

- **Config keys** (`tldw_chatbook/config.py`): `[console] sidechat_model` (default `""` = fall back to the session's provider selection) and `[console] sidechat_prompt_template` (default `"Give me more details about: {selection}"`), template block + loader string coercion. Decision: model uses qualified `provider/model` syntax (e.g. `openai/gpt-4o`) parsed at send time (a bare `model` keeps the session provider, an empty value reuses the session selection as-is); an EMPTY template falls back to the default template (never sends a blank prompt).
- **Settings surface** (`tldw_chatbook/UI/Screens/settings_screen.py`): both keys in the canonical Console Behavior category — model `Input` (placeholder "empty = current session model"), template `Input` with help text mentioning `{selection}`; staged-edit/save/revert rides the existing `_save_console_behavior_values` trail (no runtime mutation before save).
- **Headless service** (`tldw_chatbook/Chat/console_side_chat.py`): `ConsoleSideChatService` over `ConsoleProviderGateway.stream_chat` only — persistence-free by construction (the gateway's own contract bypasses Console history and the chat store; no store imports). `render_prompt` substitutes `{selection}` via `str.replace` (never `str.format`), missing placeholder appends the selection on a new line, other braces stay literal, empty template → default. `SideChatOutcome` status is `complete | cancelled | provider_error`; reply buffer tail-capped at `SIDE_CHAT_BUFFER_CAP = 100_000`. Decision: a gateway-blocked resolution surfaces as `status="provider_error"` with safe copy (never raises through the modal boundary).
- **Modal** (`tldw_chatbook/Widgets/Console/console_side_chat_modal.py`): `ConsoleSideChatModal(SafeModalDismissMixin, ModalScreen[None])` — More Details auto-sends the rendered prompt on mount; Ask mode shows the quote read-only with a freeform prompt + Send. Streaming reply updates live; Stop cancels (shows "Cancelling…", then the cancelled outcome with Retry); provider errors render inline with Retry; Escape/backdrop cancels the worker and dismisses (task-16211 contract). Async worker runs `exclusive=False, group="console-side-chat"` so it never cancels or blocks `console-run-{session_id}` session workers; monotonic request-id stale guard; `on_unmount` cancels. Added to the dismissal inventory (reachable 38).
- **Menu + wiring** (`console_selection_menu.py` / `console_transcript.py` / `chat_screen.py`): menu trio Add to chat / More Details / Ask in Side Chat; transcript `MoreDetails`/`AskInSideChat` handlers post `ConsoleSideChatRequested(quote=cap_quote(...), mode=...)` with the same cleanup as Add to chat; ChatScreen handler resolves config + gateway, renders the template for more-details, pushes exactly one modal. Decision (T5 review): a whitespace-only quote (row range cleared while the menu was open) no-ops exactly like the add-to-chat guard — no modal pushed, no send, no toast.

**Tests** (all green): `Tests/test_config_console_defaults.py -k sidechat` (5), `Tests/UI/test_settings_console_side_chat.py`, `Tests/Chat/test_console_side_chat_service.py` (23), `Tests/UI/test_console_side_chat_modal.py`, `Tests/UI/test_console_modal_dismissal.py` (inventory + AST completeness), plus the full selection suites `_core/_rows/_transcript/_menu/_end_to_end/_app_smoke` (87) incl. the new empty-quote-guard test and drag→More Details/Ask end-to-end paths.

**Baselines (pre-existing, unchanged)**: native transcript 3 / chat-flow 1 / markdown-widget 4 — verified after the wrap-up guard; failing tests (speak action-rows, inline image row, markdown flavor) are untouched by the branch.

**Ruff**: `uvx ruff check` on every file this phase touched; all remaining findings intersected against branch-added line ranges (base 08513b5da) are pre-existing main-branch findings — the 3 branch-owned ones (UP035/I001/F401) were fixed in this wrap-up.

**Live-terminal-only verification outstanding** (same as phase 1, per `backlog/docs/lessons-live-verification.md`): real-provider streaming feel, Stop/Retry timing against a live stream, and modal dismissal feel in a real terminal. Task stays In Progress pending that live spike.
<!-- SECTION:NOTES:END -->

## Live spike result

Live spike PASSED 2026-08-15 with phase 1 (same session; menu actions incl. side-chat entries verified).
