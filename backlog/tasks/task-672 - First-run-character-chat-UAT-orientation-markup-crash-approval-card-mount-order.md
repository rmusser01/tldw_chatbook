---
id: TASK-672
title: >-
  First-run character-chat UAT orientation markup crash and approval-card
  mount order
status: Done
assignee: []
created_date: '2026-07-25 16:13'
updated_date: '2026-07-25 21:54'
labels:
  - testing
  - ui
  - bug
dependencies: []
priority: high
---

# task-672 - First-run character-chat UAT: orientation markup crash + approval-card mount order

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

A headless UAT of the first-time user journey (boot → import a SillyTavern
character card → Start Chat → send a message) was needed to prove the
character-card import fixes (task-100, task-109) hold up in the real app flow.
The UAT surfaced two crash-class bugs that hit first-time users specifically:

1. The Console first-run empty state (`#chat-empty-state`) crashed with
   `textual.markup.MarkupError` whenever the selected provider was not ready
   (missing API key or unknown provider) — i.e. exactly the first-run state.
   `ProviderReadiness.user_message` embeds the literal TOML path
   `[api_settings.<provider>]`, and `Static.update()` parsed the brackets as a
   Rich style tag. Two existing tests in
   `Tests/UI/test_chat_approvals_and_resume.py` were red on `dev` from this.
2. `ChatApprovalCard.on_mount` queried `#approval-batch-body` before its
   composed children attached, raising `NoMatches` on Console mount — a latent
   crash on any approval-card mount.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] A first-time user can import a SillyTavern PNG character card, start a
  chat, send a message, and see the reply, with the conversation persisted —
  verified end-to-end by an automated UAT against the real app object.
- [x] With no provider configured, Start Chat is blocked gracefully with an
  actionable explanation (disabled button + tooltip, no crash, no dead end).
- [x] The first-run orientation empty state renders without MarkupError for a
  missing-API-key provider, an unknown provider, and a ready provider, and the
  literal `[api_settings.<provider>]` TOML path remains visible to the user.
- [x] `ChatApprovalCard` mounts without `NoMatches` regardless of compose
  timing.
- [x] Previously failing `test_chat_approvals_and_resume.py` tests pass.

<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->

ADR required: no
ADR path: N/A
Reason: routine bug fixes (markup escaping at a render boundary; widget mount
ordering) plus test-only additions. No storage, sync, security, or
cross-module contract changes; the user-facing copy is deliberately unchanged.

1. Capture the first-time journey as
   `Tests/UI/test_uat_first_time_character_chat.py` (headless Pilot against
   the real `TldwCli`, isolated temp ChaChaNotes DB, real card file; only the
   provider network call mocked).
2. Escape `readiness.user_message` with `rich.markup.escape` in
   `ChatWindowEnhanced.build_first_run_orientation_text` (audit all other
   `user_message`/`recovery` consumers for markup exposure first).
3. Defer `ChatApprovalCard`'s batch-body query to `call_after_refresh` with a
   `NoMatches` guard.
4. Add focused markup regression tests; confirm the two red
   approval-and-resume tests go green.

<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

- **Markup crash fix** (`tldw_chatbook/UI/Chat_Window_Enhanced.py`): the
  orientation text is the only markup-rendering consumer of
  `ProviderReadiness.user_message`. The Console run-inspector rows,
  staged-context tray, and native transcript were audited and are all
  `markup=False` or tuple-styled; the legacy chat window mounts the message in
  a `Markdown` widget (no Rich markup parse). The bracketed TOML copy is
  intentional (asserted by `Tests/Chat/test_provider_readiness.py` and
  `Tests/test_config_console_defaults.py`), so the fix escapes at the render
  boundary rather than changing copy.
- **Approval card fix** (`tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`):
  `on_mount` now hides via `display = False` and defers the
  `#approval-batch-body` query to `call_after_refresh` with a `NoMatches`
  guard, so compose timing can no longer crash the mount.
- **UAT harness**: `Tests/UI/test_uat_first_time_character_chat.py` runs the
  full journey (boot → Personas → import `Ann1.png` → provider-gated Start
  Chat → configure provider → handoff → send → mocked reply → persistence).
  Two harness-only lessons encoded in the test: the handoff consumption runs
  off a mount timer so the test waits for `screen.is_mounted`, and the Console
  composer treats its paste-aware segments as canonical, so drafts are loaded
  via `composer.load_draft()` rather than setting the hidden `Input.value`.
- **Regression tests**: `Tests/UI/test_first_run_orientation_markup.py`
  (missing key / unknown provider / ready, each pushed through
  `Content.from_markup`); the UAT itself covers import gating and send.
- Modified: `tldw_chatbook/UI/Chat_Window_Enhanced.py`,
  `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`.
- Added: `Tests/UI/test_uat_first_time_character_chat.py`,
  `Tests/UI/test_first_run_orientation_markup.py`.
- Note: this branch stacks on `fix/character-card-trailing-text-chunks`
  (task-109) because the UAT imports a post-IDAT PNG card; the diff against
  `dev` shrinks automatically once that PR merges.
- **Qodo hardening round (PR #896)**: all five review findings addressed in
  the UAT/markup tests — (1) realistic-looking API keys replaced with
  clearly-fake placeholder constants; (2) Google-style docstrings on all new
  public callables; (3) the machine-specific `E:\...Ann1.png` dependency
  removed — the UAT now generates its SillyTavern-layout post-IDAT PNG via
  the shared chunk-surgery helpers from
  `test_character_card_lenient_import.py`, with an optional
  `TLDW_UAT_CARD_PATH` env override that fails loudly when set-but-missing
  (never a silent skip); (4) provider-call patching switched to
  `monkeypatch.setattr` on the real seam (`Chat_Functions.chat_api_call`,
  which the gateway resolves lazily per call) and the tautological
  `... or True` handoff assertion replaced with real payload-field checks
  (`intent`/`selected_kind`/`selected_record_id`) captured before Console
  consumption; (5) the persistence check now diffs
  `db.get_all_conversation_ids()` before/after the send and asserts the new
  conversation holds ≥2 messages (verified: greeting + user + assistant = 3),
  replacing the `conversations or provider_calls` tautology and the
  nonexistent `list_conversations` probe.
- **Rebase onto dev after task-577 retirement**: while this branch was in
  review, dev retired the entire `ChatWindowEnhanced` family as dead code
  (commit `0659f103b`, "never instantiated since 8ea71071f") and removed
  `Tests/UI/test_chat_approvals_and_resume.py` with it. The rebase therefore
  dropped the `Chat_Window_Enhanced.py` markup escape as moot (the crash it
  fixed only ever manifested in those now-removed tests, never in live code)
  and removed `Tests/UI/test_first_run_orientation_markup.py` (its subject
  module no longer exists). What survives on the rebased branch: the
  `ChatApprovalCard` mount-order fix (live Console code), the first-run
  character-chat UAT, and the stacked task-109 character-card import fixes.

<!-- SECTION:NOTES:END -->
