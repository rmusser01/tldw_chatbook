# Console `/rewind` Menu — Design (SP2)

**Date:** 2026-07-24
**Status:** Approved decisions (user, 2026-07-24): **D1 = conversation-field boundary summary (A)**, **D2 = v1 ships Restore + Summarize-up-to-here**, **D3 = session's provider via the Console gateway + editable Internal_Prompts entry**. Pending final spec read-through before planning.
**Program:** "Console `/rewind`" — SP1 branching foundation COMPLETE (Phase A #799, Phase B #811 merged; Phase C #827 at gate). This spec is SP2, the user-facing payoff.

## Why

Claude Code's `/rewind` lets a user open a menu of their prior prompts and either **restore** the conversation to that point or **summarize** part of it to free context. The Console now has the branching foundation that makes restore *non-destructive*: "restore to here" is pure tree navigation (the abandoned tail stays reachable by swiping), not deletion. SP2 builds the menu and its two action families.

## Grounding (research 2026-07-24, worktree base dev b19317f5e)

- **Command plumbing is templated.** `console_command_grammar.py` registry (4 commands today: `/prompt`, `/system`, `/skills`, `/prefill`); parse at the send choke point *before* readiness gating (menu opens even when Send is blocked); dispatch via `_CONSOLE_COMMAND_NAME_TO_HANDLER_ID` + `dispatch_map` in `chat_screen.py:11120-11165`. `/prompt` is the end-to-end template (command → picker modal → composer mutation).
- **Modal template:** `ConsoleSessionSwitcherModal` — result is a tagged union `(kind, entry)`, exactly the "which action + which prompt" shape rewind needs; borrow `ConsolePromptPickerModal`'s non-focusable-rows + synthetic-highlight keyboard discipline.
- **Restore is pure navigation.** Active path USER prompts = `messages_for_session(sid)` filtered to `role is USER`. For prompt `U` at path index `i`: `P = active_path_message_ids(sid)[i-1]` (or `None` when `U` is the root — `set_active_leaf` explicitly accepts `None` = empty transcript). Restore = `store.set_active_leaf(sid, P)` + put `U.content` into the composer. **Composer seam is `_insert_prompt_text_into_composer(text, replace=True)`** (the `/prompt` path; paste-collapse for long prompts) — NOT `one_shot_prefill`, which is the *assistant-reply* prefill and never touches the composer.
- **Esc-Esc is rejected.** Single-Escape already has three consumers (transcript selection-clear, modal dismiss, collapsed-composer expand + composer re-home with deliberate priority layering). Trigger = the `/rewind` slash command; a dedicated keybinding (like the switcher's `ctrl+k`) can follow.
- **Summaries need a payload story.** `_provider_message_payloads` hard-filters transcript rows to USER/ASSISTANT — a summary stored as a SYSTEM *node* would display but never reach the provider. The token-budget trimmer (`bound_messages_to_window`, TASK-322) preserves the leading contiguous *system prefix* and drops oldest whole turns. Existing summarization machinery (`Summarization_General_Lib.analyze` + per-provider workers) exists but nothing in `Chat/` uses it; conversation compaction is net-new orchestration.

## Open decisions (user gate)

- **D1 — Summary storage model.** Recommended: **(A) conversation-field boundary summary.** "Summarize up to here" runs an LLM summary of the active path *before* the selected prompt, stores it on the conversation (new local metadata: `context_summary` text + `summary_boundary_message_id`), and the payload builder replaces pre-boundary turns with the summary injected into the **system prefix** (which the trimmer already preserves). The visible transcript is UNCHANGED — full history stays readable, only the provider context is compacted (arguably better than Claude Code's replace-with-marker). Non-destructive, no tree surgery, resume-safe, composes with the trimmer. Alternative **(B) summary-node fork**: true Claude-Code parity by forking a branch where the span is replaced by a summary node — heavier (tree copying/re-parenting), deferred unless you want it.
- **D2 — SP2-v1 action set.** Recommended: **Restore to here** + **Summarize up to here** (the "free context" workhorse). Claude Code's third action, *Summarize from here* (compress the recent side-quest, keep early detail), maps awkwardly onto model (A) — proposed as a follow-up (it is ≈ restore-to-here + auto-summarize-the-abandoned-tail-into-the-boundary-summary, which we can add once (A) is proven).
- **D3 — Summarization provider.** Recommended: the session's own resolved provider via the Console's provider gateway (respects readiness, local llama.cpp, etc.) with a dedicated internal prompt (new `Internal_Prompts` entry, editable in Settings per the internal-prompts program) — NOT `Summarization_General_Lib`'s parallel provider matrix (bypasses Console resolution). Non-streaming, worker-off-thread, with a visible "Summarizing…" state and failure = no-op + notify.

## Design

### Trigger & menu

`/rewind` (grammar constants + registration; `argument_hint` `""`). Handler collects the active path's USER prompts (index, preview of content, native id) and pushes `ConsoleRewindModal` (modeled on the session switcher): a filterable list of prompts, newest first, each row showing `#n` + a single-line preview. Selecting a row offers the action choice (second-level kinds on the same modal, switcher-style): **Restore to here**, **Summarize up to here**, **Never mind**. Result type: frozen `ConsoleRewindChoice(kind: str, message_id: str, prompt_text: str)` (or `None`).

### Restore to here

Callback: compute `P` by **id lookup** — `path = active_path_message_ids(sid)`; `i = path.index(U.id)`; `P = path[i-1]` or `None` when `i == 0`. (Never positional correspondence with the `messages_for_session` view: display-only TOOL markers pad the view but are absent from the tree walk, so view positions and path positions diverge in agent conversations — spec-review fix.) Then `store.set_active_leaf(sid, P)`; `_insert_prompt_text_into_composer(prompt_text, replace=True)` (draft-clobber matches `/prompt`'s established `replace=True` precedent); focus composer; `_sync_native_console_chat_ui()`. The dropped tail remains reachable (sibling swipe / future tree browser); the durable pointer persists the choice across restarts (SP1 Task 8 machinery). Blocked while a run is streaming (same `is_send_allowed` gate as regenerate).

**Known v1 limitation (spec-review):** restoring to before the FIRST prompt (`set_active_leaf(None)`) yields an empty transcript that does **not** survive an app restart — the persisted pointer becomes NULL, and resume's fallback treats NULL as "unset → most-recent branch." One nullable column cannot distinguish "explicitly empty" from "never set." Benign (nothing is lost; the refilled composer draft is the working artifact and the send path stashes drafts), documented rather than engineered around in v1.

### Summarize up to here (model A, per D1)

1. Span = active-path messages BEFORE the selected prompt (from the previous boundary when re-summarizing — rolling compaction; the prompt includes the prior summary).
2. Off-thread worker: build a plain-text transcript of the span; call the session's resolved provider (non-streaming) with the new internal summarization prompt; visible run-state copy "Summarizing conversation…" (exclusive `console-run` group — no interleaving with sends/regenerates/restores).
3. On success: store `context_summary` + `summary_boundary_message_id` as **local-only conversation metadata** (same local-column pattern as `active_leaf_message_id`: plain nullable columns, bare-UPDATE setter, no sync, no trigger redefinition — ChaChaNotes migration v24→v25, re-verified against dev at implementation time).
4. **Transcript affordance is render-derived, NEVER a transcript node** (spec-review fix): the transcript renderer shows a banner above the boundary message ("⤵ N earlier turns summarized for context — full history still visible above") whenever `summary_boundary_message_id` is set and on the active path — exactly the transient-render pattern the sibling `n/m` counters use. Rationale: a SYSTEM "marker row" would be a REAL tree node (only TOOL rows are display-only) — it would land at the transcript TAIL (append advances the active leaf), become the next send's parent, and, being `persist=False`, vanish on resume while the summary itself persists. A render-derived banner sits in the right place, survives resume for free, and touches no tree state.
5. **Payload compaction rule (spec-review fix, prevents a future-information leak):** compaction applies **only when the boundary message is present in the payload being built**. When present: drop the payload rows before the boundary and append the summary to the leading system message (system prefix → preserved by `bound_messages_to_window`). When absent — e.g. regenerating a message that sits BEFORE the boundary, whose ancestors-only payload ends pre-boundary — send full history untouched; a summary covering later turns must never be substituted into an earlier point's context. Hook the compaction at the single dispatch choke point (where `bound_messages_to_window` already runs) so agent and direct sends compact uniformly.
6. Interactions: restore to a point at-or-before the boundary leaves the stored summary in place but inert (boundary no longer in payloads → rule 5 sends full history; validated like the active-leaf pointer, fail-open to full history when dangling); a later re-summarize replaces it. Branch switches: the summary applies only while the boundary is on the active path.

### Out of scope (follow-ups)

Summarize *from* here (D2); a dedicated keybinding; Esc-Esc (rejected); summary-node fork model (D1-B); surfacing the dropped tail in a tree browser (SP1 follow-up C).

## Testing sketch

Grammar/dispatch unit tests (template: `/prefill`'s); modal tests (switcher-style: rows, keyboard nav, tagged dismissal); restore tests (mid-path → active path truncates + composer filled + pointer persisted + tail swipe-reachable; first-prompt → `set_active_leaf(None)` empty transcript; streaming-blocked); summarize tests (fake provider: boundary+summary stored locally, payload replaces pre-boundary turns with system-prefix summary, trimmer preserves it, dangling boundary falls back to full history, re-summarize rolls); e2e: converse → rewind-restore → re-send edited prompt (forks — SP1) → summarize-up-to-here → persist/drop/resume → payload still compacted, transcript full.

## Risks

- Migration collision (v24→v25): re-verify `_CURRENT_SCHEMA_VERSION` on latest dev at merge time (three collisions so far in this program).
- Summary quality/size: cap summary tokens (reuse `count_console_messages_tokens`); a bad summary is recoverable (clear-summary action or re-summarize; full history never destroyed).
- Concurrency: summarize is a run (exclusive `console-run` group) — no interleaving with sends/regenerates.
