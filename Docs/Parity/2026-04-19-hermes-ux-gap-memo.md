# tldw_chatbook Hermes UX Gap Memo

## Purpose

Capture only the `hermes-agent` interaction patterns that materially improve `tldw_chatbook`. This memo is an overlay, not the primary parity target.

## Tool UX

- Hermes prioritizes concise tool previews, diff-friendly summaries, and truncated long output instead of flooding the session (`agent/display.py`, `RELEASE_v0.8.0.md`).
- `tldw_chatbook` already has worker-based execution and notifications (`UI/Chat_Window_Enhanced.py`, `app.py` worker handling), but tool and job feedback is still fragmented and mostly notification-driven.
- The transferable improvement is not a CLI clone. It is a Textual-native activity surface that shows active task name, status, last result summary, and expandable logs for long-running or tool-heavy actions.

## Session UX

- Hermes has explicit session lifecycle affordances: browse, rename, export, prune, resume, and clearer session identity across runs (`hermes_cli/main.py`, `RELEASE_v0.8.0.md`).
- `tldw_chatbook` preserves chat screen state and supports multi-session chat behavior, but local history, exported artifacts, and cross-surface continuation are not yet one cohesive interoperability model.
- The useful carryover is stronger local session metadata discipline: stable IDs, timestamps, titles, and export hooks that can later map cleanly onto `tldw_server` entities.

## Safety And Approvals

- Hermes centralizes dangerous-command detection and approval state, with per-session isolation and explicit approval resolution paths (`tools/approval.py`).
- `tldw_chatbook` is not primarily a shell agent, so full Hermes-style approval workflows would be overbuilt for the current product.
- The part worth borrowing is lightweight confirmation and audit affordances for destructive local actions or future sync/import flows: delete, overwrite, conflict resolution, remote pull, and bulk import/export.

## Background Tasks

- Hermes has a real process registry for long-running work, including `notify_on_complete`, watch patterns, buffered output, and session-scoped tracking (`tools/process_registry.py`, `RELEASE_v0.8.0.md`).
- `tldw_chatbook` already uses Textual workers and worker event handlers, which is a strong base, but it does not expose a durable job center for import/export, ingestion, media processing, evals, or future sync-style pulls.
- The best adaptation is a local job/status center rather than host-process management. Completion notifications, resumable status, and visible failures matter more than raw process control.

## Model / Provider Controls

- Hermes routes model switching through one shared pipeline: alias resolution, provider normalization, capability lookup, and warnings for non-agentic models (`hermes_cli/model_switch.py`).
- `tldw_chatbook` currently has split model controls across the sidebar and the new compact model bar (`Widgets/compact_model_bar.py`, `UI/Screens/chat_screen.py`), plus command-palette provider actions in `app.py` that are still placeholders.
- The transferable improvement is a single application-level model/provider switch path that updates UI state, validates availability, persists the choice, and avoids control drift between surfaces.

## Recommendations For tldw_chatbook

- Keep Hermes as a secondary overlay. `tldw_server` parity still determines the roadmap.
- Promote Hermes-derived work only when it strengthens parity verticals that already matter: job visibility for import/export and ingest, unified model switching for prompt/chat execution, concise action/result previews, and scoped confirmations for destructive operations.
- Defer Hermes ideas that do not materially help offline-first interoperability: terminal-centric approval flows, delegation/subagent ergonomics, broad slash-command parity, and gateway-specific interaction patterns.
- A job/status surface for import/export, ingestion, evals, and later sync.
- A unified model/provider switching service behind both chat UI control surfaces.
- Compact action/result previews inside chat and other long-running workflows.
