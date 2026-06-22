# Console UAT Parallelization Acceptance Matrix

Date: 2026-06-21
Base: `origin/dev` after PR #546 (`0222294e`)

## Purpose

Coordinate parallel Console UAT work without allowing multiple agents to rediscover setup, overwrite each other's scope, or claim approval without actual rendered evidence.

The Console goal is user acceptance, not visual mockup parity: a user must be able to create and resume chats, select providers/models/settings, send messages, operate on messages, switch workspace context, and return later to saved work.

## Approval Rule

No Console screen state is approved without an actual rendered CDP/Textual-web screenshot or recording. Do not use generated SVGs, static mockups, code layout diagrams, screenshots from non-running stand-ins, mocked provider responses, or seeded assistant transcripts as approval evidence.

Automated regressions may use deterministic fakes. UAT approval must use the running app with real UI interactions, real persistence/services, and a live provider/API response for any workflow that claims completed assistant output. If no provider is reachable, capture the real blocked/unavailable path and mark provider-response UAT blocked rather than substituting fake output.

Evidence protocol: `Docs/superpowers/qa/console-uat-parallelization/cdp-evidence-protocol.md`

## Workstream Matrix

| Workstream | Task | Branch | Owner Scope | Status | Visual Evidence | Regression/UAT Evidence | Approval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UAT Harness | TASK-128 | `codex/console-uat-harness-coordination` | CDP setup, fixtures, screenshot naming, acceptance matrix | In progress | Pending | Pending | Not approved |
| Chat Lifecycle | TASK-129 | `codex/console-chat-lifecycle-parallel` | New chat, close tab, send, blocked-send recovery, transcript baseline, in-session return | Done | `task-129-chat-lifecycle-cdp-2026-06-21.png`; `task-129-composer-text-visible-cdp-2026-06-21.png` | `Tests/UI/test_console_native_chat_flow.py` passed locally: 76 passed | Approved |
| Provider + Model Configuration | TASK-130 | `codex/console-provider-model-uat-current` | Provider/model/settings selection, endpoint preservation, readiness, streaming fallback | Done | `TASK-130-provider-model-modal-controls-cdp.png`; `task-130-provider-credential-source-cdp-2026-06-21.png` | `Tests/UI/test_console_session_settings.py -k "provider or model or endpoint or credential or generation or summary"` passed locally: 64 passed; generic non-streaming fallback and UI completed-message checks passed locally: 2 passed | Approved |
| Message Actions | TASK-131 | `codex/console-message-actions-uat-current` | Select message, keyboard/click actions, Copy/Edit/Save as/Regenerate/Continue/Thumbs/Delete | Done | `task-131-selected-message-inspector-expanded-cdp-2026-06-21.png`; `task-131-save-as-context-cdp-2026-06-21.png` | `Tests/UI/test_console_native_chat_flow.py -k "message_action or selected_message or continue_action or regenerate_action or save_as or workspace_conversation"` passed locally: 18 passed | Approved |
| Workspace + Resume | TASK-127 | `codex/console-workspace-resume-uat-current` | Workspace switcher, saved conversation list, resume prior chats, Default workspace policy | Done | `task-127-live-llamacpp-response-cdp-2026-06-21.png`; `task-127-saved-conversation-resume-cdp-2026-06-21.png`; `task-127-polish-pass-cdp-chromium-2026-06-21.png` | Mounted regressions verify fake-service and real local DB-backed resume paths; focused workspace/resume subset passed locally: 16 passed; rail-safe conversation label polish regressions passed locally: 3 passed. Live llama.cpp UAT verified through Textual-web/CDP on `127.0.0.1:9099`: user prompt rendered, assistant returned `OK-127`, saved conversation row resumed persisted transcript. | Approved |

## Required Evidence Per PR

- Backlog task updated with implementation plan before code changes.
- ADR check documented in the task implementation plan.
- Focused regression or UAT script added or updated.
- Actual CDP/Textual-web screenshot or recording attached under `Docs/superpowers/qa/console-uat-parallelization/`.
- User approval checkpoint recorded as approved or not approved.
- Final implementation notes added before task is marked Done.

## Coordination Rules

- No agent owns "Console"; each owns one user outcome.
- The main checkout is not a base for implementation. Use fresh worktrees from current `origin/dev`.
- Avoid branching from stale `.worktrees/` or prior PR branches unless explicitly documenting a dependency.
- If two streams need `chat_screen.py`, Chat Lifecycle owns shell/composer/tab behavior and Message Actions owns transcript action behavior. Conflicts are resolved in a final integration PR.
- Provider + Model Configuration owns Settings-to-Console runtime configuration and should not rewrite transcript, workspace, or message-action behavior.
- Workspace + Resume owns workspace/conversation state and should not rewrite provider or transcript internals.

## Known Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Stale local worktrees | Agents reintroduce old UI or already-fixed bugs | Start every stream from freshly fetched `origin/dev`. |
| `chat_screen.py` contention | Merge conflicts and behavior regressions | Assign shell/composer/tab changes to Chat Lifecycle and message selection/actions to Message Actions. |
| Inconsistent CDP setup | Screenshots are not comparable | Harness owns launch commands, fixture config, ports, and screenshot naming. |
| Missing historical CDP runbook | Agents cannot reproduce prior browser QA setup | Harness restores a durable Console-specific protocol before feature streams claim approval. |
| False approvals | UI appears approved based on mockups or non-running captures | Approval requires actual CDP/Textual-web evidence only. |
| Workspace scope creep | Sync/server work derails Console usability | Workspace + Resume covers local/default workspace and handoff-ready metadata only; server sync remains WIP-labeled. |

## Suggested Fresh Worktree Commands

```bash
git fetch origin dev
git worktree add -b codex/console-chat-lifecycle /private/tmp/tldw-chatbook-console-parallel/chat-lifecycle origin/dev
git worktree add -b codex/console-provider-model-uat /private/tmp/tldw-chatbook-console-parallel/provider-model origin/dev
git worktree add -b codex/console-message-actions-uat /private/tmp/tldw-chatbook-console-parallel/message-actions origin/dev
git worktree add -b codex/console-workspace-resume-uat /private/tmp/tldw-chatbook-console-parallel/workspace-resume origin/dev
```
