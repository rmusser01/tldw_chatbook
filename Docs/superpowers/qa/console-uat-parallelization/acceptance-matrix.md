# Console UAT Parallelization Acceptance Matrix

Date: 2026-06-21
Base: `origin/dev` after PR #542 (`7c199cf7`)

## Purpose

Coordinate parallel Console UAT work without allowing multiple agents to rediscover setup, overwrite each other's scope, or claim approval without actual rendered evidence.

The Console goal is user acceptance, not visual mockup parity: a user must be able to create and resume chats, select providers/models/settings, send messages, operate on messages, switch workspace context, and return later to saved work.

## Approval Rule

No Console screen state is approved without an actual rendered CDP/Textual-web screenshot or recording. Do not use generated SVGs, static mockups, code layout diagrams, or screenshots from non-running stand-ins as approval evidence.

Evidence protocol: `Docs/superpowers/qa/console-uat-parallelization/cdp-evidence-protocol.md`

## Workstream Matrix

| Workstream | Task | Branch | Owner Scope | Status | Visual Evidence | Regression/UAT Evidence | Approval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UAT Harness | TASK-128 | `codex/console-uat-harness-coordination` | CDP setup, fixtures, screenshot naming, acceptance matrix | In progress | Pending | Pending | Not approved |
| Chat Lifecycle | TASK-129 | `codex/console-chat-lifecycle` | New chat, close tab, send, blocked-send recovery, transcript baseline, in-session return | Not started | Pending | Pending | Not approved |
| Provider + Model Configuration | TASK-130 | `codex/console-provider-model-uat` | Provider/model/settings selection, endpoint preservation, readiness, streaming fallback | Not started | Pending | Pending | Not approved |
| Message Actions | TASK-131 | `codex/console-message-actions-uat` | Select message, keyboard/click actions, Copy/Edit/Save as/Regenerate/Continue/Thumbs/Delete | Not started | Pending | Pending | Not approved |
| Workspace + Resume | TASK-127 | `codex/console-workspace-resume-uat` | Workspace switcher, saved conversation list, resume prior chats, Default workspace policy | Not started | Pending | Pending | Not approved |

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
