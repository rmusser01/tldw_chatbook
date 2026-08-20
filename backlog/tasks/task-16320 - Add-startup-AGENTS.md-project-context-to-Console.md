---
id: TASK-16320
title: Add startup AGENTS.md project context to Console
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-20 15:31'
updated_date: '2026-08-20 16:46'
labels:
  - console
  - agents
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let new Console agent sessions safely load repository-authored root AGENTS.md guidance from one selected workspace folder while preserving Chatbook's trust, persistence, provider, and workspace boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 New Console agent sessions enable project instructions, resolve exactly one eligible selected workspace-folder binding, and require destination-scoped first-use consent before the first provider request; legacy restored sessions remain disabled.
- [ ] #2 Root discovery implements `AGENTS.override.md`/`AGENTS.md` precedence, strict UTF-8, stable bounded reads, symlink/reparse refusal, no ascent or global fallback, whole-source byte/token admission, and content-free warnings.
- [ ] #3 Startup instructions are labeled untrusted user-level context, reach every supported Console agent send path exactly once, and never enter transcripts, agent steps, run logs, automatic tool results, exception text, `/rewind` summaries, or exchange-capture bodies.
- [ ] #4 Versioned project-context control state persists in a local-only conversation column without changing conversation version, sync timestamps, or `sync_log`; existing conversation update/delete/restore/import paths preserve it and new imported conversations start disabled.
- [ ] #5 Enabled runs compose local filesystem/git tools at the selected binding root, retain existing behavior when disabled, and omit write/edit/patch tools for read-only bindings.
- [ ] #6 The Console rail and Context surface expose the basic enabled/binding/source/warning state using metadata only, with setup recovery for zero or multiple eligible bindings.
- [ ] #7 Focused resolver, migration, session, provider-transport, persistence-leak, `/rewind`, and UI tests pass; existing Console, database, local-tool, and provider regression suites remain green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Follow the [approved delivery plan](../../Docs/superpowers/plans/2026-08-20-agents-md-support.md) for [TASK-16320](<task-16320 - Add-startup-AGENTS.md-project-context-to-Console.md>), the [approved design](../../Docs/superpowers/specs/2026-08-20-agents-md-support-design.md), and [ADR-069](../decisions/069-console-project-instruction-local-state-and-preflight.md).
2. Establish branch, pull-request, schema-head, and focused regression baselines before feature edits.
3. Add versioned project-instruction control state, destination/binding fingerprints, bounded configuration, and secure root AGENTS.md resolution with whole-source admission tests.
4. Add the next ChaChaNotes migration and explicit local-only persistence paths, proving conversation sync metadata, sync_log, imports, restores, and legacy sessions preserve the accepted boundary.
5. Inject startup guidance exactly once through every supported Console agent provider path, keep it ephemeral and untrusted, and prove it does not enter transcripts, steps, logs, rewind summaries, automatic tool results, or exchange captures.
6. Compose enabled runs at the selected authorized binding root, preserve disabled behavior, omit mutating tools for read-only bindings, and expose metadata-only rail/Context recovery and notice UX.
7. Run the plan's focused and regression verification, static checks, self-review, and task/documentation close-out.

ADR required: yes
ADR path: backlog/decisions/069-console-project-instruction-local-state-and-preflight.md
Reason: this changes provider/runtime trust boundaries and local-only durable state.

Baseline evidence (2026-08-20):
- Schema head: v32.
- Collision check: only the current codex/agents-md-support branch was found; the open AGENTS.md PR search returned zero results.
- Command: python -m pytest Tests/Agents/test_tool_catalog_owner_cache.py Tests/Agents/test_agent_runtime_review_hook.py Tests/Chat/test_console_local_review_hook.py Tests/DB/test_chachanotes_context_summary_migration.py Tests/Workspaces/test_workspace_folder_bindings.py Tests/UI/test_console_context_modal.py -q
- Result: 71 passed, 1 warning in 6.03s.
<!-- SECTION:PLAN:END -->
