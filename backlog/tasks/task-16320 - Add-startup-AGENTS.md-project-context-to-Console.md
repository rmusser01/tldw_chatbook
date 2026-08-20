---
id: TASK-16320
title: Add startup AGENTS.md project context to Console
status: To Do
assignee: []
created_date: '2026-08-20 15:31'
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
- [ ] New Console agent sessions enable project instructions, resolve exactly one eligible selected workspace-folder binding, and require destination-scoped first-use consent before the first provider request; legacy restored sessions remain disabled.
- [ ] Root discovery implements `AGENTS.override.md`/`AGENTS.md` precedence, strict UTF-8, stable bounded reads, symlink/reparse refusal, no ascent or global fallback, whole-source byte/token admission, and content-free warnings.
- [ ] Startup instructions are labeled untrusted user-level context, reach every supported Console agent send path exactly once, and never enter transcripts, agent steps, run logs, automatic tool results, exception text, `/rewind` summaries, or exchange-capture bodies.
- [ ] Versioned project-context control state persists in a local-only conversation column without changing conversation version, sync timestamps, or `sync_log`; existing conversation update/delete/restore/import paths preserve it and new imported conversations start disabled.
- [ ] Enabled runs compose local filesystem/git tools at the selected binding root, retain existing behavior when disabled, and omit write/edit/patch tools for read-only bindings.
- [ ] The Console rail and Context surface expose the basic enabled/binding/source/warning state using metadata only, with setup recovery for zero or multiple eligible bindings.
- [ ] Focused resolver, migration, session, provider-transport, persistence-leak, `/rewind`, and UI tests pass; existing Console, database, local-tool, and provider regression suites remain green.
<!-- AC:END -->
