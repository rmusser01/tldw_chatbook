---
id: TASK-1538
title: Enforce single-pass service composition and runtime dependency binding
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-28 14:27'
updated_date: '2026-07-28 16:09'
labels:
  - architecture
  - lifecycle
  - packaging
dependencies:
  - TASK-906
references:
  - backlog/decisions/032-immutable-installed-distribution-assets.md
  - backlog/decisions/033-application-session-state-ownership.md
  - backlog/decisions/036-application-service-composition-lifecycle.md
  - >-
    Docs/superpowers/specs/2026-07-28-application-service-composition-lifecycle-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent TldwCli startup from replacing live service graphs or attaching long-lived services to stale provider and sync owners.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TldwCli composes the Writing and Chat conversation service graphs exactly once, and their scope services retain the exact local and server service identities created for the application lifetime.
- [ ] #2 The app-composed ServerWritingService resolves through TldwCli's long-lived server_context_provider rather than a private legacy config provider.
- [ ] #3 ChatConversationScopeService and MediaReadingScopeService receive the current sync_scope_service during initial production-app composition without altering the existing post-construction Sync reassignment behavior or claiming provider-wiring reentrancy.
- [ ] #4 Focused source and full production-app tests prove single-pass calls and dependency identities without surrogate or simplified application classes.
- [ ] #5 The clean installed-wheel production-app probe completes the fresh ChaChaNotes migrations and proves the same composition contract outside the checkout; the exact v26-to-v27 citation-provenance and v27-to-v28 character-authority SQL assets are present in both sdist and wheel and enforced by the release checker.
- [ ] #6 Affected tests, the full repository suite, static checks, formatting, and diff hygiene pass, and the separate remaining legacy-provider inventory is documented without claiming global closure.
- [ ] #7 The two verified current-dev collection blockers are reconciled with the surviving worker and chat-shell APIs, without restoring retired StreamDone or TabState state and without adding a test application.
- [ ] #8 Current-dev tests use the surviving public runtime-config and trusted-directory contracts, production-app interaction tests wait for rendered controls, the reviewed diagnostic inventory matches current source and persistent-sink topology, and newly introduced diagnostic paths do not persist user-authored selector or eval-dataset content.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/036-application-service-composition-lifecycle.md and existing backlog/decisions/032-immutable-installed-distribution-assets.md
Reason: This task changes application service construction, runtime provider ownership, Sync dependency binding, and—after installed verification exposed missing runtime migrations—applies ADR-032 to the two exact required SQL assets without broad package-data inclusion.

1. Add narrow AST, full production TldwCli, and installed-wheel regression contracts; verify they fail on the observed duplicate/provider/Sync defects.
2. Add artifact and release-checker coverage for the exact ChaChaNotes v26-to-v27 and v27-to-v28 runtime SQL migrations, then explicitly include and enforce those files in the sdist and wheel.
3. Remove only the later duplicate Writing and Chat calls, bind Writing to server_context_provider, and inject sync_scope_service into Chat and Media initial composition.
4. Reconcile verified current-dev test and sentinel drift against the surviving worker, chat-shell, runtime-config, trusted-directory, rendered-widget, and persistent-diagnostic contracts, without restoring retired APIs, weakening production guards, or adding a test application.
5. Run focused provider/Sync/citation/packaging tests, all ProductionApp and Packaging tests, the full repository suite, compile/Ruff/diff checks, and record the remaining executable legacy-provider inventory.
6. Self-review the bounded diff, record verification evidence, complete acceptance criteria and implementation notes, then mark the task Done only after every Definition of Done item passes.
7. Rebase onto current dev, open the PR, verify and address every review thread/check, rebase again, and merge after the required evidence remains green.

Detailed executable plan: Docs/superpowers/plans/2026-07-28-application-service-composition-lifecycle.md
<!-- SECTION:PLAN:END -->
