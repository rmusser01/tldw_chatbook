---
id: DRAFT-42312
title: Design bounded plugin UI contributions from proven Console panels
status: Draft
created_date: 2026-11-05 16:09
updated_date: 2026-11-22 15:55
labels:
- plugins
- console
- design
priority: low
dependencies:
- TASK-38935
- TASK-38940
- TASK-38941
- TASK-38942
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
- https://code.claude.com/docs/en/plugins/mods/overview
documentation:
- Docs/Architecture/console-run-hooks.md
- Docs/Architecture/tool-catalog.md
- backlog/decisions/162-managed-agent-plugins.md
- backlog/decisions/163-expanded-console-hook-runtime.md
- backlog/docs/design-language.md
- backlog/docs/component-patterns.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Determine whether reusable plugin UI contributions would serve demonstrated Console customization needs after concrete panels have been qualified. Define a narrow, reviewable extension boundary before committing to a new UI plugin API or runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A written comparison uses evidence from concrete output/context/validation/replay panel work to identify which contributions require an extension seam and which are adequately served by built-ins, settings, commands, or hooks.
- [ ] #2 The proposed first release specifies allowed data/event access, UI slots and component vocabulary, activation scope, lifetime, budgets, error isolation, and unload/shutdown behavior.
- [ ] #3 The contract preserves permission floors, workspace exclusions, instruction-body visibility, runtime ownership, accessibility, keyboard conventions, and design-token governance.
- [ ] #4 Alternatives include retaining built-in panels and using existing plugin/hook facilities; no JavaScript runtime or compatibility promise is assumed without a justified decision.
- [ ] #5 An ADR records the accepted or deferred direction, unresolved questions, and the smallest separately reviewable implementation release; existing plugin and hook governance is explicitly reconciled.
- [ ] #6 The deliverable is reviewed design and linked evidence only; it neither enables third-party UI code nor changes permissions or the live plugin runtime.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Automated tests (unit + integration) cover any new logic, the linter and formatter pass, and `./scripts/preflight.sh` passes, per CLAUDE.md's Definition of Done.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
- [ ] #4 Design-only completion includes a reviewed written contract and ADR disposition; implementation, installation, or activation is not part of this draft.
<!-- DOD:END -->

## Renumbering provenance

Renumbered from draft-38943 to DRAFT-42312 on 2026-11-22. Drafts are their own Backlog.md id series (`draft-<n>` filenames, `DRAFT-<n>` frontmatter, as dev's archived DRAFT-1 shows) and no backlog guard scans `backlog/drafts/`, but 38943 also names a different task: TASK-38943, "Integrate native shared-worktree workflows with current dev and open PR" (PR #3411's branch), which dev's TASK-39147 cites. This draft was created at 16:09 and that task at 16:47 the same day, so the draft is the older claimant; it moved anyway because it was unmerged and uncited while the task is cited on dev. The 2026-11-22 sweep, after fetching origin, read every task and draft path in all objects reachable from every ref (539,451 paths), the backlog buckets of all 38 worktrees, and the content of the 93,746 distinct blobs under backlog/, Docs/, tldw_chatbook/, Tests/, scripts/ and .github/ at the tips of 723 refs. The highest id anywhere was 42281 (the one larger match, Task-46066, is an asyncio task name in a QA log). With the 30-id lead from scripts/next_backlog_task_ids.py, the split-out TASK-42311 and this draft took 42311 and 42312.
