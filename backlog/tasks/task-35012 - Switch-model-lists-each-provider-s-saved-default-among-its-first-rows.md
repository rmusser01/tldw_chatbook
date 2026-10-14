---
id: TASK-35012
title: Switch model lists each provider's saved default among its first rows
status: To Do
created_date: 2026-10-14 17:10
labels:
- console
- model-switcher
- providers
- model-config
- ux-review-2026-10-14
dependencies:
- TASK-35010
- TASK-33008
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified finding C-M1: a model saved to a provider's list is appended at the end. With no Find text, Switch model (Alt+M) shows the first three models of each ready provider in list order, and lifts only the open chat's current model. So for a provider that already holds three or more models, the model the user just saved sits under '… N more' and looks unsaved.

After TASK-35011, the model a user picks in Settings becomes that provider's saved default, so the default is the model they are looking for. The verifier's constraints hold: keep the three-row cap, and do not reorder [providers] on disk, because other readers use that order. Rows are therefore ordered at display time.

Phase 8 (TASK-33008) already adds the saved model as a row when the list reopens after a return from Settings. Phase 9's TASK-33005.12 decides which models a server listing still offers as rows. This task only orders the rows those rules produce.

Plan: `backlog/docs/plan-2026-10-14-s1-discovered-model-default-saved-lists.md`. Findings closed: C-M1 (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With no Find text, each ready provider's group in Switch model shows that provider's saved default for new chats first, or second after the open chat's current model when both belong to that provider, even when the provider has more than three models
- [ ] #2 Each group lists the same models as before, only reordered, and the '… N more' row counts the remaining ones
- [ ] #3 A shipped sample model id is never moved forward as a provider's default
- [ ] #4 Find results and their ranking are unchanged
<!-- AC:END -->
