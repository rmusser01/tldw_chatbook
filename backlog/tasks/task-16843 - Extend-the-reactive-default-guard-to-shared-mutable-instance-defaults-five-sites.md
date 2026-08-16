---
id: TASK-16843
title: 'Extend the reactive-default guard to shared mutable instance defaults (5 sites)'
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - ui
  - architecture
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-15771 (PR #1699) converted all literal/`list()`-call mutable reactive defaults to
callable factories and landed the AST guard
`Tests/Architecture/test_reactive_mutable_default_inventory.py`. Its review's F2 flagged
the natural next hole, still open at dev `ee741cf10`: **`reactive(SomeClass())` — a
shared mutable *instance* default — is the same one-object-per-class aliasing bug and is
not detected** (the guard flags `list()/dict()/set()` call results specifically, not
arbitrary constructor calls). Five occurrences confirmed live at HEAD:

- `Widgets/Console/console_context_modal.py:62-64` — `snapshot = reactive(ConsoleContextSnapshot(current_messages=[], next_send_payload={}))` (carries mutable list/dict fields)
- `UI/Screens/watchlists_collections_screen.py:547` — `region_layout = reactive(RegionLayout())`
- `UI/Screens/watchlists_collections_screen.py:582` — `selected_scope = reactive(TreeScope(kind="all"))`
- `UI/Screens/watchlists_collections_screen.py:583` — `tree_scope = reactive(TreeScope(kind="all"))`
- `UI/Watchlists_Modules/watchlists_workbench.py:98` — `region_layout = reactive(RegionLayout())`

Whether each is *actually* mutated in place is per-type and untraced (the review did not
trace them) — that classification is this task's first job, exactly as 15771's LIVE/LATENT
table did for the literal sites. Note task-15775 already reconciled the
`RegionLayout` default's *value*; identity-sharing across instances is a separate
question. Convert genuinely mutable defaults to factories (`reactive(RegionLayout)` /
lambdas), then teach the guard to flag instance-call defaults — with an explicit
allowlist mechanism for frozen/immutable types so the guard states its contract honestly
instead of overclaiming (the 15771 F1/F3 lesson).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Each of the 5 sites is classified LIVE/LATENT/immutable with in-place-mutation evidence, and every mutable one is converted to a per-instance factory
- [ ] #2 The AST guard detects `reactive(SomeClass())`-shaped defaults (proven born-red by temporary reintroduction), with immutable types handled by a documented allowlist rather than silence
- [ ] #3 Existing 15771 aliasing/guard tests and the touched widgets' suites stay green
<!-- AC:END -->
