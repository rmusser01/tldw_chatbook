---
id: TASK-19571
title: >-
  Decide once what to do with 170 unreachable modules, 78 of which still carry a
  test suite
status: To Do
assignee: []
created_date: '2026-08-21 20:21'
labels:
  - architecture
  - tech-debt
  - policy
priority: medium
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 1 (architecture & reachability) — its
census (**#14, #15**) plus the named subsystem findings **#6, #7, #9, #10,
#11-13**. Spot-checked at this branch base.

**Method matters here, because the number is only useful if it is trustworthy.**
The lane built an AST import graph over all **1,767 modules (0 parse errors)**
and ran a transitive BFS from `app.py`. Before counting, it corrected **four
false-orphan shapes** (package-`__init__` relative imports, lazy `__getattr__`
module maps, dotted registry targets, `importlib` string builders) and
calibrated against 13 known-live modules with **0 wrongly flagged**. It also
correctly **excluded the 87 splash effects**, which are `importlib`-loaded.

**Census: 170 modules / 62,375 LOC (5.5% of the package) unreachable from
`app.py`. 78 of them (37,408 LOC) still carry a test suite** — a recurring
verification tax on code that cannot run.

Named subsystems, all confirmed present at this base:

| subsystem | LOC | note |
|---|---|---|
| Media (`UI/Screens/media_screen.py` + friends) | ~6,174 | `resolve_screen_target("media")` → `LibraryScreen`; the alias is applied **before** route lookup |
| Notes-import island (`Notes/note_import_executor.py` + friends) | 10,590 | incl. **5 DB tables** selected only from unreachable code, and a registered SQLite owner policy for a module that never runs; `NoteImportExecutor` is constructed **only in `Tests/`** |
| Subscriptions scraper stack | 4,792 | `RateLimiter`/`ContentExtractor` are **redefined live** in `monitoring_engine.py`; per-site extraction config has **no reachable UI at all** |
| Tamagotchi | 2,843 | includes its own SQLite layer |
| CodeRepoCopyPaste | 1,215 | |
| `Chat/console_visual_evaluation.py` | 1,285 | |
| `Widgets/emoji_picker.py` | 1,136 | |
| RAG `pipeline_loader.py` | 958 | zero production importers; **two shipped config files instruct users to configure middleware against it**, and the live loader has no middleware handling |
| `Widgets/NewIngest/` | 963 | **imports `unittest.mock` in production** (`BackendIntegration.py:8`) and returns `{"status": "simulated"}` |
| `UI/Screens/schedules_screen.py` | 516 | stale duplicate |

Plus: two competing screen registries; a `"customize"` route
(`screen_registry.py:128`) pointing at a **deleted module**, masked by an alias
— a landmine for the next consumer of that API; dead event vocabulary at scale
(`Event_Handlers/Chat_Events/chat_messages.py` has **47 classes with zero
importers**; 13 of 34 `Event_Handlers` modules unreachable, **including two
genuinely zero-byte files** —
`llm_management_events_llamacpp.py` and `llm_management_events_llamafile.py`);
and **dead schema shipped to every user** — `mindmaps`, `mindmap_nodes`,
`conversation_dictionaries` (superseded), `workspace_handoff_audit` (never read
or written), `learning_paths` (write-only, already noted in a code comment).

**Reachability honesty — the point of this task.** The lane deliberately
**downgraded two of its own sub-agents' findings** on this basis: the media
save-analysis-as-note defect (`app.notes_db` is assigned nowhere) and the
Reading Highlights buttons (implemented handlers, no wiring) are **both real
and both unreachable**. It also corrected a sub-agent's claim that a live
pipeline path existed (zero production importers) and another that an import
path did not exist (it is a live alias shim). Anything filed out of this census
must carry the same discipline: these are **wire-or-retire** decisions, not
bugs to fix in place.

**Two counts to re-measure rather than inherit:** the review reported
`ccp_messages.py` as "44 of 45 classes dead with 14 posted at runtime landing
nowhere", but at this base `UI/CCP_Modules/ccp_messages.py` has **8 classes** —
either the figure covered a different file or it is stale, so re-derive it.
`ScraperBuilderWindow` no longer exists in the tree at all.

**This is a decide-once policy task, not a deletion task.** The value is a
recorded disposition per subsystem so the census stops being re-derived every
few months, and a guard so the number cannot silently grow.

## Acceptance Criteria

- [ ] Each named subsystem above has a recorded disposition — **wire** (with
      the route/entry point that makes it reachable) or **retire** (removed,
      along with its tests) — with the decision and its reason written down
- [ ] Retiring a subsystem removes its tests too; the 37,408 LOC of tests on
      unreachable code stops being maintained
- [ ] The dead schema (`mindmaps`, `mindmap_nodes`,
      `conversation_dictionaries`, `workspace_handoff_audit`, `learning_paths`)
      is resolved, including what happens to existing user databases
- [ ] `Widgets/NewIngest/BackendIntegration.py` no longer imports
      `unittest.mock` in production code, whatever its disposition
- [ ] The two shipped config files that tell users to configure middleware
      against the dead RAG `pipeline_loader` are corrected — a shipped config
      must not document a code path that cannot run
- [ ] The `"customize"` route pointing at a deleted module is removed, and the
      two competing screen registries are reconciled to one
- [ ] The two zero-byte `Event_Handlers` modules are deleted
- [ ] The `ccp_messages.py` figure is re-measured rather than inherited from
      the review
- [ ] A reachability check runs in CI (see TASK-19572) and fails when the
      unreachable-module count grows, so the census does not silently re-inflate
- [ ] Nothing in this task is filed or fixed as a user-facing bug without first
      confirming the surface is reachable
