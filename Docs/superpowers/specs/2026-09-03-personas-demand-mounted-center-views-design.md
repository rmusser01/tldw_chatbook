# Personas demand-mounted center views — design

- **Date:** 2026-09-03
- **Status:** Approved
- **Scope:** TASK-31215, Personas/Roleplay presentation lifecycle only
- **Decision record:** Accepted [ADR-115](../../../backlog/decisions/115-personas-demand-mounted-center-views.md)

## 1. Problem and evidence

TASK-2725 removed the four heaviest hidden Personas views from `compose`, but its
post-first-paint loader still mounts all four before it restores selection or
finishes the initial library load. The user sees the first frame, then the message
pump performs the same large CSS/mount workload regardless of which workflow is
needed.

A scratch-profile mounted census on current `dev` found 574 descendants after
settle, 458 effectively hidden. The four deferred roots contribute 350 hidden
descendants:

| Heavy view | Hidden subtree descendants |
| --- | ---: |
| `PersonasCharacterEditorWidget` | 140 |
| `PersonaProfileEditorWidget` | 83 |
| `PersonasDictionaryDetailWidget` | 67 |
| `PersonasLoreDetailWidget` | 60 |

This is the same failure shape fixed for Lab > Models by TASK-31002: painting a
small active route is followed by constructing unrelated inactive routes. Personas
has more restore and editor state than Models, so the fix needs one explicit
screen-owned lifecycle rather than scattered `QueryError` tolerance.

## 2. Goals

1. Let Personas settle into an interactive default Characters surface without
   mounting any of the four heavy inactive bodies.
2. Mount only the body required by the user's first selection/create/edit action.
3. Cache each mounted body for the screen lifetime so unsaved input, focus, and
   presentation state survive mode switching.
4. Preserve restore, runtime-source, actor-pack, visual-identity, dictionary/lore,
   and Console-handoff behavior.
5. Make concurrent first use, mount failure, and teardown ordering deterministic.
6. Pin responsiveness under the production CSS bundle.

## 3. Non-goals

- Changing Personas information architecture, labels, or visible layout.
- Deferring the default character card, library pane, inspector, preview, character
  attachments, conversation transcript, or Try-It chrome.
- Unmounting a body after first use or shrinking steady-state memory during a long
  Personas visit.
- Refactoring the character/persona services, storage, Console handoff contract, or
  the individual editor widget implementations.
- Fixing the separate Library rendered-Markdown hang or applying the pattern to
  Schedules in this change.

## 4. Chosen design

### 4.1 Stable slots and identities

`#personas-detail-stack` keeps the historical document order but replaces the
four absent-body anchor gaps with empty lightweight slots:

```text
character card
character editor slot          -> mounts #ccp-character-editor-view
character attachments
persona card
persona editor slot            -> mounts #ccp-persona-editor-view
conversation actions
dictionary detail slot         -> mounts #personas-dictionary-detail
lore detail slot               -> mounts #personas-lore-detail
conversation transcript
placeholder / empty guidance
```

The slots have internal IDs and no user-visible content. The heavy roots retain
their existing IDs, types, and CSS classes. An empty slot has zero content demand;
after population, the existing root continues to own its own visibility and
layout. This preserves the query and styling contract without remounting siblings
or using fragile `after=` anchors during a user action.

### 4.2 One screen-owned first-use boundary

`PersonasScreen` owns an immutable view definition map:

```text
view key -> root selector, slot selector, factory, post-mount hydrator
```

An async `_ensure_center_view(view_key)` method is the only population path. It:

1. validates the requested key;
2. returns the already-mounted root immediately;
3. acquires the per-view async lock and checks again, coalescing concurrent first
   requests without creating an unowned background task;
4. constructs the body only inside the successful attempt;
5. mounts it hidden into its stable slot;
6. checks the screen lifecycle generation and mounted ownership;
7. applies view-specific retained state; and
8. records readiness only after all steps succeed.

Failure leaves no readiness marker. Any partial body is detached when possible,
and the next explicit workflow action makes a fresh attempt. The caller receives a
bounded recoverable error through the existing guarded action path; raw exception
payloads are logged only under existing diagnostic policy.

The cache lasts only for this `PersonasScreen` instance. Navigating away still
disposes the entire screen as it does today.

### 4.3 Workflow admission points

Only operations that require a heavy body cross the boundary:

| View | Admission paths |
| --- | --- |
| Character editor | Create character/actor pack; edit selected character |
| Persona editor | Create persona/actor pack; edit selected persona |
| Dictionary detail/editor | Select/restore dictionary; create or edit dictionary |
| Lore detail/editor | Select/restore lore book; create or edit lore book |

The admission occurs before the workflow mutates selection, edit mode, dirty state,
or starts body-owned workers. Once admitted, the existing deeper handlers may keep
their direct widget queries because the workflow now owns a mounted body. Save,
cancel, media, policy, and visual-identity messages originate from a mounted editor
and therefore do not need an additional population call.

Merely switching among Characters, Personas, Dictionaries, Lore, and Prompts does
not populate a heavy body. Dictionary/lore row selection populates the corresponding
detail; character/persona selection continues to use the eager read-only card.

### 4.4 Restore and retained state

`restore_state` remains pre-mount and data-only. `_load_after_mount` no longer batch
mounts the four views. It loads the active library and calls
`_apply_pending_restore`; the selected-kind path then admits exactly the required
body before applying its record.

Character and persona read-only restores use their eager cards and mount no editor.
Dictionary and lore restores await their detail body. Pending restore data remains
screen-owned until replay succeeds or the existing stale/deleted-record handling
classifies it terminal.

The persona editor hydrator applies the latest normalized local/server runtime
source before the admitting create/edit function writes editor data. Other bodies
receive record data only from their existing selection/create/edit functions after
mount. A body is never populated from a snapshot belonging to another active mode,
entity, runtime source, or screen generation.

### 4.5 Concurrency and teardown

Each view has an independent lock so unrelated first-use requests do not share a
global latch. Textual mutation still occurs on the screen event loop. Repeated
requests for one view converge on the same mounted root.

`PersonasScreen` increments a lifecycle generation during unmount. A population
attempt captures the generation before mounting and rechecks it before hydration
or readiness publication. If the screen detached or the generation changed, the
attempt removes/abandons its body and returns without applying user state. Existing
worker-specific generations continue to protect actor-pack, visual-identity, TTS,
and search results; the new generation protects only the view-lifecycle boundary.

## 5. Error and recovery behavior

- An unknown view key is a programming error and fails immediately.
- Missing/detached slots or a Textual mount failure leave the view retryable.
- User-triggered failures flow through `_run_guarded` and show a bounded error while
  keeping the previous stable center view intact.
- Initial restore does not convert a transient body-mount failure into a stale
  entity deletion. The pending payload remains eligible for a later explicit retry.
- Navigating away makes every late population result stale; it cannot focus fields,
  change selection, or show an editor on another screen.
- A failed view does not block the other three views from mounting.

## 6. Verification strategy

TDD starts by replacing the old "all four exist after load" assertion with mounted
contracts that fail on current `dev`:

1. after the real initial load settles, all four heavy root selectors remain absent;
2. character card selection remains usable without mounting any editor/detail;
3. first selection/create/edit mounts only its required root;
4. a second request returns the same root and preserves an edited field;
5. two concurrent ensure calls construct and mount one body;
6. a failed first mount does not consume readiness and a second attempt succeeds;
7. dictionary/lore restore waits for the requested detail and applies the exact
   saved entity only after mount;
8. unmount during a controlled population delay produces no stale hydration or
   focus mutation;
9. all four existing mode workflows retain their current visible results; and
10. a production-CSS heartbeat sees no single event-loop stall above 250 ms while
    opening Personas with inactive bodies absent.

Verification remains targeted per repository policy: the deferred-center-view
module, affected Personas workbench/editor/dictionary/lore/restore tests, blocking
I/O architecture guard, scoped Ruff/compile checks, and `git diff --check`. A full
repository sweep is not part of this task unless the user opts in.

## 7. Documentation and rollout

No user-facing guide changes are expected because labels and behavior are unchanged;
the change is observable only as faster navigation and a one-time cost when a heavy
workflow is first opened. TASK-31215 records measured before/after evidence and any
deviation from this design. ADR-115 is accepted when the owner approves this spec.
