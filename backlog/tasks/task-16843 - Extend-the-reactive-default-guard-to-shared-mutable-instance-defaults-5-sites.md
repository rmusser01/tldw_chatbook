---
id: TASK-16843
title: >-
  Extend the reactive-default guard to shared mutable instance defaults (5
  sites)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-16'
updated_date: '2026-08-16 18:43'
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
- [x] #1 Each of the 5 sites is classified LIVE/LATENT/immutable with in-place-mutation evidence, and every mutable one is converted to a per-instance factory
- [x] #2 The AST guard detects `reactive(SomeClass())`-shaped defaults (proven born-red by temporary reintroduction), with immutable types handled by a documented allowlist rather than silence
- [x] #3 Existing 15771 aliasing/guard tests and the touched widgets' suites stay green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify the 5 filed sites are still live at HEAD (094748b3e): grep-based
   AST scan of the whole package for reactive(PascalCaseCall(...)) defaults
   -- confirmed exactly 5, matching the filing (console_context_modal.py:62,
   watchlists_collections_screen.py:547/582/583, watchlists_workbench.py:98).
2. Classify each: read ConsoleContextSnapshot (frozen dataclass but with
   list/dict fields -- mutable underneath), RegionLayout and TreeScope
   (frozen dataclass, every field itself immutable: frozenset/Literal
   str/int|None/Enum -- no mutable container to leak).
3. Fix the one dangerous site: convert console_context_modal.py's
   `snapshot = reactive(ConsoleContextSnapshot(...))` to a lambda factory.
4. Extend Tests/Architecture/test_reactive_mutable_default_inventory.py to
   detect reactive(SomeClass())-shaped defaults via a PascalCase-callee
   heuristic, with an explicit IMMUTABLE_INSTANCE_ALLOWLIST for
   RegionLayout/TreeScope (documented reasoning per entry), refactor for
   testability (_violations_in_tree), and add unit tests proving: an
   unlisted instance-call default is flagged, an allowlisted one is not, an
   unallowlisted-but-actually-immutable one is still flagged (allowlist is
   opt-in, not inferred), and the lowercase-factory blind spot is pinned as
   a documented gap.
5. Prove born-red against the real pre-fix site content (checked out via
   `git show HEAD:<path>`) with the allowlist temporarily emptied -- all 5
   real sites flagged at their exact filed line numbers.
6. Add inline comments at the 3 documented-safe declaration sites
   (region_layout x2, selected_scope/tree_scope) explaining why they're
   allowlisted, without disturbing the existing 15775/15778 seeding
   commentary.
7. Add a cross-instance leak test for ConsoleContextSnapshot in
   Tests/Widgets/test_reactive_default_aliasing.py, following the existing
   15771 pattern (two modal instances, one mutates current_messages/
   next_send_payload in place, the other must not observe it) -- using a
   blocked snapshot_factory (asyncio.Event) so both instances stay on the
   shared class-level default for the whole assertion window, matching the
   real loading-spinner window a user sees.
8. Run: the guard test, the aliasing test file, the existing
   console_context_modal suites, and the 15775/15778 Watchlists pin suites
   (cold-open layout, cold-read swap, scoped rebuilds) to confirm the
   documentation-only changes to region_layout/selected_scope/tree_scope
   didn't disturb set_reactive seeding. ruff on all touched files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Classification (all 5 sites, at HEAD `094748b3e`):**

| Site | Class | `frozen=True`? | Field types | Mutated in place anywhere? | Verdict | Fix |
|---|---|---|---|---|---|---|
| `console_context_modal.py:62` `snapshot` | `ConsoleContextSnapshot` | Yes | `current_messages: list`, `next_send_payload: dict` — **mutable containers under a frozen wrapper** | Not currently (all reads in the file are `.get`/iteration), but the fields are genuinely mutable and the class docstring only promises the *store* can't reach in, not that in-widget code can't | **LATENT** (frozen blocks reassignment, not in-place mutation) | Converted to `reactive(lambda: ConsoleContextSnapshot(current_messages=[], next_send_payload={}))` |
| `watchlists_collections_screen.py:547` `region_layout` | `RegionLayout` | Yes | `collapsed: frozenset[Region]`, `solo_region: Region \| None`, `_pre_solo: frozenset[Region] \| None` — every field itself immutable | N/A — no mutable container exists to mutate | **IMMUTABLE / harmless** | Left as-is; documented + allowlisted |
| `watchlists_collections_screen.py:582` `selected_scope` | `TreeScope` | Yes | `kind: Literal[...]`, `watchlist_id: int \| None`, `source_id: int \| None` | N/A | **IMMUTABLE / harmless** | Left as-is; documented + allowlisted |
| `watchlists_collections_screen.py:583` `tree_scope` | `TreeScope` | Yes | same as above | N/A | **IMMUTABLE / harmless** | Left as-is; documented + allowlisted |
| `watchlists_workbench.py:98` `region_layout` | `RegionLayout` | Yes | same as row 2 | N/A | **IMMUTABLE / harmless** | Left as-is; documented + allowlisted |

Note the equality-skip mechanism the task flagged (`Reactive._set` no-ops when
`new == current`): it's irrelevant to the 4 harmless sites regardless, since
there's no in-place mutation to leak in the first place; it *would* have
mattered for `ConsoleContextSnapshot` had `_load_snapshot` ever reassigned an
equal-valued instance, but dataclass equality there is by field value and a
freshly-fetched non-empty snapshot is essentially never `==` the empty
default, so the aliasing exposure window was really "between mount and the
first `_load_snapshot` completing" (the loading-spinner window) — exactly
what the new leak test exercises.

**Fix:** `console_context_modal.py`'s `snapshot` reactive now uses a lambda
factory, giving each modal instance its own `ConsoleContextSnapshot` (and its
own empty `list`/`dict`). The other 4 sites were left unchanged — rewriting
them into factories would have been functionally-identical churn against a
class that cannot leak — and instead got an inline comment at each
declaration plus a guard allowlist entry.

**Guard extension**
(`Tests/Architecture/test_reactive_mutable_default_inventory.py`): added
detection for `reactive(SomeClass())`-shaped defaults via a PascalCase-callee
heuristic (`_looks_like_class_instantiation`, matching this repo's
PascalCase-class/snake_case-function convention), gated by an explicit
`IMMUTABLE_INSTANCE_ALLOWLIST` dict naming `RegionLayout`/`TreeScope` with
the field-type reasoning that earns each entry. Refactored the detector's
core into a path-free `_violations_in_tree(tree) -> list[(lineno, reason)]`
so unit tests can drive it with synthetic source instead of only real
package files. New unit tests: an unlisted instance-call default is flagged;
an allowlisted one is not; a *different*, unallowlisted class is still
flagged even though it's equally trivial/immutable-shaped (the allowlist is
opt-in by name, never inferred from shape — a new mutable-fielded class
can't slip through by resembling an allowlisted one); and the known
lowercase-factory blind spot (a snake_case function returning a mutable
instance) is pinned as a documented gap, not silently claimed as covered.

**Born-red proof** (temporary, not committed): ran the extended detector
against the real pre-fix file contents (`git show HEAD:<path>`) with
`IMMUTABLE_INSTANCE_ALLOWLIST` monkey-patched to `{}` — all 5 filed sites
were flagged at their exact original line numbers (62, 547, 582, 583, 98).
With the real allowlist restored, only `ConsoleContextSnapshot` (line 62)
was flagged, confirming the allowlist correctly narrows detection to the one
genuinely dangerous site.

**Leak test**
(`Tests/Widgets/test_reactive_default_aliasing.py::test_console_context_modal_snapshots_do_not_leak_across_instances`):
mounts two `ConsoleContextModal` instances behind a `snapshot_factory` that
blocks on an unset `asyncio.Event`, so both stay on the shared class-level
default for the whole window (matching the real loading-spinner window
between opening the modal and the snapshot arriving). Instance A appends a
real `ConsoleChatMessage` to `current_messages` and sets a key on
`next_send_payload` in place; pre-fix this was visible on instance B
(`modal_b.snapshot is modal_a.snapshot`, both assertions failed — verified
born red) and even crashed a widget build in an earlier probe (a leaked bare
string broke `_build_current_context_widgets`'s `msg.content` access,
demonstrating real corruption, not just an identity curiosity). Post-fix,
green.

**Verification:** guard tests (10 in the file, all green), aliasing tests (6
in the file, all green), `Tests/UI/test_console_context_modal.py` +
`test_console_modal_dismissal.py` + `test_chat_screen_context_modal.py` (131
passed), and the named 15775/15778 pin suites —
`Tests/Watchlists/test_watchlists_cold_open_layout.py`,
`test_watchlists_cold_read_swap.py`, `test_watchlists_scoped_rebuilds.py` (38
passed) — confirming the doc-comment-only edits to
`region_layout`/`selected_scope`/`tree_scope` didn't disturb the
`set_reactive` seeding paths. `ruff check` clean on all touched files.

**Files touched:**
- `tldw_chatbook/Widgets/Console/console_context_modal.py` — lambda-factory fix
- `Tests/Architecture/test_reactive_mutable_default_inventory.py` — guard extension + allowlist + 4 new unit tests
- `Tests/Widgets/test_reactive_default_aliasing.py` — new cross-instance leak test
- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` — doc comments only (`region_layout`, `selected_scope`/`tree_scope`)
- `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py` — doc comment only (`region_layout`)
<!-- SECTION:NOTES:END -->
