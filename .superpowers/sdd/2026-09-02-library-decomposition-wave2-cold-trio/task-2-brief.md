# Wave-2 Task 2: Export series 1/3 — characterization spot-check + LibraryExportState

From `Docs/superpowers/plans/2026-09-02-library-decomposition-wave2-cold-trio.md` ("Tasks 2–4: Export series", first task) with the recipe (`backlog/docs/library-decomposition-recipe.md`) as the mechanics authority and the conversations series as the worked example.

## Deliverables (one PR-shaped commit series)

1. **Characterization spot-check first** (recipe rule; foundation Task-5 shape): enumerate the export cluster's methods (`ast` walk of `LibraryScreen` for names containing `export`; ~51 at the 2026-09-02 snapshot — re-derive, don't trust), find `@on` handlers and externally-reachable behaviors never exercised through the DOM by existing tests (grep + `.press()` verification, not just id mentions — the foundation Task 5 report shows the method). Write pins ONLY for genuine gaps into `Tests/UI/test_library_export_characterization.py`; they must PASS against current code (inverted TDD); live bugs get pinned, never fixed; skip decisions recorded with one-line rationales.
2. **Ownership analysis** (recipe §2 script, export prefix set: `_library_export`): classify every export-prefixed `__init__` field (expected ~12) move/stay. Known boundary to check: `_library_export_origin_row_id` is written by rail-switch shell code (`_select_library_rail_row*` path) — if shell/plumbing-only consumers, it MOVES (shims cover the transition); if another subsystem's methods consume it, it STAYS. Record the table.
3. **`LibraryExportState`** in `tldw_chatbook/UI/Library_Modules/library_export_state.py`: exclusive fields moved with verbatim defaults (mutable defaults via `field(default_factory=...)` preserving per-instance semantics — conversations state is the template); computed defaults → constructor args at the same `__init__` position; comments carried verbatim.
4. **Screen shims, programmatic** (the foundation mechanism, NOT literal properties): sentinel-wrapped install loop after the class; `_n=` default-arg closure binding both directions; define the export plural/prefix mapping in the STATE module as the single source (the conversations series' `CONVERSATIONS_PLURAL_STATE_FIELDS` pattern — check whether any export field needs a non-default prefix at all; if all fields map under `_library_export_`, say so and skip the plural set).
5. **Wiring test** `Tests/Architecture/test_library_export_wiring.py`, written FIRST and watched failing: state-object fields ↔ shim surface (the conversations wiring test's final shape, adapted).

## Verification battery

- Wiring test RED → GREEN; characterization file all-PASS.
- Size ratchet ceiling AND slack green (this task should be ~line-neutral like the conversations state PR; if the slack guard trips, measure and lower — never raise).
- Recompose ratchet (now with its own slack guard) + support-layer surface test green.
- `-k "export and library"` suite with stash-baseline comparison; full xdist paired-baseline sweep per recipe §7 (with its documented pre-existing failures list); preflight.

## Commits

`test(library): characterization pins for the export extraction series` (if any pins written), then `refactor(library): export state object + shims (export series 1/3)`. No blame-ignore entries (state PRs are not body moves; foundation precedent).
