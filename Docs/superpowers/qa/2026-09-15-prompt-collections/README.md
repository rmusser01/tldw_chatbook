# Prompt Collections journey review — TASK-32632

More actions → Collections now reveals Info from Basic, closes More actions,
and opens the manager from its visible Manage control. Advanced and Info retain
their projection. Previously the menu tried to press a hidden button and did
nothing. Done and Cancel use the shared guarded dismissal and explicitly reveal
their returned opener; at 80×24 it otherwise kept focus at y=27 below the screen.

The manager retains both input drafts when selection, catalog results, or
mutation outcomes replace its children. Selecting a row previously erased the
name being entered. Apply preserves the dirty Prompt draft and restores keyboard
focus to Apply after failure or Manage after success, unless newer focus or a
replacement editor owns the interaction. Storage and session contracts are unchanged.

## Targeted verification

**180 distinct targeted checks passed**, without a full repository sweep:

| Command, using `.venv/bin/python -m pytest` | Result |
| --- | --- |
| `Tests/UI/test_library_prompt_collection_journeys.py Tests/UI/test_library_prompt_collections.py -q --disable-warnings --tb=short --show-capture=no` | 66 passed |
| `Tests/UI/test_library_prompt_collections.py Tests/UI/test_library_prompt_history_journeys.py Tests/UI/test_library_prompt_action_journeys.py -q --disable-warnings --tb=short --show-capture=no` | 82 passed, including 22 neighboring journeys and the same 60 collection cases |
| `Tests/Prompt_Management/test_prompt_collection_catalog.py Tests/Prompt_Management/test_prompt_collection_membership.py Tests/Library/test_library_prompts_state.py -k 'collection or membership' -q --disable-warnings --tb=short --show-capture=no` | 66 passed, 191 deselected |
| `Tests/UI/test_design_token_governance.py Tests/UI/test_css_bundle_sync_guard.py Tests/Architecture/test_library_prompts_wiring.py -q --disable-warnings --tb=short --show-capture=no` | 26 passed |

Six new production-CSS/real-SQLite journeys cover keyboard menu entry, Done,
Cancel, Apply failure/retry, persistent memberships and unchanged unsaved content,
both sizes in both themes, name collision, literal labels and unsubmitted search
drafts. Existing tests cover 207-row pagination, exact-page retry, catalog
create/rename, mutation cancellation, immutable identities, stale responses and
transactional service validation. Two baseline assertions now wait for mounted
manager readiness and separate the normal modal-return refresh from Apply's
refresh. The targeted runs report existing temporary-directory cleanup warnings.

The new journey file and native runner pass Ruff and formatting checks.
[Lint comparison](lint-results.json) shows no new diagnostics in existing files.
No token values or CSS changed. The final diff was self-reviewed.

## Native verification and persistence

The [runner](native_check.py) used the real TldwCli and LinuxDriver in an owned
tmux session, with all 12 configured storage roots inside a fresh synthetic
profile and exclusive profile ownership asserted. It created and saved a Prompt,
opened Collections from Basic, created a literal Unicode name, corrected a
collision through Rename, staged a membership, applied it while retaining a dirty
title, and canceled a subsequent membership change. The Prompt's visible Cancel
then discarded only its title draft.

[Native results](native-results.json) record successful 170×48 dark and 80×24
light journeys, normal Ctrl+Q, app.run returning, exit 0, and the observed zsh
shell. The owned session was closed. [Read-only SQLite checks](persistence.json)
confirm both Prompts retain their original names and bodies at version 1, the
renamed collections remain active, and only the applied memberships persist.
Collection version 3 includes both the rename and membership update.

All six final captures were rendered and inspected:

| State | Wide dark | Compact light |
| --- | --- | --- |
| Collision retains entered name | [Collision](collision-170.svg) | [Collision](collision-80.svg) |
| Staged membership and focused Apply | [Staged](staged-170.svg) | [Staged](staged-80.svg) |
| Applied membership and returned Manage | [Applied](applied-170.svg) | [Applied](applied-80.svg) |

The compact status line below Apply may require scrolling. Newly staged names
can appear as `Collection #ID` until Apply refreshes their labels; this existing
presentation limitation remains visible in the evidence. Native service failures
and large catalogs were not injected; targeted tests cover those paths.
No provider, full-suite or integration verification is claimed.

ADR required: no. Applies existing ADR-086, ADR-150 and ADR-161 and the TASK-198
collection contracts. Next component: Use in Console. Integration into dev is pending.
