# Buddy feature integration verification — 2026-09-10

Base: dev `16c72b5b1e`. Source: `codex/buddy-import-design` at `b4e460f75142b37ff3df277fd712bdb1812007f9`.

## Changes and reconciliation

The merge preserves both original design commits and playback, attribution,
conversion and Petdex implementation commits in order. It restores Dynamic/Static
Console expressions and F9 preferences, checksummed Actor Pack artwork and
hash-bound conversion lineage, reviewed independent character publication from
native/saved Persona Buddy snapshots, and bounded pinned-HTTPS/local Petdex import.

Dev retains independent Buddy ownership and revision checks, snapshot metadata,
no-follow path checks and sanitized source-read errors. The Actor Pack carrier now
uses the existing `Persona_Visual/artwork.py` record validator rather than copying
native validation into the character subsystem. Saved snapshots retain source
context, description and source kind. Native import uses one shared archive
validation helper. Publication preserves source context under current identity and
Buddy ownership checks; mapping provenance retains the source branch's restricted
scalar validation.

The older source test expected absent artwork context for unknown terms. Current
dev explicitly stores unknown terms; its assertion now verifies null creator,
license and source URL, empty notices and version 1. No imported license is invented.

ADR required: no new decision for integration. Existing ADR-074 amendments apply.
Incoming playback/Petdex ADRs are renumbered 144/145, preserving landed ADR128/134
identities. The initial scan covered 580 cached refs, maximum ADR143; all incoming
file/header/index/spec/plan/task references were updated. A final pre-commit scan
of 581 cached refs found no existing 144/145 claims. Existing independent
ownership remains governed by ADR139.

No downloaded Petdex assets or packs are committed. The two original source review
screenshots remain with their verification links. Source verification records are
explicitly labeled historical; these integration results cover combined code.

## Commands and evidence

Commands below run from `/private/tmp/chatbook-buddy-qualification` with:

```sh
export PYTHONPATH=.:packages/tldw_profile_core/src
PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
```

1. Core feature and independent Buddy regression:

```sh
$PY -m pytest -q Tests/Chat/test_character_expression_playback.py Tests/Character_Chat/test_artwork_attribution.py Tests/Character_Chat/test_buddy_conversion.py Tests/Character_Chat/test_buddy_conversion_lineage.py Tests/Actor_Packs/test_actor_pack_attribution.py Tests/Actor_Packs/test_buddy_conversion_lineage.py Tests/Persona_Visual Tests/Petdex Tests/Persona_Buddy
```

980 passed, 1 skipped, 1 failed in 192.68s. The sole failure was the obsolete unknown
artwork assertion described above. After correcting it and preserving saved
snapshot metadata, the final affected run:

```sh
$PY -m pytest -q Tests/Persona_Visual/test_buddy_snapshot.py Tests/Persona_Visual/test_native_artwork_roundtrip.py
```

27 passed, 1 skipped in 6.76s. Skip: optional collection requires
`TLDW_BUDDY_COLLECTION`; no collection qualification is claimed here.

2. Additional publication/import and architecture checks:

```sh
$PY -m pytest -q Tests/Persona_Visual/test_buddy_snapshot.py Tests/Persona_Visual/test_native_artwork_roundtrip.py Tests/Character_Chat/test_expression_set_io.py Tests/Character_Chat/test_visual_identity_publication.py Tests/Architecture/test_actor_pack_boundary.py Tests/Architecture/test_persona_visual_runtime_boundary.py Tests/Architecture/test_persona_buddy_boundary.py
```

147 passed, 2 skipped, 1 failed in 46.25s. The architecture failure is unchanged:
`Persona_Visual/builtin_pixel_migu.py` imports `Character_Chat.builtin_pixel_migu`.
Exported all Persona Visual Python sources plus that test directly from baseline
16c72b5b1e into `.superpowers/sdd/2026-09-10-buddy-feature-integration/baseline`, then:

```sh
$PY -m pytest -q -c /dev/null .superpowers/sdd/2026-09-10-buddy-feature-integration/baseline/Tests/Architecture/test_persona_visual_runtime_boundary.py::test_persona_visual_modules_stay_outside_other_runtime_and_ui_boundaries
```

Reproduced the same forbidden Character_Chat dependency in unchanged builtin_pixel_migu.py
(1 failed in 1.43s; set iteration selected local_character_persona_service first
instead of builtin_pixel_migu). Other skip is a Windows-native
fallback contract. No change to that unrelated built-in import is included.

3. UI evidence:

```sh
$PY -m pytest -q Tests/UI/test_character_expression_avatar.py Tests/UI/test_console_character_avatar.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_buddy_character_review.py Tests/UI/test_petdex_import_review.py Tests/UI/test_buddy_management_journey.py Tests/UI/test_buddy_management_modal.py Tests/UI/test_persona_buddy_app_mount.py Tests/UI/test_persona_buddy_widget.py Tests/UI/test_buddy_entry_points.py
$PY -m pytest -q Tests/UI/test_settings_configuration_hub.py -k 'appearance and not focused_input'
```

Mounted workflow selection: 262 passed, 11 failed in 379.74s. All failures were
Petdex imports of a public `MAX_NOTICE_BYTES` constant removed by Ruff during
validator consolidation. The public constant now explicitly re-exports the native
limit; a fresh-process import passes. The affected final selection:

```sh
$PY -m pytest -q Tests/Petdex Tests/UI/test_petdex_import_review.py Tests/Character_Chat/test_artwork_attribution.py Tests/Actor_Packs/test_actor_pack_attribution.py Tests/Actor_Packs/test_buddy_conversion_lineage.py Tests/Character_Chat/test_buddy_conversion.py Tests/Character_Chat/test_buddy_conversion_lineage.py
```

206 passed in 29.62s. This reruns the previously failing Petdex review cases as well
as all Petdex parsing/network/publication and shared attribution/conversion consumers.
The remaining 262 UI cases need no rerun for a restored constant re-export. They
include independent Buddy management journey/modal, app mounting, widget lifecycle,
entry points, Console expression rendering and Buddy character review.
Appearance: 14 passed, 404 deselected in 31.60s, including expression selection,
save and revert. The known focused-input width case is unrelated to this restoration
and was excluded; the historical source verification documents its baseline issue.

The initial UI run included the entire Settings hub module and was manually
interrupted after 107 passed in 244.71s to narrow to the affected Appearance cases.
No interrupted-run completion claim is made. Separate focused save/motion selection
was 2 passed, 416 deselected in 6.61s. Final selections are recorded below.

4. Static/build checks:

```sh
$PY -m tldw_chatbook.css.build_css
$PY tldw_chatbook/css/check_bundle_sync.py
```

Both pass; generated widget defaults are updated and all ten generated CSS outputs
reproduce. Ruff diagnostic comparison over 58 modified Python files: 808 current,
810 baseline, zero added diagnostics. Reconciled artwork/snapshot/export/test files
pass Ruff check and format check. Python 3.12 AST parsing passes for all 58 files,
with no duplicate top-level definitions. `git diff --check HEAD` passes.

Targeted runs only; no full-suite or physical-terminal claim. Tests used the existing
main venv with this worktree and profile-core source on PYTHONPATH. Environment
warnings include requests dependency-version mismatch and pre-existing pytest temp
cleanup warnings. Live Petdex fetch/save and server compatibility qualification
belong to the root task, not this integration evidence.

## Remaining independent-management seams

- `UI/Navigation/buddy_management.py:BuddyManagementCoordinator` and
  `Widgets/Persona_Widgets/buddy_management_modal.py:BuddyManagementModal` only
  accept native archive paths; they expose neither Petdex review nor Create character.
- `Petdex/review.py:review_petdex_import` captures the Personas screen's
  `_persona_visual_authoring`, saved local Persona snapshot, generation and guard.
  Its reviewed result has source and native archive bytes, reusable by an independent
  management coordinator with an explicit captured profile/destination guard.
- `Persona_Buddy/library.py:BuddyLibrary.review_archive` and `publish_review` already
  own independent Buddy creation. Route reviewed Petdex native content through those
  guards rather than creating a dummy Persona.
- `Persona_Visual/snapshot.py:read_saved_buddy` still accepts a Persona ID, while
  `PersonaVisualRepository.get_active_persona_pack_for_export` supports `buddy_id`.
  Extend the saved source boundary with an explicit independent owner/revision guard
  before creating characters from independently saved Buddies.
- `UI/Persona_Modules/buddy_conversion.py` currently captures Persona workbench state;
  its `BuddyCharacterReviewDialog` and existing Actor Pack activation can be reused by
  management with independent guards and explicit Console navigation. Native archives
  already work via Characters Import without creating a dummy Persona.

These are genuine missing independent-management user journeys, not fulfilled by
Persona-only paths. The next task must amend ADR139 and qualify install/offline
reload/character creation from independent management.

## Supplemental command audit

The initial affected red snapshot probe was the same snapshot/native roundtrip
selection above with `-x`: 25 passed, 1 skipped, 1 failed in 6.22s. Its obsolete
assertion was subsequently fixed and the final green run is recorded above.
The interrupted UI command was exactly the final mounted selection above with
`Tests/UI/test_settings_configuration_hub.py` inserted immediately after
`Tests/UI/test_settings_appearance_defaults.py` (107 passed before KeyboardInterrupt).
The supplemental Settings command was:

```sh
$PY -m pytest -q Tests/UI/test_settings_configuration_hub.py -k 'appearance and (save or cancel or expression or motion)'
```

2 passed, 416 deselected in 6.61s; the later broader Appearance run supersedes it.
Raw execution logs are retained under `/private/tmp/buddy-integration-*.log`; the
root handoff report lists their location. No full-suite inference is made from
these overlapping targeted selections.

## Review fix: native metadata roundtrip

Review baseline: `028f7358b5`. Fixed the native exporter to serialize the snapshot
description and default to its validated source context. An explicitly supplied
context, including `{}`, replaces that default; validated artwork terms remain
attached. The saved exporter now uses that pinned snapshot directly, removing its
redundant second context read.

Investigation found matching loss at both inbound boundaries: `_pack` discarded
description, the import draft replaced it with fixed text, and the archive snapshot
left description unset. The fix carries a present description through all three
using the existing authoring 4096-character/UTF-8 validation. Missing-description
archives retain existing defaults: empty snapshot description and the import
review's `Imported Persona Visual pack` label. Null, non-string and oversized
values fail before snapshot/review. This narrow expanded scope is necessary for
an actual roundtrip, and was explicitly authorized by root. No new ADR is required
for this preservation bug fix. Also removed the blank line breaking the ADR144/145
index table.

The regression exercises native snapshot reading and review, default context,
explicit replacement and empty overrides, invalid metadata, missing-description
compatibility, and real import/save/edit/saved-export/reimport.

Exact commands (same worktree and venv/PYTHONPATH as above):

```sh
PYTHONPATH=.:packages/tldw_profile_core/src /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Visual/test_native_artwork_roundtrip.py -k 'description or native_import_save_edit'
PYTHONPATH=.:packages/tldw_profile_core/src /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Visual/test_native_artwork_roundtrip.py Tests/Persona_Visual/test_buddy_snapshot.py Tests/Persona_Visual/test_persona_visual_importer.py Tests/Petdex Tests/UI/test_petdex_import_review.py
```

Before implementation: **7 failed, 11 deselected in 3.39s**, reproducing the missing
metadata and validation. After the fix: **186 passed, 1 skipped in 25.81s**. The
skip is the optional collection probe; existing requests/temp-cleanup warnings
remain disclosed. Logs: `/private/tmp/buddy-metadata-red.log` and
`/private/tmp/buddy-metadata-green.log`.

Ruff check passes for exporter, snapshot and regression test. The importer's five
pre-existing diagnostics are unchanged (zero added, compared with HEAD). All four
modified Python files pass `ruff format --check`; the owned diff passes
`git diff --check`. Root-owned Task32238, integration plan and ADR139 are excluded.
