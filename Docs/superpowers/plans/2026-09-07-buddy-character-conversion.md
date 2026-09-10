# Buddy character conversion implementation plan

> **For agentic workers:** Use subagent-driven-development with bounded file ownership and integrated review. Execute continuously.

**Goal:** Create independent editable characters from saved Buddies or native archives through a reviewed mapping and preview.
**Architecture:** Validate immutable native snapshots, convert explicitly selected sprite timelines into ordinary image assets, and publish through existing Actor Pack activation. Runtime bindings stay separate.
**Tech Stack:** Python, existing Pillow/WebP, Textual, SQLite, native Actor Pack services.
**Spec:** [Approved conversion design](../specs/2026-09-07-buddy-to-character-design.md).

ADR required: yes
ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
Reason: reviewed snapshot conversion and portable lineage across otherwise separate runtime boundaries.

## Interfaces and global constraints

- `Persona_Visual/snapshot.py`: `BuddyAssetSnapshot(metadata, data)` and `BuddySnapshot(title, manifest_json, assets, artwork, source_sha256, _guard)`; frozen snapshots expose `is_current()`. `artwork` is the validated pack-level tldw/artwork record (original creator/terms or explicitly unspecified). `_guard` never exports. `read_buddy_archive(path)` and `read_saved_buddy(repository, persona_id, profile_root)` return snapshots. Reuse native importer checks; no dummy Persona.
- `Character_Chat/buddy_conversion.py`: `suggest_buddy_mappings(snapshot)` returns mapping rows with `source_state`, `expression_key`, `fallback`, `frame_count`; `convert_buddy(snapshot, mappings=None, animate=True, portrait_state="idle", portrait_frame=None)` returns `BuddyConversion` with `snapshot`, `expressions`, `portrait`, `warnings`. Each expression exposes `source_state`, `expression_key`, `data`, `metadata`, `fallback`. `metadata` is the native image metadata dictionary.
- `publish_buddy_character(conversion, *, name, personality="", first_message="", db, local_service, profile_root, authority_guard)` returns existing `ActorPackActivationResult`. Prepare a private Actor Pack, then consume the real import/activation path. Revalidate source and destination in its transaction, never publish a dummy actor.
- Memory: serialize conversions; reserve decoded source canvases plus retained output frames before allocation, total 64 MiB; native image and pack byte/pixel limits still apply. Reject excess selection instead of truncating. Native nested animations use selected raster frame zero. Alignment is normalized on stable transparent canvases.
- Loop false means one play, loop true infinite. Validate WebP output composited timelines and total duration, including encoder-coalesced frames. Coalesced single frames become PNG. If unsupported, return explicit static fallback warning. No source image regeneration.
- Source operational states map idle→neutral, thinking→thinking, speaking/error/listening→custom keys. Canonical emotion normalization applies; other states and unused animations retain reviewed custom keys. Duplicate keys reject before encoding. User exclusions are explicit null mappings.
- Portrait choice is independent of expression Static frame zero. Global animation/reduce-motion policy suppresses preview animation as it does Console playback.
- `tldw/buddy_conversion` v1 per-image record: exact keys version, source_sha256, source_state, source_asset_sha256 (sorted unique list), converted_at (UTC ISO), fallback (boolean), output_sha256. Public digest replaces local identity. Creator/terms stay in tldw/artwork. Known records validate and bind to output hashes; replacements clear them.
- Carry lineage using attribution carrier version 2 with an additional `conversions` expression map and required feature `visual-buddy-conversion/v1`; attribution/v1 remains required. Existing v1 output is unchanged when no lineage exists. Older readers reject the new required feature. Native visual manifests stay unchanged; no server-side preservation claim without testing.

## Tasks

### 1. Validated source snapshots
Owner files: Persona_Visual snapshot/importer, Character_Chat/expression_set_io.py, related source tests.
- [x] Write failing archive integrity/saved identity tests and real-seven-archive probes.
- [x] Extract reusable native validation; return immutable bytes and a current-source guard, known public artwork only.
- [x] Route legacy native expression extraction through native validation before generic ZIP limits.
- [x] Run native importer/snapshot/expression IO tests; report exact unsupported source metadata without inventing terms.

### 2. Timeline conversion, lineage and publication
Owner files: Character_Chat/buddy_conversion.py and artwork_attribution.py, Actor_Packs carrier integration, dedicated tests.
- [x] Write red tests: two-frame pixel/timing loop equivalence, identical frame coalescing, collision, fallback, canvas budget, portrait selection, atomic publication and stale source.
- [x] Implement bounded sequential frame conversion and output verification. Use existing image validation before building an Actor Pack.
- [x] Add validated portable lineage and retain/drop semantics through existing context projection.
- [x] Publish through private Actor Pack staging and existing character transaction; exercise reopen/edit/export/reimport and injected failures.

### 3. Review and product entry points
Owner files: a focused Buddy conversion dialog/controller, Personas visual widget/screen integration, UI tests.
- [x] Add Create character from Buddy for saved packs and a native-archive source option; disable saved conversion for dirty drafts.
- [x] Show name/personality/greeting, every suggested mapping including fallback labels, explicit exclusions, portrait selection and Dynamic/Static previews. Review errors keep the dialog editable; no implicit overwrites.
- [x] Workers own decoding/publication; bind source and destination authority across asynchronous work, and remove previews on close.
- [x] Creation does not change Console character or Buddy settings. Offer explicit Open in Console using the established character action.
- [x] Test compact mounted review, cancellation/stale destination, and actual created-character selection path.

### 4. Integration and review
- [x] Run targeted source/conversion/Actor Pack/UI tests and all seven finished collection archives.
- [x] Inspect native server validation in disposable storage if available; qualify any unsupported portable server contract.
- [x] Review against spec, fix actionable findings, record evidence, update Backlog/ADR and commit.

## Execution ledger

Ruling: Conversion may reuse the existing Actor Pack archive-review boundary internally. This preserves the established atomic transaction and cleanup authority without exposing a second confirmation step to users.
Ruling: The additive lineage carrier uses a new required feature, so older readers cannot silently drop it. Artwork-only archives keep their original v1 carrier.

Ruling: Character Library Import also admits native Buddy archives directly, so
conversion never requires creating or selecting a dummy Persona. The existing
shared picker pins destination authority before user selection.
Ruling: Visible-pixel verification requires exact alpha and exact RGB at nonzero
alpha; fully transparent RGB is immaterial. All seven actual collection archives
now encode without fallback warnings and publish independently.
Ruling: Final source/destination validation occurs after visual activation writes
inside the existing character transaction; failures use native cleanup/rollback.
Ruling: The current server lacks the lineage carrier contract. Read-only inspection
is recorded instead of claiming a server roundtrip. No server code was changed.

Execution evidence: [verification record](../reviews/2026-09-07-buddy-character-conversion-verification.md).
