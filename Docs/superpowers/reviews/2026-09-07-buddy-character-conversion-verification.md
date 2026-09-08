# Buddy character conversion verification

Task: TASK-32025. Branch: `codex/buddy-import-design`.
ADR: [ADR-074](../../../backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md).
Plan: [Implementation plan](../plans/2026-09-07-buddy-character-conversion.md).

## Implemented boundary

Saved local Buddy graphs and native archives produce pinned, immutable snapshots.
The native import validator owns archive declarations, paths, checksums and image
limits; conversion never creates a dummy Persona. Damaged native archives cannot
fall through to ordinary image-ZIP import.

The review preserves supported states and unused animations, exposes fallback
labels and requires explicit mapping collision resolution. Conversion uses bounded,
serialized frame composition with native cropping and alignment. Static expressions
use frame zero; portrait selection independently honors the native preview pose or
an explicit frame. WebP verification compares visible pixels, exact alpha, timing
and loop behavior, allowing adjacent identical frames to coalesce. RGB values under
fully transparent pixels carry no visible content and do not trigger a fallback.

Publication goes through native Actor Pack review and activation. Source and local
destination checks run before and after writes inside the owning SQLite transaction.
Injected post-write failures roll back the actor, visual graph and identity and
remove newly published files. Private staging cleanup failure after a successful
commit returns the committed identity with `cleanup_pending`; it does not claim
creation failed or invite duplicate publication.

Carried artwork terms and asset-bound conversion lineage survive an independent
copy, reopen, retained-image edit, export and reimport. Image replacement clears stale
lineage. Attribution carrier v2 requires `visual-buddy-conversion/v1`; artwork-only
v1 packs remain unchanged. The lineage contains public content digests and source
state names, never local paths or source Persona IDs.

## Evidence

- 82 tests passed together across snapshots, conversion, lineage and mounted review/
  pack controls. This includes real SQLite publication and export/reimport, post-write
  invalidation, binding failure, cleanup failure, collision recovery, transparent
  pixels, loop counts, identical-frame coalescing, portrait independence and limits.
- 101 additional targeted regression tests passed across native import, legacy
  expression import, existing artwork attribution and CSS bundle synchronization.
- All seven published collection archives were exercised through real decoding and
  conversion. The six basic packs each produce six expressions, five animated;
  pixel-migu produces 31 expressions, 12 animated. No animation fallback warnings.
  All seven also published through real Actor Pack activation in a disposable local
  profile with matching expression counts and no cleanup pending. The optional
  collection test accepts a `TLDW_BUDDY_COLLECTION` directory rather than depending
  on a developer-specific checkout path.
- Final targeted UI/import/CSS coverage: 51 passed; after the final unavailable-profile
  recovery fix, 20 focused UI/import tests passed. The collection snapshot probe
  passed all 16 cases with its explicit collection directory configured. These runs
  overlap earlier selections and are not summed into a unique-test claim.
- Independent implementation review found and fixed synthetic state-name collisions,
  missing final transaction validation, clipped warnings and the asynchronous
  character-hydration race before Console handoff.
- New and focused modules pass Ruff lint/format checks. Existing large modules retain
  their unrelated lint debt; a comparison with `a552b8eda` found no new Ruff
  diagnostics. Changes do not claim a repository-wide cleanup.

The targeted runs emit the existing Requests dependency-version warning. No full
repository sweep was requested or run. The unrelated five Actor Pack baseline failures
remain documented in [the attribution verification](2026-09-07-artwork-attribution-verification.md);
they are not presented as new regressions or silently reclassified as passes.

## Product checks and limits

Mounted tests use the real review dialog, converter, local database and established
Personas character-selection and Console handoff path. They cover both saved and
archive sources, reduced-motion preview, explicit navigation and preservation of the
existing Buddy/chat selection until the user chooses Open. Warning text is scrollable
and acknowledgment gates publication. Compact pixel-migu previews were also rendered and inspected:
[expression](../qa/buddy-character-conversion/expression-preview.png) and
[portrait](../qa/buddy-character-conversion/portrait-preview.png).
These are headless Textual UI tests, not a
claim of physical-terminal or live-provider chat verification.

The seven current collection archives contain no embedded creator/license/notices
metadata. Conversion displays unspecified terms for those sources instead of inferring
attribution from filenames or adjacent README files. Source records that do contain
terms preserve their exact validated public fields.

Read-only inspection of tldw_server at `27cd02755563d326d218787ac60e0bc8e0524cb8`
found no Actor Pack or lineage-v2 contract. Its generic visual archive importer skips
JSON metadata and writes image assets without the carried context. Server preservation
is unsupported and was not claimed or tested as a successful roundtrip. Petdex URL
import and generation of additional Buddy artwork remain separate work.
