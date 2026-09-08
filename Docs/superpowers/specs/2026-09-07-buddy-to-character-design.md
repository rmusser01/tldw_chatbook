# Create an independent character from a Buddy

Status: Design review
Attribution prerequisite: implemented on `codex/buddy-import-design`; see
[verification](../reviews/2026-09-07-artwork-attribution-verification.md).
Conversion itself remains unimplemented.
Date: 2026-09-07
Creator: tldw-project
Programme: [Buddy imports and characters](2026-09-07-buddy-character-programme-design.md)

## User flow

Offer Create character from Buddy on a saved local Buddy's visual-pack detail.
Also accept an imported native Buddy archive as a conversion source. Review shows:

1. Source Buddy, creator and immutable revision/archive digest.
2. New character name, portrait and optional personality/greeting text.
3. Available animations and suggested expression mappings, playable in Dynamic or
   Static preview. Static shows the first composited frame of the encoded result;
   the character portrait has a separate source-pose choice.
4. Missing expressions, explicit fallback choices and source/license notices.
5. Create character, followed by an explicit Open in Console action.

Creation never changes the active chat character or turns the floating Buddy on/off.
Names may be edited; same-name characters are never overwritten implicitly. The
new character is an independent local snapshot with fresh identity. Updating,
disabling or deleting the source Buddy cannot affect it.

## Contract boundary

Persona Visual operational states and Shared Visual Identity expressions remain
different models. Conversion is an explicit, reviewed, one-time transformation.
It must not make either runtime resolve through the other's live binding.

Native archive validation belongs to the existing Persona Visual importer. Extract
the reusable validated snapshot boundary if needed, so reading an archive for
conversion does not require creating a dummy Persona or bypassing checksum/path
checks. Existing saved packs likewise yield a validated immutable snapshot.

Fix the legacy expression-set import route to consume this snapshot: current code
reads the archive's outer manifest instead of `pack.visual_manifest`, skips native
checksum semantics, and has a generic ZIP member budget unsuitable for some complete
Buddy packs. Detect native archives before generic ZIP handling, and apply the
native importer's existing limits. Ordinary image ZIPs retain their own limits.

## Expression mapping

| Buddy state | Suggested character expression |
| --- | --- |
| idle | neutral, and the character portrait |
| thinking | thinking |
| speaking | custom:speaking |
| error | custom:error |
| listening | custom:listening |
| Explicitly named emotions | Existing canonical normalization, reviewed for collisions |
| Other named states/reactions | Explicit custom key, preserving original label |

Do not guess that a tool action, jump or running animation denotes a particular
emotion. Custom names normalize through the existing expression-key rules. Two
source labels mapping to one key require a visible choice; do not overwrite silently.
Preserve all available supported sequences, including unmapped extras, in the
conversion review until the user explicitly excludes them. Do not claim missing
reactions were created. Optional new-image generation is a separate explicit action.

Resolve source fallbacks through the native validator. Record when a suggested
expression uses a fallback, so duplicated idle imagery is not represented as an
authored speaking reaction. Require a valid neutral/portrait selection to publish.

## Asset conversion and provenance

For each selected sequence, render frame regions/alignment on a stable transparent
canvas. Flatten the native frame selection exactly: a Persona Visual frame pointing
at an animated source image uses the native selected raster frame and does not
implicitly expand nested animations. Preserve the resulting visible timeline and
loop behavior. Encode multi-frame sequences as
lossless animated WebP using existing Pillow capabilities; verify the codec is
available. Single-frame sequences become PNG. When animation encoding is unavailable,
offer the reviewed static result with a clear explanation; do not report animation
as preserved. Encoded output must pass the native Visual Identity image validator.
Preflight the destination limits as well as source limits: Persona Visual permits
256 assets/custom-state sets while Shared Visual Identity allows 128 expression
assets. Refuse silent truncation; review must let the user reduce the selection.
Bound conversion canvas pixels and total decoded/encoded bytes before allocation.

An encoder can coalesce adjacent equal frames. Derive frame count and durations
from the decoded output, not the source list. Compare the composited pixel timeline
at source/output transition boundaries and total duration, rather than requiring
equal frame indices. Record finite-loop conversion explicitly: source loop=false
means one play; source loop=true means infinite. Normalize GIF repetition counts
and WebP loop counts to total plays at the decoder boundary. A generated sequence
that coalesces to one frame is published as a static PNG. If the codec cannot retain
the required pixels, alpha, timing or loop semantics, offer static conversion or
cancel with an explanation, not an allegedly faithful animation.

Store the chosen static portrait separately in the character's normal image field.
Keep conversion provenance in a validated namespaced entry of the existing
`source_context_json`: `tldw/buddy_conversion`, version 1. It records source identity
and revision/digest, original creator/license/source URL, conversion date, and
per-expression source state, source/output hashes and fallback use.
Only bounded known fields are consumed; no embedded instructions are executed.

This provenance is not runtime authority and does not select animation frames.
Bind per-expression records to output SHA-256. Replacing an image clears its old
conversion record; retaining an image through an edit preserves the record.

Publication and portability need explicit changes: the current editor publisher
replaces source context with local bookkeeping and stamps `SAMIRA_LICENSE`; Actor
Pack export defaults root provenance/license and import drops asset context. Those
behaviors must not be reused for foreign art. Carry source terms unchanged, including
unspecified terms; never grant AGPL or Apache merely because conversion ran here.
Add a bounded, versioned provenance/notice carrier to the portable contract used
by conversion, preserve it through export/import and editing, and include it in
archive integrity checks. Distinguish public source URLs, creator and notices from
local authority IDs, paths and credentials, which must never be exported. Source
notices must not be truncated to the current 256-character display-summary limit.
The conversion implementation plan must define this additive contract and its
compatibility fixtures before code; do not claim server notice preservation until
its native validation/import path passes. Playback alone needs no such extension.

Use the existing Actor Pack character activation boundary: privately prepare and
verify assets first, then atomically insert the character, portable identity, visual
version and binding in one outer SQLite transaction. Adapt a reviewed conversion
snapshot into that service; do not call `publish_visual_identity_candidate` inside
the transaction. That editor function rejects active transactions, and its candidate
constructor cannot create an unbound character graph. Do not publish a dummy graph.
A failure leaves no selectable half-character or binding. Reuse the Actor Pack's
owned-file cleanup and expose cleanup-pending honestly. Crash recovery may leave
unreferenced private assets but cannot leave a partially committed actor. No second
cross-store journal is needed for SQLite-owned characters.

Recheck source revision and destination authority before commit. A changed source
requires a fresh review; cancellation removes private staging and creates no actor.
No source preferences, Persona prompt, voice credentials or tool permissions are
copied. A character does not inherit the Persona user's identity.

## Acceptance and evidence

- All seven finished tldw-stuff archives produce neutral plus their actual supported
  operational expressions through the real product import path. Pixel-migu retains
  its additional poses/sequences; actual counts are asserted from the source manifest.
- Native checksum failure, path traversal, undeclared members, resource-limit excess,
  stale reviews and duplicate normalized mappings fail before publication.
- Animated pixels, transparency, timings, finite/infinite loops and static frame-zero
  previews survive conversion, native import, database reopen and export/reimport.
- Equal adjacent frames and all-identical frames prove metadata is derived from the
  output; source/destination asset and memory limits fail before oversized work.
- Retaining/replacing an expression, profile forking and export/reimport preserve
  the correct provenance without stale output hashes, invented licenses or local IDs.
- Delete/edit the original Buddy and verify the character still renders and edits.
- Inject database, staging and binding failures and confirm no partial actor leaks.
- Open the created character in Console: live operational transitions and manual
  reactions select the copied expressions; Dynamic/Static behaves as specified.
- Verify native exported visual data against server import validation in disposable
  storage. This does not claim a server conversion endpoint or live remote UI support.

ADR required: yes. Narrowly amend ADR-074 to permit explicit reviewed snapshot
conversion while retaining separate runtime/storage semantics and ADR-037 identities.
