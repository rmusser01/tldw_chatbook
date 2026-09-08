# Create an independent character from a Buddy

Status: Design review
Date: 2026-09-07
Creator: tldw-project
Programme: [Buddy imports and characters](2026-09-07-buddy-character-programme-design.md)

## User flow

Offer Create character from Buddy on a saved local Buddy's visual-pack detail.
Also accept an imported native Buddy archive as a conversion source. Review shows:

1. Source Buddy, creator and immutable revision/archive digest.
2. New character name, portrait and optional personality/greeting text.
3. Available animations and suggested expression mappings, playable in Dynamic or
   Static preview, with a selectable representative frame.
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
canvas. Preserve order, duration and loop behavior. Encode multi-frame sequences as
lossless animated WebP using existing Pillow capabilities; verify the codec is
available. Single-frame sequences become PNG. When animation encoding is unavailable,
offer the reviewed static result with a clear explanation; do not report animation
as preserved. Encoded output must pass the native Visual Identity image validator.

Store the chosen static portrait separately in the character's normal image field.
Keep the expression's preview index in a validated namespaced entry of the existing
`source_context_json`: `tldw/buddy_conversion`, version 1. It records source identity
and revision/digest, original creator/license/source URL, conversion date, and
per-expression source state, source frame hashes, preview index and fallback use.
Only bounded known fields are consumed; no embedded instructions are executed.

Expose the validated preview index to the local playback DTO. A missing or invalid
index defaults to frame zero. The animation retains its original order. Export
preserves source attribution and conversion metadata; consumers without that
metadata can still display the ordinary animated image and first-frame fallback.

Create character rows, immutable visual version and binding through existing local
repositories/publication services under a coordinated transaction. Stage assets in
the private destination filesystem before publication. A failure must leave no
selectable half-character or binding; clean unpublished assets with the existing
publication recovery/cleanup mechanism. Do not introduce a second cross-store journal.

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
- Animated pixels, transparency, timings, finite/infinite loops and selected static
  preview survive conversion, native import, database reopen and export/reimport.
- Delete/edit the original Buddy and verify the character still renders and edits.
- Inject database, staging and binding failures and confirm no partial actor leaks.
- Open the created character in Console: live operational transitions and manual
  reactions select the copied expressions; Dynamic/Static behaves as specified.
- Verify native exported visual data against server import validation in disposable
  storage. This does not claim a server conversion endpoint or live remote UI support.

ADR required: yes. Narrowly amend ADR-074 to permit explicit reviewed snapshot
conversion while retaining separate runtime/storage semantics and ADR-037 identities.
