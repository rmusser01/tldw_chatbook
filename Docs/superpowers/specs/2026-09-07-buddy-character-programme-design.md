# Buddy imports and animated character expressions

Status: Playback, independent character conversion, and Petdex import implemented in the isolated branch
Date: 2026-09-07
Creator: tldw-project

## Agreed product behavior

Users can import a Petdex companion as a Buddy, and create an independent,
editable character from an existing Buddy. The character appears in the Console
character portrait area and reacts during interactions. It does not require the
floating Buddy to remain enabled, installed, or selected.

Settings → Appearance exposes Character expressions: Dynamic or Static. Dynamic
plays available animation, with static fallback. Static still changes expression
with the interaction; it freezes animation within the selected expression.
Existing reaction-off behavior retains the neutral portrait unless a manual
reaction is explicitly selected, preserving current manual-selection precedence.
Reduce motion and disabled application animations take precedence over Dynamic.

The user explicitly selected independent copies and animation with static fallback.
The Dynamic default and immediate settings application were presented before the
user requested continuation. No per-character setting override is introduced.

## Three independently reviewable parts

1. [Dynamic and Static Console playback](2026-09-07-character-expression-playback-design.md).
   Works with existing animated expression images, independently of Petdex.
2. [Buddy-to-character conversion](2026-09-07-buddy-to-character-design.md).
   Creates native character and expression data from a reviewed Buddy snapshot.
3. [Petdex import](2026-09-07-petdex-buddy-import-design.md).
   Produces native Buddy content; users can then invoke the ordinary conversion.

Implement playback first, conversion second, and Petdex third. Each part must
deliver its own working user path and targeted evidence. Petdex does not need a
new character renderer or an external CLI installation.

## Existing architecture and findings

Source inspected at Chatbook dev `d6d792ecdfe8b201e70e25a71e9b77544cbcfa49`.
The earlier local-checkout probe of `resolve_local_expression_set` against all
seven tldw-stuff finished Buddy archives returned no images: the legacy helper
reads operational states from the outer archive manifest. The same incorrect
manifest lookup remains in this clean dev revision. That is source confirmation,
not a new execution test against this revision.

The native Persona Visual importer reads `metadata/pack.json` →
`pack.visual_manifest`, validates archive declarations and checksums, and creates
private unpublished staging. The conversion must reuse that validation boundary.

Shared Visual Identity already records `is_animated`, frame count, duration and
immutable asset bytes. Its resolver returns animation capability. The Console
character controller currently prepares a single image; the Buddy widget has a
timed-frame rendering path. These are useful existing parts, not evidence that
animated Console expressions already work.

## Governance

- ADR required: yes, before implementation planning.
- Existing ADR paths: `backlog/decisions/067-bundled-samira-visual-identity-pack.md`,
  `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`,
  `backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md`,
  and `backlog/decisions/122-bundled-pixel-migu-character-and-buddy.md`.
- Reason: explicit conversion between previously separate visual contracts,
  external import trust boundaries, local ownership and long-lived motion UX.
- Playback decision: [ADR-144](../../../backlog/decisions/144-character-expression-playback.md)
  records the motion preference and rendering lifecycle.
- [ADR-074](../../../backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md)
  includes the artwork attribution and independent snapshot conversion amendments.
- [ADR-145](../../../backlog/decisions/145-reviewed-petdex-import-and-pinned-https.md)
  records the Petdex mapping, native notice carrier, and pinned HTTPS boundary.
  Explicit snapshot conversion does not introduce live runtime coupling.

Playback and conversion require no database migration. Playback uses existing
animated-image bytes and their first composited frame. The implemented conversion
preserves a bounded versioned attribution/lineage carrier through actual publication
and export; Petdex adds the canonical native Buddy artwork carrier. See each part's
verification record for executable evidence and server interoperability limits.

## Scope and preservation

The first product flow runs in Chatbook against profile-local actors. Server-backed
actors require the established explicit local-copy path; conversion never writes
to a remote actor implicitly. Native Buddy/Visual Identity exports must be tested
against the existing server import contracts before claiming interoperability.
Server-native conversion endpoints and a mirrored server-side UI setting require
a separate contract design and are not implied by this local playback preference.

Petdex discovery stays on Petdex for the first release: paste a pet URL/slug or
choose a downloaded package. No in-app public gallery, account sync, auto-update,
registry publishing, background polling, or forced CLI installation.

Project-authored collection content uses creator `tldw-project`. Imported pets
retain their actual creators and license statements. The metadata change already
published in tldw-stuff does not reattribute third-party pets.

## Avatar content work

The separate concept sheet covers woodpecker, rubber duck, werewolf, circle,
square, rhombus, octagon, triangle, trenchcoat figure, Shiba-inu, Dipsy and Ghosty.
It is a visual proposal, not twelve importable packs. Art production needs separate
consistent transparent poses and frames, native pack validation and gallery previews.
The mechanics in these specs do not depend on approval of that visual style.
New optional packs belong in tldw-stuff and do not expand application defaults.

## Verification strategy and current status

Each component spec lists discriminating tests. Product-path tests must show the
rendered result, not merely settings values or a timer invocation. Use disposable
profiles and databases. Full-suite execution requires the repository's normal
user opt-in; targeted import, database, renderer and settings tests are the default.

See the [design review](../reviews/2026-09-07-buddy-character-design-review.md)
for source-backed corrections and the encoder experiment.

The original implementation is preserved in the integration branch. Its source-branch
verification is historical; see the
[integration verification](../reviews/2026-09-10-buddy-feature-integration-verification.md)
for combined-code evidence and the remaining independent Buddy management entry points.
