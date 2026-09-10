# Buddy and character design review

Date: 2026-09-07
Reviewer: Codex
Scope: four programme specs on `codex/buddy-import-design`
Code baseline: `d6d792ecdfe8b201e70e25a71e9b77544cbcfa49`

## Findings and resolutions

| Severity | Finding | Correction |
| --- | --- | --- |
| High | The proposed atomic character creation called a publisher that rejects an active transaction and cannot start an unbound character graph. | Use the existing Actor Pack character activation transaction, with assets prepared before the transaction. No dummy character/pack or nested editor publication. |
| High | Editor materialization stamps `SAMIRA_LICENSE` and replaces asset provenance; Actor Pack export/import does not retain the proposed conversion context. | Explicitly include publication and versioned portable notice/provenance preservation in conversion scope. No imported asset is automatically relicensed. This must be specified before conversion implementation planning. |
| High | Animation preparation's 64 MiB wording did not constrain allocations before caching or concurrent old/new frames. | Preflight buffers, decode sequentially, downscale immediately, account active plus in-flight buffers and retain only current/next renderables. Test native codec memory separately. |
| Medium | Source preview-frame indices can become invalid after encoding. | Use encoded frame zero for Static expressions; retain an independent selectable character portrait. Avoid a new playback schema just for a poster index. |
| Medium | Encoding can merge frames and change reported counts without changing the visible animation. | Validate composited timelines and durations, derive metadata from output, and emit PNG for a sequence that becomes one frame. |
| Medium | The Petdex v2 proposal inferred row meanings/counts/timing from geometry. | Only use supported declarations or a tested version mapping; otherwise require explicit manual mapping before publication. Reject conflicting declarations. |
| Medium | A motion-setting save could be ignored when the resolver's asset identity stayed unchanged. | Include effective motion and geometry in render-generation invalidation and test a same-asset settings change. |
| Medium | Reaction-off wording would regress the Console's existing manual override. | Preserve explicit manual-reaction precedence; automatic-off with no manual reaction stays neutral/static. |
| Medium | Destination expression capacity is lower than the Buddy source's capacity. | Preflight both sets of limits; require a reviewed reduction instead of dropping excess expressions. |
| Medium | Network and provenance rules omitted decoded HTTP limits, actual connection validation and local notice files. | Bound decoded streaming bodies, preserve notices as data and recheck destination authority across asynchronous stages. The existing egress guard explicitly excludes DNS pinning; require a tested connection-pinning adapter before enabling remote import. |

## Source evidence

- `Character_Chat/visual_identity.py`: `create_visual_identity_candidate` only
  offers an empty graph for Personas; `publish_visual_identity_candidate` rejects
  `connection.in_transaction`. `_materialize_visual_identity_candidate` replaces
  asset context and sets `license` to `SAMIRA_LICENSE`.
- `Actor_Packs/activation.py`: `_activate_character` already inserts character,
  portable identity and shared visual binding within one outer transaction.
- `Actor_Packs/export.py`: root license/provenance defaults and `_bounded_provenance`
  are not a complete carrier for source URLs or long copyright/license notices.
- `UI/Console_Modules/character.py`: `_current_request` gives explicit manual
  reactions precedence; asset-identity cache alone does not represent motion settings.
- `Persona_Visual/contracts.py` and `Character_Chat/visual_identity.py`: different
  asset/frame and decoded-image budgets must both be enforced.
- Petdex `src/lib/sprite-atlas.ts` proves geometric versions, while the inspected
  `pet-states.ts` only establishes the nine-row state table. Geometry is not semantics.
- `Utils/egress.py` documents DNS-rebinding IP pinning as a non-goal: the HTTP
  client resolves again after the policy check. Existing guarded fetches alone do
  not satisfy the proposed actual-destination guarantee.

## Executed probe

Used installed Pillow's WebP encoder with synthetic 4×4 frames: red, red, blue,
durations 100/200/300 ms, lossless encoding, infinite loop. The decoded output had
two frames: red for 300 ms and blue for 300 ms. This directly demonstrates why
source frame count/index equality is not a valid encoding invariant. No source
artwork, application profile or database was changed by this probe.

## Remaining implementation risks

No live Console animation, Petdex download, conversion publication or server
round-trip was executed in this review. Those remain required implementation
evidence. Codec loop/disposal differences, bounded render allocations, asynchronous
authority changes and portable license preservation need targeted tests.

The playback design can advance independently. The conversion plan must first
define and test its versioned notice/provenance carrier; it must not assume the
existing string metadata fields preserve everything. The Petdex plan must carry
real v1/v2 metadata fixtures or the explicit manual-mapping fallback.

No runtime changes were made. The design corrections preserve the user's chosen
Dynamic/Static experience and independent-character ownership.
