# ADR-144: Dynamic and Static Console character expressions

Status: Accepted
Date: 2026-09-07
Spec: [Character expression playback](../../Docs/superpowers/specs/2026-09-07-character-expression-playback-design.md)
Related: [ADR-067](067-bundled-samira-visual-identity-pack.md),
[ADR-074](074-portable-actor-packs-and-local-persona-visual-runtime.md)

## Decision

Present existing animated Shared Visual Identity assets in the Console character
portrait area. Add the profile-local appearance preference
`character_expression_mode`, Dynamic or Static, default Dynamic. Application
animation-off and Reduce motion override playback without changing the saved
choice. Static still follows expression selection. Automatic reactions disabled
means neutral/static unless an explicit manual reaction is selected, preserving
the existing manual override.

Use existing image bytes and animation metadata. Static displays encoded frame
zero; the character portrait remains independently editable. No database migration,
portable playback extension, external rendering dependency or state-catalog merge.

The visible avatar owns bounded preparation and a disposable timer. Preparation
runs off-thread, accounts at most 64 MiB of RGBA buffers across active/in-flight
work, and uses at most current/next renderables. Presentation is capped at 30 paints
per second. The timer does no I/O, image decoding, database access or rail remount.
Motion, geometry, session and immutable asset identity fence preparation and paint.
Hidden avatars pause; removed avatars release playback resources.

## Alternatives

- Convert every expression to static PNG: loses the requested motion and source data.
- Store a second Buddy runtime on each character: couples operational and expression
  models contrary to ADR-067/074 and complicates independent copies.
- Add a configurable per-expression poster index now: creates a portability contract
  not needed for Dynamic/Static; codec frame coalescing makes source indices unstable.
- Let each frame remount the rail avatar: risks flicker, focus/layout work and stale
  asynchronous mounts. Only content updates belong on the frame path.

## Consequences and boundaries

Existing static content and expression precedence remain compatible. Invalid or
unsupported animation falls back visibly without weakening validation. Codec and
native allocation behavior still requires measurement; the buffer limit is not a
total-process RSS guarantee.

This decision covers playback only. Buddy conversion and Petdex import remain
separate reviewed designs with their own publication/provenance decision before
implementation. They do not gain a license or cross-model write permission here.

Validation must enter Settings save/cancel and the mounted Console path, proving
distinct rendered frames, a frozen Static image, reaction changes, lifecycle cleanup,
same-asset preference updates and stale-worker rejection. Recheck ADR numbering
against dev/open work at merge time.
