# ADR-243: Source-bound portrait component compositing

Status: Accepted for planning; implementation not started
Date: 2026-10-24
Related tasks: TASK-36551, TASK-36552, TASK-36553, TASK-36554, TASK-36555
Amends: [ADR-242](242-console-radio-portrait-performance.md), generation/authoring portions only
Design: [Console radio portraits](../../Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md)
Plan: [Compositing implementation](../../Docs/superpowers/plans/2026-10-24-portrait-compositing-implementation.md)
Implementation clarifications: [Three findings resolved and revised-plan review](../../Docs/superpowers/reviews/2026-10-24-portrait-compositing-plan-review.md#revised-plan-review-after-three-findings). Existing decision statements remain unchanged.

## Context

User review rejected independently generated six-pose artwork because exposure
varied and mouth shapes did not match across blink pairs. Referencing the same
source did not prevent broad model retouching. Explicitly authorized local
compositing then produced exact pairs from one original base and shared patches;
full-resolution review and the mounted app journey passed. The user now requests
this approach inside generation and accepts one-time manual region selection.

## Decision

1. Keep the existing six playback roles, renderer, speech owners and explicit Apply.
   Generation prepares three components: closed eyelids, small mouth and open mouth.
   Every edit references one bounded, metadata-free prepared base. A confirmed base
   with open eyes and closed mouth supplies `rest` without a paid request. Sources
   not meeting that requirement remain eligible for ordinary six-frame imports;
   there is no implicit fourth request or source replacement.
2. Mark three non-overlapping rectangles in prepared-source pixel coordinates:
   image-left eye, image-right eye and mouth. Provide two-corner pointer selection,
   exact numeric keyboard adjustment and detailed mask review, including a fallback
   for terminals without graphics. Do not introduce face detection or an inpainting
   provider contract. Request whole reference edits through the existing capability
   gate, then discard their pixels outside the masks locally.
3. Use a versioned, deterministic inward feather mask. Preserve the base alpha
   everywhere and all RGBA pixels outside mask support exactly. Compose each raw
   component onto the base once; assemble blink combinations by copying the finished
   eyelid region onto finished mouth planes without feathering twice. Enforce the
   exact paired-mouth/paired-eyelid invariants on generation and recovery of reusable
   components. These checks guarantee pixel relationships, not facial quality.
4. Save regions explicitly and independently of performance Apply. One current
   bounded recipe per local character/expression stores exact PerformanceSource,
   prepared-base digest/dimensions, three rectangles, algorithm version, neutral-base
   confirmation and an independent monotonic revision. Use a new character-FK table
   and the next executed schema migration; allocate its version at implementation.
   Source/profile/editor/revision checks match existing authoring authority. A source
   change requires explicit selection/save against that source; old coordinates are
   not silently scaled or transferred. Saving settings changes no active artwork. An unchanged current-revision Save is
   a no-op; stale-revision saves still conflict.
5. Add optional bounded composition-recipe provenance to the existing performance
   parent and typed record, defaulting to absent for all existing/manual sets. Apply
   validates source, recipe/invariants and region revision, then stores provenance
   and all six frames in the same existing transaction. Store no raw provider images
   or new frame blobs beyond the six existing frames. On reopen, `small`, `open` and
   `blink_rest` are sufficient finished component planes if the stored recipe/base
   and all six pixel invariants validate. Never feather those recovered planes again.
   Invalid/missing provenance disables component reuse, not ordinary valid playback.
6. Regenerate a component, not an independent pose: small/open mouth replaces its
   two eye variants; eyelids replaces all three blink variants. Stage affected groups
   atomically and preserve unaffected frames. A full generation starts a fresh
   candidate after consent; partial successes remain within that candidate. Apply
   requires a complete valid set. Uncertain paid calls are never retried automatically.
   A successful individual frame import or explicit clear switches the draft to manual frames, clears
   composition lineage and preserves saved regions; existing six-slot import remains.
   Cancelled/invalid imports preserve the previous frames and component lineage.
7. Full-profile backups carry settings/provenance. Soft deletion preserves them;
   permanent character deletion cascades them. Remove performance keeps saved regions
   and clears applied provenance. Existing cards, Actor Packs, Chatbooks and sync
   retain their current formats and exclude this metadata. No new dependency or
   provider is introduced; retain shared memory budgets, off-thread image work,
   exact async owner fencing, source originals and existing scoped TTS cancellation.

## Alternatives

- Stronger six-frame prompts: already failed real user review; cannot enforce pairs.
- Automatic landmarks/segmentation: extra model/runtime and correction controls;
  style/pose reliability remains uncertain. User accepts manual selection.
- Provider-native masks: uneven support and a broader adapter contract; local
  compositing works with existing reference-capable providers.
- Session-only regions: violates reuse after reopening. External sidecars or portable
  recipe exports add unnecessary ownership/backup/file-format boundaries.
- Persist raw edits or extra component PNGs: unnecessary; verified finished planes
  can be recovered from the existing six-frame set without repeated blending.

## Consequences and limits

Three edit requests replace the original six; destination/cost disclosure and retry
labels must change together. Saving masks is a new explicit local write with its own
CAS revision, not performance publication. A schema migration, SQL allowlist update,
backup/fresh-open/edit test and export regression are required. Existing databases
upgrade without changing frame bytes or performance revisions.

Wrong region bounds, moved generated features or local tonal mismatch can still
produce poor seams. Show masks and a full-resolution 2×3 review as well as the small
Console preview. Refuse wrong output geometry; do not silently stretch, crop or
retarget. No generated-content semantic detector is claimed. Changing saved regions
stops only the current draft sample, invalidates disposable component work and
requires a fresh batch; the currently
applied set keeps playing. Live in-app provider qualification remains distinct from
external ChatGPT-generated patch evidence.
