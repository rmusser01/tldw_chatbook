# ADR-076: Library Lifecycle Progressive Disclosure

Status: Accepted
Date: 2026-08-20
Related Tasks: TASK-19022
Supersedes: N/A

## Decision

Library uses destination-local, profile-persisted lifecycle composition rather
than a global Beginner/Expert mode or a second onboarding wizard. The existing
application first-run wizard remains the only startup/setup owner. Library reads
the startup admission fact that a profile was created in the current run but
does not write or reinterpret the wizard's completion state.

The Library rail lifecycle is `unknown`, `starter`, `expanded`, or `graduated`.
It is stored beside the existing `library.rail_state.sections` preferences by
the same config owner and is coerced independently so corrupt lifecycle state
cannot discard section-collapse preferences. Existing profiles without the new
value default to `expanded`; a corrupt value also defaults to `expanded`.
Deleting content never moves a profile backward. `Explore all tools` moves a
starter profile to `expanded` and is remembered. Any authoritative eligible
user content moves `starter` or `expanded` to `graduated` permanently.

Each existing source owner exposes only tri-state graduation evidence:
`unknown`, `empty`, or `has_user_content`. The seam returns no records or
private content. Library may conclude `starter` only after every relevant
source in one guarded generation reports `empty`; one positive result may
graduate immediately. Bundled/system/sample content, Trash-only records,
inaccessible records, and failed or incomplete imports are not positive
evidence. When a source cannot distinguish provenance, it reports `unknown`
rather than letting Library guess from a broad count.

Evidence loads and persistence are fenced by the active profile, route, screen
lifecycle, and request generation. Late results cannot recompose or persist an
old screen. A partial failure keeps persisted state unknown but still exposes
Import, New note, Explore all tools, and explicit recovery so uncertainty does
not block first value.

Starter is a filtered presentation of the existing rail and production landing
actions, not a second navigation model. Deep links and command-palette routes
bypass starter filtering. Source canvases continue to own their own loading,
empty, filtered-zero, stale, mutation, and recovery composition. Prompt and
Skill Basic/Advanced preferences are separate profile-local values; forced
safety/compatibility presentation never overwrites them.

## Context

The full Library rail exposes many destinations before a new user has content
to browse, while several empty canvases retain list mechanics that cannot yet do
useful work. A global novice mode would misclassify technical first-time users
and burden regular users with a permanent product-wide preference. Inferring
emptiness from the broad landing snapshot would also be unsafe: its counts may
be partial, unavailable, or unable to distinguish bundled from user content.

The repository already owns Library section preferences under
`library.rail_state`, has production Import and New note flows, and uses
generation-guarded source requests. Extending those owners is smaller and safer
than adding a lifecycle framework or tutorial data model.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Global Beginner/Expert preference | User expertise is contextual, and a global mode would hide power affordances outside Library. |
| First-run Library wizard | Duplicates the application setup wizard and delays real work. |
| Graduate from any positive broad count | Counts can include bundled, inaccessible, partial, or otherwise ineligible records. |
| Recompute Starter after every deletion | Makes navigation regress unexpectedly and punishes users who intentionally clear content. |
| One generic lifecycle controller/canvas | Moves source-specific authority and recovery away from existing owners. |
| Persist lifecycle in a new settings store | Duplicates the existing Library config read/write and restart-fallback machinery. |

## Consequences

### Benefits

- New empty profiles reach Import or New note without first parsing the complete
  Library information architecture.
- Existing and technical users keep immediate access through Explore all,
  deep links, and the command palette.
- Empty-state decisions are evidence-based and cannot fabricate zero from a
  failed or partial source read.
- The change reuses existing navigation, config, focus, and async-ownership
  paths.

### Accepted trade-offs

- Source owners that lack provenance may need one narrow boolean/tri-state
  evidence query.
- Starter cannot be declared while any required negative authority is unknown.
- `expanded` and `graduated` render the same full rail but differ in whether a
  user may intentionally return to Starter.
- Source-specific empty-state improvements ship in later atomic tasks rather
  than through one shared compositor.

## Links

- [Design specification](../../Docs/superpowers/specs/2026-08-20-library-lifecycle-progressive-disclosure-design.md)
- [ADR-067: Library Top-Level Pagination Contracts](067-library-top-level-pagination-contracts.md)
