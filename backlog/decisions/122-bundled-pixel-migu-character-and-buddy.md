# ADR-122: Bundle pixel-migu as an optional character and Buddy

Status: Accepted
Date: 2026-09-05

## Decision

Ship the maintainer-supplied pixel-migu sprite assets as selectable application content. Seed a separate character with 18 Shared Visual Identity expressions and a local Persona with its operational Persona Visual animations. Follow ADR-067 for immutable character resources and create-only provenance, ADR-074 for the Buddy runtime, and the existing Actor Pack coordinator for cross-store Persona ownership. No schema or remote synchronization change is introduced.

The maintainer explicitly requested inclusion in both application repositories. Preserve the supplied-art provenance (LicenseRef-User-Supplied) and source checksums; this decision does not assert a new artist identity or grant unrelated trademark or merchandise rights.

Seeding must never change the current assistant, Persona, or Buddy preference. Persisted built-in identities, including tombstones and user edits, are terminal. Character card/pack insertion is atomic. Buddy installation uses the existing profile-owned Persona Visual boundary and coordinated Persona creation/recovery; the immutable packaged source remains untouched. First-install work runs in the existing background startup/readiness flow, not a new synchronous application-constructor path.

## Alternatives

Manual imports leave fresh-install users without a selectable entry. Reusing the character expression pack for the Buddy conflates two established runtime contracts. Replacing the active assistant or existing marker Migu is outside the request.

## Validation

Exercise fresh profiles, all 18 character expressions and baseline Buddy states, restart preservation, rollback, and resources from built wheel/sdist artifacts.
