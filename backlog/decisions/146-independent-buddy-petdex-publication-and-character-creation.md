# ADR-146: Independent Buddy Petdex publication and character creation

Status: Accepted
Date: 2026-09-10
Task: TASK-32238
Extends: ADR-139's management entry points; its independent ownership, binding and
Console execution decisions remain in force.
Partially supersedes: ADR-145, only for its saved-Persona-only Petdex destination.
The saved Persona destination and every source, transport, review and attribution
decision remain in force.

## Decision

Console Buddy management may stage a reviewed Petdex source without selecting or
creating a Persona. The existing Petdex review produces unpublished native archive
data, and the independent Buddy library keeps authority over validation and
publication. This adds an independent Buddy destination to ADR-145; the saved local
Persona destination remains available through Persona authoring.

ADR-145's source and transport trust decisions remain unchanged. Local and remote
sources converge on the same bounded review, remote requests retain pinned HTTPS and
host restrictions, source bytes and identities remain guarded through publication,
and creator, source, terms and mapping provenance remain attached to the published
artwork. No downloaded instructions execute and no unvalidated path or network
fallback is introduced.

Preparation alone installs nothing. Cancelling Petdex review or Buddy management
before Apply publishes nothing and changes no preferences. Apply publishes reviewed
artwork before persisting the selected Buddy and presentation settings. Those file
and settings writes are intentionally not one atomic transaction: if publication
succeeds and settings persistence fails, the installed Buddy remains durable while
the previous settings remain selected. Recovery reuses the installed record when
the user retries the same form, or asks the user to reopen management and verify it
before importing again. Cancellation after that partial Apply does not roll back the
installed artwork.

Temporary source bytes, source guards, profile authority and retry identity belong
to the management operation. Source, selection, profile and destination changes
invalidate pending publication. Cleanup removes only operation-owned staging after
its work has drained.

Create character reads a saved independent Buddy snapshot with the exact Buddy owner,
revision, immutable pack version and verified asset bytes. The existing character
review and guarded Actor Pack publication create a separate editable local character
that retains artwork attribution and expressions. Explicit Create is durable even if
the surrounding management form is later cancelled, and it never applies staged
follow-target, Persona, motion or presentation settings.

This supplements ADR-139 without changing its independent ownership, explicit binding
or Console execution decisions. It retains ADR-074's conversion and attribution
contract and ADR-144's motion contract. No fabricated Persona, hidden Persona, second
importer, second character renderer, schema migration or live runtime coupling is
introduced.

## Alternatives

- Require a saved Persona for every Petdex import: rejected because independent Buddy
  ownership already separates artwork from prompt identity.
- Publish during preview or review acceptance: rejected because review must remain
  side-effect free until the management Apply/Persona Save action.
- Roll back installed files after a settings failure: rejected because publication
  has already linearized and destructive rollback could remove a valid installed
  Buddy; retry identity safely reuses it.
- Create a Petdex-specific character path: rejected because the existing independent
  Buddy snapshot and Actor Pack conversion already provide the guarded copy boundary.

## Consequences

The UI and documentation must qualify cancellation guarantees as applying before
Apply or Save. Partial Apply errors must state that the Buddy was installed, previous
settings were retained, and the user should retry the same form or reopen and verify
the installed Buddy before importing again. Publication errors remain path-free and
distinguish changed sources, which require a fresh review, from actionable profile
storage failures.

Tests cover review and management cancellation, source/profile/selection changes,
publication followed by settings failure and retry reuse, independent character
durability, preserved attribution and Dynamic/Static playback. Platform-capability
tests keep local Petdex folder, `pet.json` and ZIP sources available without POSIX
descriptor flags while retaining link and changed-source rejection.

## Links

- [ADR-074](074-portable-actor-packs-and-local-persona-visual-runtime.md)
- [ADR-139](139-independent-buddy-conversation-and-workspace-bindings.md)
- [ADR-144](144-character-expression-playback.md)
- [ADR-145](145-reviewed-petdex-import-and-pinned-https.md)
- [Buddy character programme](../../Docs/superpowers/specs/2026-09-07-buddy-character-programme-design.md)
- [Integration plan](../../Docs/superpowers/plans/2026-09-10-buddy-feature-integration.md)
