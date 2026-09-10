# ADR-139: Independent Buddy ownership and explicit conversation/workspace bindings

Status: Accepted
Date: 2026-09-08
Amends: the Persona-required Buddy selection/seeding in ADR-074 and ADR-122,
and ADR-079 §5 automatic provisioning when the user explicitly selects None.

## Decision

Keep one visible Buddy in v1. It owns a local visual binding independent of Persona
records and explicitly follows one conversation or one workspace. Reuse existing
immutable visual pack versions/assets and import validation. Introduce a genuine
Buddy owner/binding instead of a hidden Persona or a fabricated Persona identifier.
Copy a legacy selection's binding into that owner so subsequent Persona changes or
deletion cannot invalidate the Buddy. Preserve notices and original pack metadata.
Keep private placement/settings out of portable exports.

The existing app Buddy overlay hosts presentation. ConsoleRuntime, its controller and
store continue to own execution. The management and interaction modals submit commands
with explicit target identities and project existing run/approval/transcript state.
Switching the active Console tab, workspace, modal or screen does not retarget a Buddy
or terminate a run. Normal navigation uses the existing reusable Console route from
TASK-31520; this feature does not replace application navigation or add a job service.

Workspace Persona defaults use the existing ADR-079 reference-backed setting. Resolve
it only for newly created conversations. Explicit None is distinct from unspecified;
existing/copy/move flows retain assignments. An explicitly absent workspace default
must remain absent through provisioning/backfill; omission can retain the existing
auto-create convenience. Artwork never grants Persona or tool
authority and never modifies accepted request settings.

Conversation mode allows directed text and supported voice input. Workspace mode
allows directed text and optional named spoken output, with no voice input. One
application-owned speech queue serializes Buddy announcements, retaining pause/skip/
mute independently of run state. Only explicit user responses resolve questions or
approvals; hearing speech or opening the inbox does not acknowledge everything.

An explicit conversation opener may initialize the existing headless launch runtime
and restore that exact local saved conversation with `activate=False`. Passive inbox
inspection must not construct provider gateways or agent bridges. A failed bootstrap
keeps an explicit Open Console recovery action; it never selects another conversation.

Opening a Buddy interaction also retains decision availability for that exact live
session object and binding revision. This amends the existing no-view question posture
only for explicitly opened Buddy targets: their questions, approvals and confirmations
remain parked and recoverable after the modal closes. Wake-only sessions with no
Console view or retained Buddy interaction keep their existing fail-closed posture.
This capability neither supplies fake UI setters nor approves actions. Existing
per-round owner checks, permissions, Stop and shutdown cancellation remain authoritative.
Finite decision clocks count only visible, answerable cards; the worktree review link
alone does not spend that budget. Open Console uses the existing handoff to synchronize
the selected conversation's transcript and composer.

The same user-facing behavior is the target for the later tldw_server port. Local
profile paths, DB keys, UI geometry and Textual lifecycle are not server contracts.
Multiple visible Buddies remain v2; explicit bindings avoid an implicit active-session
contract that would prevent that extension.

## Alternatives

- Keep requiring a Persona: prevents the requested artwork-only experience and makes
  deleting a prompt identity accidentally delete its UI companion.
- Fabricate a Persona identifier: obscures data ownership and leaks dummy assistant
  identities into selectors and exports.
- Follow whatever Console currently selects: makes off-screen replies ambiguous and
  can direct a private reply to the wrong conversation.
- Build a second execution system for Buddy replies: duplicates approvals, tools,
  persistence and cancellation, risking divergent behavior.
- Automatically change older conversations when workspace defaults change: violates
  explicit conversation choices and would change established assistant behavior.

## Consequences

Storage changes require versioned migration and real SQLite coverage. Selection and
migration failures must leave the previous usable selection intact. Tests must cover
source Persona deletion, notices, restart, exact-target reply routing, navigation,
explicit-None precedence, and directed speech. Server implementation follows Chatbook
validation; no cross-product storage coupling is introduced.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-09-08-console-buddy-management-design.md)
- [ADR-074](074-portable-actor-packs-and-local-persona-visual-runtime.md)
- [ADR-079](079-workspace-assistant-defaults.md)
- [ADR-094 lifecycle objective](094-console-turn-lifetime-and-navigation-boundary.md)

## Chatbook artwork implementation (TASK-32079)

ChaChaNotes schema 70 adds `buddy_profiles` and `buddy_visual_bindings`. The existing
immutable pack/version/asset tables remain shared storage. A read-only
`visual_owner_bindings` projection retains explicit, mutually exclusive Persona and
Buddy identifiers; Buddy graph identities have no Persona identifier. Publication
and rendering use the same validation and exact-identity checks for both owner kinds.

`Persona_Buddy/library.py` supplies read-only archive review, list/read/preview, native
publication, guarded Persona artwork copying, and built-in installation. Independent
copies own new pack/version rows and private file copies. Legacy selection migration
uses a unique source key, publishes first, and writes preferences before changing the
selected owner; failed attempts keep the old selection. Built-in and import records
preserve public artwork notices, and unknown license statements remain null.

## Petdex and character management entry points (TASK-32238)

Console Buddy management may stage a reviewed Petdex source without selecting or
creating a Persona. Petdex's existing bounded review produces native archive data;
the independent Buddy library retains its existing validation/publication authority.
Apply publishes staged artwork. Cancelling either review or management publishes
nothing and changes no preferences. Temporary source bytes and revalidation guards
belong to that management operation and are cleaned on cancellation or completion.

Create character reads a saved independent Buddy snapshot with the exact Buddy
owner, revision and immutable pack version. Existing Persona snapshot callers remain
supported. The ordinary character review and guarded publication create a separate,
editable local character retaining artwork attribution and expressions. It does not
apply any staged follow-target or Persona settings. The explicit character creation
action is durable even if the surrounding management form is later cancelled.

Source changes and profile/destination changes invalidate pending work. No fabricated
Persona identifier, hidden Persona, parallel importer or second character renderer
is introduced. ADR074, ADR144 and ADR145 retain the conversion, motion and Petdex
trust contracts. These entry points extend existing ownership without a migration.
