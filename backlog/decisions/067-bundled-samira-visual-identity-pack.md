# ADR-067: Bundle Samira through a local Visual Identity bridge

Status: Accepted
Date: 2026-08-14
Related Task: [TASK-16319](../tasks/task-16319%20-%20Bundle-Samira-character-and-full-Visual-Identity-reaction-pack.md)
Related Spec: [Samira built-in Visual Identity pack design](../../Docs/superpowers/specs/2026-08-14-task-16319-samira-builtin-visual-identity-pack-design.md)
Amends: [ADR-026](026-brand-asset-source-and-export-boundary.md)
Supersedes: N/A

## Decision

Chatbook will ship Samira “Sammy” Vadem as an included, editable character card
with an immutable, bundled Visual Identity expression pack. The character card,
canonical portrait, and reaction images are application assets licensed under
`AGPL-3.0-or-later`.

This decision authorizes Samira's supplied `Sammy.png` portrait and the derived
reaction pack as production in-application assets. It narrowly amends the public
release gate in ADR-026 and the earlier Chatbook brand-mascot design. It does not
authorize the separate logo system, wordmark, advertising, merchandise, campaign
art, or campaign copy. Those uses remain behind their existing production and
clearance gates. The bundled character also does not become Chatbook's default
assistant or default Persona in this decision.

The local implementation follows the Shared Visual Identity Expression Pack
contract in `tldw_server` development commit
`385afa951922c8a9dc2002c675bb6cad65e4ac23`. Chatbook will use the server's exact
expression-key normalization rules and a four-table semantic subset that preserves
the server's field names and vocabulary for packs, immutable versions, version-bound
assets, and actor bindings. The local asset table requires `pack_version_id` and
omits the server's draft-only `draft_id` column and foreign key; the local schema is
therefore intentionally not byte-for-byte identical to the server schema. It will
not port server drafts, jobs, idempotency, APIs, or authenticated ownership in this
tranche. Chatbook's profile-local database uses a documented local owner sentinel;
local primary keys are not server IDs, and future sync must translate this subset
through an explicit adapter.

The installed package is immutable under ADR-032. Built-in reaction assets are
read from explicit package resources and are not copied into every profile. A user
edit never writes beneath the package root: the first edit forks the built-in pack
into profile-owned storage resolved through `get_user_data_dir()`. Staging occurs
inside that private destination filesystem so publication can use an atomic replace;
package reads use `importlib.resources` rather than source-tree path assumptions.
A single staged save then creates one new immutable version and updates the binding.

Samira is seeded by stable provenance rather than display name. The V2 card carries
`tldw/builtin_id: samira`; the pack carries a stable built-in identifier and content
digest in `source_context_json`. Seeding is create-only and restart-safe. It never
overwrites an existing card or pack, never restores a tombstone, and never changes a
renamed or customized card. A same-name user card causes a deterministic
“(Built-in)” disambiguation instead of replacement.

A valid active binding from Samira to a profile-owned fork is a terminal customized
state for startup seeding. It is preserved without package revalidation, rebinding,
or warning; the seed must not mistake intentional copy-on-write authoring for an
incomplete built-in installation.

Soft-deleting the Samira character does not delete or tombstone its binding. The
binding becomes dormant because resolvers reject unavailable actors, so an explicit
character restore naturally reactivates the prior relationship. Explicit pack or
binding deletion remains a tombstone and startup never recreates it.

The existing `character_expression_images` table remains the legacy operational
fallback. The new resolver returns a structured result and resolves, in order:

1. session-local manual expression override;
2. requested operational expression;
3. the bound pack's default and neutral asset;
4. the legacy character expression image;
5. the character card portrait; and
6. the existing empty/placeholder state.

Manual overrides are scoped to the active local Console session and actor. They are
not written into `messages.metadata_json` or image-generation metadata because
those stores have different ownership contracts. Durable message replay and sync
of visual identity data require a later authenticated protocol decision.

The Persona Visual Pack runtime remains separate. This decision does not merge its
operational state catalog with character Visual Identity expression packs.

## Context

Chatbook currently supports only four automatic character portrait states:
`idle`, `thinking`, `speaking`, and `error`. The latter three are stored as BLOBs in
`character_expression_images`; `idle` reuses the character card image. The current
import/export and editor surfaces therefore cannot represent the standard
28-label SillyTavern/GoEmotions reaction vocabulary requested for an included
feature demonstration.

The `tldw_server` development branch already defines a Shared Visual Identity
system with canonical expression keys, custom labels, immutable versions, actor
bindings, deterministic fallbacks, and future message replay metadata. A separate
server Persona Visual Pack system uses runtime states such as listening and tool
running. Copying that second state machine into character reactions would create
the wrong future sync boundary.

The earlier Chatbook brand-mascot design approved Samira's identity direction but
classified its concept images as non-production references and kept public release
behind later clearance. The maintainer has now explicitly authorized the narrower
in-app character and reaction use under `AGPL-3.0-or-later`; broader brand uses are
unchanged.

This decision requires an ADR because it changes the local schema, asset ownership,
versioning, binding and fallback contracts, package contents, sync boundary, and a
previous public-release decision.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Expand `character_expression_images` to 31 state BLOBs | It preserves a character-specific legacy table, duplicates large immutable assets per profile, and moves away from the server's shared pack/version/binding model. |
| Port the complete server Visual Identity subsystem | Draft jobs, APIs, authenticated ownership, idempotency, animation, and VN integration are not needed to demonstrate a bundled local pack. |
| Reproduce only the server's four public tables byte-for-byte | The server asset table references its draft table even when an asset is version-bound; omitting drafts while retaining that foreign key produces an invalid local schema. A documented semantic subset preserves the sync seam honestly. |
| Reuse Persona Visual Packs | Their operational runtime-state contract is distinct and the server still treats unification as future work. |
| Store every reaction inside the V2 card extension | It would create a large non-standard card blob, weaken independent asset validation, and provide no immutable version or actor-binding boundary. |
| Copy bundled assets into every profile | It wastes storage and creates an avoidable package-to-user-data installation transaction. Immutable package reads plus copy-on-write edits are simpler. |
| Persist manual reactions in local message metadata | `messages.metadata_json` is intentionally local-only and closed to this contract; using it would imply replay and sync guarantees that do not exist. |
| Make Samira the default assistant now | It changes established ID-1 and assistant-selection behavior and is a separate product decision requested only as follow-up work. |

## Consequences

### Benefits

- New users receive a complete, visible reaction-pack demonstration without setup.
- Local data shapes and expression normalization have a defined future server seam.
- Installed assets stay immutable and are not duplicated until a user edits them.
- Existing characters and legacy three-slot expression behavior continue to work.
- Samira remains an optional character, so current default assistant behavior is
  unchanged.

### Accepted Trade-offs

- The distribution grows by the bounded portrait and 31 WebP reaction assets.
- Local repository IDs and the local owner sentinel require translation during a
  future authenticated sync implementation.
- Manual expression choices disappear when the local session ends and historical
  messages do not replay an exact past reaction yet.
- Editing a bundled pack performs a one-time profile-owned copy of its assets.
- Existing users receive the built-in card through idempotent startup seeding rather
  than a destructive migration of similarly named cards.

## Rollback Plan

- Stop seeding new Samira records and hide the new pack-aware UI while leaving the
  additive tables and existing rows intact.
- Continue resolving legacy expression images and card portraits.
- Do not delete or rewrite user-forked packs or conversations.
- A later migration may archive the built-in binding, but automatic destructive
  down-migration is prohibited.

## Links

- [TASK-16319 design](../../Docs/superpowers/specs/2026-08-14-task-16319-samira-builtin-visual-identity-pack-design.md)
- [ADR-026: Brand asset source and export boundary](026-brand-asset-source-and-export-boundary.md)
- [ADR-032: Immutable installed distribution assets](032-immutable-installed-distribution-assets.md)
- [ADR-037: Character and Persona identity separation](037-roleplay-assistant-identity-and-persona-user-profile-separation.md)
- [ADR-040: Profile-owned state and shared asset paths](040-profile-owned-state-and-shared-asset-paths.md)
- [Chatbook brand mascot design](../../Docs/superpowers/specs/2026-07-27-chatbook-brand-mascot-design.md)
