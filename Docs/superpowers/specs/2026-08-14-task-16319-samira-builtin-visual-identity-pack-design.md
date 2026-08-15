# Samira Built-in Character and Visual Identity Reaction Pack Design

Date: 2026-08-14
Status: Approved for spec review
Backlog: TASK-16319
ADR: [ADR-067](../../../backlog/decisions/067-bundled-samira-visual-identity-pack.md)
Server reference: `tldw_server` `origin/dev` at
`385afa951922c8a9dc2002c675bb6cad65e4ac23`

## Summary

Ship Samira “Sammy” Vadem as an included, public V2 character card with her
canonical `Sammy.png` portrait and a complete reaction pack. The pack contains the
28 standard SillyTavern/GoEmotions labels plus `thinking`, `speaking`, and `error`,
for 31 distinct 1024×1024 WebP reaction assets.

Chatbook will bridge its current four operational character-avatar states to a
minimal local implementation of the server's Shared Visual Identity Expression Pack
model. Users can inspect the full pack, lazily preview one reaction, and set or clear
a session-local manual reaction in Console. Automatic emotion classification,
durable per-message reaction replay, server sync, and making Samira the default
Persona are separate future work.

All bundled Samira card and image assets are licensed under
`AGPL-3.0-or-later`. No packaged card, image metadata, manifest, license notice, or
generated asset may mention `VademHQ`.

## Goals

- Give fresh and upgraded profiles one included Samira character card without
  replacing similarly named or user-edited cards.
- Demonstrate a full character reaction pack through a discoverable, usable UI.
- Preserve the exact 28-label standard vocabulary while aligning internal keys with
  the server's normalization contract.
- Keep current `idle`, `thinking`, `speaking`, and `error` automation working.
- Preserve legacy character expression images as fallbacks.
- Use immutable package resources and profile-owned copy-on-write edits.
- Keep the local persistence and resolver shapes compatible with a later explicit
  server sync adapter without claiming that sync exists now.
- Produce a portable JSON V2 card and PNG V2 card whose decoded data agree.

## Non-Goals

- Do not change character ID 1, `Default Assistant`, or default conversation
  selection.
- Do not make Samira the default Persona in this task.
- Do not add automatic emotion classification, prose regexes, or model-directed
  expression inference.
- Do not persist reaction choices into message history or promise historical visual
  replay.
- Do not implement Visual Identity server APIs or actual client/server sync.
- Do not port server drafts, jobs, idempotency, animated assets, or VN composition.
- Do not merge Persona Visual Packs with character Visual Identity packs.
- Do not ship or clear the separate Sammy logo, wordmark, campaign, advertising, or
  merchandise system.
- Do not add a new runtime dependency.

## Release and Brand Boundary

ADR-067 narrowly amends the earlier brand decision. The supplied `Sammy.png` is the
canonical in-app portrait for this character and the sole visual source for the
reaction pack. The earlier brand concept images remain direction references for the
separate logo and campaign system.

The in-app V2 card, portrait, and reactions are production assets under
`AGPL-3.0-or-later`. Broader brand uses retain their existing clearance and
production gates. Shipping this card must not imply that Chatbook uses one model,
one assistant personality, or a romantic companion.

## Selected Architecture

Use a server-aligned local Visual Identity bridge rather than extending the legacy
three-row BLOB table or porting the complete server subsystem.

The bridge has five bounded responsibilities:

1. **Expression contract** — exact server normalization plus the Samira manifest's
   explicit standard-label mappings.
2. **Persistence** — packs, immutable versions, assets, and active character/Persona
   bindings using the server's core schema shapes.
3. **Asset loading** — immutable package-resource reads and profile-owned user-asset
   reads behind one validated loader.
4. **Resolution** — one structured result with deterministic fallbacks.
5. **Presentation** — a pack browser/editor in Personas and a session-local picker in
   Console.

The existing `character_expression_images` table remains intact. Characters without
a Visual Identity binding continue through the current three-slot editor and runtime.

## Character Card Contract

The canonical external card is a V2 JSON file. The canonical PNG retains the supplied
1254×1254 RGB pixels and embeds the same V2 JSON in the standard base64 `chara` PNG
metadata field. Embedding metadata may change PNG file bytes but must not change the
decoded pixel content.

The sanitized card keeps the approved character writing and makes these metadata
changes:

- `creator` becomes `tldw_chatbook`.
- `creator_notes` describes an included editable demonstration character and her
  reaction pack; it removes “private” language and the nonexistent profile-emblem
  reference.
- all `vademhq/*` extensions are removed;
- private flags are removed;
- `tldw/builtin_id` is `samira`;
- role, personality-mix, and nature metadata move under the `tldw/*` namespace;
- `tldw/license` is `AGPL-3.0-or-later`;
- `tldw/visual_identity_pack_id` names the stable bundled pack;
- tags describe the public character and feature demonstration without private
  provenance tags.

The surname “Vadem” remains part of the character's name. The forbidden string is
the removed organization name `VademHQ`.

An `ASSET_LICENSE.md` beside the packaged assets records the license, the supplied
portrait as the source reference, the generation tool and date for derived reactions,
and the exact packaged inventory. It makes no unsupported authorship claim.

## Reaction Inventory and Key Mapping

Every asset has a portable lowercase filename, an exact user-facing original label,
and one internal expression key. Colons appear only in metadata keys, never in
filenames.

| File/original label | Internal expression key | Visual direction |
| --- | --- | --- |
| `admiration.webp` | `custom:admiration` | softened eyes and quiet impressed respect |
| `amusement.webp` | `custom:amusement` | restrained closed-mouth smile and bright eyes |
| `anger.webp` | `angry` | controlled lowered brow and firm jaw, never rage |
| `annoyance.webp` | `custom:annoyance` | slight brow pinch and restrained exasperation, milder than anger |
| `approval.webp` | `custom:approval` | small affirming nod and composed positive regard |
| `caring.webp` | `custom:caring` | attentive concern and gentle protective warmth |
| `confusion.webp` | `confused` | asymmetrical brow and searching focus |
| `curiosity.webp` | `custom:curiosity` | alert eyes, slight head angle, investigative interest |
| `desire.webp` | `custom:desire` | focused yearning toward an idea or objective, never flirtation |
| `disappointment.webp` | `custom:disappointment` | lowered gaze and restrained letdown |
| `disapproval.webp` | `custom:disapproval` | steady evaluative gaze and lightly pressed lips |
| `disgust.webp` | `custom:disgust` | subtle nose tension and aversion without caricature |
| `embarrassment.webp` | `custom:embarrassment` | averted gaze and contained self-consciousness |
| `excitement.webp` | `excited` | widened bright eyes and energized posture, not theatrical |
| `fear.webp` | `custom:fear` | guarded eyes and contained alarm |
| `gratitude.webp` | `custom:gratitude` | softened eyes and sincere appreciative warmth |
| `grief.webp` | `custom:grief` | heavy eyes and controlled deep loss, no melodrama |
| `joy.webp` | `happy` | genuine warm smile and open eyes |
| `love.webp` | `custom:love` | deep warm regard and trust, explicitly nonromantic |
| `nervousness.webp` | `custom:nervousness` | slight tension and uncertain focus, no cartoon cues |
| `neutral.webp` | `neutral` | canonical quiet recognition and composed warmth |
| `optimism.webp` | `custom:optimism` | lifted focus and restrained confidence about what comes next |
| `pride.webp` | `custom:pride` | upright composure and earned satisfaction, never smugness |
| `realization.webp` | `custom:realization` | newly focused eyes and subtle “I see it” recognition |
| `relief.webp` | `custom:relief` | released facial tension and a small exhale |
| `remorse.webp` | `custom:remorse` | lowered gaze and accountable regret |
| `sadness.webp` | `sad` | quiet sorrow and softened posture |
| `surprise.webp` | `surprised` | widened eyes and subtly parted lips, controlled intensity |
| `thinking.webp` | `thinking` | pensive focus and slight off-axis gaze |
| `speaking.webp` | `custom:speaking` | natural mid-sentence engagement with restrained mouth opening |
| `error.webp` | `custom:error` | concerned, apologetic recovery focus without a sweatdrop or symbol |

The explicit mappings intentionally preserve all 31 distinct assets. Generic server
normalization remains exact:

- canonical keys: `neutral`, `happy`, `excited`, `sad`, `angry`, `thinking`,
  `confused`, and `surprised`;
- server aliases remain byte-for-byte equivalent to the pinned server module;
- unknown valid labels normalize to `custom:<sanitized_label>`.

Samira's manifest explicitly maps `excitement` to `excited`, `sadness` to `sad`,
`confusion` to `confused`, and `surprise` to `surprised`; those four mappings are not
currently generic server filename aliases. It also maps `joy` to `happy` and `anger`
to `angry`, matching server aliases. `idle` is an operational client state and maps
to `neutral` at the resolver boundary rather than becoming `custom:idle`.

## Visual Production Contract

The canonical portrait establishes invariant identity and composition:

- same adult face, age, skin tone, facial proportions, and natural asymmetry;
- same dark page-ribbon low knot and bookmark-shaped pin;
- same rectangular/index-tab earring and minimal temple line;
- same dark futuristic/editorial uniform, crop, camera angle, and lighting family;
- same circular orange archive interface on black;
- no added people, hands obscuring the face, text, logos, watermarks, symbols, or new
  accessories.

Only facial expression and a slight head/shoulder posture change. Expressions are
clear at avatar size but restrained, composed, and never theatrical, flirtatious,
smug, romantic, chibi, anime-symbolic, or emoji-like.

`neutral.webp` is a high-quality 1024×1024 WebP derivative of the supplied canonical
portrait. Every other reaction is generated as an independent edit from the original
`Sammy.png`, never from another reaction, to avoid cumulative identity drift.

Each final reaction must:

- decode as a single-frame 1024×1024 WebP;
- use sRGB-compatible color;
- have a SHA-256 digest recorded in the manifest;
- be at most 1 MiB;
- contain no embedded private or legacy metadata; and
- pass individual inspection plus a 31-image contact-sheet consistency review.

The 31 WebP reactions together must not exceed 16 MiB. The complete packaged Samira
directory, including the canonical PNG, JSON, manifest, and license, must not exceed
20 MiB. Generated prompts and tool/date provenance are retained in the manifest or
adjacent production record, not rendered into the images.

## Pack Manifest

The packaged directory is explicit package data beneath
`tldw_chatbook/assets/characters/samira/` and contains:

```text
Samira.character.json
Sammy.png
visual_identity_pack.json
ASSET_LICENSE.md
expressions/
  admiration.webp
  ...
  surprise.webp
  thinking.webp
  speaking.webp
  error.webp
```

The manifest has a versioned schema identifier and records:

- stable built-in pack ID and title;
- license and source provenance;
- default expression key `neutral`;
- source server commit and normalization-contract version;
- pack content digest;
- for every asset: expression key, exact original label, display label, relative
  filename, content type, byte count, dimensions, SHA-256, animation fields, and
  generation provenance.

Paths are normalized relative POSIX paths. They may not be absolute, contain `..`,
backslashes, NULs, or leave the Samira package directory. The manifest contains one
asset per expression key and exactly the 31 labels above.

## Persistence Contract

The ChaChaNotes schema gains the server-shaped core tables:

- `visual_identity_packs`;
- `visual_identity_pack_versions`;
- `visual_identity_assets`; and
- `visual_identity_bindings`.

Their columns, status values, foreign keys, checks, and active-binding uniqueness
match the pinned server implementation. The omitted server tables are drafts and
idempotency; no local empty facsimiles are created for them. The local owner sentinel
is `0` because each Chatbook database is already profile-owned. It is documented as
local-only and must never be sent as a server user ID.

Built-in asset rows store package-relative `storage_relpath` values such as
`characters/samira/expressions/joy.webp`. `source_kind = 'builtin'` selects the
immutable package-resource root. User-owned packs resolve beneath a private
`get_user_data_dir()/visual_identities/` root. The loader validates every relative
path against its selected root before reading.

Activated versions are immutable by service contract. Pack editing stages all
changes in a private temporary directory, validates the complete candidate, atomically
moves new files into profile-owned storage, and commits one version plus its asset
rows and binding update in one database transaction. “Generate all” therefore makes
one version, not 31 partial versions. A failed candidate leaves the active version
and packaged assets untouched.

Editing a built-in pack first copies its complete active asset set into the staged
user-owned candidate, creates a new user-owned pack/version, and changes only that
actor's binding. Packaged rows and files remain immutable.

## Seeding and Lifecycle

Seeding runs after schema and FTS trigger initialization. It is bounded and
idempotent:

1. Read and validate the packaged V2 JSON, embedded PNG card, pack manifest, and
   required resource inventory.
2. Locate a character by `tldw/builtin_id = samira`, including soft-deleted rows.
3. If an active built-in row exists, preserve every card field and use its stable
   numeric actor ID.
4. If the row is soft-deleted, do not restore it and do not activate a binding.
5. If no built-in row exists, insert the card and portrait. If its preferred name is
   occupied by any row, use `Samira “Sammy” Vadem (Built-in)`; if that is also
   occupied, append the lowest available deterministic integer suffix.
6. Validate the complete pack before creating pack metadata. Then create the pack,
   version, 31 assets, and binding in one transaction.
7. If pack validation or activation fails, keep the usable character/avatar, report a
   bounded warning, and retry only the absent built-in pack on next startup.

A renamed or edited built-in card is never replaced because identity comes from its
extension and existing binding, not its name or byte equality. A deleted pack,
deleted binding, or deleted character is a tombstone and is never resurrected by
startup. An archived pack remains archived. A missing pack after an interrupted
pre-activation failure may be retried because no tombstone exists.

The seed is create-only: a later packaged revision does not silently rewrite an
existing user's card or active version. An explicit reset/update design is required
if built-in updates become necessary.

## Resolution Contract

The resolver accepts actor identity, the current requested operational state, and an
optional session-local manual key. It returns a frozen structured result containing:

- actor kind and local actor ID;
- pack, version, expression, and asset IDs when present;
- requested and resolved expression keys;
- storage source and relative path;
- content type and animation flag;
- fallback reason; and
- resolution source.

Priority is deterministic:

1. a valid manual override for the current session and actor;
2. requested operational state mapped as `idle → neutral`, `thinking → thinking`,
   `speaking → custom:speaking`, or `error → custom:error`;
3. the active version's declared default;
4. the active version's `neutral` asset;
5. the matching legacy operational BLOB, where applicable;
6. the character card portrait; and
7. the current empty/placeholder rendering.

Unknown or missing requested assets fall through without mutation. Fallback reasons
are stable, testable strings rather than log prose. Corrupt or missing assets are
treated as unavailable, logged without private paths, and fall through.

Manual override state is keyed by local session ID plus actor identity. It clears on
explicit user action, actor replacement, session disposal, or application restart.
Selecting an absent expression reports an error and preserves the prior selection.
It is not written to `messages.metadata_json`, generation metadata, the card, or the
database.

## Personas Workbench UI

For a character with a Visual Identity binding, the expression section becomes a
pack-aware browser/editor:

- a filterable list of expression labels and a visible `current / total` count;
- one selected asset decoded into one lazy preview;
- exact display label plus internal-key diagnostics where useful;
- replace/generate/clear staging actions;
- a dirty-state summary and one Save action that creates one version;
- Generate All stages the candidate and remains cancellable without changing the
  active version; and
- built-in status and the copy-on-write consequence are stated before the first edit.

The screen must not decode 31 full images or compose 31 thumbnails at once. It keeps
at most the selected preview plus the existing bounded runtime cache. Labels remain
visible so meaning is not conveyed by imagery or color alone.

Characters without a binding retain the current fixed legacy
`thinking`/`speaking`/`error` controls. Creating a general pack from a legacy
character is not required for Samira's release unless it falls out naturally from the
same bounded save path; no migration prompt is added.

## Console UI

Samira's Console avatar uses the structured resolver. A pack action near the existing
avatar controls opens a filterable selector with one lazy preview. Selecting a label
sets the session-local override; Clear returns control to operational automation.

The UI shows the active manual label while overridden. It does not add a global
hotkey or advertise an unimplemented footer action. The existing reaction-enable
configuration still controls automatic operational switching; manual selection is an
explicit user action and remains available when the pack is visible.

At narrow terminal sizes, the selector remains keyboard-operable, the preview may be
hidden before labels/actions, and neighboring controls stay within the screen. No
`/emote` command is added in this tranche.

## Error Handling and Recovery

- Missing packaged JSON or portrait: skip new card creation and report one bounded
  startup warning.
- Card valid but pack invalid: keep the card/avatar, omit the binding, and retry the
  absent pack on restart.
- Missing/corrupt runtime reaction: return the next fallback; never crash Console.
- Same-name character: disambiguate; never overwrite or undelete.
- Existing built-in tombstone: preserve it and suppress reseeding.
- Pack-save generation/upload failure: retain the staged error and old active
  version; do not publish partial assets.
- Database commit failure after staged asset move: leave the previous binding active
  and make unreferenced staged files eligible for bounded cleanup on the next pack
  operation; never delete an active version's files.
- Unsupported or unsafe path: reject before reading.
- Application shutdown during generation: cancel workers where supported and leave
  the active version unchanged.

Logs contain stable identifiers and fallback/error categories, not raw card content,
user paths, or image bytes.

## Packaging

`pyproject.toml` currently has `include-package-data = false`, so the Samira directory
must be added through an explicit recursive package-data rule under the
`tldw_chatbook` package. It is not enough to create a filesystem directory that
source-checkout tests can see.

Distribution verification follows ADR-032:

- build wheel and sdist from a temporary source copy;
- inspect the exact Samira inventory in both artifacts;
- enforce the per-file and total size budgets;
- install the wheel into an isolated target with no editable-source shadowing;
- initialize a fresh redirected profile from the installed package; and
- prove the installed loader resolves package resources without writing beneath the
  installed package root.

The reaction contact sheet and temporary generation intermediates are review artifacts
and are not package data.

## Validation

### Asset and card tests

- Decode all 31 WebPs and assert exact dimensions, format, frame count, byte caps,
  checksums, inventory, and unique expression keys.
- Extract `chara` metadata from `Sammy.png` and compare the parsed V2 object with
  `Samira.character.json`.
- Compare canonical PNG pixels before and after metadata embedding.
- Scan every packaged text and binary asset for `VademHQ` and private legacy keys.
- Validate manifest paths, filenames, labels, mappings, digest, and license.
- Review every reaction individually and as a contact sheet for identity drift,
  ambiguity, watermarks, unwanted text, and expression distinctness.

### Database and lifecycle tests

- Fresh production constructor creates one searchable Samira card and one bound
  31-asset version after FTS triggers exist.
- The first real edit of the seeded card succeeds and leaves FTS healthy.
- An upgraded database gains Samira without changing existing cards.
- Repeated startup creates no duplicates or new versions.
- Name collision, renamed card, customized card, deleted card, deleted binding,
  archived pack, and partial pack failure follow the lifecycle rules above.
- One staged multi-asset save creates exactly one immutable version.
- Built-in edits fork into the active profile and never write to package resources.

### Server parity tests

- Run the same normalization fixture matrix against the local module and the pinned
  server behavior.
- Assert the four local table schemas match the pinned server columns, constraints,
  indexes, and status vocabulary selected by this design.
- Assert structured resolver fields and fallback categories retain the planned sync
  seam while local IDs and owner `0` remain marked local-only.

### Runtime and UI tests

- Manual override wins, Clear restores operational behavior, actor changes clear the
  override, and restart does not persist it.
- `idle`, `thinking`, `speaking`, and `error` choose the expected pack assets.
- Missing pack assets fall through to legacy BLOBs and card portraits.
- The Personas browser filters all 31 labels, decodes one preview, and stages without
  publishing until Save.
- The Console selector works by keyboard and remains in-bounds at 80×24 and a normal
  terminal size.
- Legacy characters retain their current controls and behavior.

### Installed and live verification

- Build, inspect, install, and run both distribution artifacts as described above.
- Start the real TUI with a fresh redirected profile, open Samira, inspect/filter the
  pack, begin a chat, select and clear reactions, and observe all four automatic
  operational states.
- Induce a missing reaction and a failed staged save and confirm graceful fallback
  and unchanged active version.
- Mutation-check the seed idempotency guard, path-boundary guard, fallback order, and
  lazy-preview assertion so each test demonstrably fails when its protection is
  removed.

## Scope Deferred to Follow-up Work

- Making Samira the first default Persona or replacing Default Assistant.
- Automatic 28-label classification or model-directed expression metadata.
- Durable, syncable per-message pack/version/asset replay.
- Authenticated Visual Identity sync and remote-ID assignment.
- Full draft/job/idempotency workflow and arbitrary ZIP review.
- `/emote` commands, multi-actor VN staging, and Persona Visual Pack unification.
- Public logo, wordmark, advertising, campaign, or merchandise use.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/067-bundled-samira-visual-identity-pack.md`

Reason: The feature adds schema and immutable versioning, defines package versus
profile asset ownership, establishes resolver and binding contracts, preserves a
future sync boundary, and narrowly amends an earlier public-release decision.
