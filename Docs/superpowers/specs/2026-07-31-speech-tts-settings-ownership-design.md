# Speech & TTS Settings Ownership — Product Requirements and Design

**Status:** Approved design; ready for task decomposition
**Date:** 2026-07-31
**Target branch reviewed:** `origin/dev` at `503e2eeb7`
**Canonical ADR:** [ADR-039](../../../backlog/decisions/039-global-and-studio-tts-settings-ownership.md)
**Related decisions:** [ADR-012](../../../backlog/decisions/012-provider-credential-settings-boundary.md), [ADR-023](../../../backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md), [ADR-028](../../../backlog/decisions/028-character-tts-generation-profile-ownership.md), and [ADR-037](../../../backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md)

## Document purpose

This document is the approved program-level product requirements document for
making Speech & TTS configuration understandable, truthful, and safe across
global Settings, the Lab Speech studio, and character roleplay.

It deliberately does **not** contain an implementation plan or Backlog task
breakdown. The next agent must re-audit the then-current `dev` branch, use the
stable requirement IDs in this document, and create atomic Backlog tasks in
dependency order. Task boundaries may change as the code evolves; the approved
product behavior and ownership rules may not be weakened without a new design
decision.

## Executive summary

Chatbook already has a native external audio.cpp adapter, global TTS defaults,
a Lab Speech playground, legacy provider settings, and character-specific TTS
generation profiles. The current UI places almost all TTS settings in the Lab,
mixes application-wide configuration with one studio's generation controls,
and sometimes reports local dependency availability as if it were the health
of every speech provider. This makes setup hard to discover and makes it
unclear whether an edit affects the whole application, only the Studio, or a
character.

The product will establish four explicit owners:

| Owner | Owns | Does not own |
| --- | --- | --- |
| **Settings → Speech & TTS** | Application-wide defaults, credentials, endpoints, local initialization resources, external audio.cpp configuration, and provider safety limits | Live network checks, catalog refresh, generation, playback, character assignment |
| **Lab → Speech → Studio TTS Preferences** | Persisted, provider-scoped Studio overrides and the current Studio generation draft | Global defaults, credentials, runtime-global initialization, implicit character assignment |
| **Character TTS profiles** | Exact character-specific provider/model/voice/tuning selections and assignment | Global or Studio mutation |
| **TTS runtime / Lab operations** | Readiness checks, catalog and voice discovery, refresh, synthesis progress, playback, and runtime diagnostics | Durable configuration ownership |

Settings becomes the discoverable global entry point. It exposes connection or
initialization fields for every existing TTS provider, but audio.cpp is the
only provider receiving a complete redesigned provider experience in this
program. Existing providers keep their current generation behavior and tuning
semantics while their fields are separated by ownership. Lab remains the place
to test a connection, refresh a catalog, generate audio, and hear the result.

Studio preferences are stored separately and never alter global defaults.
Character profiles remain separately stored and win over global defaults for
roleplay. A character/profile preview loaded into Studio does not change Studio
preferences until the user explicitly adopts and saves it.

Managed audio.cpp binary or `server.json` launching and supervision is
deferred. This program configures and uses an independently running external
audio.cpp server only.

## Background and current-state audit

The design was reviewed against the current TTS architecture and UI, including:

- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/UI/Speech/speech_settings_pane.py`
- `tldw_chatbook/UI/Speech/speech_settings_model.py`
- `tldw_chatbook/UI/Speech/speech_settings_mixin.py`
- `tldw_chatbook/UI/Lab_Modules/lab_speech_status.py`
- `tldw_chatbook/TTS/preferences.py`
- `tldw_chatbook/TTS/TTS_Generation.py`
- `tldw_chatbook/TTS/legacy_bridge.py`
- the character TTS profile repository and service

The implementation already has useful foundations:

- an app-scoped TTS adapter registry and native external audio.cpp adapter;
- immutable global `TTSPreferencesSnapshot` admission data;
- explicit audio.cpp model modes (`exact` or `first_available`) and voice modes
  (`exact` or `server_default`);
- targeted provider reconfiguration and configuration revisions;
- catalog and voice discovery revisions;
- complete WAV generation, playback, export, and request provenance;
- a durable, exact, character-addressed TTS profile system; and
- a rebuilt Lab Speech settings pane with a field-completeness inventory.

The remaining product issues are ownership and interaction issues rather than
a missing TTS engine:

1. **Poor discoverability.** The main Settings destination has no first-class
   Speech & TTS category even though recovery copy refers users to Settings.
   A first-time user must find the Lab and interpret a large provider form.
2. **Mixed scopes.** Application defaults, credentials, endpoints, runtime
   initialization, provider tuning, live actions, and Studio generation
   controls appear together. Saving does not communicate the blast radius.
3. **Untruthful readiness.** A combined local TTS/STT dependency chip can imply
   that speech is unavailable even when an external audio.cpp server is usable.
   Configuration validity, runtime health, and catalog freshness are distinct.
4. **Weak exact-choice UX.** Global audio.cpp model and voice defaults need to
   use known catalog data without causing hidden network access. Missing or
   stale exact selections must not silently become another model or voice.
5. **Provider forms are operationally dense.** The current inventory spans
   global connection fields, request-scoped tuning, file pickers, voice-blend
   management, discovery, test, and save actions. Advanced safety limits crowd
   normal first-run setup.
6. **Ambiguous recovery.** Users need a reliable path between a saved global
   configuration and the Lab action that verifies it, while dirty drafts and
   asynchronous results remain protected.
7. **No separately persisted Studio layer.** The Studio needs durable,
   provider-scoped overrides that inherit from global defaults and can be
   removed without copying global values.
8. **Character preview can be mistaken for persistence.** A roleplay voice
   profile loaded into the Studio must remain preview-only unless explicitly
   adopted.
9. **Credential mutation is mixed with ordinary settings persistence.** A
   masked placeholder or environment-sourced credential must never be written
   back as if it were a secret value.
10. **Privacy boundaries need explicit product copy and tests.** External TTS
    transmits synthesis text to the configured server; that text must not
    appear in logs, diagnostics, metrics, caches, or error messages.

## Users and core journeys

### First-time external audio.cpp user

The user installed `audiocpp_server`, owns an external server configuration,
and has started it independently. They want to enter one URL, save it as the
global default, open the Lab, verify the connection, discover models and
voices, and generate playable WAV audio without editing TOML.

### Existing legacy-provider user

The user already relies on OpenAI, ElevenLabs, Kokoro, Chatterbox, Higgs, or
AllTalk. They need their existing values and behavior preserved while
credentials/endpoints/runtime initialization move to a truthful global
Settings surface. They should not be forced through an audio.cpp redesign.

### Studio experimenter

The user wants the Lab Speech studio to remember its own provider/model/voice
and request-scoped tuning between visits without changing the application-wide
voice used elsewhere.

### Roleplay user

The user assigns a dedicated TTS profile to a character. That character's
responses should use the assigned exact voice while unrelated characters and
normal responses continue to use global defaults. Loading the profile into the
Studio should be safe to audition and should not silently become the Studio's
new default.

### Environment-managed credential user

The user supplies a provider credential through an environment variable. The
UI must report that effective source without exposing the value, copying it
into local configuration, or letting an ordinary Save overwrite it.

## Goals

- Make global TTS configuration findable from the primary Settings
  destination and Settings search.
- Make scope explicit before the user edits or saves anything.
- Provide a complete, coherent external audio.cpp global configuration UX.
- Move connection, credential, and runtime-initialization fields for every
  existing provider into global Settings.
- Preserve the current behavior of legacy providers behind their existing
  adapter bridge.
- Give the Lab Studio a separately persisted, provider-scoped override layer.
- Resolve request settings deterministically across explicit request,
  character, Studio, global, and provider fallback layers.
- Distinguish configuration validity, runtime health, and catalog freshness.
- Preserve exact selections safely across catalog refresh and configuration
  revision changes.
- Protect secrets and submitted synthesis text.
- Keep the UI keyboard-operable and understandable at narrow terminal sizes.
- Deliver an audibly validated first-time audio.cpp journey without requiring
  network access in ordinary CI.

## Non-goals

- Launching, supervising, restarting, adopting, or stopping
  `audiocpp_server`.
- Accepting a binary path, `server.json` path, server bind setting, or managed
  process option.
- Downloading, building, packaging, or updating audio.cpp or a model.
- Supporting more than one active audio.cpp instance.
- Adding authentication or arbitrary request headers to external audio.cpp.
- Adding a new TTS provider.
- Migrating the six legacy providers to native adapters.
- Redesigning every legacy provider's tuning controls into a generic schema.
- Adding dynamic provider plugins, entry points, or a schema-driven form
  framework.
- Replacing or redesigning character TTS profile authoring, assignment,
  portability, or repair.
- Making a character preview implicitly persistent.
- Adding hidden catalog discovery during Settings mount, navigation, or save.
- Implementing true incremental audio streaming. Complete WAV responses remain
  exposed through the existing asynchronous response interface.
- Moving voice-blend add/import/export into global configuration. Those are
  Voice Profile library operations.
- Removing legacy config keys in this program.

## Product vocabulary

The UI and diagnostics must use these terms consistently:

| Term | Meaning |
| --- | --- |
| **Global defaults** | Persisted application-wide TTS selection used when a more specific owner supplies no value |
| **Studio preferences** | Separately persisted Lab Speech overrides; optional values inherit from global defaults |
| **Current Studio controls** | Unsaved values in the mounted Studio form; highest precedence inside Studio only |
| **Character TTS profile** | Persisted exact TTS generation selection assigned to one canonical character identity |
| **Configuration state** | Whether stored or drafted values are complete and locally valid |
| **Runtime state** | Whether a provider was checked and can currently answer an operation |
| **Catalog freshness** | Whether known model/voice choices match the latest accepted catalog observation |
| **Exact selection** | One opaque model or voice identifier that must not be silently substituted |
| **Dynamic selection** | `First available` model or `Server default` voice, resolved by the admitted provider operation |
| **External audio.cpp** | A server started and owned outside Chatbook, addressed by an HTTP(S) origin |

“Configured” must never be used as a synonym for “Ready.” “Saved” must never
be used as a synonym for “Checked.”

## Authoritative ownership model

### OWN-001 — Global Settings ownership

`Settings → Speech & TTS` is the only UI owner for durable application-wide
TTS defaults, credentials, endpoints, runtime initialization resources, and
provider safety limits. An editor elsewhere may link to this category but may
not mutate those values.

### OWN-002 — Studio ownership

`Lab → Speech → Studio TTS Preferences` owns a separate durable record of
optional, request-scoped Studio overrides. Saving Studio preferences never
writes global TTS sections, credentials, endpoint fields, character profiles,
or adapter initialization fields.

### OWN-003 — Character ownership

Character TTS profiles and their canonical character assignments retain their
existing repository and service authority. Global and Studio persistence must
not copy, mutate, repair, or delete a character assignment.

### OWN-004 — Runtime ownership

Lab/runtime services own test, health, catalog refresh, synthesis, progress,
playback, and export. Global Settings performs local validation and persistence
only; mounting or saving Settings does not contact a provider.

### OWN-005 — Built-in provider inventory

The implementation must maintain an explicit, testable inventory that assigns
every existing built-in Speech settings field to exactly one of:

- global configuration;
- Studio preference;
- Voice Profile library action;
- runtime action/readout; or
- intentionally unsupported/retired behavior with a documented reason.

The inventory is a bounded built-in mapping, not a generalized plugin or
schema-driven UI framework. A completeness test compares it with the fields
actually mounted by the replaced Lab settings UI so no setting disappears
silently.

### OWN-006 — Configure does not select

`Default TTS Provider` and `Configure Provider` are distinct controls. Opening
or editing a provider's configuration does not silently make it the global
default. Changing the global default does not discard another provider's
saved configuration.

## Field ownership inventory

The following inventory is the approved classification baseline. The task
decomposition agent must verify the exact config keys and current runtime
support against the latest `dev` before creating tasks. Verification may refine
key names or group labels, but it may not move connection/credential/init
fields into Studio or make Studio values global.

### Shared selection fields

| Field | Global Settings | Studio | Character profile |
| --- | --- | --- | --- |
| Provider | Global default | Optional override | Exact persisted provider |
| Model policy/value | `Exact` or `First available` | Optional override using supported provider modes | Exact model |
| Voice policy/value | `Exact` or `Server default` | Optional override using supported provider modes | Exact or provider-declared server default |
| Output format | Global default | Optional supported override | Exact supported format |
| Speed | Global default | Optional supported override | Exact supported speed |

Provider constraints apply at every scope. In the current native audio.cpp
contract, format is locked to WAV, speed is locked to `1.0`, and arbitrary
options are rejected.

### audio.cpp

| Global Settings | Studio preferences | Runtime / Lab |
| --- | --- | --- |
| External mode readout; server URL; connect timeout; synthesis timeout; maximum input characters; maximum response bytes; maximum metadata bytes; maximum catalog models; maximum voices per model; maximum identifier characters; privacy notice | Shared provider/model/voice fields only; WAV and speed `1.0` shown as fixed; no unsupported tuning | Test connection; refresh models; refresh voices; generation; playback; export; revisioned health/catalog status |

audio.cpp is the only provider receiving the full provider-detail redesign.
All limits live under an `Advanced safety limits` disclosure except the URL
and the two timeouts needed to explain slow external inference. The server URL
is a canonical HTTP(S) origin. No authentication, file path, binary,
`server.json`, bind address, launch, restart, or stop field is permitted.

### OpenAI

| Global Settings | Studio preferences | Runtime / Lab |
| --- | --- | --- |
| API key; base URL; organization ID | Shared request selection fields and only request-scoped options supported by the existing adapter | Generation, playback, and safe provider status |

### ElevenLabs

| Global Settings | Studio preferences | Runtime / Lab |
| --- | --- | --- |
| API key | Model; output format; stability; similarity boost; style; speaker boost, plus supported shared selection | Generation, playback, and safe provider status |

### Kokoro

| Global Settings | Studio preferences | Voice Profile / runtime |
| --- | --- | --- |
| Device; ONNX enablement; ONNX model file; voices JSON file; file-picker affordances for those global paths | Maximum tokens; voice-mixing enablement; performance tracking, plus supported shared selection | Voice-blend add/import/export belong to the Voice Profile library rather than the global action strip; generation and playback remain runtime actions |

### Chatterbox

| Global Settings | Studio preferences | Runtime / Lab |
| --- | --- | --- |
| Device; voice-resource directory and its path-picker affordance | Exaggeration; CFG weight; temperature; candidates; seed; preprocessing; audio validation; normalization/target level; maximum/chunk sizes; streaming/chunking; crossfade enablement/duration, plus supported shared selection | Generation, playback, and safe provider status |

### Higgs

| Global Settings | Studio preferences | Runtime / Lab |
| --- | --- | --- |
| Model path; voice-resource directory; path-picker affordances; device; flash attention; dtype | Language; maximum reference duration; voice cloning; multi-speaker behavior; speaker delimiter; performance tracking; maximum new tokens; temperature; top-p; repetition penalty, plus supported shared selection | Generation, playback, and safe provider status |

### AllTalk

| Global Settings | Studio preferences | Runtime / Lab |
| --- | --- | --- |
| Server URL | Voice; language; output format, plus supported shared selection | Generation, playback, and safe provider status |

If current adapter behavior proves that a listed tuning value is
runtime-global rather than operation-scoped, it remains global or read-only
until the adapter supports operation-scoped use. It must not be presented as a
Studio override that the request cannot honor.

## Information architecture and discoverability

### IA-001 — First-class Settings category

Add `Speech & TTS` as a main Settings category with a description that says it
contains application-wide defaults and provider setup. It must be reachable by
keyboard navigation and Settings search.

### IA-002 — Search vocabulary

Settings search must match, at minimum:

- `speech`
- `TTS`
- `voice`
- `audio.cpp`
- `audio_cpp`
- `OpenAI`
- `ElevenLabs`
- `Kokoro`
- `Chatterbox`
- `Higgs`
- `AllTalk`

Search results must open the category with the relevant provider selected when
the query names a provider.

### IA-003 — Global scope banner

The category begins with persistent, non-color-only copy equivalent to:

> You are editing application-wide Speech & TTS defaults. The Speech Studio
> can keep separate preferences without changing these values.

It includes an `Open Speech Lab` action. The banner must remain visible or
quickly recoverable when the form scrolls.

### IA-004 — Studio scope banner

Rename the existing Lab subview to **Studio TTS Preferences**. It begins with
persistent copy equivalent to:

> These preferences affect only the Speech Studio. They never change global
> defaults or character TTS profiles.

It includes an `Open Global Speech & TTS Settings` action that preserves the
selected provider context.

### IA-005 — Deep-link and return context

Navigation between Settings and Lab carries a canonical provider ID and an
optional intent (`configure`, `test`, `refresh-models`, or `refresh-voices`).
It does not carry credentials, field values, synthesis text, or an arbitrary
widget selector. Returning restores the relevant provider and exact selection
when still representable.

If the source screen has a dirty draft, navigation uses the draft-protection
flow in CFG-012 before leaving. A deep link may focus a provider but may not
silently save, discard, refresh, test, or generate.

## Global Settings experience

### CFG-001 — Page structure

The global category has these sections in order:

1. scope banner and concise ownership explanation;
2. **Global defaults**;
3. **Provider setup**, containing distinct `Configure Provider` selection;
4. a selected-provider form;
5. a collapsed **Advanced safety limits** section when applicable;
6. a configuration inspector; and
7. the global action strip.

Only the selected provider's form is mounted or expanded. Users are not
required to open a sequence of identical collapsed provider boxes to learn
which one is configured.

### CFG-002 — Global defaults

The defaults section exposes:

- `Default TTS Provider`;
- model policy and exact value when relevant;
- voice policy and exact value when relevant;
- output format; and
- speed.

Provider capability rules update the enabled fields and explanatory copy. A
constraint does not disappear: for audio.cpp, the UI shows WAV and speed
`1.0` as fixed values and explains that the current adapter contract requires
them.

### CFG-003 — Provider setup summary

Each provider is summarized as a configuration state, not a runtime health
claim. The selected provider form exposes the globally owned fields in the
approved inventory. Legacy providers retain recognizable field labels and
saved behavior; only their ownership and placement change.

### CFG-004 — audio.cpp setup

The audio.cpp provider form identifies the mode as **External server** and
shows the server URL as the primary field. It states that Chatbook connects to
a server the user starts and owns. It includes all external adapter timeouts
and bounds from the inventory and excludes every managed-process concept.

The privacy notice says that generation sends the submitted text to the
configured server. It does not imply that loopback is guaranteed because the
user may configure another HTTP(S) origin. The audio.cpp origin rejects
userinfo, query, fragment, and non-origin path components so credentials or
request data cannot be smuggled into the configured URL.

A non-loopback plain-HTTP origin is permitted for compatibility but receives a
visible warning that submitted text and returned audio are not transport-
encrypted. HTTPS is recommended for non-loopback servers. The warning does not
claim that Chatbook authenticated the remote server.

### CFG-005 — Local validation only

Global Save validates field shape, numeric ranges, supported provider IDs,
provider constraints, URL canonicalization, and path syntax where applicable.
It does not open a socket, refresh a catalog, initialize a local runtime, read
a model merely to prove it exists, or synthesize hidden audio.

A locally valid configuration may save while the provider is unavailable.
After save, its runtime state remains `Not checked`, `Stale`, or `Unavailable`
until a user-initiated runtime operation produces newer evidence.

### CFG-006 — Global actions

The ordinary global action strip provides:

- `Save`;
- `Revert`;
- `Restore Non-secret Defaults`; and
- `Open Speech Lab`.

`Restore Non-secret Defaults` changes only non-secret fields in the draft and
does not save automatically. It never clears, replaces, copies, or reveals a
credential.

Test connection, catalog refresh, voice refresh, generation, playback,
resource library management, voice-blend management, and file import/export
are not ordinary global Save actions.

### CFG-007 — Configuration inspector

The category exposes a concise inspector for the selected provider and global
selection. It reports, without secret values:

- effective source (`Environment`, `Saved local config`, `Default`, or
  `Inherited` where applicable);
- configuration state;
- saved configuration revision;
- runtime revision used by the latest observation;
- latest runtime state and observation time;
- catalog revision and freshness when known; and
- whether an unsaved change would affect global selection, one provider's
  adapter configuration, or Studio only.

### CFG-008 — Separate credential mutation

Credential operations are separate from ordinary form persistence:

- `Set credential` when no local credential exists;
- `Replace credential` when a local credential exists; and
- `Clear saved credential` for the local-config value.

The operation opens an empty masked editor. A rendered placeholder, bullets,
an environment-sourced secret, or a previously masked value is never treated
as input. The user must explicitly confirm Set/Replace/Clear. Ordinary Save
omits credential mutation entirely.

When an environment variable supplies the effective credential, the UI shows
the source and variable name, never its value. A separately stored local
fallback may be managed explicitly, but the inspector explains when it is
shadowed by the environment.

The Set/Replace flow labels a local-config credential as local secret storage
and presents an environment variable as the safer portable option already
accepted by ADR-012. This program does not introduce a new keyring or encrypted
credential store.

### CFG-009 — Reconfiguration semantics

Saving a locally valid global draft persists it atomically using the existing
configuration owner. Only providers whose effective adapter-affecting global
configuration changed are submitted for targeted reconfiguration. Selection-
only changes do not unnecessarily recreate unrelated provider adapters.

Persistence success and runtime reconfiguration success are distinct results.
If persistence succeeds but reconfiguration fails, the saved values remain
saved and the affected provider becomes truthfully `Unavailable` or
`Reconfiguring`; the application does not roll back silently or select a
different provider.

### CFG-010 — Revert semantics

`Revert` restores the current form to the last successfully persisted global
snapshot and clears local validation errors. It does not change runtime health
evidence, refresh a catalog, or mutate Studio/character state.

### CFG-011 — Exact-selection integrity

An exact model or voice ID is opaque and case-sensitive. If it is no longer in
the applicable accepted catalog revision, the UI keeps the exact value visible
and marks it `Invalid` or `Unavailable`. It does not select the first list item,
clear the value, change to a dynamic mode, or substitute a similarly named ID.

Generation that depends on the missing exact value is blocked with a safe
recovery action. Unrelated providers and previously generated artifacts remain
usable.

### CFG-012 — Dirty-draft navigation

Leaving a dirty global or Studio form through category navigation, deep link,
screen dismissal, or provider switch offers:

- `Save and continue`;
- `Discard and continue`; and
- `Cancel`.

Save failures keep the user on the current form. Cancel preserves the draft
and focus. Changing the selected configuration provider is a scope change and
uses the same Save/Discard/Cancel flow when the current provider draft is
dirty. The UI does not maintain a hidden collection of unsaved provider
drafts.

## Catalog and exact-choice behavior

### CAT-001 — No hidden discovery

Settings mount, category search, provider selection, field editing, ordinary
Save, Revert, and default restoration perform no provider network operation.
Catalog and voice refresh occur only after an explicit action in Lab.

### CAT-002 — Known choices

Global exact selectors may use the latest accepted in-memory catalog and voice
observations already owned by the TTS service. This program does not add a new
disk catalog merely to populate Settings.

When no accepted catalog is available, audio.cpp offers only these choices for
a new selection:

- model: `First available`; and
- voice: `Server default`.

The UI links to `Open Speech Lab → Refresh models/voices` rather than silently
discovering choices. If an exact value was already persisted, it remains
pinned and visible as the current unverified value until an authoritative
refresh classifies it; it is not offered as proof of availability or erased
from the draft.

### CAT-003 — Freshness presentation

- A fresh choice is selectable and labeled normally.
- A stale choice is selectable but has explicit stale copy and the observation
  time or age.
- A saved exact choice absent from the latest complete observation remains
  visible as missing and blocks affected generation.
- An ambiguous or failed voice observation is `Unverified`, not authoritative
  evidence that the voice list is empty.

An approximate legacy-provider catalog cannot establish authoritative
absence. It may offer known choices, but cannot invalidate an existing exact
selection solely because that value is not in the approximate list.

`Unverified` is not `Invalid`. An explicit generation operation may verify a
pinned exact selection through its normal readiness/catalog path and proceed
when the resulting authoritative revision contains it. It must fail safely,
without substitution, when that operation establishes that it is missing.

### CAT-004 — Model-scoped voices

Voice observations are keyed by canonical provider ID, exact model ID,
provider configuration revision, and catalog revision. Changing a model clears
or invalidates only the mounted voice options for the superseded model; it does
not erase a saved exact voice value before the user resolves the mismatch.

### CAT-005 — Stale asynchronous result rejection

Every catalog, voice, readiness, and synthesis result carries enough immutable
identity to compare provider, configuration revision, catalog revision, model,
and initiating screen request. A result that no longer matches the active
request may update a safe shared cache only when the service contract permits,
but it may not overwrite the current form, status row, or generated artifact.

### CAT-006 — Dirty-config result attribution

If Settings contains unsaved connection changes, a Lab observation obtained
from the currently active saved configuration is labeled as applying to the
previously saved settings. It must not be displayed as proof that the dirty
draft works.

## Studio TTS Preferences

### CFG-020 — Separate persistence

Studio preferences are stored independently from global settings under the
logical schema:

```toml
[speech_studio]
schema_version = 1

[speech_studio.selection]
# Optional provider/model/voice/format/speed overrides.

[speech_studio.provider_options.<canonical_provider_id>]
# Optional, validated, request-scoped provider tuning.
```

The namespace is written through the existing atomic configuration owner. It
is a separate preference scope, not a second configuration writer or a new
file lifecycle. Choosing a different physical store requires an ADR amendment,
not an ordinary task decision.

### CFG-021 — Sparse inheritance

Studio values are optional overrides. Absence means inherit the effective
global value at request time. The UI shows the inherited value and its source
without copying it into Studio persistence.

`Reset to Global` deletes all Studio selection overrides and provider-option
subsections, leaving only the Studio schema/version envelope. It does not copy
current global values, save a second source of truth, reset global settings, or
change a character profile.

### CFG-022 — Provider-scoped options

Provider tuning is stored under an exact canonical provider ID. Switching
providers restores that provider's Studio options. Unknown provider IDs and
unknown or unsupported option keys fail closed and are not forwarded to an
adapter.

### CFG-023 — Studio actions

Studio provides:

- `Save Studio Preferences`;
- `Revert Studio Preferences`;
- `Reset to Global`; and
- `Open Global Speech & TTS Settings`.

Saving Studio preferences performs local validation and atomic Studio
persistence only. It does not reconfigure a provider because it changes no
credential, endpoint, initialization resource, or runtime-global setting.
Revert reloads the last successfully saved Studio snapshot without changing
global, character, catalog, or runtime state.

### CFG-024 — Current controls precedence

Within the mounted Studio, the current validated controls take precedence over
persisted Studio preferences for the next generation. This lets a user
experiment before saving. A failed or cancelled generation does not save the
draft.

### CFG-025 — Character/profile preview safety

Opening a character TTS profile in Studio loads a labeled preview state. It
does not change the Studio store, global defaults, or character assignment.

The only way to turn the preview into Studio preferences is an explicit
`Adopt as Studio Preferences` action followed by a successful Studio save.
`Save Studio Preferences` by itself never absorbs a preview that was not
adopted. Leaving or replacing an unadopted preview discards only the preview.

### CFG-026 — Unsupported tuning

Studio mounts only tuning proven operation-scoped and accepted by the selected
provider path. For audio.cpp, no arbitrary tuning is exposed; WAV and speed
`1.0` are read-only constraints. A legacy provider option that the bridge reads
only at adapter construction stays global or read-only until it has a real
per-request contract.

## Effective-setting resolution

### STATE-001 — Normal and roleplay requests

For normal chat, roleplay, media reading, and other non-Studio callers, resolve
each applicable request selection in this order:

1. explicit value intentionally supplied by the caller;
2. assigned character TTS profile, when the request has an authoritative
   assistant `CharacterRef` and the profile supports the operation;
3. global defaults; and
4. provider-declared fallback.

An invalid higher-precedence exact value blocks the affected request. It does
not fall through to a lower layer or another provider. Requests without an
authoritative assistant character identity do not use a character assignment.

### STATE-002 — Studio requests

For Lab Studio generation, resolve in this order:

1. current validated Studio controls, including an explicitly loaded preview;
2. persisted Studio preferences;
3. global defaults; and
4. provider-declared fallback.

Merely having a character selected elsewhere in the app does not inject that
character's profile into Studio. The profile must be explicitly opened as a
preview.

### STATE-003 — Resolution is one coherent snapshot

The resolver produces one immutable, validated effective-selection snapshot
before request admission. Provider, model mode/value, voice mode/value, format,
speed, validated options, source metadata, and relevant preference revisions
are frozen together. Adapter lease admission then follows the existing
configuration-revision contract.

The snapshot reports a source for each effective axis (`explicit`,
`character_profile`, `studio_draft`, `studio_saved`, `global`, or
`provider_fallback`) for UI explanation and deterministic tests. It does not
contain credential values or synthesis text.

### STATE-004 — Dynamic modes

`First available` resolves exactly once at request admission against the
accepted provider catalog. `Server default` is represented by omitting the
voice according to the adapter contract. Neither mode writes the resolved
ephemeral identifier back to global, Studio, or character storage.

### STATE-005 — No silent fallback

The resolver may inherit only when a higher layer has no value for that axis.
It may not inherit past an invalid, missing exact, unsupported, or
revision-incoherent value. Provider selection is never silently changed after
an error.

## State and status model

### STATE-010 — Configuration vocabulary

Configuration uses only these primary states:

| State | Meaning |
| --- | --- |
| `Inherited` | No value exists at this scope; a named lower-precedence owner supplies it |
| `Default` | Shipped/provider fallback supplies the value |
| `Saved` | The scope's value is durably stored and locally valid |
| `Unsaved` | The mounted valid draft differs from its saved snapshot |
| `Incomplete` | A required field is absent |
| `Invalid` | A supplied value fails local or authoritative exact-selection validation |

Configuration labels never claim network reachability.

### STATE-011 — Runtime vocabulary

Runtime uses only these primary states:

| State | Meaning |
| --- | --- |
| `Not checked` | No runtime observation applies to the saved configuration revision |
| `Checking` | An explicit runtime operation is in progress |
| `Ready` | A successful observation applies to the named saved configuration revision |
| `Stale` | A prior observation exists but is too old or applies to an earlier revision |
| `Unavailable` | A completed observation found a safe, actionable failure |
| `Reconfiguring` | Saved adapter-affecting configuration is draining or being replaced |

“Never checked” is not Ready. A stale Ready observation is Stale, not Ready.

### STATE-012 — Independent capability rows

Speech UI reports at least these independently:

- selected TTS provider configuration;
- selected TTS provider runtime;
- TTS catalog/voice freshness when relevant; and
- STT/local dependency availability.

Missing Kokoro, Chatterbox, Higgs, or STT local dependencies must not mark an
external audio.cpp configuration unavailable. Conversely, a reachable
audio.cpp server says nothing about STT readiness.

### STATE-013 — Revisioned runtime snapshot

Each displayed runtime observation carries:

- canonical provider ID;
- saved provider configuration revision;
- runtime adapter revision when available;
- catalog revision when applicable;
- observed-at timestamp;
- freshness classification; and
- a bounded safe diagnostic and recovery action.

The UI renders a result only in the matching scope. An older result cannot
overwrite a newer check or a newly saved provider revision.

### STATE-014 — Artifact independence

Changing configuration, Studio controls, a catalog, or runtime status does not
invalidate an already completed audio artifact. Playback and export remain
available from the artifact's immutable provenance unless the artifact itself
is cleared or replaced.

## Persistence and migration

### MIG-001 — Versioned Studio schema

Studio preferences have an explicit positive schema version. Parsing is
strict for known structural types and permissive only for forward-compatible
container presence where doing so cannot cause an unsupported option to be
executed.

### MIG-002 — One-time classification migration

On first read of Studio preferences after this program ships, migration may
copy only legacy fields proven to be request-scoped Studio tuning. It must:

- preserve all existing global connection, credential, initialization, and
  default-selection values;
- copy no secret, masked placeholder, environment value, endpoint, or runtime
  resource path into Studio;
- be versioned and idempotent;
- record only safe, field-name-level diagnostics;
- fall back field-by-field rather than discard every valid field because one
  value is malformed; and
- avoid a startup write when no migration is necessary.

### MIG-003 — Compatibility reads

Existing global and legacy provider keys remain readable while callers move to
the shared resolver. This program does not delete or rename legacy keys merely
to clean up configuration. Canonical writes may continue current dual-write
behavior where an accepted ADR already requires it.

### MIG-004 — Corruption isolation

A corrupt Studio record resets or quarantines only Studio preferences after a
safe warning. Global settings, character profiles, assignments, credentials,
and legacy provider behavior remain intact. Recovery never silently promotes a
character profile or global value into Studio storage.

### MIG-005 — Atomicity and concurrency

Global and Studio writes use the repository's single configuration owner,
atomic replacement, and existing process/thread safety rules. Each scope has a
revision or equivalent compare-before-publish snapshot so stale UI drafts do
not overwrite a newer successful save without warning.

### MIG-006 — No down-migration requirement

Rollback relies on additive storage and compatibility reads. Older code may
ignore `[speech_studio]`; no destructive down-migration is required. A failed
Studio migration preserves prior global behavior and leaves the Studio layer
inactive.

## Security and privacy

### SEC-001 — Credential boundary

Follow ADR-012. Credentials are loaded only through the established
environment/config boundary. Secret values never enter Studio preferences,
character profiles, catalog/status snapshots, navigation context, diagnostics,
or generated artifact provenance.

### SEC-002 — Masked-input safety

Masked display strings are presentation only. Save payloads are constructed
from explicit mutation intent, not widget text. Environment-owned values are
read-only. Clearing a saved local credential never attempts to mutate the
process environment.

### SEC-003 — Synthesis text privacy

The external audio.cpp form and generation UX disclose that submitted text is
sent to the configured server. Synthesis text and provider response bodies must
not be written to persistent logs, diagnostics bundles, runtime status,
metrics, catalog caches, exception strings, or migration records.

Generated audio artifacts retain only the already-approved in-memory or user-
selected export behavior. This PRD does not create a new automatic audio or
text history store.

### SEC-004 — Safe diagnostics

Diagnostics may contain canonical provider ID, bounded error category,
revision numbers, timestamps, configured-origin classification such as
`loopback`/`remote` when safely derived, and recovery action. They must not
contain credentials, query strings, raw URLs with embedded user info, local
model contents, submitted text, raw upstream response bodies, or arbitrary
exception strings.

### SEC-005 — Test and screenshot privacy

Automated fixtures, screenshots, and manual UAT documentation use synthetic
text and non-secret endpoints. Live UAT credentials and model paths are
user-supplied and never committed.

## Errors and recovery

### STATE-020 — Field-specific validation

Local validation places the error adjacent to the responsible control, keeps
focus reachable, and uses safe copy that does not echo secret or unbounded
input. A summary may link to invalid fields but cannot be the only error
presentation.

### STATE-021 — Saved but unavailable

When valid configuration cannot connect, the page continues to show `Saved`
for configuration and `Unavailable` for runtime. Recovery offers `Open Speech
Lab to test`, `Edit connection`, or the provider-specific safe action. It does
not undo the save or select a fallback provider.

### STATE-022 — Reconfiguration failure

If an adapter handoff fails after persistence, the provider remains saved and
truthfully unavailable. Other providers remain available. A later explicit
operation may retry according to the existing TTS service contract; there is
no automatic configuration rewrite.

### STATE-023 — Missing exact selection

The missing ID remains displayed. Recovery offers refresh, choose another
exact value, or intentionally select the relevant dynamic mode. Only the user
may make that selection change.

### STATE-024 — Corrupt Studio preferences

The UI explains that Studio-only preferences could not be loaded and offers a
Studio reset. It does not suggest resetting global TTS configuration or
character profiles.

## Accessibility and responsive behavior

### A11Y-001 — Programmatic labeling

Every input, selector, disclosure, status, and action has a programmatic label
and concise help text where the field's scope or units are not obvious.
Placeholders are examples, not labels.

### A11Y-002 — Keyboard flow

The category and Studio pane are fully usable without a mouse. Focus order
follows visual order, collapsible Advanced content is reachable, validation
returns focus to the first invalid field on request, and navigation actions do
not trap focus.

### A11Y-003 — Non-color state

Every configuration, runtime, freshness, dirty, invalid, and read-only state
uses text or an icon with an accessible label. Color is supplementary.

### A11Y-004 — Disabled reasons

A disabled selector or action has adjacent copy explaining why and the
recovery action. Audio.cpp's fixed WAV/speed controls are labeled as contract
constraints, not merely grayed out.

### A11Y-005 — Narrow terminal

At the repository's supported narrow terminal gate, the page remains
scrollable, labels do not overlap values, action strips wrap or stack, the
current scope remains understandable, and Save/Cancel recovery remains
reachable. Horizontal scrolling is not required for primary setup.

### A11Y-006 — Stable status announcements

Checking, save, reconfiguration, catalog refresh, generation, and error state
changes are announced through the app's accessible status mechanism without
moving keyboard focus unexpectedly.

## Functional acceptance criteria

The program is complete only when all of the following are true:

- [ ] **IA-001/IA-002:** A user can find Speech & TTS from main Settings and
  provider-name search.
- [ ] **IA-003/IA-004:** Global and Studio editors always state their scope and
  link to the other surface.
- [ ] **OWN-001 through OWN-006:** Every existing built-in field/action is
  inventoried exactly once; connection/credential/init values are global,
  supported request tuning is Studio-scoped, and runtime actions stay in Lab.
- [ ] **CFG-001 through CFG-011:** Global defaults and the selected provider
  can be edited with local validation, explicit credential intent, truthful
  save/reconfiguration outcomes, and no exact-selection substitution.
- [ ] **CFG-004:** audio.cpp exposes the complete accepted external adapter
  configuration and no managed-mode field or action.
- [ ] **CAT-001:** Mounting, searching, editing, saving, reverting, and restoring
  Settings cause no network request.
- [ ] **CAT-002 through CAT-006:** Cached exact choices, missing values,
  freshness, model-scoped voices, revisions, and dirty-config attribution are
  deterministic and truthful.
- [ ] **CFG-020 through CFG-026:** Studio preferences persist separately,
  inherit sparsely, reset by deletion, remain provider-scoped, never trigger
  provider reconfiguration, and do not absorb character previews implicitly.
- [ ] **STATE-001 through STATE-005:** Normal/roleplay and Studio resolution
  follow the approved precedence matrices and fail closed on invalid exact
  values.
- [ ] **STATE-010 through STATE-014:** Configuration, runtime, and catalog
  states remain separate and old artifacts remain playable.
- [ ] **MIG-001 through MIG-006:** Migration is versioned, additive,
  idempotent, secret-free, corruption-isolated, and backward compatible.
- [ ] **SEC-001 through SEC-005:** Credentials and synthesis text remain inside
  their approved boundaries.
- [ ] **A11Y-001 through A11Y-006:** Both surfaces pass keyboard, labeling,
  non-color-state, disabled-reason, announcement, and narrow-terminal gates.
- [ ] Existing OpenAI, ElevenLabs, Kokoro, Chatterbox, Higgs, and AllTalk
  generation behavior has no unapproved regression.
- [ ] A first-time external audio.cpp user can save a server URL, deliberately
  test/refresh in Lab, choose an exact or dynamic selection, generate a
  complete WAV, and play it.
- [ ] A roleplay response with an assigned character profile uses that exact
  profile while another response without the assignment uses global defaults.

## Verification strategy

### Deterministic automated coverage

Normal CI must use fakes and pinned fixtures and must not require network,
audio hardware, a local speech model, or a running audio.cpp server.

Required automated coverage includes:

1. **Field inventory completeness:** every legacy Speech control/action/status
   is classified once; ownership and mounted destination agree.
2. **Precedence matrices:** table-driven tests for every presence/absence and
   invalid-exact boundary in normal, roleplay, and Studio resolution.
3. **Studio storage:** sparse inheritance, provider switching, atomic save,
   reset-by-deletion, revisions, corruption isolation, and schema migration.
4. **Migration:** repeat execution is a no-op; secrets/endpoints/init paths are
   never copied; malformed fields do not erase valid independent fields.
5. **Credentials:** environment source, local Set/Replace/Clear, masked
   placeholder rejection, shadowed local fallback, and redacted diagnostics.
6. **Search and deep links:** search vocabulary, provider context, no
   auto-action, and exact return state.
7. **Draft navigation:** Save/Discard/Cancel for category, provider, deep-link,
   and screen-dismiss paths.
8. **No hidden network:** instrumented service asserts zero provider calls on
   mount/search/edit/save/revert/default restoration.
9. **audio.cpp field completeness:** every external adapter setting round-trips
   with bounds validation; no managed setting is accepted or rendered.
10. **Catalog integrity:** fresh/stale/missing/unverified states, model-scoped
    voices, configuration/catalog revisions, and stale async-result rejection.
11. **Reconfiguration:** only changed provider-global adapter inputs trigger a
    handoff; Studio saves and selection-only changes do not.
12. **Runtime truthfulness:** external audio.cpp can be Ready while unrelated
    local dependencies are absent; TTS and STT rows remain independent.
13. **Playback handoff:** a complete pinned WAV fixture reaches the existing
    playback artifact path without claiming incremental streaming.
14. **Responsive/accessibility:** Textual pilot tests at normal and narrow
    dimensions cover focus order, labels, disabled reasons, wrapped actions,
    status copy, and keyboard-only completion.
15. **Legacy regression:** existing provider configuration and generation
    fixtures still resolve and produce the same supported request shape.

### Manual live UAT

Live UAT is required before final release but is not part of ordinary CI. It
uses a user-supplied, already running external `audiocpp_server` and model.
Chatbook must not download or start either resource.

The acceptance record distinguishes:

- **headless proof:** valid complete WAV bytes, artifact provenance, and
  playback-control handoff; and
- **human proof:** the user or tester hears the expected spoken synthetic text
  through the console page's playback flow.

## UAT journeys

### UAT-01 — First-time external audio.cpp setup

1. Start from an app configuration with no audio.cpp values.
2. Find `Speech & TTS` through Settings search within 60 seconds, without docs
   or raw TOML.
3. Select audio.cpp under `Configure Provider`, enter the user-supplied URL,
   and save.
4. Verify Settings reports `Saved` and runtime `Not checked` rather than Ready.
5. Open Lab using the provided action, test the connection, and refresh models.
6. Open the Console/Roleplay response flow, generate a synthetic assistant
   character response, invoke its TTS action, and verify the complete WAV plays
   audibly through the response's playback control.

### UAT-02 — Offline save and recovery

Save a locally valid external URL while the server is stopped. Confirm the
configuration remains Saved, the explicit Lab test reports Unavailable, no
fallback provider is selected, and a later test becomes Ready after the user
starts the server.

### UAT-03 — Exact and dynamic choices

Refresh a multi-model catalog, choose an exact model/voice globally, navigate
away and back, and verify exact restoration. Then intentionally choose `First
available` and `Server default` and verify those modes persist without writing
ephemeral resolved IDs.

### UAT-04 — Studio persistence and isolation

Save a Studio-only selection, leave and return, and verify restoration. Confirm
the global inspector and a normal non-Studio request are unchanged.

### UAT-05 — Reset Studio to global

Change a global default after saving a Studio override. Confirm the Studio
still uses its override. Choose `Reset to Global`, save, change the global
value again, and confirm the Studio inherits the new value because its override
was deleted rather than copied.

### UAT-06 — Character roleplay precedence

Assign a supported exact audio.cpp TTS profile to one canonical character.
Generate and play that character's response from the Console/Roleplay flow and
confirm its profile wins. Generate a response without that assignment and
confirm global defaults apply. Confirm Studio preferences are unchanged.

### UAT-07 — Character preview safety

Open the character profile in Studio, generate and play a preview, leave
without adopting, and verify saved Studio preferences are unchanged. Repeat,
choose `Adopt as Studio Preferences`, save, and verify only Studio persistence
changes.

### UAT-08 — Environment-managed credential

Start with a supported provider credential in an environment variable. Confirm
the source is shown without its value, ordinary Save does not create a local
secret, masked text is never persisted, and clearing a local fallback does not
affect the environment.

### UAT-09 — Existing legacy provider

Open each existing provider's global connection/init fields, confirm legacy
values are preserved, save a Studio-supported tuning change, and run the
provider's existing generation fixture or available live smoke without an
unapproved request-shape change.

### UAT-10 — Independent dependency status

Run with local Kokoro/Chatterbox/Higgs or STT dependencies absent and an
external audio.cpp server available. Confirm audio.cpp can be Ready and
generate/play audio while the unrelated local dependency rows independently
report their true state.

## Rollout and task-decomposition seams

The next agent must translate the program into atomic, testable Backlog tasks
in dependency order. The following are delivery seams, not pre-created tasks:

1. **Authority foundation:** ADR adoption, built-in field ownership manifest,
   shared resolver/source metadata, revisioned status projection, and bounded
   navigation context.
2. **Studio persistence foundation:** versioned sparse store, migration,
   corruption isolation, and resolver integration without changing visible
   Lab ownership yet.
3. **Global Settings category:** discoverability, global defaults, all-provider
   connection/init fields, credential intent, inspector, and local-only save.
4. **audio.cpp provider experience:** full external form, Advanced limits,
   cached exact choices, freshness/missing states, and Lab recovery links.
5. **Studio transition:** rename, scope banner, provider-scoped controls,
   Save/Revert/Reset, character preview adoption, and removal of global fields
   only after their Settings replacements exist.
6. **Truthful runtime and navigation:** independent capability rows,
   revision-safe result display, dirty draft handling, and bidirectional deep
   links.
7. **Hardening and UAT:** accessibility, narrow layouts, privacy regression,
   legacy-provider regression, deterministic end-to-end tests, and manual live
   audio.cpp UAT.

Each PR must deliver one independently verifiable outcome and must not remove a
legacy UI path before its replacement is available. An agent may combine or
split seams based on the then-current code, but may not create an omnibus task
that attempts the whole program in one PR.

Every implementation plan must include:

```text
ADR required: yes
ADR path: backlog/decisions/039-global-and-studio-tts-settings-ownership.md
Reason: The task implements the accepted global/Studio TTS ownership,
        persistence, precedence, runtime-status, or navigation boundary.
```

If a task is purely mechanical or test-only, it may say `ADR required: no` but
must still list ADR-039 under conformance and explain why it makes no new
decision.

Before task creation, the decomposition agent must:

1. fetch and inspect the latest `dev`;
2. read the corresponding Backlog task files and current canonical ADR index;
3. re-run the field/control inventory against the current Speech UI;
4. identify already-landed behavior so tasks do not duplicate work;
5. give each task outcome-oriented acceptance criteria with requirement IDs;
6. order foundations before consumers and UI removal after replacement; and
7. leave managed audio.cpp and native legacy-provider migration out of scope.

## Release gates and success measures

- A first-time user can reach global audio.cpp setup from Settings search in 60
  seconds without documentation or raw TOML.
- Global, Studio, character, and runtime ownership is explicit in UI copy and
  mechanically enforced by persistence boundaries.
- Normal Settings activity performs zero provider network calls.
- A Studio save produces zero global config mutations and zero adapter
  reconfigurations.
- No missing exact model/voice is silently replaced.
- No credential or submitted synthesis text appears in prohibited stores or
  diagnostics.
- Deterministic CI passes without a live server; manual live UAT proves audible
  playback through the console page.
- Existing provider behavior remains covered through the legacy bridge.
- No managed audio.cpp setting, process action, or ownership claim appears.
- No release-blocking priority-zero finding remains.
- Every priority-one finding is fixed or rejected with technical evidence and
  explicit user approval.
- A priority-two finding may be deferred only when it violates no acceptance
  criterion and has a separately created follow-up Backlog task.

## Rollback strategy

- The rollout is additive until the replacement global fields and Studio store
  are proven.
- Existing global/legacy keys are not deleted.
- `[speech_studio]` is inert to an older reader and requires no down-migration.
- Disabling or reverting the Studio resolver reader restores prior global
  behavior without touching stored Studio data.
- A failed migration keeps prior behavior and does not publish a partial Studio
  snapshot.
- Removing the old Lab global editor occurs in a separate change after its
  replacement passes regression tests; reverting that removal restores the
  prior UI.
- Rollback never copies character assignments into another scope, silently
  selects a provider, or rewrites an invalid exact choice.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Splitting one form causes a setting to disappear | Explicit field inventory plus mounted-control completeness test |
| Studio and global values drift into two sources of truth | Sparse Studio overrides, source metadata, reset-by-deletion, one shared resolver |
| Catalog refresh races with selectors | Configuration/catalog/model revisions and stale-result rejection |
| Environment credential is accidentally persisted | Intent-based secret mutation; ordinary Save excludes credential widgets |
| Legacy provider tuning is mislabeled as request-scoped | Verify actual adapter consumption; unsupported operation-scoped claims fail closed |
| External audio.cpp is reported unavailable because a local dependency is missing | Independent provider/runtime/dependency status rows |
| Settings save unexpectedly blocks on a server | Local-only validation and explicit Lab operations |
| Character preview changes Studio behavior | Preview state is non-persistent until explicit adoption and save |
| Task decomposition becomes an oversized rewrite | Required dependency seams, atomic PR rule, stable requirement IDs |
| Managed-server work leaks back into scope | Explicit field rejection, tests, non-goal, and ADR boundary |

## Traceability matrix

| Audit issue | Requirements | Primary verification |
| --- | --- | --- |
| Speech configuration is hard to find | IA-001, IA-002, IA-003 | Automated search/deep-link tests; UAT-01 |
| Global and Studio scopes are conflated | OWN-001, OWN-002, CFG-020, IA-004 | Storage mutation tests; UAT-04 and UAT-05 |
| Configuration and runtime health are conflated | CFG-005, STATE-010, STATE-011, STATE-012 | Runtime projection tests; UAT-02 and UAT-10 |
| Exact audio.cpp choices are weak or silently mutable | CFG-011, CAT-002, CAT-003, CAT-004, CAT-005 | Catalog revision tests; UAT-03 |
| Provider form is dense and actions are mixed with settings | OWN-005, CFG-001, CFG-003, CFG-004, CFG-006 | Field inventory and narrow-layout tests; UAT-01/UAT-09 |
| Recovery navigation can lose context or drafts | IA-005, CFG-012, CAT-006 | Draft/deep-link tests; UAT-02 |
| Studio has no durable isolated preference layer | CFG-020 through CFG-024, MIG-001, MIG-005 | Store/resolver tests; UAT-04/UAT-05 |
| Character preview could be mistaken for persistence | CFG-025, STATE-002 | Preview tests; UAT-06/UAT-07 |
| Credential placeholders could become persisted values | CFG-008, SEC-001, SEC-002 | Credential mutation tests; UAT-08 |
| External synthesis privacy is implicit | CFG-004, SEC-003, SEC-004, SEC-005 | Redaction tests and synthetic live UAT |
| Managed audio.cpp could leak into this program | CFG-004, non-goals, release gates | Field completeness/rejection tests; UAT-01 |
| Legacy providers could regress during field movement | OWN-005, CFG-003, MIG-003 | Legacy fixtures; UAT-09 |

## Handoff contract

This PRD and ADR-039 are the authoritative design inputs for task
decomposition. The next agent may clarify implementation mechanics by reading
the latest code, but must return for product approval before changing:

- which scope owns a field;
- the normal/roleplay or Studio precedence order;
- separate Studio persistence;
- the explicit preview-adoption rule;
- credential mutation boundaries;
- no-hidden-network behavior;
- exact-choice fail-closed behavior;
- the external-only audio.cpp boundary; or
- the release requirement for audible manual UAT.
