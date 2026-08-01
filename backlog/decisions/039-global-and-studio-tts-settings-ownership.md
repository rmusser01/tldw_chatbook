# ADR-039: Global and Studio TTS Settings Ownership

Status: Accepted
Date: 2026-07-31
Related Task: N/A — program task decomposition follows the approved PRD
Extends:
[ADR-012 Provider Credential Settings Boundary](012-provider-credential-settings-boundary.md),
[ADR-023 TTS Adapter Registry and audio.cpp Runtime Boundary](023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md),
[ADR-028 Character TTS Generation Profile Ownership](028-character-tts-generation-profile-ownership.md), and
[ADR-037 Roleplay Assistant Identity and Persona/User Profile Separation](037-roleplay-assistant-identity-and-persona-user-profile-separation.md)

## Context

Chatbook has an app-scoped TTS adapter registry, a native external audio.cpp
adapter, complete-WAV generation, global TTS defaults, a Lab Speech playground,
and exact character TTS generation profiles. Six existing providers remain
behind the accepted temporary legacy bridge.

The current product surface still places application-wide defaults,
credentials, endpoints, local runtime initialization, request-scoped tuning,
voice-resource actions, readiness checks, catalog discovery, and Studio
generation controls in one Lab settings experience. The primary Settings
destination does not expose a first-class Speech & TTS category, even though
durable provider configuration and credential ownership belong there under
ADR-012. Users cannot reliably tell whether an edit affects the whole
application, the Speech Studio, or one character.

The UI also needs to distinguish three facts that can differ at the same time:

1. a configuration is locally valid and saved;
2. a provider was observed ready for a particular configuration revision; and
3. a model/voice catalog observation is current for a particular provider,
   model, and revision.

A combined local-dependency status is not authoritative for an external
audio.cpp server. A missing local TTS or STT dependency must not make a usable
external provider appear unavailable.

The Speech Studio needs durable convenience preferences, but reusing global
settings would make experiments change normal and roleplay behavior. Copying
global values into a Studio record would create two defaults that drift. The
existing character profile repository must remain separately authoritative and
must not be turned into Studio storage.

This is a significant storage, data-ownership, provider-runtime, precedence,
credential, and long-lived navigation decision. A canonical ADR is therefore
required before task planning or implementation.

## Decision

### Four explicit owners

Chatbook will use four non-overlapping product owners:

- **Settings → Speech & TTS** owns durable application-wide defaults,
  credentials, endpoints, runtime initialization resources, external
  audio.cpp configuration, and provider safety limits.
- **Lab → Speech → Studio TTS Preferences** owns separately persisted,
  provider-scoped, request-level Studio overrides and the current Studio
  generation draft.
- **Character TTS profiles** retain ownership of exact character-specific TTS
  generation selections and canonical character assignments.
- **The TTS runtime and Lab operational surface** own health checks, catalog
  and voice refresh, synthesis, progress, playback, export, and runtime
  diagnostics.

One field or action may have only one durable UI owner. Other surfaces may
display effective values and link to the owner, but may not create a second
write path.

### Global Settings surface

The main Settings destination will add a searchable `Speech & TTS` category.
It will state that the user is editing global application defaults and will
link to the Lab for runtime operations.

The category will separate `Default TTS Provider` from `Configure Provider` so
editing a provider does not make it the default. It will show only the selected
provider's setup form rather than mounting every provider's full form.

Global Settings will expose connection, credential, or runtime-initialization
fields for all existing providers:

- audio.cpp: external server origin, connection and synthesis timeouts, input
  and response bounds, metadata/catalog/voice/identifier bounds, and privacy
  disclosure;
- OpenAI: API key, base URL, and organization ID;
- ElevenLabs: API key;
- Kokoro: device, ONNX choice, model file, and voices file;
- Chatterbox: device and voice-resource directory;
- Higgs: model path, voice-resource directory, device, flash attention, and
  dtype; and
- AllTalk: server URL.

The exact field-key inventory will be maintained as an explicit, testable map
of built-in fields. It is not a generic plugin, dynamic schema, or form
framework. audio.cpp is the only provider that receives a complete redesigned
provider-detail experience in this program. Existing providers keep their
legacy generation contracts and recognizable tuning behavior.

Global Save performs local validation and persistence only. Mounting,
searching, editing, saving, reverting, and restoring settings do not check a
connection, initialize a model, refresh a catalog, or synthesize audio.
Persistence and targeted provider reconfiguration have separate, truthful
outcomes. A configuration may be Saved while runtime is Not checked,
Unavailable, Stale, or Reconfiguring.

### External-only audio.cpp boundary

This program supports connecting to one independently started external
`audiocpp_server`. The global form will not accept or expose a binary,
`server.json`, bind address, launch policy, restart action, log supervisor, or
process status. Chatbook does not launch, adopt, restart, supervise, or stop an
audio.cpp process in this program.

The native adapter retains its accepted complete-WAV asynchronous response
contract, WAV-only output, speed exactly `1.0`, one active instance, and
explicit no-fallback behavior from ADR-023.

The configured audio.cpp value is a canonical HTTP(S) origin without userinfo,
query, fragment, or a non-origin path. A non-loopback plain-HTTP origin remains
supported but receives an explicit warning that submitted text and returned
audio are not transport-encrypted.

### Studio persistence

Studio preferences will use a separate, versioned namespace written through
the existing atomic configuration owner:

```toml
[speech_studio]
schema_version = 1

[speech_studio.selection]
# Sparse provider/model/voice/format/speed overrides.

[speech_studio.provider_options.<canonical_provider_id>]
# Sparse, validated, request-scoped options.
```

The namespace contains no credential, endpoint, runtime initialization,
provider safety limit, character assignment, or secret-derived value. Missing
values inherit global defaults at request time. `Reset to Global` deletes
Studio overrides; it never copies current global values into the Studio
namespace.

Provider tuning may enter Studio persistence only when the selected request
path accepts it per operation. Unknown provider IDs and unsupported option keys
fail closed. A legacy setting consumed only during adapter construction remains
global or read-only until a real request-scoped contract exists.

Saving Studio preferences never reconfigures a provider and never writes a
global or character store.

### Character preview boundary

A character TTS profile opened in Studio is a non-persistent preview. It may be
used by the current Studio generation, but it does not mutate Studio
preferences, global defaults, the profile, or its assignment.

The user must explicitly choose `Adopt as Studio Preferences` and then
successfully save Studio preferences to make the compatible selection durable
for Studio. A general Studio Save cannot implicitly absorb an unadopted
preview.

### Effective-setting precedence

Normal, roleplay, media, and other non-Studio requests resolve each applicable
axis in this order:

1. an explicit caller value;
2. an assigned character TTS profile when the request carries an authoritative
   assistant `CharacterRef`;
3. global defaults; and
4. provider-declared fallback.

Studio requests resolve in this order:

1. current validated Studio controls, including an explicitly loaded preview;
2. persisted Studio preferences;
3. global defaults; and
4. provider-declared fallback.

Resolution produces one immutable effective-selection snapshot before request
admission and records a non-secret source for each axis. A higher-precedence
invalid, missing exact, unsupported, or revision-incoherent value blocks the
affected request. It never silently falls through to another value or
provider.

`First available` model and `Server default` voice remain deliberate dynamic
modes. They resolve at request admission and do not write their ephemeral
result back to any preference store.

### Catalog and revision behavior

Global exact model and voice selectors may display only the latest accepted
in-memory catalog observations already owned by the TTS service. No hidden
network discovery occurs from Settings, and this decision does not add a new
disk catalog.

When no catalog exists, audio.cpp offers only `First available` and `Server
default` plus a link to an explicit Lab refresh. Fresh and stale observations
are labeled distinctly. A saved exact choice absent from the latest complete
observation remains visible and invalid; it is never automatically cleared,
substituted, or converted to a dynamic mode. Ambiguous voice discovery is
Unverified rather than authoritative empty success.

Runtime observations are associated with canonical provider ID,
provider-configuration revision, catalog revision when relevant, model,
timestamp, and freshness. Stale asynchronous results cannot overwrite newer
form, status, or artifact state. A result from the currently saved
configuration cannot prove that an unsaved connection draft works.

### Configuration and runtime state

Configuration uses the primary states `Inherited`, `Default`, `Saved`,
`Unsaved`, `Incomplete`, and `Invalid`.

Runtime uses the primary states `Not checked`, `Checking`, `Ready`, `Stale`,
`Unavailable`, and `Reconfiguring`.

The UI will report provider configuration, selected-provider runtime,
catalog/voice freshness, and STT/local dependency status independently. It will
never treat `Not checked` as Ready or a missing unrelated local dependency as
proof that external audio.cpp is unavailable.

### Credential boundary

ADR-012 remains authoritative. Environment and local-config credential sources
are shown without exposing their values. Environment-owned secrets are
read-only and never copied into local configuration.

Credential `Set`, `Replace`, and `Clear saved credential` are explicit
mutations separate from ordinary configuration Save. A masked placeholder or
environment value can never become a persistence payload. Studio preferences,
navigation context, status/catalog snapshots, diagnostics, and artifact
provenance contain no secret values.

The UI labels local-config secret storage and recommends the already-supported
environment source. This decision does not add keyring or encrypted credential
storage.

### Privacy boundary

The external audio.cpp form and Lab generation flow disclose that synthesis
sends submitted text to the configured server. Submitted synthesis text and
raw provider response bodies will not be written to persistent logs,
diagnostics, metrics, catalog caches, migration records, runtime status, or
error messages.

### Migration and rollback

Migration is additive, versioned, and idempotent. Existing global connection,
credential, initialization, default, and legacy-provider keys remain in place.
A one-time Studio migration may copy only fields proven request-scoped; it
copies no secret, endpoint, runtime resource path, environment value, or masked
placeholder.

Compatibility reads remain until all consumers use the shared resolver. This
program does not delete legacy configuration keys.

A malformed Studio record affects only Studio. Global configuration, character
profiles, assignments, credentials, and legacy behavior remain usable. Older
code may ignore the additive `[speech_studio]` namespace. Rollback can disable
the Studio reader or restore the prior Lab editor without a destructive
down-migration.

## Consequences

- Users gain one discoverable global setup surface and one explicitly isolated
  Studio preference surface.
- First-time audio.cpp setup becomes a URL-first Settings flow followed by
  explicit Lab verification and generation.
- The Settings category can truthfully save configuration while a provider is
  offline because persistence no longer implies readiness.
- Studio experiments stop changing application-wide defaults.
- Character roleplay retains exact, separately authoritative voice assignment.
- Exact missing model/voice selections may block generation until the user
  resolves them; this is intentionally safer than silent substitution.
- Catalog choices in Settings may be sparse until the user refreshes in Lab;
  this is the visible cost of prohibiting hidden network activity.
- Connection and initialization fields for legacy providers move to global
  Settings, but their tuning and generation implementations are not generally
  redesigned.
- A shared resolver and field inventory add deliberate central contracts, but
  avoid a speculative provider-plugin or generic-form framework.
- The additive Studio namespace and compatibility reads permit staged rollout
  and rollback without deleting user configuration.
- Manual live UAT remains necessary to prove audible console playback; normal
  CI remains deterministic and server-free.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep all TTS settings in Lab | Preserves the mixed-scope form, leaves global setup hard to find, and conflicts with the Settings credential boundary. |
| Move every field into global Settings | Would make request-scoped Studio experiments mutate application defaults and would erase a useful Studio-specific workflow. |
| Use one editor with a Global/Studio scope toggle | A toggle makes the same controls change ownership and persistence target, increases accidental global edits, and makes links/recovery less truthful. |
| Keep global Settings shallow and deep-link to Lab for all provider setup | Improves discoverability only cosmetically; global credentials/endpoints would still be owned by a Studio surface. |
| Copy global values into Studio on reset | Creates two full default snapshots that drift and makes later global changes appear broken. Sparse inheritance is simpler and truthful. |
| Let a character profile automatically become Studio preferences | Makes previewing a character silently persistent and couples separate stores. Explicit adoption preserves user intent. |
| Discover models automatically when Settings opens or saves | Introduces hidden network work, blocks offline configuration, and conflates Save with readiness. |
| Replace a missing exact choice with the first available item | Can generate with the wrong voice/model and hides a broken character or global selection. |
| Fully redesign every provider behind one schema now | Expands the program into a legacy adapter rewrite and speculative form framework before those providers have native capability contracts. |
| Put credentials in Studio for convenience | Violates ADR-012, duplicates write paths, and risks secret leakage into Studio persistence. |
| Add managed audio.cpp setup now | Reintroduces binary trust, `server.json`, process ownership, supervision, and lifecycle concerns explicitly deferred from the external-integration workstream. |
| Persist catalog data to disk for Settings selectors | Adds a new cache lifecycle and privacy/invalidation contract that is unnecessary when explicit Lab refresh plus in-memory observations is sufficient. |

## Links

- [Product requirements and design](../../Docs/superpowers/specs/2026-07-31-speech-tts-settings-ownership-design.md)
- [ADR-012](012-provider-credential-settings-boundary.md)
- [ADR-023](023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md)
- [ADR-028](028-character-tts-generation-profile-ownership.md)
- [ADR-037](037-roleplay-assistant-identity-and-persona-user-profile-separation.md)
