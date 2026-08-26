# audio.cpp Guided Model Setup and Clone Profiles — Product Requirements and Design

Status: Approved by the user on 2026-08-09 after iterative design review

Date: 2026-08-09

Target branch: `dev`

Supersedes: only the generated-configuration, fixed-port-for-all-Managed, and
POSIX-only deferrals in the 2026-08-02 managed-lifecycle design. Its existing
user-provided `server.json`, ownership, state, admission, diagnostics, and
shutdown contracts remain normative.

Extends:

- the native adapter and complete-WAV contract in
  [ADR-023](../../../backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md);
- the managed child contract in the
  [audio.cpp managed-lifecycle design](2026-08-02-audio-cpp-managed-lifecycle-design.md);
- global/Studio ownership in
  [ADR-039](../../../backlog/decisions/039-global-and-studio-tts-settings-ownership.md);
  and
- character TTS profile ownership in
  [ADR-028](../../../backlog/decisions/028-character-tts-generation-profile-ownership.md).

Normative decisions:

- [ADR-050: Generated audio.cpp model setup](../../../backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md)
- [ADR-051: Private TTS clone reference assets](../../../backlog/decisions/051-private-tts-clone-reference-assets.md)

## Document purpose

This document defines the next audio.cpp workstream after external-server and
user-provided-binary/`server.json` support. It answers one concrete product
question:

> Can a new user install Chatbook, install `audiocpp_server` separately,
> download or select a supported model package, configure it without writing
> JSON, and hear a valid sample?

The required answer after this workstream is **yes for every exact
model/package/platform/backend tuple Chatbook labels as supported**. The final
program goal is every audio.cpp `release-0.5.1` family tagged `TTS` or `Clone`,
including the release's community table. Interim releases must state their
exact coverage and cannot imply that every audio.cpp model works.

This is a requirements and architecture document. It deliberately does not
create Backlog tasks or an implementation plan.

## Current-state baseline

At the approved `dev` baseline, Chatbook already has:

- an application-scoped TTS adapter registry with audio.cpp as the first native
  adapter;
- an External source for an independently managed audio.cpp server;
- a Managed source using a user-provided executable and user-provided
  `server.json`;
- lazy launch, one owned child, health checks, exit supervision, explicit
  restart/shutdown, saved/applied/process generations, bounded memory-only
  diagnostics, and definitive application shutdown;
- complete PCM16 WAV responses presented through an asynchronous response
  interface;
- global Speech & TTS Settings separated from Studio preferences;
- Speech Lab catalog, generation, playback, export, and managed-runtime UX;
- reusable provider/model/voice profiles and character assignments; and
- an existing shared Model Library store for explicitly downloaded artifacts.

The remaining first-time gap is setup knowledge. A user must already know how
to translate a model package into audio.cpp's model entry, choose compatible
assets and backend, and write `server.json`. Clone-capable families also need a
private, typed owner for reference audio and transcript; the current profile
schema intentionally has neither.

## Upstream compatibility snapshot

The initial recipe registry is pinned to the official audio.cpp
[`release-0.5.1`](https://github.com/0xShug0/audio.cpp/releases/tag/release-0.5.1)
tag:

| Field | Pinned value |
| --- | --- |
| Tag | `release-0.5.1` |
| Commit | `238ab6a9e321c17de8e120559f57efeedaeb1345` |
| Release publication | 2026-08-04 |
| Server entry point | `audiocpp_server --config <server.json>` |
| Client contract | `/health`, `/v1/models`, `/v1/audio/speech` |
| Initial response contract | complete structurally validated WAV |

The release README and `model_specs/*.json` expose 21 unique families tagged
`TTS` or `Clone`, with 67 declared package entries in the pinned snapshot. The
family inventory is normative; package counts are an audit aid and may include
multiple precision, language, or checkpoint variants that require distinct
recipes.

### Core release families

| Family | Release tasks | Declared packages |
| --- | --- | ---: |
| `chatterbox` | TTS, Clone, VC | 3 |
| `confucius4_tts` | Clone | 1 |
| `dramabox` | TTS, Clone | 1 |
| `fish_audio` | TTS, Clone, control | 2 |
| `higgs_audio_tts` | TTS, Clone, control | 2 |
| `miotts` | TTS, Clone | 3 |
| `omnivoice` | TTS, Clone, design, control | 4 |
| `pocket_tts` | TTS, Clone | 11 |
| `qwen3_tts` | TTS, Clone, design, control | 9 |
| `vevo2` | TTS plus non-TTS tasks | 3 |
| `vibevoice` | TTS, dialogue | 2 |
| `voxcpm2` | TTS, Clone, design, control | 4 |
| `index_tts2` | TTS, Clone, control | 4 |
| `irodori_tts` | TTS, Clone, design, control | 6 |
| `moss_tts_nano` | TTS, Clone | 2 |
| `moss_tts_local` | TTS, Clone, control | 2 |
| `supertonic` | TTS | 4 |

### Community release families

| Family | Release tasks | Declared packages |
| --- | --- | ---: |
| `glm_tts` | TTS, Clone | 1 |
| `inflect_v2` | TTS | 1 |
| `outetts` | TTS, Clone | 1 |
| `vietneu_tts` | TTS, Clone | 1 |

`moss_tts_local` also appears in the release's community attribution table but
is counted once because it is already in the core supported-model table and
has one canonical model spec in this snapshot.

Inclusion in this inventory means “must be evaluated and either receive exact
approved recipes or remain an explicit release blocker.” It does not itself
assert that every family, package, operating system, or backend already works
through Chatbook.

## Goals

- Let a first-time user configure supported local audio.cpp packages without
  editing JSON.
- Preserve External mode and the existing user-provided `server.json` Managed
  source for expert or unsupported configurations.
- Support multiple configured TTS/Clone models in one lazy-loaded audio.cpp
  child.
- Auto-detect an already installed `audiocpp_server` and support manual browse,
  without installing or downloading it.
- Give exact, truthful compatibility and runtime evidence rather than a single
  overloaded “supported” state.
- Reuse the existing Model Library store for explicit curated downloads and
  return an installed package directly to the Settings draft.
- Support macOS, Linux, and Windows with exact platform/backend verification.
- Add typed, private, reusable clone references to TTS profiles.
- Preserve global Settings, Studio preferences, character profiles, and runtime
  operations as separate owners.
- Produce and audibly play a complete WAV in Speech Lab, including the
  reference-required path where applicable.

## Non-goals

- Downloading, bundling, building, signing, installing, or updating the
  audio.cpp server binary.
- A remote recipe registry or any recipe-supplied executable code.
- Guessing arbitrary GGUF compatibility from a filename or weak similarity.
- Editing generated JSON or making it a second persistent configuration owner.
- Multiple managed audio.cpp children, load balancing, failover, or process
  adoption.
- A general audio.cpp configuration editor exposing every upstream field.
- Copying a user-selected local model package into application data.
- Automatic model unloading, a VRAM scheduler, or a second resource manager.
- Changing Studio preference ownership or silently adopting setup into Studio.
- Native incremental playback/streaming in this workstream.
- Non-loopback managed binding or Managed CORS.
- Embedding clone audio in ordinary character cards or ordinary profile export.
- Sending a client-local clone path to an independently owned External server;
  that requires a future upload or remote-asset contract.
- A generic redesign of every provider's voice profile options.
- Claiming consent, speaker identity, provenance, authenticity, or forensic
  erasure.

## Product decisions

### GM-DEC-001 — Two Managed setup sources

Managed audio.cpp exposes exactly two setup sources:

1. **Chatbook-guided setup** — structured global Settings generate the launch
   artifact.
2. **Existing server.json** — the user supplies the executable and JSON exactly
   as in the current managed lifecycle.

Existing users remain on their current source after upgrade. Switching sources
preserves the dormant source's values so a user can switch back without
re-entering paths. Save never rewrites a user-provided JSON file.

### GM-DEC-002 — Server remains separately installed

Chatbook detects `audiocpp_server` on `PATH` and reviewed platform install
locations, and provides a file picker. It may show platform-specific
installation guidance and official links. It does not invoke Homebrew, a
system package manager, an installer, a compiler, or a download for the server.

The executable is validated on Save without running it. Version and capability
probing occur only during a deliberate Test/Start/replacement path.

### GM-DEC-003 — Built-in exact recipes

Guided setup recognizes packages through built-in declarative recipes. A recipe
is immutable within a Chatbook release, cannot execute code, and is reviewed
against an exact upstream release and package variant. Remote recipes and
silent runtime updates are prohibited.

Unknown packages are not rejected from audio.cpp generally. They remain usable
through the existing user-provided JSON source.

Implementation evidence recorded on 2026-08-10 refined the initial PocketTTS
classification. The pinned PocketTTS GGUF packages register in the shared
server, but a real release-0.5.1 synthesis requests a separate voice embedding
that is not contained in those standalone GGUF files. Their revision-2 recipes
therefore declare `Reference: Required`; they remain valid guided catalog
entries but are not eligible for the no-reference first sample. The PocketTTS
Safetensors package includes its reviewed embeddings and remains
`Reference: Optional`. The first text-ready onboarding sample uses Supertonic;
PocketTTS GGUF generation belongs to the typed voice/reference increment.

### GM-DEC-004 — One multi-model child

All accepted guided models are projected into one generated `server.json` and
one managed child. Top-level `lazy_load` is always true. audio.cpp registers all
model IDs at startup, loads the selected model on first use, and retains loaded
models until shutdown. Chatbook discloses that lazy loading is not unloading.

### GM-DEC-005 — Generated configuration is ephemeral authority evidence

Structured Settings are durable authority. A generated JSON file is an
immutable, generation-scoped launch artifact. It is materialized only when a
deliberate operation may launch or replace a child and is retained until that
exact child and generation-local clients/tasks are definitively reaped.

Chatbook never treats edits to that artifact as new settings and never
revalidates its source files on every synthesis while the matching child is
running.

### GM-DEC-006 — Complete WAV first

The native adapter keeps its asynchronous response interface but initially
emits one complete bounded, structurally validated WAV. Upstream streaming
support does not imply Chatbook streaming playback.

### GM-DEC-007 — Cross-platform, tuple-scoped verification

Generated setup targets macOS, Linux, and Windows. CPU is the baseline on each.
CUDA, Metal, Vulkan, and HIP/ROCm are labeled Verified only for exact tuples
covered by evidence. “Expected” and “Untested” are distinct from “Verified.”

### GM-DEC-008 — Typed clone references

Clone reference audio and transcript are typed profile data. They do not enter
generic provider options, connection settings, generated JSON, or character
cards. A transient first sample does not require profile creation; a successful
result may be saved later using the exact canonical reference bytes that
produced it.

### GM-DEC-009 — Onboarding ends at audible value

The primary onboarding user is new to audio.cpp package/configuration details,
not necessarily new to local software. The “aha” moment is hearing their own
valid sample, not completing a tour or reading every advanced setting.

Guided setup therefore uses the real Settings → Speech Lab workflow with
progressive disclosure and one next action. There is no separate tutorial mode,
forced walkthrough, or package-install ceremony. Experienced users skip the
guided path by retaining External or user-provided `server.json` setup.

## Ownership and component architecture

### GM-ARCH-001 — Owner matrix

| Concern | Durable owner | Runtime owner |
| --- | --- | --- |
| Setup source, binary selection, models, backend preference, global defaults | Settings/config owner | TTS service projection |
| Generated `server.json` | None; derived from structured settings | Managed generation/supervisor |
| User-provided `server.json` | User-selected file | Existing managed generation/supervisor |
| Recipe definitions | Installed Chatbook distribution | Recipe registry |
| Local selected model bytes | User-selected/shared path | audio.cpp child |
| Model Library downloads | Existing shared artifact store | Existing Model Library owner |
| Applied child, endpoint, process health, diagnostics | None | App-owned supervisor/TTS service |
| Studio provider/model/voice preferences | `speech_studio` namespace | Speech Lab |
| Reusable voice selection/reference | TTS profile repository | Profile admission/materialization |
| Current WAV and transient clone reference | None | Speech Lab current-result owner |
| Character assignment | TTS profile repository | Character-aware request admission |

No UI surface writes another owner's durable state. Settings says that edits
are global and links to Speech Lab. Speech Lab may display effective global
setup but cannot persist a second copy. A one-time setup preview intent never
overwrites Studio preferences.

### GM-ARCH-002 — Structured setup projection

The durable guided setup is a bounded structured model, not arbitrary JSON. It
contains:

- setup source;
- selected executable path or detected executable identity;
- accepted recipe projections and their stable public model IDs;
- default model ID;
- backend preference (`auto` or one supported explicit backend);
- optional device and thread controls admitted by the selected backend policy;
- bounded request/body/busy limits owned by Chatbook; and
- the existing lifecycle/synthesis safety limits.

It does not persist a runtime port, live endpoint, process ID, loaded-model set,
health, launch attempt, or Verified status. Those are observations or caches.

### GM-ARCH-003 — App-owned generation artifact

For each eligible launch, Chatbook writes one owner-private, immutable
`server.json` under the active user-data area. Its identity includes the applied
provider generation and launch attempt. The file uses atomic private creation,
is opened without following a symlink, and is not placed in the installed
package or current working directory.

The child receives exactly:

```text
[resolved_audiocpp_server, "--config", generated_server_json]
```

There is no shell and no arbitrary argument field. The existing allowlisted
non-credential environment boundary remains. The artifact and any attempt-
local directory are removed only after the exact child, pipe drains, exit
monitor, probes, HTTP clients, and generation cleanup are joined. Startup may
clean a recognized leftover only after proving no live ownership lock exists;
unknown files are left untouched.

### GM-ARCH-004 — Recipe registry boundary

The registry is a small, sealed collection of data records. It owns package
recognition and safe projection, but not process launch, HTTP, downloads,
profile storage, or UI state. A recipe cannot import Python, name a callback,
invoke a converter, execute a hook, interpolate an environment variable, or
add an unreviewed JSON/argv field.

The registry provides pure operations for:

- listing compatible recipe/package variants;
- bounded matching of a pre-scanned package description;
- validating one accepted normalized projection;
- projecting allowlisted server model fields; and
- mapping a recipe to reviewed Model Library artifact identifiers and source
  links.

### GM-ARCH-005 — Existing lifecycle authority remains

The app owns one `AudioCppSupervisor`; the registry remains lease and transition
authority; the native adapter remains HTTP/contract authority. Guided setup
does not add a second process manager, synthesis quota, catalog store, or
shutdown coordinator.

The saved configuration generation, applied provider generation, process
generation, and generated artifact identity remain distinct. Every client and
result is fenced to the exact applied/process generation as already required by
the managed-lifecycle design.

The native catalog contract expands narrowly from upstream `task = "tts"` to
the exact speech-task set `{"tts", "clone"}`. This is required for clone-only
families such as `confucius4_tts`. ASR, VC, Music, and every other audio.cpp
task remain excluded from the TTS adapter. The adapter cross-checks the
server-reported task/model ID against the applied recipe projection and
preserves typed text/clone capabilities in catalog evidence; the UI does not
infer clone support from a family name.

The provider-neutral request contract gains optional typed clone-reference
input for native audio.cpp admission. The existing generic `options` mapping
remains empty for audio.cpp profiles and is not used to smuggle `voice_ref` or
`reference_text`.

### GM-ARCH-006 — Windows ownership

On Windows, Chatbook owns only the exact process handle it creates and the
tasks/clients associated with that launch. It must use Windows-native
creation, wait, terminate, and definitive handle-close behavior. It cannot
claim ownership of arbitrary descendants or support builds that daemonize away
from the owned handle.

POSIX process-group behavior cannot be assumed to establish Windows safety.
Cross-platform release evidence must prove the matching child cannot survive
Chatbook's definitive shutdown path.

## Persistent configuration and launch projection

### GM-CFG-001 — Source selection and dormant values

The active audio.cpp Managed source is explicit. Guided and user-JSON fields
persist separately. Switching the selector changes only the active projection;
it does not erase the inactive draft or copy fields between them.

Existing configurations with a managed executable and JSON path continue to
read as the user-JSON source without a migration write. The default for a new
Managed setup is guided setup only after the user explicitly selects Managed.
External remains a separate source and retains existing behavior.

### GM-CFG-002 — Save remains side-effect free

Save performs bounded local validation, persists atomically, and publishes the
new saved generation. It does not:

- run the executable or query its version;
- choose or bind a runtime port;
- materialize the launch artifact;
- contact loopback or any remote origin;
- launch, restart, stop, or adopt a process;
- refresh catalog/voice data; or
- synthesize hidden audio.

A successful guided Save reports `Configuration saved — ready to test`. It
does not report Running, Connected, Ready, or Verified.

### GM-CFG-003 — Safe generated top-level fields

Generated JSON always writes:

- `host: "127.0.0.1"`;
- one launch-selected bounded loopback `port`;
- the resolved backend and validated device/thread projection;
- `lazy_load: true`;
- `log_request_body: false`;
- bounded request-body and busy-timeout policy; and
- the exact accepted `models` list.

Generated setup never writes `cors_origins`, never enables request-body logs,
and never accepts an arbitrary top-level extension. New upstream fields require
a recipe/schema review and a design amendment if they widen trust or privacy.

### GM-CFG-004 — Safe model fields

Each generated model entry has a stable public ID and an allowlisted projection
from its accepted recipe. The maximum initial field set is:

- `id`;
- `family`;
- absolute `path`;
- recipe-declared speech `task` (`tts` or `clone` as required by the exact
  upstream family/session contract);
- supported `mode` (`offline` initially unless a future Chatbook streaming
  design explicitly admits another mode);
- optional reviewed absolute `model_spec_override`;
- optional bounded model `busy_timeout_ms`;
- optional recipe-declared `load_options`; and
- optional recipe-declared `session_options`.

Recipe projections cannot place a clone reference or transcript in
`default_voice_preset`/`voice_presets`. Clone material remains request-scoped.
Arbitrary user JSON belongs in the existing user-provided source.

### GM-CFG-005 — Absolute paths and model specs

Generated model paths are absolute. The registry uses embedded package specs
when the exact package supports them. Otherwise it supplies a reviewed,
installed-distribution model-spec path or a validated user-selected override.
It never depends accidentally on the app's current working directory.

The generated child uses a deterministic working directory appropriate to the
installed distribution and selected binary. The existing user-JSON source
keeps its JSON-parent working-directory semantics because relative paths there
belong to the user's file.

### GM-CFG-006 — Port and endpoint

Guided setup persists no fixed port. At each new launch Chatbook selects an
available port from a finite loopback-only range, records it in the immutable
artifact, and treats the child bind/result as authoritative. Port preflight is
advisory because another process may race it.

The runtime endpoint is a generation-bound observation and may be cached by
executable identity for diagnostics, but it is never a global default. The
user-JSON source retains its existing explicit-port, occupied-port-fails-closed
behavior.

### GM-CFG-007 — Accepted projection snapshot

When the user accepts a matched package, Chatbook persists the normalized
recipe ID/version, public model ID, canonical package root identity, selected
variant, and safe projected values needed to reproduce the setup. A later
Chatbook recipe update cannot silently reinterpret that record.

If the installed recipe changes meaning, is withdrawn, or no longer accepts
the saved projection, Settings shows `Review required`. The user reviews an
explicit diff before a new projection becomes saved authority. An actively
running child keeps its immutable applied projection until explicit
replacement.

## Binary, version, and compute selection

### GM-BIN-001 — Detection

Detection checks only reviewed candidates:

- the executable already configured by the user;
- `audiocpp_server` resolved through the application's sanitized `PATH` view;
- known Homebrew locations on macOS; and
- reviewed conventional install locations for supported Windows/Linux
  distributions.

Detection reports candidates; it does not execute them, recursively search the
machine, rewrite `PATH`, or select an ambiguous candidate silently. Manual
browse always remains available.

### GM-BIN-002 — Version evidence

A deliberate Test/Start may obtain a bounded version/build observation through
an exact reviewed mechanism. `release-0.5.1` at the pinned commit is the first
Verified baseline. A recognized incompatible version fails with a stable
message. An unknown version may continue to Test after a warning, but its
recipes display `Untested` and cannot inherit Verified platform/backend labels.

Version output, executable paths, environment, and raw child errors remain out
of general logs and public exception messages.

### GM-BACKEND-001 — Auto and Advanced override

The default is `Auto`. Candidate order derives from operating system,
architecture, detected build/backend capabilities, and exact recipe evidence.
The UI summarizes the resolved choice before launch. Advanced settings permit
an explicit supported backend and validated device/thread values.

The selection result is launch evidence, not durable host truth. A binary
replacement or materially different build invalidates its cached observation.

### GM-BACKEND-002 — Bounded fallback

Automatic fallback is allowed only when:

1. backend is `Auto`;
2. the failure maps to a stable allowlisted backend-unavailable code;
3. the exact failed child, clients, tasks, and generated artifact are reaped;
4. the next candidate has recipe/platform evidence; and
5. the UI records the attempted and selected backends without raw private
   diagnostics.

All other failures stop. The user receives explicit recovery, including
`Try CPU`, rather than an unbounded backend cascade.

## Recipe registry

### GM-RECIPE-001 — Recipe identity

A recipe identifies one exact package variant/revision, not merely a model
family. Its stable identity includes:

- recipe schema version and immutable recipe ID;
- audio.cpp release/commit range;
- family and package variant;
- declared capabilities (`tts`, `clone`, design/control where relevant);
- exact bounded metadata and layout signals;
- required and optional relative assets;
- allowed server projection and option domains;
- voice/reference requirements;
- compatible platform/backend evidence;
- reviewed Model Library artifact IDs and static source links; and
- recipe verification status and evidence reference.

Recipe declarations cannot contain an absolute path, `..` traversal, a symlink
requirement, executable path, shell text, environment expansion, credential, or
arbitrary JSON fragment.

### GM-RECIPE-002 — Exact match classes

Matching produces one of:

- **Exact** — one recipe variant satisfies every required signal.
- **Ambiguous** — multiple variants remain possible; no variant is selected.
- **Unknown** — no recipe matches; guided setup cannot add it.
- **Incomplete** — a candidate variant is recognizable but required assets are
  missing or the scan was bounded/cancelled before proof.
- **Permission limited** — exact required evidence could not be inspected.

There is no fuzzy “closest” selection. Ambiguous and unknown packages can use
the manual `server.json` source.

### GM-RECIPE-003 — Compatibility states

The UI keeps three dimensions separate:

1. **File readiness** — required package files are present and currently
   readable within scan bounds.
2. **Recipe evidence** — exact recipe/version/platform/backend classification.
3. **Runtime evidence** — this applied child successfully exposed the model and
   completed the relevant contract/sample.

Valid labels include `Verified`, `Expected`, `Untested`, `Review required`,
`Backend unsupported`, `Files missing`, and `Runtime failed`. A green runtime
result cannot rewrite the persisted recipe identity; a recipe match cannot
claim the model actually loaded.

### GM-RECIPE-004 — Complete release accounting

The registry carries an auditable matrix for all 21 pinned families and every
declared package variant. Each entry is Approved, Explicitly unsupported with a
reviewed reason, or an Open gap. The program is not complete while an Open gap
exists or while an Explicitly unsupported entry contradicts the accepted goal
without renewed user approval.

Incremental releases publish the precise Approved subset in user-facing docs
and tests. “Supports audio.cpp models” is prohibited as a blanket claim before
the matrix is complete.

### GM-RECIPE-005 — Withdrawal and upgrades

A later Chatbook release may withdraw an unsafe recipe. A withdrawn saved model
is visible but cannot start a new child until reviewed or moved to manual JSON.
An already running child is not killed by passive recipe loading; the next
explicit lifecycle action applies the safe policy.

Recipe upgrades show a field-level safe projection diff. Accepting the update
preserves the stable public model ID when the exact package identity is the
same. It never changes the global default model, profile assignments, or Studio
preferences silently.

## Local package scanner and Model Library

### GM-SCAN-001 — Explicit roots only

The scanner examines only a root selected by the user for the current action or
an exact package root returned by Model Library. It does not crawl a home
directory, mounted volume, model cache, or arbitrary parent automatically.

A top-level selected symlink may be resolved after clear disclosure. Its
canonical target becomes the accepted root identity. Nested symlinks,
junctions, reparse points, devices, sockets, and aliases are not traversed. A
later top-level symlink retarget is a `Review required` change, not the same
package.

### GM-SCAN-002 — Bounded and responsive work

Scanning runs in a cancellable worker outside the Textual event loop. The
implementation defines finite, tested limits for:

- directory depth;
- visited entries;
- candidate package roots;
- bytes of metadata read per file and in aggregate;
- individual and total scan time; and
- retained unknown/error detail.

Reaching any limit returns an explicit partial/incomplete result. It never
silently treats an unvisited tree as absent. Cancellation stops publication and
late results are fenced by root/draft revision.

### GM-SCAN-003 — Recognition and deduplication

The scanner reads only allowlisted relative files and bounded metadata needed by
candidate recipes. It does not load model weights into a native runtime or hash
multi-gigabyte packages during ordinary discovery.

Candidate identity is at least:

```text
(canonical_root, recipe_variant, configuration_identity, weight_identity)
```

It is not root alone. Multi-file shards and required companion assets must be
complete. Two valid variants in one root remain distinct; repeated discovery of
the same exact identity is deduplicated.

A durable internal UUID is allocated only after the user accepts a candidate.
The public model ID is stable and separately validated for audio.cpp. Moving an
accepted package through the explicit relocation flow preserves both IDs after
the new exact identity is reviewed.

### GM-SCAN-004 — Validation points

Package evidence is checked:

1. during scan;
2. again at Add/Save against the accepted draft; and
3. immediately before a deliberate new launch or replacement artifact is
   published.

A running child retains its immutable launch snapshot even if source files are
later moved, deleted, or edited. The next replacement fails safely and explains
which model needs attention. Ordinary synthesis against the running child does
not repeatedly reopen every source file.

### GM-SCAN-005 — Error and privacy presentation

Permission failures and incomplete candidates are isolated per root/candidate.
The primary result is a bounded summary. Unknown entries are collapsed behind
an expandable, capped list. General logs and stable errors use recipe IDs,
counts, and safe basenames where needed; they do not emit complete model paths,
file contents, metadata payloads, or raw OS exceptions.

### GM-LIB-001 — Curated installs

Model Library shows only artifacts whose IDs are reviewed in a compatible
recipe. An install is an explicit user action through the existing shared
artifact owner. The setup screen never starts a download on mount, scan, save,
or Test.

On success, Model Library returns the exact installed package root and artifact
identity to the preserved Settings draft. The draft remains unsaved until the
user reviews it. Install does not select a default, start audio.cpp, create a
profile, or change Studio preferences.

### GM-LIB-002 — Removal dependencies

Before removal, Chatbook shows dependencies from:

- guided global audio.cpp setup;
- the global default model/profile;
- reusable TTS profiles, including references whose recipes require that
  package;
- character assignments reachable through those profiles; and
- any other existing Model Library lease/owner.

Removal requires explicit resolution. Chatbook does not silently retarget
profiles, switch the global default, or delete clone references. A running child
may retain already-open model state until shutdown, but later use/restart must
report the missing package truthfully.

## Lifecycle and generation semantics

### GM-LIFE-001 — Deliberate apply boundaries

When there is no live child, the latest valid guided settings may become
applied on:

- explicit Start/Test;
- explicit catalog Refresh that requires a server;
- first user-requested Speech Lab synthesis; or
- lazy Console/Roleplay synthesis.

Passive Settings mount, Speech Lab mount, status rendering, profile browsing,
recipe loading, package scan, Save, and Model Library listing never launch or
contact audio.cpp.

### GM-LIFE-002 — Running child and staged settings

When a child is Running and saved settings advance:

- the active child, endpoint, adapter clients, catalog, and loaded models stay
  bound to the applied generation;
- ordinary Test, Refresh, and synthesis remain on that applied generation;
- the UI shows the saved/applied diff and `Restart & Apply Settings`;
- only the explicit replacement transition drains leases, stops/reaps the
  exact child, generates the new artifact, and promotes the newest eligible
  saved generation; and
- a failed replacement does not relaunch an older staged projection silently.

This preserves the current managed-lifecycle contract. “Test” cannot become a
hidden restart merely because a newer configuration is saved.

### GM-LIFE-003 — Launch transaction

One launch attempt performs, in order:

1. capture the eligible saved projection under the existing publication/
   transition fence;
2. revalidate the executable and accepted package projections;
3. resolve backend and select a bounded loopback port;
4. atomically materialize the immutable private artifact;
5. launch one direct child with the existing sanitized environment;
6. publish Starting without exposing an unverified endpoint;
7. require process liveness, `/health`, `/v1/models`, expected model IDs, and
   adapter contract evidence;
8. publish Running/capability/catalog only for the exact process generation;
   and
9. retain all generation resources until exit/replacement/shutdown cleanup is
   definitive.

Any failure before Running invalidates the endpoint/evidence immediately,
terminates only the exact child, joins all owned resources, removes only its
exact artifact, and returns a stable safe failure. An eligible backend fallback
starts a new launch attempt only after that rollback completes.

### GM-LIFE-004 — Runtime revalidation

A health failure invalidates catalog/voice/runtime evidence for the process
generation. A later Test, Refresh, or synthesis revalidates the adapter contract
before relying on it. A temporary recovery of the health probe alone does not
make old catalog evidence fresh.

Unexpected exit invalidates the endpoint and generation-bound evidence before
potentially slow output/client cleanup. The first later deliberate operation
uses the newest eligible saved settings. It cannot resurrect an older applied
artifact merely because cleanup was still finishing.

### GM-LIFE-005 — Definitive shutdown

Restart, source switch, and shutdown reuse the registry's exclusive transition
and drain admitted leases. Already admitted work may complete against its exact
generation; new work is rejected. Application close seals new service and
lifecycle admission, applies one outer shutdown deadline to any already-running
startup/stop transition, and retains definitive joining until every owned child,
handle, client, monitor, probe, drain, artifact lease, and lifecycle task
reaches zero ownership.

The foreground close budget may expire, but `wait_closed()` cannot claim
terminal ownership completion while retained work remains.

## Clone-reference profile model

### GM-VOICE-001 — Profile schema v3

The TTS profile repository advances from schema v2 to v3. The existing profile
row remains the reusable provider/model/voice selection owner. A new
one-to-one reference row, keyed by profile UUID with cascade delete, contains:

| Field | Contract |
| --- | --- |
| `reference_id` | Immutable UUID, distinct from profile UUID |
| `profile_id` | Unique foreign key to the profile |
| `wav_bytes` | Canonical bounded WAV BLOB |
| `reference_text` | Bounded non-empty transcript when required |
| `sha256` | Digest of canonical WAV bytes |
| `byte_length` | Canonical byte length |
| `duration_ms` | Validated duration |
| `sample_rate_hz` | Validated canonical rate |
| `channels` | Validated canonical channel count |
| `sample_encoding` | Canonical encoding identifier |
| timestamps | Created/updated UTC metadata |

The repository enforces one reference per profile, hard per-reference size and
duration limits, hard total stored-byte/reference-count quotas, bounded
transcript length, and recipe-specific narrower limits. Quotas are checked
inside the mutation transaction.

### GM-VOICE-002 — Canonical ingest

Reference ingest accepts a regular user-selected WAV only. It uses no persisted
source path. A bounded decoder validates supported RIFF/WAVE chunks, sample
format, channels, rate, duration, and size; rejects malformed, compressed, or
unsupported input; strips arbitrary metadata and unknown chunks; and writes one
canonical WAV representation.

Digest and metadata are computed from the canonical bytes. The source is
rechecked before commit so a changed file cannot be represented by earlier
validation. BLOB reads/writes are streamed or incrementally copied; list and
normal profile-open paths load summaries only.

### GM-VOICE-003 — Typed capability combinations

An audio.cpp profile may hold:

- an exact native `voice_id` only;
- one clone reference only; or
- both only if the exact recipe explicitly supports and defines their
  precedence/combination.

A recipe that requires a reference cannot be used without one. A text-only
recipe cannot acquire a reference merely because the server accepts generic
fields. Clone support never reopens arbitrary `options` on audio.cpp profiles.

Reference-bearing execution initially requires the exact accepted guided
Managed recipe and the app-owned local child. An External server or an
unclassified user-provided JSON model may still use text and exact native voice
IDs, but Chatbook does not send it a client-local materialization path. A stored
reference remains visible and inactive with a recovery explanation when the
applied source cannot safely consume it.

### GM-VOICE-004 — Immutable admission

Admission freezes:

- profile UUID and revision;
- provider/model/voice selection;
- accepted recipe projection/version;
- reference UUID and digest;
- bounded transcript; and
- applied provider/process generation.

The materializer reads and fully validates the exact reference under the
repository generation/revision fence, creates an opaque private per-session
WAV path, and passes typed `voice_ref` and `reference_text` to the native
request. It never puts them in `server.json`, voice presets, generic options,
catalog state, or public artifact provenance.

An edit/delete admitted after this capture affects future work. Already
admitted speech may consume the exact captured reference until response close,
after which the session materialization is removed.

### GM-VOICE-005 — Session materialization

Each materialization lives in an owner-private directory under the active
user-data/runtime area and has an ownership lock tied to application session
and operation identity. Files use opaque names and owner-only access. Normal
completion, cancellation, provider failure, replacement, and app shutdown
remove the exact directory after the adapter can no longer read it.

Startup cleanup handles only directories with a recognized format and removes
one only after it proves no live owner holds the lock. It does not delete an
unknown directory, use age alone as proof, or follow a symlink/reparse point.

### GM-VOICE-006 — Transient audition

Speech Lab may stage one canonical reference for the current clone draft
without creating a profile. The staged artifact is bounded, private, and owned
by the current result workflow. It is deleted when replaced, discarded, or the
app closes.

After a successful result, `Save as Voice Profile` persists the exact canonical
bytes and transcript used by that successful request. It does not reopen the
original file. A failed generation does not offer to save an unproven reference
as though it generated successfully, although the user may return to setup and
correct it.

### GM-VOICE-007 — Privacy statement

The profile database, BLOB, pre-migration backup, backup/restore output, and
temporary materializations are local private data. On POSIX they use ADR-029's
owner-only boundary. Windows displays its actual separately implemented ACL
posture and never translates POSIX-mode assumptions into a privacy claim.

The product says plainly:

- reference audio and transcript are stored locally in plaintext;
- local filesystem controls are not encryption;
- exports/backups contain sensitive audio when explicitly requested; and
- deletion is best effort, not guaranteed forensic erasure across SQLite
  journals, copy-on-write storage, backups, or physical media.

### GM-VOICE-008 — Migration and downgrade

Migration is eager on the schema-owning repository path:

1. acquire the existing exclusive repository lifecycle boundary;
2. validate the v2 source;
3. create and retain an owner-private SQLite online backup of v2;
4. migrate transactionally to v3;
5. validate schema, foreign keys, integrity, and domain equivalence; and
6. publish v3 only after success.

Every existing v2 profile remains logically equivalent and has no reference
row. A failure leaves v2 usable or the repository explicitly unavailable; no
partial v3 store is published or silently recreated.

An older build must refuse v3. Downgrade requires closing the new build,
restoring the dedicated v2 backup, and then launching the old build. Changes
made after migration are lost. Switching to manual `server.json` or disabling
clone UI in the current build is a feature rollback, not a database downgrade.

### GM-VOICE-009 — Backup, restore, and damage isolation

Existing repository-owned SQLite online backup includes reference BLOBs and
reports bounded progress/deadline/quota outcomes. Restore validates the full
candidate, including reference digests/WAV structure and aggregate quotas,
before replacement.

Normal open remains metadata-oriented and does not read every BLOB. Full
reference verification occurs on import, restore, backup qualification,
reference edit, and exact admission. If a damaged reference row/BLOB can be
isolated safely, that profile becomes unavailable while unrelated profiles
remain usable. Structural database corruption or ambiguous integrity failure
keeps the repository unavailable.

## Portability

### GM-PORT-001 — Ordinary sanitized portability

Profiles without a reference retain the existing wire version 1. An ordinary
export of a reference-bearing profile uses wire version 2 and contains only the
sanitized profile selection plus an explicit `reference omitted` marker. It
contains no WAV bytes, transcript, digest usable as a local path oracle,
temporary path, assignment, endpoint, or generated configuration.

An older reader skips the unsupported version. Importing wire v2 never creates
or assigns a broken profile silently. The user chooses to attach a local WAV,
import a separate bundle, or skip the attachment.

### GM-PORT-002 — Explicit voice bundle

The explicit portable container is a versioned ZIP, separate from character
cards and ordinary profile export. Its allowlisted entries are exactly:

```text
manifest.json
profile.json
reference.wav
reference.txt
```

The manifest carries the bundle schema, bounded metadata, entry sizes,
canonical SHA-256 checksums, and generic user-declared provenance/consent note.
The profile is the sanitized selection, not a database row. The archive has no
model weights, character/persona data, assignments, app default, credentials,
origins, paths, recipe code, process state, or timestamps not needed by the
format.

### GM-PORT-003 — Hostile archive admission

Before extracting or storing, import rejects:

- encrypted entries;
- duplicate names and case/Unicode-normalization collisions;
- absolute paths, traversal, separators outside the exact names, symlinks,
  devices, or other special entries;
- unknown/missing entries;
- unsupported compression methods;
- per-entry, aggregate compressed/uncompressed, ratio, count, transcript, and
  manifest limits;
- malformed JSON/text/WAV;
- source archive changes during admission; and
- checksum mismatch.

Extraction uses an owner-private staging directory and never trusts archive
paths. Checksums prove byte integrity only; the UI never labels them signature,
authenticity, speaker identity, or consent proof.

### GM-PORT-004 — Explicit import result

Import shows exact profile UUID/name/model collisions and requires a user
choice. It never overwrites, assigns to a character, changes app default, or
retargets to another model automatically.

If the bundle is structurally valid but its exact compatible recipe/model is
absent, Chatbook may store it as inactive `Needs compatible model`. Runtime
admission stays blocked until that exact dependency is installed and reviewed.

## Global Settings UX

### GM-UX-001 — Scope and information architecture

Settings → Speech & TTS clearly states:

> You are editing global TTS setup and defaults. Speech Lab has testing,
> generation, playback, and Studio-only preferences.

It links directly to Speech Lab. Guided setup does not mount inside Studio and
never writes `speech_studio` preferences.

The audio.cpp provider detail uses this order:

1. setup source and External/Managed ownership;
2. server executable detection/browse and version posture;
3. configured model packages and default model;
4. compute summary with Advanced override;
5. validation/save summary; and
6. one state-specific handoff to Speech Lab.

Advanced lifecycle/safety fields remain discoverable without dominating the
first-time path. Validation expands the containing disclosure and focuses the
first invalid field.

### GM-UX-002 — Model package list

Each model row shows:

- display name and stable public model ID;
- family/package variant and capabilities;
- local or Model Library ownership;
- file readiness;
- recipe evidence, including exact release/backend posture;
- reference/voice requirement;
- dependency count when relevant; and
- runtime evidence for the matching applied generation, if any.

The row does not collapse these into one colored dot. Text labels, symbols, and
accessible descriptions communicate status without color alone. Unknown items
appear in a bounded collapsed summary rather than flooding the screen.

### GM-UX-003 — Add, scan, install, and review

`Add local package` opens a picker, scans only the selected root, and returns
exact/ambiguous/unknown/incomplete results. `Browse Model Library` opens the
curated compatible subset; a completed install returns to the preserved draft
with the package ready for review.

Adding a matched package does not Save automatically. The user reviews the
model ID, capabilities, backend posture, and any voice/reference requirements.
Default-model selection is part of the same review.

A non-destructive global provider/model change uses an inline before/after
summary; Save itself is the confirmation. A modal is reserved for destructive
dependency resolution, such as removing a package used by profiles.

### GM-UX-004 — Saved, applied, and active truth

Settings and Speech Lab distinguish:

- current draft;
- latest durably saved configuration generation;
- applied provider generation;
- active process generation and endpoint;
- recipe/file evidence;
- catalog/model evidence; and
- most recent sample result.

`Saved` never means `Applied`; `Applied` never means `Running`; `Running` never
means a specific model loaded; a recipe match never means runtime success.

### GM-UX-005 — Save outcome and handoff

A successful Save says `Configuration saved — ready to test`. The guided CTA is
`Open Speech Lab & Hear a Sample`. It carries a one-time setup-preview intent
containing the saved provider/default-model identity and expected action. It
does not persist Studio preferences or overwrite an existing Studio draft.

If a live child uses older settings, the handoff says that the active
configuration remains unchanged and focuses Speech Lab's dynamic primary
action, not a generic Refresh control.

## Speech Lab UX

### GM-UX-010 — One immutable action projection

One pure projection derives the visible label, operation, enabled state,
disabled reason, tooltip, progress label, and post-operation focus target from
the same immutable observation. The click handler executes that projected
operation rather than recomputing from stale hidden state.

Representative primary actions are:

| State | Primary action |
| --- | --- |
| Guided setup saved, stopped, text-ready | `Start & Generate Sample` |
| Guided setup saved, stopped, reference required | `Start & Set Up Voice` |
| Live child, newer saved generation | `Restart & Apply Settings` |
| Compatible server, missing clone reference | `Create Voice & Generate` |
| Prior sample failed, server still healthy | `Retry Sample` |
| Runtime observation unknown | `Test Connection` |

Shutdown remains secondary. A state whose primary is Restart does not also show
an enabled duplicate Restart in the runtime card.

### GM-UX-011 — Provider switching and stale results

The user may switch providers while lifecycle, catalog, voice, or generation
work runs. Late results carry provider, saved/applied/process, draft, and
operation revisions. They may update their retained owner but cannot disable,
relabel, or execute an action for the newly selected provider.

When audio.cpp is hidden and shown again, the pane clears or atomically renders
its cached observation before accepting a click. A visible `Test Connection`
can never execute a hidden stale Restart/Shutdown projection.

### GM-UX-012 — Clone setup flow

For a reference-required recipe, deliberate Start/Test first establishes the
server and matching catalog. Speech Lab then pauses at a voice setup step with:

- choose reference WAV;
- enter/confirm bounded transcript;
- local plaintext/privacy notice;
- exact recipe guidance and validation;
- `Create Voice & Generate`; and
- an option to use an existing compatible Voice Profile.

The user can audition without naming or saving a profile. After successful
playback, the result offers `Save as Voice Profile`. Saving opens a concise
profile name/assignment review and uses the exact successful reference
artifact.

### GM-UX-013 — Current result

The current complete-WAV result is visually primary and includes:

- Play/Pause;
- duration and playback/generation status;
- exact captured provider/model/voice-or-reference-safe provenance;
- applied/process generation identity expressed without private paths;
- `Generate again`;
- `Save WAV as…`; and
- `Save as Voice Profile` when the captured typed selection is eligible.

Autoplay follows only the existing optional Studio autoplay preference. Setup
does not silently enable it. A new result replaces the old current result only
after successful validation; a failed retry does not erase the last playable
WAV.

### GM-UX-014 — Diagnostics and accessibility

Diagnostics remain bounded, memory-only, sanitized, expandable, focusable, and
keyboard-scrollable. The diagnostics viewport has a visible focus state and no
nested scroll trap. Copy copies only the bounded sanitized projection.

Every disabled action has a current reason. In-progress Test/Start/Restart/
Generate states update labels and tooltips together. On a passive observation
failure, controls leave their busy state and expose a safe Test/Retry path whose
visible label matches its actual operation.

The complete guided flow is keyboard-operable at supported narrow terminal
sizes. Live regions announce scan, validation, lifecycle, generation, playback,
and save outcomes without announcing every progress tick. Focus returns to a
stable relevant control after dialogs and cancellation.

## User journeys

### Journey 1 — First-time text-ready package

1. User installs Chatbook and installs `audiocpp_server` separately.
2. In Settings → Speech & TTS, they select audio.cpp → Managed → Guided setup.
3. Chatbook detects the executable or the user browses to it.
4. User selects a supported local package or explicitly installs one from
   Model Library.
5. The bounded scanner finds one exact recipe; the user reviews the package,
   model ID, backend summary, and default model.
6. Save validates and persists without launching or contacting audio.cpp.
7. Settings says `Configuration saved — ready to test` and opens Speech Lab.
8. Speech Lab projects `Start & Generate Sample`.
9. The deliberate action generates the immutable launch artifact, starts one
   child, verifies catalog/model evidence, requests one sample, validates a WAV,
   and presents Play.
10. User hears the sample. The child remains available and the loaded model may
    remain resident until Shutdown/app exit.

### Journey 2 — First-time clone-required package

1. Steps 1–7 above apply.
2. Speech Lab projects `Start & Set Up Voice`.
3. Start verifies the server and catalog without generating hidden audio.
4. User chooses a bounded WAV and transcript; Chatbook canonicalizes a transient
   private reference.
5. `Create Voice & Generate` materializes the exact typed request reference,
   generates and validates a WAV, cleans the session path, and presents Play.
6. User hears the result and optionally saves the exact successful reference as
   a reusable Voice Profile and assigns it to a character.

### Journey 3 — Multiple models and lazy use

1. User accepts several compatible packages in guided Settings and chooses one
   global default.
2. One generated config registers them all with `lazy_load: true`.
3. Startup does not load every model.
4. First synthesis for a selected model loads it; later requests reuse its
   session.
5. Selecting another configured model loads it in the same child.
6. UI explains that both may remain in memory until shutdown; Chatbook does not
   claim to unload them.

### Journey 4 — Save while running

1. Applied generation A is Running.
2. User changes model/backend setup and saves generation B.
3. A continues serving ordinary Test/Refresh/synthesis; no files are re-read for
   each request.
4. Settings and Speech Lab show saved B versus applied A.
5. `Restart & Apply Settings` drains A, definitively reaps it and its artifact,
   validates B, generates B's artifact, and launches B.
6. Failure leaves runtime truthfully unavailable or on the still-valid outcome
   defined before mutation; it never silently relaunches A or claims B applied.

### Journey 5 — Unknown package or version

1. Scanner cannot identify a package exactly, or the server build is unknown.
2. Guided setup shows `Unknown`/`Untested` with the missing evidence and does
   not guess.
3. User may choose another reviewed package, follow a static official link, or
   switch to the existing user-provided `server.json` source.
4. An unknown server version may Test after warning, but the result cannot
   inherit Verified recipe/platform/backend status.

### Journey 6 — Model Library round trip

1. User opens Model Library from the preserved guided Settings draft.
2. Library shows only reviewed recipe artifacts compatible with the pinned
   support surface.
3. User explicitly installs one package into the shared store.
4. Library returns the exact installed root/identity to the draft.
5. User reviews and Saves; neither install nor return launches audio.cpp or
   changes Studio/global default implicitly.

### Journey 7 — Export and import a clone voice

1. User explicitly exports a Voice Profile as a voice bundle.
2. Chatbook warns that the archive contains plaintext reference audio and
   transcript and records only a generic user declaration, not proof of consent.
3. On another installation, import validates the hostile archive and exact
   checksums before displaying the sanitized profile.
4. A UUID/name collision and model dependency are reviewed explicitly.
5. Import creates an unassigned profile or inactive `Needs compatible model`;
   it never changes a character assignment or app default.

### Journey 8 — Console/Roleplay lazy first use

1. Valid guided setup is saved, but no child has run this session.
2. A user explicitly invokes Speak in Console/Roleplay.
3. Admission captures exact global or character profile selection.
4. The one shared startup applies the latest eligible saved setup, and speech
   waits on that retained startup.
5. Complete WAV generation succeeds or returns a stable recovery action. No
   passive character/profile browsing launches a child.

## Error, privacy, and recovery contract

### GM-ERR-001 — Stable phases

Public failures identify a stable phase and safe recovery:

- executable missing/not executable/unsupported build;
- package missing/changed/incomplete/ambiguous/unknown;
- recipe withdrawn/review required/backend unsupported;
- generated configuration invalid;
- loopback port/bind/startup/contract failure;
- process exited/unhealthy/stopping/reconfiguring;
- model absent/load failed/server busy;
- reference invalid/quota exceeded/transcript required;
- clone request unsupported/generation failed;
- WAV invalid/too large;
- profile store/migration/reference unavailable; and
- bundle malformed/unsafe/integrity failed/dependency missing.

Raw child, HTTP, OS, archive, SQLite, native decoder, or hook exception strings
never cross the public boundary or remain reachable through chained exception
context. Safe errors contain no full executable/model/reference/config/temp
path, transcript, prompt, audio bytes, credential, or environment value.

### GM-ERR-002 — Phase-local recovery

Recovery preserves work that is still valid:

- scan failure keeps the Settings draft;
- Save validation failure expands/focuses the exact field or model row;
- server startup failure leaves no child or artifact;
- voice setup failure keeps a healthy server running;
- generation/reference failure keeps the last valid playable WAV;
- playback failure does not restart the server or regenerate;
- profile save failure keeps the transient successful result/reference until
  the user retries or discards it; and
- bundle import failure commits no profile/reference/assignment.

No failure silently falls back to another provider, model, voice, package,
reference, or backend except the narrowly admitted Auto backend fallback.

### GM-ERR-003 — Diagnostic containment

General/persistent logs remain metadata-only. Managed child stdout/stderr stay
in the existing bounded in-memory sanitized ring. For clone operations,
diagnostic suppression/redaction covers:

- reference/transcript request fields;
- temporary materialization paths;
- original source paths;
- HTTP debug request bodies;
- subprocess/debug-loop command lines containing private configured paths; and
- nested exception graphs.

The UI warns that native child output is only best-effort sanitized and should
be reviewed before copying. Generated setup forcibly disables audio.cpp
request-body logging.

## Testing strategy

### GM-TEST-001 — Pure configuration and recipe tests

- Guided/manual source round trips and dormant-value preservation.
- Side-effect-free Save guards: no subprocess, socket, HTTP, artifact, restart,
  stop, or catalog call.
- Strict generated JSON golden/projection tests with forbidden-field mutation
  guards.
- Every recipe declaration validates traversal, field allowlists, assets,
  option domains, evidence, and package identity.
- Exact/ambiguous/unknown/incomplete matching matrices for all declared package
  variants.
- A generated compatibility-accounting test fails on an unclassified pinned
  family/package gap.

### GM-TEST-002 — Scanner and Model Library tests

- Depth/entry/candidate/metadata/time limits and explicit partial results.
- Cancellation and stale draft/root result rejection.
- Top-level symlink disclosure/retarget review; nested symlink, junction,
  reparse, device, and traversal rejection on each OS.
- Shard completeness, variant dedupe, relocation identity, permission isolation,
  and unknown-result caps.
- Model Library artifact filtering, explicit install, exact-root return, draft
  preservation, and no server-binary/package-manager side effects.
- Dependency-aware removal with global/profile/assignment truth and no silent
  retargeting.

### GM-TEST-003 — Generated lifecycle tests

- Concurrent first use creates one artifact and one child.
- Auto port selection, bind race, no CORS, no body logging, sanitized
  environment, no shell, exact argv, absolute paths, deterministic workdir.
- Multi-model lazy registration and first-use loading without an unload claim.
- Save while running remains staged; ordinary operations stay on the applied
  generation; only explicit Restart & Apply replaces.
- Source deletion/change while Running does not break admitted use; next
  deliberate replacement revalidates and fails safely.
- Failed launch/fallback/restart/exit reaps exact process, Windows/POSIX handle,
  tasks, clients, artifact, and locks before another attempt.
- Health/exit evidence invalidation and generation fences cover every
  interleaving already required by the managed-lifecycle design.
- Application shutdown during scan/startup/restart/synthesis/reference
  materialization stays within the shared foreground budget and retains
  definitive ownership joining.

### GM-TEST-004 — Profile/reference tests

- v2→v3 migration, domain equivalence, retained v2 backup, atomic failure,
  unsupported newer schema, explicit downgrade procedure.
- Reference/profile one-to-one/cascade, optimistic revision, repository
  generation, quota, aggregate quota, and concurrent admission/edit/delete.
- WAV parser/canonicalizer matrices for chunk order, padding, metadata stripping,
  unsupported encoding, truncation, overflow, duration/rate/channels, and
  changed source.
- Streaming BLOB I/O and metadata-only list/open assertions.
- Typed voice/reference combination matrices per recipe.
- Session private modes/ACL posture, ownership locks, crash leftovers, no
  unsafe sweep, cleanup on success/failure/cancel/restart/shutdown.
- Exact admitted revision/reference digest and raw-error/privacy exception-graph
  guards.
- Isolated damaged-reference behavior versus structural database corruption.

Schema tests use isolated temporary profile databases. Development/UAT must
never point a schema-bumping build at the user's live profile store.

### GM-TEST-005 — Portability tests

- Wire v1 remains byte/behavior compatible for non-reference profiles.
- Wire v2 omits WAV/transcript and older readers skip it safely.
- Ordinary import requires attach/bundle/skip and never creates a broken
  assigned profile silently.
- Bundle round trip and deterministic manifest/checksum behavior.
- Archive rejection: encryption, duplicates, case/Unicode collision, traversal,
  special entries, unknown entries, compression, bombs, limits, malformed
  content, changed source, and mismatch.
- Collision/model-missing import remains explicit, inactive when needed, and
  never overwrites/assigns/defaults/retargets.
- Backup/restore includes full references and respects progress/deadline/quota
  validation.

### GM-TEST-006 — Textual UX tests

- Scope copy/global link and no Studio write.
- Detection, package rows, status-dimension truth, scan/install return, review
  diff, Save outcome, and Speech Lab handoff focus.
- One immutable action projection: visible label and executed operation cannot
  diverge under provider switches, observation failures, staged settings, or
  late results.
- Busy/disabled reasons and tooltips for every lifecycle/generation phase.
- Clone setup, transient audition, current result, save-as-profile, and last
  good WAV retention.
- First invalid hidden field expands and receives focus.
- Keyboard-only flow, focus restoration, live announcements, no color-only
  status, narrow terminal geometry, and focusable/scrollable diagnostics.

Reduced apps that imitate only part of the production application are not
acceptable evidence for cross-owner flows. Tests compose the real production
screen/service boundary or the smallest real owner below it.

### GM-TEST-007 — Real process and compatibility gates

For every recipe tuple labeled Verified:

- run the exact supported `audiocpp_server` binary and exact package;
- prove generated JSON is accepted;
- verify health/catalog/default-model identity;
- synthesize a structurally valid WAV using text or reference flow as declared;
- verify lazy Console/Roleplay first use where applicable; and
- prove shutdown leaves no owned child, handle, task, client, artifact, or
  private materialization.

At minimum, CPU real-process gates cover macOS, Linux, and Windows. Accelerated
labels require their own real tuple gate. Unknown-version behavior and the
manual user-JSON regression are separate gates.

Normal CI remains hermetic and does not download audio.cpp, models, or network
content. Large/accelerated compatibility suites use explicitly provisioned
artifacts with checksums and opt-in runners.

## Manual UAT and release evidence

The release gate uses a clean temporary Chatbook config/data/profile database,
never the developer's live files. Evidence records:

- exact Chatbook commit;
- audio.cpp tag, commit, binary origin, and executable identity;
- OS version, architecture, compute backend/device;
- recipe ID/version, family, package variant, model artifact identity;
- whether the package was local or installed through Model Library;
- generated configuration digest with private paths removed;
- lifecycle state and model/catalog identity;
- sample text class or sanitized digest, not private content;
- reference metadata/digest without path/audio/transcript where applicable;
- WAV format, rate, channels, duration, byte length, and structural validator;
- shutdown/cleanup evidence; and
- human audible-playback confirmation.

Required journeys:

1. clean first-time guided local-package setup, Save without launch, handoff,
   Start & Generate, Play;
2. Model Library install → exact root return → guided Save → sample;
3. clone-required transient reference → Generate → Play → Save as Profile →
   character roleplay response → Play;
4. multiple configured models and lazy first use;
5. save while running → staged truth → explicit Restart & Apply;
6. unknown version warning and unknown/ambiguous package manual-source recovery;
7. crash/health failure → evidence stale → later deliberate recovery;
8. explicit External and user-JSON regressions, proving configured origin/file
   ownership remains exact;
9. ordinary sanitized export versus explicit voice-bundle warning/import; and
10. keyboard-only/narrow-terminal setup and diagnostics.

Audible confirmation is a required human datum; structural WAV validation alone
does not prove a useful sample. Conversely, “I heard it” without pinned binary,
model, app commit, and teardown evidence is not sufficient release evidence.

## Delivery boundaries

These are product release boundaries, not implementation tasks or a plan. Task
decomposition occurs only after this design receives final approval.

1. **Generated text-ready vertical** — guided setup, exact recipes for the
   approved initial package subset, a no-reference Supertonic first sample,
   truthful PocketTTS GGUF voice-required classification, POSIX real-process
   evidence, and first-time sample UAT.
2. **Clone-reference foundation** — profile v3, canonical private reference,
   transient audition, typed admission, and explicit voice bundle.
3. **Model Library and dependency UX** — reviewed artifact mapping, install
   return, dependency-aware removal, and complete Settings review flow.
4. **Windows lifecycle parity** — owned-handle launch/reap/shutdown,
   path/ACL/scanner behavior, and Windows CPU UAT.
5. **Complete release-0.5.1 recipe coverage** — every pinned TTS/Clone package
   variant classified and all supported tuples evidenced.

Each releasable increment states the exact family/package/platform/backend
subset it supports. The program cannot call itself complete before boundary 5
has zero unapproved gaps.

## Rollout and rollback

- External and existing user-JSON sources remain available throughout rollout.
- Existing configs require no write and retain their source.
- Guided fields may remain inert in builds where the feature is disabled.
- A failed guided launch affects no legacy provider and leaves no owned child or
  generated artifact.
- A withdrawn recipe blocks only a new guided launch/replacement; it does not
  passively kill a valid running generation.
- Clone feature rollback disables new reference admission without deleting v3
  data.
- Application downgrade uses the dedicated v2 pre-migration backup and is
  explicitly lossy for post-migration profile changes.
- No rollback deletes user-selected packages, Model Library artifacts, user
  JSON, exported WAVs/bundles, or unknown files.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Upstream package/schema churn | Pinned immutable recipes, accepted projection snapshots, explicit review, manual JSON escape hatch |
| “All models” becomes an untestable promise | Exact 21-family/package accounting matrix and tuple-scoped labels |
| Scanner freezes the TUI or walks too much | Explicit roots, finite limits, cancellable worker, partial result truth |
| Package heuristics select the wrong family | Exact signals only; ambiguity blocks guided selection |
| Generated JSON becomes a second owner | Structured settings authoritative; generation artifact immutable and never imported |
| Auto backend hides real faults | Fallback only for stable allowlisted backend-unavailable codes after definitive cleanup |
| Multiple model loads exhaust memory | Lazy-load disclosure; one child; no false unload promise; explicit shutdown |
| Saved settings disrupt running speech | Existing saved/applied/process generations and explicit Restart & Apply |
| Source files change while a child is live | Immutable applied snapshot; revalidate on next launch, not every synthesis |
| Windows child survives app exit | Exact process-handle ownership, early close sealing, shared deadline, definitive joining, real UAT |
| Clone reference leaks through paths/logs | Canonical BLOB owner, opaque session path, request-body logging off, safe errors/context severing |
| SQLite BLOBs grow without bound | Per-reference and aggregate quotas, streamed I/O, summary reads |
| Users assume local means encrypted | Explicit plaintext disclosure and no forensic-erasure claim |
| Voice bundle is an archive attack | Exact entries, no traversal/special/encrypted files, strict size/ratio/checksum validation |
| Imported voice is assigned silently | Explicit collision/dependency review; never overwrite/assign/default/retarget |
| Migration makes downgrade unsafe | Retained v2 online backup and explicit closed-store downgrade procedure |
| Tests pass but real audio is unusable | Pinned real-process gates plus human audible UAT |

## Functional acceptance criteria

- [ ] GM-AC-001: Existing External and user-provided `server.json` users retain
  their source and behavior without a migration write or guided side effect.
- [ ] GM-AC-002: A user can select Guided Managed setup, detect/browse a
  separately installed server, add a supported package, choose a default, and
  Save without any process, socket, HTTP, artifact, or model side effect.
- [ ] GM-AC-003: A deliberate first use materializes one private immutable
  loopback/no-CORS/no-body-log generated config and launches exactly one owned
  no-shell child.
- [ ] GM-AC-004: One generated server registers multiple accepted models with
  lazy loading; first use loads the selected model and the UI does not claim
  later unloading.
- [ ] GM-AC-005: Saved, applied, process, file/recipe, catalog, and sample state
  remain distinct across save, restart, failure, exit, and provider switching.
- [ ] GM-AC-006: A running child keeps its immutable applied artifact and model
  snapshot; source changes are revalidated only for the next deliberate launch
  or replacement.
- [ ] GM-AC-007: Backend Auto and explicit override work across the supported
  platforms; fallback happens only for recognized failures after exact cleanup.
- [ ] GM-AC-008: Scanning is explicit-root, bounded, cancellable, off-loop,
  symlink-safe, exact-match-only, and reports partial/permission outcomes.
- [ ] GM-AC-009: Recipe accounting covers every package variant in all 21
  pinned TTS/Clone families, and user-facing support claims name the exact
  evidenced subset until no unapproved gap remains.
- [ ] GM-AC-010: Model Library shows reviewed compatible artifacts, installs
  only on explicit action, returns the exact package root to the preserved
  draft, and does not install the server binary.
- [ ] GM-AC-011: Package removal discloses and explicitly resolves global,
  profile, character-assignment, and artifact-owner dependencies without
  silent retargeting or reference deletion.
- [ ] GM-AC-012: Guided setup works on macOS, Linux, and Windows CPU with
  exact-child definitive shutdown; accelerated status is Verified only with
  tuple-specific evidence.
- [ ] GM-AC-013: Reference ingest creates canonical bounded WAV bytes and a
  bounded transcript in one profile-owned v3 reference row without persisting
  the source path or arbitrary RIFF metadata.
- [ ] GM-AC-014: Clone request admission freezes exact profile revision,
  recipe, reference UUID/digest, transcript, and runtime generation, uses an
  opaque private session path only with the compatible guided app-owned child,
  and cleans it definitively; External/unclassified sources never receive it.
- [ ] GM-AC-015: A transient reference can generate a sample without creating a
  profile; Save as Voice Profile uses the exact successful canonical artifact.
- [ ] GM-AC-016: v2→v3 migration is guarded by a retained private v2 backup,
  transactional validation, domain-equivalent existing profiles, and explicit
  lossy downgrade instructions.
- [ ] GM-AC-017: Profile listing/open remains metadata-focused, BLOB operations
  are streamed and quota-bound, backup/restore includes full references, and
  an isolatable bad reference blocks only its profile.
- [ ] GM-AC-018: Ordinary portability omits reference audio/transcript; explicit
  voice bundles are strictly validated, plaintext-disclosed, unassigned on
  import, and never auto-overwrite/default/retarget.
- [ ] GM-AC-019: Settings states global ownership, preserves Studio
  preferences, reports `saved — ready to test`, and hands off to the correct
  dynamic Speech Lab primary action.
- [ ] GM-AC-020: Speech Lab provides coherent state-specific actions, clone
  setup, prominent Play/Pause/current-result controls, optional existing
  autoplay preference, retry, WAV export, and profile save.
- [ ] GM-AC-021: Provider switches, late results, observation failures, and
  busy states cannot make a visible action execute a different stale operation
  or leave recovery controls falsely disabled.
- [ ] GM-AC-022: The guided and clone flows meet keyboard, focus, announcement,
  non-color, narrow-layout, and scrollable-diagnostics accessibility gates.
- [ ] GM-AC-023: Stable errors and exception graphs contain no full executable,
  config, model, reference, temporary path, transcript, prompt, audio,
  credential, environment, or raw upstream exception detail.
- [ ] GM-AC-024: Normal CI is hermetic; provisioned real-process gates validate
  each Verified tuple; manual UAT records exact evidence and human audible
  playback.
- [ ] GM-AC-025: A clean first-time user can complete local package selection or
  Model Library install, Save, Speech Lab generation, Play, and definitive
  shutdown without editing JSON.
- [ ] GM-AC-026: A clean clone-required flow can create a transient voice,
  generate/play audio, save a reusable profile, assign it to a character, and
  audibly play a roleplay response without leaking reference data.
- [ ] GM-AC-027: The native catalog admits only upstream `tts` and `clone`
  speech tasks, preserves their typed capabilities, supports clone-only
  families, and continues excluding ASR/VC/Music/other tasks.

## ADR check

ADR required: yes

ADR paths:

- `backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md`
- `backlog/decisions/051-private-tts-clone-reference-assets.md`

Reason: generated setup changes durable configuration authority, runtime launch
artifacts, recipe/model discovery, cross-platform process ownership, and the
Settings/Lab contract. Clone support independently changes the profile schema,
private-data storage, admission snapshot, backup/downgrade, and portability
boundary. Keeping these as two decisions prevents process configuration from
becoming coupled to sensitive voice-reference storage.

Existing ADRs followed rather than replaced wholesale:

- ADR-023 — registry/native adapter/complete-WAV/one-child authority;
- ADR-028 — profile and character-assignment ownership;
- ADR-029 — local private data and metadata-only logs;
- ADR-039 — global Settings versus Studio/runtime ownership; and
- ADR-040 — shared model assets and active-user private paths.

## References

- [audio.cpp repository](https://github.com/0xShug0/audio.cpp)
- [audio.cpp release-0.5.1](https://github.com/0xShug0/audio.cpp/releases/tag/release-0.5.1)
- [Pinned audio.cpp README](https://github.com/0xShug0/audio.cpp/blob/238ab6a9e321c17de8e120559f57efeedaeb1345/README.md)
- [Pinned audio.cpp server guide](https://github.com/0xShug0/audio.cpp/blob/238ab6a9e321c17de8e120559f57efeedaeb1345/app/server/README.md)
- [Pinned audio.cpp model specs](https://github.com/0xShug0/audio.cpp/tree/238ab6a9e321c17de8e120559f57efeedaeb1345/model_specs)
- [Existing adapter-registry design](2026-07-23-audio-cpp-tts-adapter-registry-design.md)
- [Managed lifecycle design](2026-08-02-audio-cpp-managed-lifecycle-design.md)
- [Speech & TTS ownership design](2026-07-31-speech-tts-settings-ownership-design.md)
- [Character TTS profile design](2026-07-25-character-tts-generation-profiles-design.md)
- [Voice-profile expansion design](2026-08-04-voice-profiles-expansion-design.md)
- [ADR-050](../../../backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md)
- [ADR-051](../../../backlog/decisions/051-private-tts-clone-reference-assets.md)
