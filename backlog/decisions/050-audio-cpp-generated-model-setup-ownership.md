# ADR-050: Generate audio.cpp model setup from structured global settings

Status: Accepted
Date: 2026-08-09
Related Task: N/A — task decomposition follows the approved design
Extends: ADR-023, ADR-039, ADR-040
Supersedes: Only the conflicting generated-configuration and Windows deferrals
in ADR-023 and ADR-039; all other boundaries remain in force

## Decision

Chatbook will add a second Managed audio.cpp setup source: structured global
Settings that materialize an application-owned, generation-specific
`server.json` only when a deliberate operation is eligible to launch or replace
the managed child. The existing user-provided `server.json` source remains
supported and unchanged.

Structured global Settings are the sole durable authority for generated setup.
The materialized JSON is an immutable launch artifact, not a second editable
configuration store. Chatbook never imports edits from it and never presents it
as user-editable. A saved configuration can remain staged while the active
child continues using its exact applied artifact. Save validates and persists;
it does not create the artifact, bind a port, probe, launch, restart, or stop a
process.

Generated setup uses a built-in, immutable recipe registry. Recipes are
declarative, versioned with Chatbook, and pinned to reviewed audio.cpp release
evidence. They recognize exact model-package layouts and project only
allowlisted server fields. They cannot execute code, download content, add
arbitrary launch arguments, enable CORS, enable request-body logging, or accept
non-loopback managed binding. Runtime remote recipe updates are prohibited.

The initial compatibility baseline is audio.cpp `release-0.5.1`, commit
`238ab6a9e321c17de8e120559f57efeedaeb1345`. The compatibility goal is every
released core or community family tagged `TTS` or `Clone` in that snapshot and
every approved package variant for those families. A release may claim only
the exact recipe/package/platform/backend tuples for which evidence exists;
Chatbook cannot claim general release coverage while an unapproved gap remains.

Models selected from disk are referenced in place with absolute paths. Chatbook
does not copy them into application data. Model Library installations remain
explicit user actions and use the existing shared managed-model store. A
recipe's reviewed artifact identifiers may connect the two flows, but setup
never invokes a package manager or downloads the audio.cpp server binary.

The generated launch contract is:

- exact `127.0.0.1` binding, CORS disabled, and request-body logging disabled;
- a port selected at launch from a bounded application-owned loopback range;
- `lazy_load: true`, with all accepted configured models registered in one
  server and loaded by audio.cpp on first model use;
- absolute model and model-spec paths;
- one direct, no-shell `audiocpp_server --config <artifact>` child;
- one application-owned managed child per Chatbook process; and
- the existing saved/applied/process-generation, lease-drain, supervision,
  diagnostics, and bounded-shutdown rules from ADR-023.

The native TTS catalog admits only audio.cpp's exact speech tasks `tts` and
`clone`. It preserves those typed capabilities and cross-checks them against
the applied recipe; it does not broaden into ASR, voice conversion, music, or
other audio.cpp task routing. Native clone requests extend the provider-neutral
request with typed reference input rather than reopening generic options.

Chatbook may detect an existing `audiocpp_server` on `PATH` or in a reviewed
platform install location and may let the user browse to one. It does not
install, update, bundle, or adopt the server. Unknown server versions are
allowed to Test with an explicit warning but never inherit Verified status.

Generated setup targets macOS, Linux, and Windows. Verification is an exact
tuple of Chatbook version, audio.cpp release and commit, operating system,
architecture, binary identity, model recipe and package, and compute backend.
CPU is the cross-platform baseline; accelerated backends are Verified only on
the tuples actually exercised. Backend `Auto` selects from observed host and
binary evidence. Automatic fallback is permitted only for a stable,
recognized backend-unavailable failure and only after the failed child and
artifact are definitively reaped. Otherwise recovery is explicit, including a
`Try CPU` action.

The local package scanner examines only roots explicitly chosen by the user. It
is bounded, cancellable, runs off the Textual event loop, does not recursively
follow nested symlinks, and reports exact, ambiguous, unknown, incomplete, and
permission-limited outcomes separately. A recipe match is configuration
assistance, not proof that arbitrary bytes are safe or loadable; audio.cpp
remains the runtime contract authority. Chatbook persists the normalized recipe
projection accepted by the user so a later recipe update cannot silently
reinterpret an installed model.

The Settings surface owns setup and global defaults. Speech Lab owns deliberate
Test/Start/Restart/Shutdown, synthesis, playback, and diagnostics. Studio
preferences remain separate and are never overwritten by global setup. The
first generation result remains one complete validated WAV delivered through
the existing asynchronous response interface.

## Context

ADR-023 deliberately began with an external server, then added a managed child
whose binary and `server.json` are both user-provided. That boundary is usable
for experienced audio.cpp users but does not meet the first-time path now being
designed: install Chatbook, install audio.cpp separately, download or select a
supported model package, configure it without hand-authoring JSON, and hear a
sample.

audio.cpp `release-0.5.1` exposes multiple TTS and voice-cloning families through
one server, package model specs, lazy model loading, and OpenAI-shaped speech
requests. Its configuration remains sufficiently model-specific that accepting
an arbitrary GGUF and guessing fields would be unreliable. A bounded recipe
registry gives Chatbook an auditable compatibility promise without embedding
upstream executable logic or becoming a general audio.cpp configuration editor.

This is an architecture decision because it changes configuration authority,
managed-runtime launch inputs, artifact lifecycle, cross-platform process
ownership, model-discovery contracts, and the long-lived division between
Settings and Speech Lab.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep user-authored `server.json` as the only Managed path | Preserves expert control but leaves the stated first-time flow dependent on manual JSON and upstream schema knowledge. |
| Treat generated JSON as another editable durable file | Creates two authorities, makes staged/applied truth ambiguous, and allows edits that bypass Settings validation. |
| Copy every selected model into application data | Duplicates very large assets, breaks user-managed layouts, and conflicts with the existing shared-artifact boundary. |
| Accept any GGUF and infer its family heuristically | A GGUF filename or weak metadata match cannot safely determine required companion assets, task, mode, voice contract, or model-spec projection. |
| Download or install audio.cpp from Chatbook | Adds binary distribution, signing, update, platform-toolchain, licensing, and trust responsibilities outside this workstream. |
| Fetch recipes from a remote service | Makes compatibility and launch behavior change outside the installed application version and expands the supply-chain boundary. |
| Start one child per model | Multiplies lifecycle, memory, port, diagnostics, and shutdown ownership despite audio.cpp already supporting multiple registered models. |
| Add automatic model unloading or a VRAM scheduler | audio.cpp intentionally retains loaded sessions; a second resource manager is not required for the first usable setup flow. |
| Silently fall back through every backend | Can hide broken builds and leave multiple failed children or inconsistent evidence; fallback is safe only for stable recognized failures. |

## Consequences

### TASK-13207 amendment: pinned Model Library packages

Reviewed audio.cpp packages offered by Model Library are described by a static
Chatbook catalog pinned to commit
`597048d9a920592808d7d4e2acd7b9c4596a143a` of the official
`audio-cpp/audio.cpp-gguf` Hugging Face repository. Hugging Face hosts the
bytes; the built-in recipe registry remains the compatibility authority. Recipe
support and official-artifact availability are independent: an approved recipe
may remain local-only, while an explicitly unsupported recipe cannot become
downloadable merely because a similarly named file exists. Every visible
package needs exact file sizes and SHA-256 values, artifact-specific license
evidence, a complete companion-file closure, and an exact recipe mapping.
Runtime does not browse or reinterpret moving repository state.

An explicit Model Library install provisions one self-contained managed package
root without activating it, saving Settings, selecting a TTS/Studio default, or
launching audio.cpp. Guided Settings persists an optional exact managed
artifact identity alongside its existing package evidence. At a deliberate
Start/Test/apply boundary, the runtime activates and acquires that exact
artifact and retains its shared lease for the whole staged/live generation.

Removal is owned by the shared artifact service and begins with a versioned
dependency preview covering Guided Settings and drafts, profiles and clone
references, character assignments, runtime generations, and artifact leases.
Live/staged leases block removal. Durable consumers may remain only after the
user explicitly acknowledges that they will become unavailable; Chatbook never
retargets or deletes them as a removal side effect. The final removal
uses one service-owned authority that acquires the existing lifecycle then
artifact locks exactly once, revalidates the preview under that authority, and
commits without calling the public lock-acquiring delete path recursively.

Package companions live in the same managed root as the selected variant. The
scanner and generated configuration both require one canonical package root;
Chatbook does not synthesize a linked multi-root view merely to deduplicate
companion files.

### Benefits

- A new user can configure supported audio.cpp model packages without writing
  JSON while expert users retain the existing manual source.
- Durable settings, generated launch artifacts, applied process state, and
  runtime evidence have explicit non-overlapping owners.
- Compatibility claims are reviewable at family, package, platform, and
  backend granularity.
- Multiple configured models share the already accepted one-child lifecycle.
- The design stays offline-capable and does not introduce remote code or a new
  runtime dependency.

### Accepted trade-offs

- Recipe work must track upstream package variants; an unknown or ambiguous
  package remains manual-only until explicitly reviewed.
- Referencing user-selected packages in place means moves, deletion, symlink
  retargeting, or byte replacement can require review or fail the next
  deliberate launch.
- Generated setup cannot expose every upstream option. Unsupported expert
  fields require the user-provided JSON source.
- A Verified label is narrower than “audio.cpp supports it”; it requires exact
  Chatbook evidence for the displayed tuple.
- Windows lifecycle and privacy guarantees must be implemented and verified
  explicitly rather than inferred from POSIX behavior.
- Lazy loading reduces startup cost but does not unload a model after first use;
  the UI must disclose that memory remains resident until shutdown.

## Rollback

- Disable the generated setup source while retaining its structured settings
  as inert data.
- Preserve the user-provided JSON and External sources as recovery paths.
- Stop any owned child through the existing definitive lifecycle before
  disabling generated launch.
- Do not delete selected model packages, Model Library artifacts, user JSON, or
  unknown generation artifacts during rollback.
- A build that no longer understands a stored recipe projection must present it
  as needing review; it must not reinterpret or silently launch it.

## Links

- [Guided model setup design](../../Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md)
- [ADR-023: TTS adapter registry and audio.cpp runtime](023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md)
- [ADR-039: Global and Studio TTS settings ownership](039-global-and-studio-tts-settings-ownership.md)
- [ADR-040: Profile-owned state and shared asset paths](040-profile-owned-state-and-shared-asset-paths.md)
- [Managed lifecycle design](../../Docs/superpowers/specs/2026-08-02-audio-cpp-managed-lifecycle-design.md)
- [audio.cpp release-0.5.1](https://github.com/0xShug0/audio.cpp/releases/tag/release-0.5.1)
