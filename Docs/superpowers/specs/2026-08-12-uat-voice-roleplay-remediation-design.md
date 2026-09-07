# New-User Voice and Roleplay UAT Remediation Design

**Status:** Approved

**Date:** 2026-08-12

**Baseline:** `origin/dev` at `5414d811b8720c1c32c5813f96925a82c60c5f72`

## Objective

Make the clean-install journey from first-run setup through an imported
character roleplay and spoken character reply reliable, understandable, and
consistent. This tranche fixes the state contracts and focused interaction
problems found during UAT while preserving the current overall information
architecture.

The broader setup and Speech Lab redesign is a separate follow-up tranche. It
may replace screen composition, but it must retain the configuration and
routing contracts defined here.

## UAT Scenario

The acceptance journey is:

1. Start Chatbook with a clean user profile.
2. Complete Quick Setup or Full Setup.
3. Configure an OpenAI-compatible PocketTTS endpoint, explicit authentication
   mode, model, voice, output format, and sample.
4. Save the selection as the application default.
5. Import a character-card PNG.
6. Start a roleplay conversation from that card.
7. Receive an in-character response.
8. Enable per-conversation Speak replies, or manually select Speak.
9. Confirm the response is sent to the configured TTS endpoint with the exact
   model, voice, format, and authentication behavior.

## Product Decisions

- Automatic reply speech is opt-in per conversation and starts off.
- OpenAI-compatible TTS authentication is explicit: `api_key` or `none`.
- Quick Setup and Full Setup both include Voice setup.
- Quick Voice setup contains endpoint, authentication, model, voice, sample
  text with Test and Hear, and Use as default.
- Provider-neutral Voice Profiles and Kokoro Voice Blends are separate tools.
- A valid offline configuration may be saved without a successful sample.
- Existing information architecture remains in this tranche.
- Full setup and Speech Lab redesign is deferred to the next tranche.

## Architecture

### Effective Chat Configuration

Introduce one pure effective-chat resolver. It returns provider, model,
endpoint, credential source metadata, readiness inputs, and provenance without
network access or secret projection.

Precedence is:

1. Explicit session or handoff selection
2. Model profile
3. `chat_defaults`
4. Legacy provider-specific fallback

`chat_defaults.provider` and `chat_defaults.model` are the authoritative global
defaults. Provider-specific sections own transport details such as endpoint and
credential references. Existing provider-specific model values remain
read-compatible fallbacks when no canonical model exists. They are not allowed
to override a present canonical default.

Legacy provider aliases are normalized in the read model and UI. Opening a
configuration never rewrites it. An explicit Save writes the canonical value.

### Effective TTS Configuration

Introduce one pure effective-TTS resolver used by Settings, Speech Lab,
Console Speak, roleplay auto-speak, and onboarding. It returns:

- Provider identity and provenance
- Model mode and model ID
- Voice mode and voice ID
- Response format and speed
- Provider configuration reference
- Authentication mode and credential-source metadata
- Default or character profile source
- Saved, applied, and active provider configuration revisions

Existing profile, character, Studio, global, and provider fallback precedence
remains supported. Every UI surface displays and dispatches the same resolved
selection.

### Authentication

Add `app_tts.OPENAI_AUTH_MODE` with accepted values `api_key` and `none`.
Missing or invalid values resolve to `api_key`, preserving fail-closed behavior
for existing installations. The Settings state field is
`authentication_mode`, and it is part of
`GLOBAL_TTS_PROVIDER_FIELD_IDS["openai"]` so every save path has the same
ownership contract.

The Official OpenAI preset always selects `api_key` and does not permit `none`
while its endpoint is active. Switching from an unauthenticated compatible
endpoint to Official OpenAI resets the draft to `api_key` before validation.
Custom OpenAI-compatible endpoints expose both explicit choices, except that
the normalized Official OpenAI origin always requires `api_key` regardless of
which preset produced the draft.

`none` has end-to-end meaning:

- Readiness does not require a credential.
- No credential lookup occurs.
- No Authorization header is sent.
- The UI identifies the configuration as explicitly unauthenticated.

Selecting `none` for a non-loopback plaintext HTTP endpoint requires explicit
confirmation. The non-secret confirmation is tied to a fingerprint of the
normalized endpoint origin. Changing the endpoint origin or authentication
mode invalidates it. Userinfo and query strings are rejected and never enter
the fingerprint. Loopback HTTP and all HTTPS endpoints do not show that
warning.

Local credential and authentication changes invalidate the existing provider
configuration revision; no second persistent credential-revision system is
introduced. Connection and sample verification is process-scoped and returns
to Needs test after restart unless the provider can re-establish availability
without synthesis. This avoids implying that an endpoint or external secret is
still usable when Chatbook has not observed it in the current process.

### Configuration Revisions

The existing saved, applied, and active runtime revision distinction remains
authoritative. Successful saves follow this order:

1. Validate the draft.
2. Persist atomically.
3. Increment the saved provider configuration revision.
4. Reload the runtime configuration snapshot.
5. Apply provider adapter changes.
6. Publish the applied and active runtime revisions after successful handoff.

Consumers compare the saved, applied, and active runtime revisions and resolve
fresh state from the active snapshot. A persistence or reload failure does not
advance any revision. An adapter handoff failure may leave a valid draft saved,
but the UI reports Saved, activation failed and generation continues with the
previous active runtime. No surface may advertise the saved draft as active
until the applied and active revisions match it.

## First-Run Experience

### Provider Choice

The provider control must never place `Static` headers or other non-radio
widgets inside a Textual `RadioSet`. The replacement may use an option list or
another grouped selection control, but keyboard traversal must reach only real
provider options.

Required regression: Down and Space across every visible row cannot assert,
crash, or select a group heading.

### Resume And Recovery

Persist only the active step and non-secret completed-step values in a
versioned setup-draft namespace. That draft is not an active application
configuration and is never consumed by chat or speech resolvers. API keys and
other secrets remain memory-only until an explicit credential operation.
Start over deletes only the setup draft and transient setup state; it does not
delete unrelated settings or a credential the user already explicitly saved.

An interrupted setup produces a recovery prompt on the next launch:

- Resume setup
- Start over
- Later

Resume opens the saved step. Before that push, setup stores a resume-attempt
marker and clears it only after the target step mounts successfully. If the
next launch finds the marker uncleared, it does not push the wizard again; Home
shows Resume setup and Start over actions instead. Required provider and model
step failures show Retry, Use manual setup, and Finish later. Only optional
steps may be skipped automatically.

### Voice Setup

Both setup tracks include Voice setup. Quick Setup uses a compact single-page
flow. Full Setup may expose advanced provider details while writing through the
same save command.

The local path is labeled OpenAI-compatible endpoint, with PocketTTS as a
recommended preset. Official OpenAI is a separate preset with API-key
authentication selected.

Save is available for locally valid values. A successful sample marks the
selection Verified. Saving an unverified selection is allowed and visibly
marked Needs test.

Use as default writes the canonical global TTS preference axes: provider,
model mode and ID, voice mode and ID, response format, and speed. It references
the provider configuration resolved at generation time; it does not copy an
endpoint, credential, or secret into a profile. When selected, setup saves the
provider configuration first and only updates the global default after the new
runtime revision becomes active. If activation fails, the previous default is
preserved and the user can retry without re-entering the draft.

### Progress And Network Activity

Progress count and active styling derive from the selected track. Quick Setup
has five steps after Voice setup is added. Active, completed, and upcoming
states must remain distinct in dark and light themes.

Setup model discovery contacts only the selected provider. Unrelated provider
catalogs do not refresh during first run. Background refresh remains a user
setting and does not emit unsolicited first-run toasts.

## Settings And Provider Testing

### Information Hierarchy

Primary Settings content uses task language and current status. Runtime owner,
raw setting keys, revisions, and provenance move behind a Details disclosure at
all widths. The Scope Inspector is not the default reading path.

The default provider view contains no raw configuration keys, owner IDs,
provenance chains, or revision numbers. Details and Scope Inspector are
collapsed initially at desktop and narrow widths and preserve no expanded state
across application restarts.

Legacy aliases appear under Advanced, except when an existing configuration
uses one and needs a visible migration path.

Local dependency status names the affected features. It must state that a
configured OpenAI-compatible endpoint remains usable when local synthesis or
transcription packages are absent.

### Save Behavior

Clickable Save and Revert controls remain available while text inputs are
focused. Letter shortcuts never fire inside a text-entry widget. A save action
cannot append its shortcut character to an endpoint.

Test state is keyed by a non-secret fingerprint containing the normalized
tested fields, provider identity, and saved provider configuration revision.
The fingerprint never contains credential values. Saving unchanged fields
preserves Verified within the current process. Editing a tested field or
changing the provider configuration revision changes the state to Needs
retest. After restart, prior connection or sample evidence is not restored as
Verified unless current provider capability discovery re-establishes it.

### Structured Readiness

Readiness contains independent values:

- Configuration: valid, incomplete, or invalid
- Connection: reachable, unreachable, not tested, or unsupported

UI summaries never describe a complete test as passed when the connection
probe failed. Valid but offline configurations remain saveable.

For speech-only OpenAI-compatible services, a successful bounded sample POST
to the configured speech operation is the authoritative reachable result.
Model or voice catalog discovery is optional and reported separately; an
unsupported catalog cannot turn a successful speech sample into a connection
failure. Manual model and voice entry remains available when discovery is not
supported.

Endpoint normalization parses known terminal paths, including:

- `/v1/chat/completions`
- `/chat/completions`
- `/v1/models`
- `/v1/audio/speech`

It then derives a provider-appropriate discovery or health URL. Unknown paths
are reported as Connection not tested unless an existing provider adapter
declares a safe health target. Probes use the shared URL validation and SSRF
policy, remain on the normalized origin, disable redirects, and never guess a
path, forward userinfo, or carry query strings. Known paths are never extended
by blindly appending `/v1/models`. Explicitly configured loopback destinations
remain permitted for local PocketTTS use; redirect targets and automatically
derived URLs receive no broader access than the original validated origin.

## Speech Lab And Voice Profiles

### Live Synchronization

Speech Lab listens for configuration revisions and resolves fresh global and
Studio state. It does not retain construction-time global defaults after
Settings changes. Provider, model, voice, format, and speed show their actual
value and source.

### Voice Profiles

Provider-neutral Voice Profiles use the existing TTS profile service. A profile
stores provider identity and synthesis axes, not endpoint URLs or credentials.
Provider configuration is referenced at resolution time.

The existing profile repository, draft types, generation provenance, and
provider-configuration revision model are reused; this tranche does not create
a second profile store or persistence schema. The profile availability service
is extended with a process-scoped evidence cache so an OpenAI-compatible
profile created from a successful sample can carry current provider-neutral
test evidence. Evidence is keyed by profile ID and revision, exact synthesis
axes, and active provider configuration revision. Today, non-audio.cpp profiles
deliberately remain Unverified; treating that state as Verified without
extending the service is not acceptable.

Verified state records the active provider configuration revision and exact
tested synthesis axes. A provider configuration change or edited synthesis
axis marks the profile Needs retest. On restart, the profile remains usable but
returns to Needs test unless current provider capability discovery can
re-establish availability without generating speech.

### Voice Blends

The current Kokoro-specific blend editor is renamed Voice Blends and remains a
separate tool. Copy, labels, navigation, and tests must not imply that a Kokoro
blend is a generic provider profile.

### Dependency And Mapping Behavior

The local dependency message distinguishes local TTS, local STT, and remote or
OpenAI-compatible availability. Recovery guidance names the exact optional
dependency group.

`openai_tts_mappings.json` is packaged and required by installed-distribution
tests. Built-in mappings remain a defensive fallback and log informationally,
not as a startup warning.

## Roleplay And Console

### Character Import

Character import uses the enhanced picker with context `character_import`.
Start-directory precedence is:

1. Remembered character-import directory
2. `~/Documents` when it exists
3. User home directory

The picker never defaults to the process working directory. The selected file
row has a clear focused and selected state. Display sanitization handles
decoding replacement characters, invalid controls, and unsupported
terminal-width sequences without changing stored card data.

### Character Handoff

Chat now reuses an untouched initial `Chat 1` session. It creates another tab
when the existing session contains user work or non-default state. The active
character, greeting, and conversation title are visible immediately.

The handoff consumes the canonical effective chat configuration, including the
correct `chat_defaults` model when no higher-precedence selection exists.

### Speak Replies

Each Console conversation persists `auto_speak`, defaulting to false. Enabling
Speak replies shows a confirmation that completed assistant text is sent to
the selected TTS provider. The copy names the effective provider and sanitized
destination and mentions possible charges only when applicable.

Consent is stored per conversation against a non-secret destination
fingerprint containing provider identity and normalized origin. Before every
automatic dispatch, Console resolves the current effective TTS configuration
and compares the destination. A changed provider or origin pauses automatic
speech and asks for confirmation before any text is sent. The fingerprint
contains no userinfo, query, credential, or message text.

Auto-speak uses the existing speech sequencer and playback cancellation path.
Hands-free mode owns speech while active and suppresses duplicate auto-speak.

Eligible content is a completed assistant or character response. Tool output,
errors, system messages, partial streams, and cancelled responses are excluded.
Enabling the toggle does not speak earlier messages. A response that completes
after its conversation loses focus is not played automatically; it retains
Manual Speak and shows that speech is ready. Returning to the conversation does
not trigger delayed playback.
One auto-speak failure changes the conversation state to Paused and exposes
Retry speech and Resume auto-speak. The paused state persists with the
conversation so restart cannot silently retry. Manual Speak remains available.

Manual Speak or Stop remains visible in the assistant message header. Other
message actions remain in the selected-message action row. Playback state is
Generating, Playing, Stopped, or Failed.

### Paste Blocks

Long pastes remain collapsed as distinct blocks. Each block displays:

`Pasted text | N characters | Expand`

Adjacent collapsed blocks are joined with one newline on submission unless the
user expands and edits them. The term Unfurl is removed from this workflow.

## Accessibility And Visual Quality

- Increase first-run text and secondary-copy contrast.
- Strengthen active-step, selected-row, and keyboard-focus indicators.
- Reduce unused first-run vertical space without crowding controls.
- Keep all controls and text within their containers.
- Verify dark and light themes.
- Verify 120x40, 177x45, and narrow supported terminal dimensions.
- Add token assertions and rendered snapshots for active, selected, focused,
  disabled, warning, and error states.

## Error Handling And Privacy

- Required setup failures offer recovery and never exit the application.
- Optional setup failures may be skipped with an explicit summary entry.
- Invalid drafts cannot replace the saved active configuration.
- Connection failure is retryable and does not invalidate locally valid data.
- TTS failure leaves the text response intact and restores manual Speak.
- Auto-speak pauses after a failure rather than retrying or disabling silently.
- Logs may include provider ID plus sanitized scheme and host category.
- Logs and toasts never include userinfo, query strings, credentials, request
  text, card text, or generated speech text.

## Delivery Plans

This design is implemented through four sequential plans on one branch:

1. First-run crash recovery, resume behavior, progress, and visual resilience
2. Canonical configuration, authentication, Settings, provider testing, and
   compact Voice setup in both first-run tracks
3. Speech Lab synchronization, Voice Profiles, and Voice Blends
4. Roleplay import, Console handoff, auto-speak, message actions, and paste UX

Each plan uses test-driven development and ends in independently working,
regression-tested software. Later plans consume contracts established by
earlier plans rather than duplicating their own configuration logic.

## Verification Strategy

### Unit Tests

- Chat and TTS precedence, provenance, migration, and alias normalization
- Setup-draft isolation from active chat and TTS configuration
- Explicit authentication behavior and transport headers
- Plaintext unauthenticated confirmation invalidation by normalized origin
- Saved, applied, and active configuration revisions
- Process-scoped connection and sample verification
- Structured readiness and endpoint derivation
- Profile verification invalidation
- Auto-speak eligibility and pause behavior
- Auto-speak destination-consent invalidation and active-conversation gating
- Paste-block boundaries

### Textual Integration Tests

- Keyboard traversal across every first-run provider option
- Interrupted setup, Resume, Start over, Later, and crash-loop fallback
- Save while an input is focused
- Official OpenAI preset transition resets authentication to API key
- Use as default commits only after runtime activation and preserves the prior
  default on activation failure
- Test fingerprint preservation and invalidation
- Details and Scope Inspector are collapsed on each application start
- Speech Lab refresh after a successful configuration revision
- OpenAI-compatible profile verification, invalidation, and restart behavior
- Profile and blend navigation labels
- Character import start directory and selected row
- Untouched-tab reuse and preserved worked-on tabs
- Speak visibility and playback state
- Auto-speak/hands-free exclusion
- Destination-change reconfirmation and background-conversation suppression
- Paste block display, expand, and submission

### Service And End-To-End Tests

Fake OpenAI-compatible services capture exact chat and TTS requests. Acceptance
requires:

- Chat uses the resolved provider and model.
- TTS uses `pocket-tts`, the selected voice, and WAV in the reference UAT.
- Authentication None sends no Authorization header.
- Configuration-valid/connection-failed copy is not a passing test.
- Official OpenAI cannot be saved with authentication None.
- Endpoint probes reject cross-origin redirects and do not guess unknown paths.
- Use as default resolves the tested provider, model, voice, format, and speed
  without copying endpoint or credential data into the profile.
- Character greeting, response, and spoken response remain in one active chat.

### Packaging And Visual Tests

- Installed distribution contains `openai_tts_mappings.json`.
- Dark and light rendered states meet the contrast-token contract.
- Supported dimensions have no overlap, clipping, or inaccessible actions.

### Manual Release Check

Repeat the clean-profile UAT against a real PocketTTS service and listen for
successful playback, acceptable latency, intelligibility, and voice identity.
Objective voice quality remains a manual release criterion.

## Finding Coverage Matrix

| UAT finding | Owning plan | Required evidence |
| --- | --- | --- |
| Provider keyboard selection crashes first run | 1 | Keyboard integration regression |
| Restart abandons setup | 1 | Resume/recovery integration tests |
| Speech/TTS missing from Quick Setup | 2 | Five-step Quick Setup contract |
| Low contrast and unclear active progress | 1 | Theme tokens and snapshots |
| Excess whitespace and weak discovery state | 1 | Layout and selected-provider discovery tests |
| Unsolicited first-run catalog refresh | 1 | Selected-provider-only network test |
| Dense Settings and technical jargon | 2 | Details disclosure and primary-copy tests |
| Legacy aliases selectable but invalid | 2 | Alias normalization/migration tests |
| Provider test passes while endpoint fails | 2 | Structured readiness tests |
| Probe creates malformed `/v1/models` URL | 2 | Endpoint derivation tests |
| Save invalidates an unchanged test | 2 | Test fingerprint tests |
| Save shortcut mutates endpoint input | 2 | Focused-input shortcut regression |
| Local endpoint incorrectly requires key | 2 | Authentication None readiness/transport test |
| Saved chat model ignored by roleplay | 2 and 4 | Resolver and handoff tests |
| Speech Lab ignores inherited globals | 3 | Configuration-revision refresh test |
| Generic Voice Profiles opens Kokoro blend | 3 | Separate profile/blend UI tests |
| Local dependency message is ambiguous | 3 | Capability-specific copy tests |
| Missing OpenAI mapping resource warning | 3 | Installed-distribution resource test |
| Import starts in process directory | 4 | Picker-context directory test |
| Character file selection is ambiguous | 4 | Focus/selection snapshot |
| Chat now leaves a confusing blank tab | 4 | Untouched-session reuse test |
| Speak action is hidden and ambiguous | 4 | Persistent Speak/Stop integration test |
| No opt-in automatic character speech | 4 | Persisted toggle and eligibility tests |
| Playback state lacks useful feedback | 4 | Playback lifecycle UI test |
| Paste tokens concatenate and say Unfurl | 4 | Paste-block boundary tests |
| Unsupported card display glyphs | 4 | Display-only sanitization tests |

## Out Of Scope

- Replacing the overall Settings or Speech Lab information architecture
- Redesigning every provider setup flow
- Automatically enabling speech in any conversation
- Persisting secrets in wizard-resume state or profiles
- Defining an automated subjective voice-quality score
- Removing legacy configuration read compatibility

## Completion Gate

The tranche is complete when every row in the finding coverage matrix is fixed
and covered, or explicitly moved into the approved redesign tranche with a
written reason. The original clean-profile UAT must pass without a crash,
configuration mismatch, dummy credential, malformed probe, hidden Speak action,
or incorrect TTS payload.
