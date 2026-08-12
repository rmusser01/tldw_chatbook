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
- Quick Voice setup contains endpoint, authentication, model, voice, sample,
  Test and Hear, and Use as default.
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
- Configuration and credential revisions

Existing profile, character, Studio, global, and provider fallback precedence
remains supported. Every UI surface displays and dispatches the same resolved
selection.

### Authentication

Add `app_tts.OPENAI_AUTH_MODE` with accepted values `api_key` and `none`.
Missing or invalid values resolve to `api_key`, preserving fail-closed behavior
for existing installations.

`none` has end-to-end meaning:

- Readiness does not require a credential.
- No credential lookup occurs.
- No Authorization header is sent.
- The UI identifies the configuration as explicitly unauthenticated.

Selecting `none` for a non-loopback plaintext HTTP endpoint requires a one-time
confirmation. Loopback HTTP and all HTTPS endpoints do not show that warning.

Credential operations increment a non-secret credential revision. Tests and
profiles depend on that revision without hashing or recording secret values.

### Configuration Revisions

Successful saves follow this order:

1. Validate the draft.
2. Persist atomically.
3. Reload the runtime configuration snapshot.
4. Apply provider adapter changes.
5. Publish a monotonic configuration revision.

Consumers ignore older revisions and resolve fresh state from the runtime
snapshot. A persistence, cache, or adapter failure cannot advertise the draft
as the active configuration.

## First-Run Experience

### Provider Choice

The provider control must never place `Static` headers or other non-radio
widgets inside a Textual `RadioSet`. The replacement may use an option list or
another grouped selection control, but keyboard traversal must reach only real
provider options.

Required regression: Down and Space across every visible row cannot assert,
crash, or select a group heading.

### Resume And Recovery

Persist only the active step and non-secret completed-step values. API keys and
other secrets remain memory-only until an explicit credential operation.

An interrupted setup produces a recovery prompt on the next launch:

- Resume setup
- Start over
- Later

Resume opens the saved step. A repeated startup failure falls back to the Home
Resume setup action instead of repeatedly pushing the wizard. Required provider
and model step failures show Retry, Use manual setup, and Finish later. Only
optional steps may be skipped automatically.

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

Legacy aliases appear under Advanced, except when an existing configuration
uses one and needs a visible migration path.

Local dependency status names the affected features. It must state that a
configured OpenAI-compatible endpoint remains usable when local synthesis or
transcription packages are absent.

### Save Behavior

Clickable Save and Revert controls remain available while text inputs are
focused. Letter shortcuts never fire inside a text-entry widget. A save action
cannot append its shortcut character to an endpoint.

Test state is keyed by a non-secret fingerprint containing tested fields,
provider identity, configuration revision, and credential revision. Saving
unchanged fields preserves Verified. Editing a tested field changes the state
to Needs retest.

### Structured Readiness

Readiness contains independent values:

- Configuration: valid, incomplete, or invalid
- Connection: reachable, unreachable, not tested, or unsupported

UI summaries never describe a complete test as passed when the connection
probe failed. Valid but offline configurations remain saveable.

Endpoint normalization parses known terminal paths, including:

- `/v1/chat/completions`
- `/chat/completions`
- `/v1/models`
- `/v1/audio/speech`

It then derives a provider-appropriate discovery or health URL. Unknown paths
are probed directly when safe, or reported as Connection not tested. They are
never extended by blindly appending `/v1/models`.

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

Verified state records the provider configuration and credential revisions.
Changing either marks dependent profiles Needs retest.

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
Speak replies shows a one-time confirmation that completed assistant text is
sent to the selected TTS provider and may incur charges.

Auto-speak uses the existing speech sequencer and playback cancellation path.
Hands-free mode owns speech while active and suppresses duplicate auto-speak.

Eligible content is a completed assistant or character response. Tool output,
errors, system messages, partial streams, and cancelled responses are excluded.
One auto-speak failure changes the conversation state to Paused and exposes
Retry speech and Resume auto-speak. Manual Speak remains available.

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
- Explicit authentication behavior and transport headers
- Configuration and credential revisions
- Structured readiness and endpoint derivation
- Profile verification invalidation
- Auto-speak eligibility and pause behavior
- Paste-block boundaries

### Textual Integration Tests

- Keyboard traversal across every first-run provider option
- Interrupted setup, Resume, Start over, Later, and crash-loop fallback
- Save while an input is focused
- Test fingerprint preservation and invalidation
- Speech Lab refresh after a successful configuration revision
- Profile and blend navigation labels
- Character import start directory and selected row
- Untouched-tab reuse and preserved worked-on tabs
- Speak visibility and playback state
- Auto-speak/hands-free exclusion
- Paste block display, expand, and submission

### Service And End-To-End Tests

Fake OpenAI-compatible services capture exact chat and TTS requests. Acceptance
requires:

- Chat uses the resolved provider and model.
- TTS uses `pocket-tts`, the selected voice, and WAV in the reference UAT.
- Authentication None sends no Authorization header.
- Configuration-valid/connection-failed copy is not a passing test.
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
