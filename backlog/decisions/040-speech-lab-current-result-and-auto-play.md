# ADR-040: Speech Lab Current Result and Auto-play Ownership

Status: Accepted
Date: 2026-08-02
Related Task: TASK-1700
Extends: [ADR-039 Global and Studio TTS Settings Ownership](039-global-and-studio-tts-settings-ownership.md)

## Context

Live UAT showed a completed WAV status while the Speech Lab's playback actions
were clipped outside the visible result pane. The pane also mounted an empty
multi-take history that generation never populated, displayed `0:00 / 0:00`
before any duration was known, and gave runtime diagnostics more visual weight
than the generated artifact. A first-time user could generate successfully and
still have no discoverable way to hear the result.

The approved remediation adds an optional auto-play preference. That choice is
persistent product state, so its owner and rollback behavior must be explicit.

## Decision

### One current result

Speech Lab presents one session-scoped current result. A successful generation
replaces the prior current result and exposes Play and Export in the result
region. Multi-take history and comparison are not part of the active product
contract in this change and will not be represented by non-functional controls.

The result reports only known facts: readiness, actual format, and a positive
duration when supplied by validated artifact metadata. Unknown duration is
omitted rather than rendered as zero. The UI states that the current result is
temporary and must be exported to retain a user-owned copy.

### Auto-play belongs only to Studio

`speech_studio.auto_play` is a boolean Studio preference. It defaults to
`false`, is written through the existing revisioned `speech_studio` store, and
is included in Reset to Global's deletion of Studio-local choices. Although no
global auto-play value exists, reset returns playback to the safe off default.

Auto-play does not belong to global TTS defaults, provider configuration,
character TTS profiles, assignments, or request provenance. It changes only
what the Speech Lab does after receiving a complete generated artifact; it is
never sent to a provider.

The field is an additive optional member of schema version 1. It is serialized
only when true, so old records continue to mean off. Invalid non-boolean values
recover to off with a bounded `speech_studio.auto_play` issue. Older builds may
ignore the additive field; a rollback can therefore lose only this convenience
preference and cannot affect provider or character behavior.

### Audition behavior

With auto-play off, successful generation moves keyboard focus to the enabled
Play action. With auto-play on, Speech Lab invokes its existing playback path
and keeps Stop reachable. Playback failure leaves the current result available
for retry or export.

### Information hierarchy

The visible primary flow is Configure, Generate, Audition. One concise provider
readiness line stays in the main flow. Runtime/catalog/local-dependency details
and the generation log use collapsed disclosures. A known non-applicable axis,
including Language for audio.cpp, remains mounted for compatibility but is not
shown or placed in the focus path.

## Consequences

- A completed generation always has an obvious next action at supported widths.
- Studio auto-play is opt-in and cannot surprise first-time users or alter
  global/roleplay behavior.
- Artifact duration and retention copy remain truthful.
- Existing async generation and complete-WAV adapter contracts do not change.
- Session comparison remains absent until it has real storage, routing, and
  playback behavior.
