# Speech Lab Current-result UX Design

**Status:** Approved during TASK-1989 UAT
**Date:** 2026-08-02
**Canonical decisions:** [ADR-039](../../../backlog/decisions/039-global-and-studio-tts-settings-ownership.md) and [ADR-040](../../../backlog/decisions/040-speech-lab-current-result-and-auto-play.md)

## Problem

Speech Lab can complete generation and say a WAV is ready while clipping every
playback action out of the visible result pane. The surrounding interface makes
the failure harder to diagnose: an unimplemented take-history empty state leads
the result area, an unknown duration is shown as `0:00 / 0:00`, and seven
diagnostic rows compete with the artifact the user came to hear.

## Approved direction

The primary loop is:

1. Configure the provider and generation axes.
2. Generate one result.
3. Audition or export that current result.

This change delivers one current result. It does not present multi-take
comparison until that capability has real artifact retention and routed
per-result actions.

## Interaction contract

### Empty

- The result heading and Play/Export actions are visible.
- Play, Pause, Stop, and Export are disabled.
- Copy says no audio has been generated and directs the user to Generate.
- No progress timer or zero duration is shown.

### Generating

- Existing asynchronous generation and progress behavior remains.
- The prior delivered result may remain playable until a replacement is
  delivered; stale operations still cannot replace a newer result.

### Ready

- The result says `Ready · FORMAT` and appends `MM:SS` only for a validated,
  positive duration.
- It says `Temporary result — export to keep a copy.`
- Play and Export are enabled and visible without scrolling at supported
  desktop widths; stacked/narrow layouts remain scrollable.
- With auto-play off, Play receives focus after delivery.
- With auto-play on, existing playback starts and Stop remains reachable.

### Playback and failure

- Play, Pause/Resume, and Stop use their existing paths and truthful enabled
  states.
- Playback failure names the problem without discarding the generated artifact;
  Play and Export remain available where safe.
- Progress/time appears only after the player reports a positive duration.

## Information hierarchy and copy

- Use `Sample text` instead of `Random` and `Clear text` instead of `Clear`.
- Hide Language for audio.cpp because that provider does not accept it.
- Keep one concise provider readiness line in the main flow.
- Put configuration/runtime/catalog/local dependency facts in collapsed
  `Connection details`; keep `Generation log` collapsed separately.
- Compact Speech-specific collapsible titles to one terminal row.
- Use semantic design tokens for separators and status colors.
- Keep existing keyboard bindings and make Play focus movement visible; action
  tooltips may include relevant shortcut labels.

## Persistence and ownership

The Studio settings surface adds `Play generated audio automatically` with
explicit On/Off consequence copy. It is stored only as
`speech_studio.auto_play`, defaults off, and is removed by Reset to Global.
It never changes global defaults, provider settings, or character profiles.

## Acceptance checks

- At 120×40 and 80×24, the current-result action strip and enabled Play/Export
  actions are contained by the result pane and viewport after delivery.
- At a narrow stacked width, all actions are reachable by scrolling.
- Unknown duration never renders as `0:00 / 0:00`.
- Known audio.cpp `audio_duration_ms` metadata yields a truthful formatted
  duration.
- Default/off auto-play focuses Play without invoking playback; saved/on
  auto-play invokes playback once.
- The diagnostics disclosure starts collapsed and the audio.cpp language cell
  is not displayed or focusable.
- Studio preference round-trip, corrupt recovery, reset, dirty state, and scope
  isolation are covered by automated tests.
