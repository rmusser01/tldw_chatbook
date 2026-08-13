---
id: TASK-15771
title: Sweep non-callable reactive list/dict defaults shared by identity across widget instances
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - bug
  - textual
priority: medium
---

## Description

Found and confirmed during task-15479 (input-latency burn-down), flagged as
independent of that task's own fix and explicitly out of scope: Textual's
`Reactive._initialize_reactive` installs `default_or_callable() if
callable(...) else default_or_callable` — since a bare `[]` or `{}` literal
is not callable, the exact same list/dict object is installed as the
reactive's backing attribute on every widget instance that has not
explicitly reassigned it. This is the classic Python mutable-default-argument
trap, applied to a Textual `reactive()` default.

The proven instance is `Widgets/TTS/character_voice_widget.py`'s
`characters = reactive([])`: two `CharacterVoiceWidget` instances that both
start from the un-set default and never reassign `characters` share and
cross-mutate the same underlying list. Task-15479 found this only because it
made its own new tests order-dependent within one pytest session (one test's
`.append()` leaked into a later test's row count via the shared default);
its tests work around it defensively by assigning a fresh list before use,
documented in that file's test-module docstring. This is a live production
hazard, not just a test-isolation nuisance: any two instances of the same
widget class that both rely on the declared default will alias state.

## Acceptance Criteria

- [ ] `character_voice_widget.py`'s `characters` reactive uses a factory
      (e.g. `reactive(list)` / a `default=` callable) so each instance gets
      its own list, not a shared one
- [ ] A test proves two `CharacterVoiceWidget` instances no longer alias:
      mutating one's `characters` does not affect the other's
- [ ] A repo-wide sweep for the same shape — `reactive([...])` /
      `reactive({...})` with a literal (not a callable) default — classifies
      each hit and fixes any other widget instantiated more than once per
      app session where cross-instance aliasing is plausible; sites that are
      genuinely singleton-per-app or already reassign the attribute before
      first read are recorded as reviewed, not touched
- [ ] Existing `character_voice_widget` and STTS test suites stay green
