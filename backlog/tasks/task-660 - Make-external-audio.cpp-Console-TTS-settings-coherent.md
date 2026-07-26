---
id: TASK-660
title: Make external audio.cpp Console TTS settings coherent
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-26 04:41'
updated_date: '2026-07-26 05:43'
labels:
  - tts
  - audio-cpp
  - console
  - settings
dependencies:
  - TASK-569
references:
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
documentation:
  - >-
    Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md
  - Docs/superpowers/plans/2026-07-25-external-audio-cpp-console-tts.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make newly saved external audio.cpp preferences immediately usable by Console Speak while preserving one application-owned TTS runtime, complete-WAV delivery through the asynchronous response interface, legacy-provider compatibility, and user ownership of the external server process.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Blank legacy audio.cpp model and voice values resolve to explicit compatible modes, while saves persist authoritative mode keys, dual-write exact values, and atomically remove stale canonical and legacy exact keys for dynamic modes.
- [ ] #2 Preference or request selection and the matching provider revision and lease are admitted atomically, settings completion remains bounded during active speech, admitted speech is not silently cancelled, and old and replacement audio.cpp instances never coexist.
- [ ] #3 Console Speak routes audio.cpp through the native TTSService and plays one validated complete WAV through the existing asynchronous response and playback lifecycle, while unassigned legacy providers retain their compatibility path.
- [ ] #4 The installed audio.cpp build passes the pinned-contract characterization gate before UAT, and Chatbook never launches, restarts, signals, supervises, or stops the external server.
- [ ] #5 Deterministic tests cover sentinel persistence, mixed-generation admission races, pending and superseded reconfiguration, native Console routing, complete-WAV cleanup, legacy regressions, and external-process non-ownership.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Amend ADR-023 and record the pre-existing static-analysis baseline before production changes.
2. Characterize the installed Homebrew audio.cpp build against the pinned contract; stop before runtime code if incompatible.
3. Add immutable global TTS preferences plus one atomic set/delete config mutation whose structured result distinguishes pre-replacement failure from post-replacement cache-refresh failure.
4. Make STTS settings translate Select sentinels into explicit modes and persist authoritative mode/value mutations.
5. Add distinct safe revision/reconfiguring/unavailable errors, revision-checked registry admission, and split TTSService resource admission from execution.
6. Add one app-owned request-admission coordinator that freezes preferences and acquires the matching lease under a writer-preferred gate.
7. Run config persistence off-loop inside one service-retained, serialized publication task, then perform a two-second bounded latest-generation audio.cpp handoff without cancelling admitted speech or overlapping adapters.
8. Route Console Speak through native audio.cpp complete-WAV synthesis while retaining all legacy providers behind LegacyTTSAdapter.
9. Prove external-process non-ownership with exact PID checks, run isolated first-time-user Console UAT, execute focused/repository-wide and baseline-aware static verification, update docs, and record evidence.

ADR required: yes
ADR path: backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
Reason: the task strengthens the accepted provider/runtime service contract and configuration lifecycle.
<!-- SECTION:PLAN:END -->

## Pre-implementation Static-analysis Baseline

The following commands ran at task base commit
`5ac4e6299992bf9b7dd7d7a6c6bdc33dd55f9b5b` before any production Python
changes. Each command returned the expected nonzero status. These results are
baseline evidence only and do not authorize fixes to unrelated pre-existing
debt.

The commands ran from the worktree root with Python 3.12.11, Ruff 0.15.22,
and mypy 2.3.0; the tool versions were verified again before recording this
evidence.

### Ruff check

Command (exit 1):

```text
../../.venv/bin/python -m ruff check tldw_chatbook/config.py
```

Exact result:

```text
F841 Local variable `file_validation_section` is assigned to but never used
   --> tldw_chatbook/config.py:757:5
    |
755 |     web_scraper_section = get_toml_section('WebScraper')
756 |     confluence_section = get_toml_section('Confluence')
757 |     file_validation_section = get_toml_section('FileValidation')
    |     ^^^^^^^^^^^^^^^^^^^^^^^
758 |     providers_section_from_toml = get_toml_section('providers')  # Get the [providers] table
759 |     library_section = get_toml_section('library')
    |
help: Remove assignment to unused variable `file_validation_section`

F841 Local variable `providers_section_from_toml` is assigned to but never used
   --> tldw_chatbook/config.py:758:5
    |
756 |     confluence_section = get_toml_section('Confluence')
757 |     file_validation_section = get_toml_section('FileValidation')
758 |     providers_section_from_toml = get_toml_section('providers')  # Get the [providers] table
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
759 |     library_section = get_toml_section('library')
    |
help: Remove assignment to unused variable `providers_section_from_toml`

Found 2 errors.
No fixes available (2 hidden fixes can be enabled with the `--unsafe-fixes` option).
```

### Ruff format check

Command (exit 1):

```text
../../.venv/bin/python -m ruff format --check tldw_chatbook/config.py
```

Exact result:

```text
Would reformat: tldw_chatbook/config.py
1 file would be reformatted
```

### Mypy

Command (exit 1):

```text
../../.venv/bin/python -m mypy tldw_chatbook/config.py tldw_chatbook/TTS/adapter_types.py tldw_chatbook/TTS/adapter_registry.py tldw_chatbook/TTS/TTS_Generation.py tldw_chatbook/TTS/adapter_bootstrap.py tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py tldw_chatbook/UI/STTS_Window.py
```

Exact result:

```text
tldw_chatbook/config.py:76: error: Need type annotation for "DEFAULT_DATABASE_CONFIG" (hint: "DEFAULT_DATABASE_CONFIG: dict[<type>, <type>] = ...")  [var-annotated]
tldw_chatbook/config.py:3965: error: Incompatible default for parameter "key" (default has type "None", parameter has type "str")  [assignment]
tldw_chatbook/config.py:3965: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
tldw_chatbook/config.py:3965: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
tldw_chatbook/config.py:4081: error: Argument 1 to "deep_merge_dicts" has incompatible type "Collection[str]"; expected "dict[Any, Any]"  [arg-type]
tldw_chatbook/config.py:4086: error: Incompatible return value type (got "Collection[str]", expected "dict[str, Any]")  [return-value]
tldw_chatbook/config.py:4145: error: Argument 1 to "deep_merge_dicts" has incompatible type "object"; expected "dict[Any, Any]"  [arg-type]
tldw_chatbook/config.py:4150: error: Incompatible return value type (got "object", expected "dict[str, Any]")  [return-value]
tldw_chatbook/config.py:4857: error: Name "API_MODELS_BY_PROVIDER" already defined on line 1997  [no-redef]
tldw_chatbook/config.py:4858: error: Name "LOCAL_PROVIDERS" already defined on line 2090  [no-redef]
tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py:706: error: "TTSEventHandler" has no attribute "notify"  [attr-defined]
tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py:739: error: "TTSEventHandler" has no attribute "notify"  [attr-defined]
tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py:743: error: "TTSEventHandler" has no attribute "notify"  [attr-defined]
tldw_chatbook/UI/STTS_Window.py:5612: error: Name "_get_model_for_provider" already defined on line 5358  [no-redef]
Found 12 errors in 3 files (checked 8 source files)
```

## Installed-build Stop/Go Gate Evidence

On 2026-07-25, the installed Homebrew package `audio-cpp 0.4` passed the
contract gate without changing the ADR-023 upstream pin:

- Bounded health, model, and voice responses passed the pinned parsers, with
  the characterized model and voice present.
- One bounded speech response had content type `audio/wav` and passed complete,
  non-empty mono PCM16 WAV validation at 44.1 kHz.
- The provenance schema change followed a focused TDD red/green cycle: the
  contract test first failed because compatible-build evidence was absent and
  then passed after the single bounded entry was added.
- The focused contract and adapter suite passed all 361 tests, and
  `git diff --check` passed.
- After verification, health remained `ok` and the same pre-existing process
  still owned the listener. Chatbook did not launch, restart, signal, adopt,
  reconfigure, supervise, or stop it.

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Automated unit, integration, Textual, race, and cleanup tests cover every acceptance criterion and pass.
- [ ] #2 Ruff checks and formatting, compileall, focused typing checks where configured, and git diff --check pass.
- [ ] #3 ADR-023, user documentation, compatibility limitations, external-process ownership, and UAT evidence are current.
- [ ] #4 Self-review confirms the implementation stays within Slice 1 and adds no managed process or character-profile behavior.
- [ ] #5 All acceptance criteria and DoD items are checked, concise implementation notes are added, and status changes to Done only after all evidence exists.
<!-- DOD:END -->
