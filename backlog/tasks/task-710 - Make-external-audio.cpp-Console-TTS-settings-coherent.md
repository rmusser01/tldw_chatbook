---
id: TASK-710
title: Make external audio.cpp Console TTS settings coherent
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-26 04:41'
updated_date: '2026-07-31 11:20'
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
- [x] #1 Blank legacy audio.cpp model and voice values resolve to explicit compatible modes, while saves persist authoritative mode keys, dual-write exact values, and atomically remove stale canonical and legacy exact keys for dynamic modes.
- [x] #2 Preference or request selection and the matching provider revision and lease are admitted atomically, settings completion remains bounded during active speech, admitted speech is not silently cancelled, and old and replacement audio.cpp instances never coexist.
- [x] #3 Console Speak routes audio.cpp through the native TTSService and plays one validated complete WAV through the existing asynchronous response and playback lifecycle, while unassigned legacy providers retain their compatibility path.
- [x] #4 The installed audio.cpp build passes the pinned-contract characterization gate before UAT, and Chatbook never launches, restarts, signals, supervises, or stops the external server.
- [x] #5 Deterministic tests cover sentinel persistence, mixed-generation admission races, pending and superseded reconfiguration, native Console routing, complete-WAV cleanup, legacy regressions, and external-process non-ownership.
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

## Slice 1 UAT and Verification Evidence

### Live Console UAT

Before rebase, isolated clean-config Textual Console UAT passed against the
user-owned audio.cpp listener at `http://127.0.0.1:8080`:

- provider `audio_cpp`, model mode `first_available`, and voice mode
  `server_default` were saved and used without restarting Chatbook;
- a deterministic Mira response used exactly one native adapter;
- one complete owner-only (`0600`) 594,604-byte WAV was produced: mono PCM16 at
  44.1 kHz, 297,280 frames, and 6.741 seconds;
- lifecycle counts were complete `1`, playback `1`, progress `4`, and streaming
  `0`; `/usr/bin/afplay` exited `0`;
- the same listener identity and healthy response were present before and
  after UAT, and application shutdown took no action on the external process.

After the final rebase onto `origin/dev` `5bff29934`, all 27 existing PR
commits were range-diff `=` patch-identical. A second live run was unavailable:
`/opt/homebrew/bin/audiocpp_server` from `audio-cpp 0.4` remained installed,
but no process or listener existed and the health request failed. Chatbook did
not launch it. The pre-rebase run is therefore the live UAT evidence; patch
identity and automated results are not represented as a second live run.

### Fresh post-rebase verification

- Final post-review focused Slice 1 suite: 325 passed, 1 warning in 78.89
  seconds.
- Final post-review broad TTS/STTS suite: 1,016 passed, 14 skipped, 1 warning
  in 370.04 seconds.
- Static gates: primary Ruff passed; config Ruff passed with only the two known
  `F841` findings ignored; task-scoped Ruff format passed across 73 files;
  compileall passed; focused mypy passed across seven files; and
  `git diff --check` passed.
- Baseline audits: full mypy retained exactly the same 12 errors in the same
  three files and symbols, and the `config.py` format diff retained the exact
  pre-implementation hunks.

### Repository-wide DoD limitation

The pre-rebase repository-wide run recorded 42 failed, 16,355 passed, 187
skipped, and 2 errors. Its external rerun reduced to 37 failures; an untouched
latest `origin/dev` control produced the identical exact 37 failures. The
feature-only regression delta is zero, but the project-wide suite is not green.
TASK-710 therefore remains **In Progress** and is not marked Done.

### Latest-dev closeout audit (2026-07-31)

- Rebased the clean closeout branch onto `origin/dev`
  `70f08e5bad26571c7401435d508364607c05f967`.
- A fresh repository-wide run on the immediately preceding TTS head completed
  with 24,406 passed, 171 skipped, four failures, and two setup errors. The two
  intervening `dev` commits only changed Settings styles/tests and did not
  touch TASK-710 or any red node.
- Both setup errors were sandbox-only loopback-bind denials. Each exact Console
  provider gateway node passed alone outside the sandbox (0.80 and 0.83
  seconds).
- Closed two TTS-caused verification gaps: refreshed the reviewed production
  diagnostic inventory for seven added TASK-494 diagnostic calls and one TTS
  owner file (no persistent-sink topology change), and made Personas generation
  wiring tests post button events directly when the TTS editor panel pushes
  Advanced controls below the viewport.
- Fresh relevant verification passed: 2,100 broad TTS/STTS tests with 14 skips,
  all nine Personas generation-wiring tests, all three diagnostic-inventory
  tests, Ruff check, Ruff format, and `git diff --check`.
- The remaining observed full-suite failure is unrelated wizard test-state leakage:
  `test_full_track_skip_everything_leaves_app_usable` passes alone but fails
  after the two preceding wizard tests because the resulting `HomeScreen` lacks
  the expected shell-nav button. It is intentionally not fixed in this TTS
  closeout.
- No process is listening on `127.0.0.1:8080`, so a fresh live external
  audio.cpp rerun remains unavailable. Chatbook did not launch or supervise a
  server. TASK-710 remains **In Progress**.

## Implementation Notes

- Added immutable global TTS preferences with explicit exact/dynamic model and
  voice modes, backward-compatible blank audio.cpp reads, and one atomic
  canonical/legacy set-delete mutation. Exact values dual-write; dynamic modes
  remove stale exact aliases.
- Added writer-preferred preference/revision/lease admission and a
  service-retained, off-loop, generation-aware settings publication handoff.
  Foreground completion is bounded, admitted speech is preserved, and only the
  latest pending audio.cpp generation can replace the old adapter.
- Routed Console audio.cpp **Speak** through native
  `TTSService.synthesize_default()` and the validated complete-WAV response
  lifecycle. The six existing providers remain contained by
  `LegacyTTSAdapter`.
- Extended deterministic preference, race, reconfiguration, Console playback,
  cleanup, legacy-regression, privacy, invalid-initial-provider recovery, and
  external-process non-ownership coverage. Updated ADR-023, the developer and
  user guides, the approved design, and the Slice 1 implementation plan.
- No storage migration, dependency, managed-process behavior, or character
  profile behavior was added. ADR-023 is the governing amended decision; a new
  ADR was not created.
- Added-line process-keyword review found only restart-recommendation copy and
  an in-process `asyncio.Event` close signal; it found no process launch or
  control API. The only changed profile-named file is the approved design
  document, not character-profile production code.
- Final spec review found and verified one cross-layer provider-switch race:
  Console had compared a pre-admission preference snapshot with the coherently
  admitted response. Console now treats the successful admitted response as
  authoritative for metrics, while `TTSService` rejects an adapter response
  whose provider does not match the canonical admitted lease before consuming
  stream bytes. Deterministic red/green tests cover both the valid switch and
  invalid private-provider cases, including response and lease cleanup.
- Final quality review found and verified one config-boundary privacy issue:
  a noncanonical initial provider could be sampled for a failure metric before
  admission. `TTSService` now quarantines that selection as recoverably
  unconfigured, returns fixed safe unavailable copy without a provider metric,
  and accepts a later canonical settings publication without restart.
- PR review hardening added Google-style boundary documentation, contextual
  but privacy-safe config mutation logging, 64 KiB ordered artifact writes,
  bounded cancellation and secure-delete joins, retained late-I/O draining,
  eventual cleanup after late success or failure, and matching-only cache
  release so a delayed delete cannot erase a replacement artifact. Red/green
  lifecycle regressions cover the cancellation, timeout, shutdown, batching,
  privacy, and replacement races. A final independent review reported no
  Critical, Important, or Minor findings.
- The implementation satisfies the task acceptance criteria, but project DoD
  remains blocked by the non-green repository-wide baseline and the current
  absence of a user-started server for a second live run.

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Automated unit, integration, Textual, race, and cleanup tests cover every acceptance criterion and pass.
- [x] #2 Ruff checks and formatting, compileall, focused typing checks where configured, and git diff --check pass.
- [x] #3 ADR-023, user documentation, compatibility limitations, external-process ownership, and UAT evidence are current.
- [x] #4 Self-review confirms the implementation stays within Slice 1 and adds no managed process or character-profile behavior.
- [ ] #5 All acceptance criteria and DoD items are checked, concise implementation notes are added, and status changes to Done only after all evidence exists.
<!-- DOD:END -->
