---
id: TASK-1754
title: 'The live transcription backend reads nothing from [transcription] (dotted get_cli_setting returns None)'
status: Done
assignee: []
created_date: '2026-08-01 12:00'
labels: [transcription, config, bug]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Local_Ingestion/transcription_service.py`'s `_LegacyTranscriptionBackend.__init__` (~L320-338)
reads its configuration with the dotted single-argument form, e.g.
`get_cli_setting("transcription.default_provider", "faster-whisper")`. That form returns `None`
regardless of config contents AND regardless of the supplied default -- measured directly:

    get_cli_setting("transcription.default_provider", "FALLBACK") -> None
    get_cli_setting("transcription", "default_provider", "FALLBACK") -> "parakeet-mlx"

So the backend silently ignores the entire `[transcription]` section: provider, model, language,
source/target language and device all resolve to `None` (or the inline `or "cpu"` fallback where
one exists). Users' transcription settings therefore never reach media ingest; whatever the service
picks internally wins. Found while implementing TASK-867 (macOS engine preference), where the
Console dictation resolver was correct because it uses the two-argument form.

Note the fallback-ignoring behaviour means this is two defects: the call sites are wrong, and
`get_cli_setting`'s dotted path handling drops the caller's default instead of returning it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The transcription backend's configured provider, model, language and device reach it from `[transcription]`.
- [x] #2 `get_cli_setting` either returns the caller's default for an unresolvable dotted path or the dotted form is removed from use; whichever is chosen is covered by a test.
- [x] #3 A test fails against the current code and passes after the fix, asserting a configured non-default provider actually reaches the backend.
- [x] #4 Media ingest transcription honours a user-configured provider end to end on at least one platform.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause was one accessor, not the call sites: `get_cli_setting`'s recovery heuristic for a dropped
default (`not isinstance(key, str) and default is None`) only fired for non-string defaults, so every
string default -- provider, model, language, device names -- silently became `None`. Fixed with an
explicit sentinel distinguishing "no third argument" from "None passed explicitly"; the traditional
3-arg form is byte-identically unchanged, and the dotted 2-arg form now both resolves real values and
honours defaults of any type (verified live: dotted read returns the configured `parakeet-mlx`, and an
absent key returns the caller's fallback).

Deliberate deviation, recorded: the transcription call sites were NOT rewritten to the 3-arg form.
Once the accessor is correct they are behaviourally identical (checked field by field), and rewriting
them would break ~10 test files whose mocks key on the literal dotted string. The audit also caught a
regression the accessor fix would otherwise have introduced at `library_screen.py:12279`, where an
ambiguous 2-arg dotted call would have started returning a whole table; fixed with an explicit default.

Severity note for the record: pre-fix, the constructor's `or fallback` still produced a dispatchable
provider, so a misread config could make `transcribe()` silently load the WRONG provider's model
rather than fail fast. Commits 3355acb2b (accessor) and b6c0fc8f9 (audit + call-site fix + end-to-end
proof).
<!-- SECTION:NOTES:END -->
