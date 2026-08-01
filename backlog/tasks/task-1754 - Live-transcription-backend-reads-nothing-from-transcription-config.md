---
id: TASK-1754
title: 'The live transcription backend reads nothing from [transcription] (dotted get_cli_setting returns None)'
status: To Do
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
- [ ] #1 The transcription backend's configured provider, model, language and device reach it from `[transcription]`.
- [ ] #2 `get_cli_setting` either returns the caller's default for an unresolvable dotted path or the dotted form is removed from use; whichever is chosen is covered by a test.
- [ ] #3 A test fails against the current code and passes after the fix, asserting a configured non-default provider actually reaches the backend.
- [ ] #4 Media ingest transcription honours a user-configured provider end to end on at least one platform.
<!-- AC:END -->
