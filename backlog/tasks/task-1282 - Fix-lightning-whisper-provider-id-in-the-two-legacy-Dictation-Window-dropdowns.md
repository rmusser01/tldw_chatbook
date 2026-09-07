---
id: TASK-1282
title: >-
  Fix "lightning-whisper" provider id in the two legacy Dictation Window
  dropdowns
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 15:00'
updated_date: '2026-07-29 15:05'
labels:
  - dictation
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`tldw_chatbook/UI/Dictation_Window_Improved.py` (the two provider `Select` option lists, currently at lines 351 and 359) and `tldw_chatbook/UI/Dictation_Window.py` (currently line 227) both offer `("Lightning Whisper", "lightning-whisper")` as a dropdown option. Every place that actually dispatches on a provider id -- `transcription_service.py`, and `console_voice_input.py`'s `LOCAL_PROVIDER_MODULES` -- uses `"lightning-whisper-mlx"` instead. The Console dictation service allowlist had this exact id bug and was already fixed there; these two legacy windows still carry it.

`Dictation_Window_Improved.py`'s `ImprovedDictationWindow` is live: `STTS_Window.py` imports and mounts it as the Dictation tab, so selecting "Lightning Whisper" there currently sends an id nothing recognizes. `Dictation_Window.py` (the non-"Improved" version) has no production importer today, but carries the identical bug and is worth fixing at the same time so it isn't resurrected wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `Dictation_Window_Improved.py`'s two provider dropdown option lists use the id `"lightning-whisper-mlx"`, matching what `transcription_service.py` and `console_voice_input.py`'s `LOCAL_PROVIDER_MODULES` actually dispatch on.
- [x] #2 `Dictation_Window.py`'s provider dropdown option list uses the same corrected id.
- [x] #3 Selecting "Lightning Whisper" in the STT/TTS settings Dictation tab (`STTS_Window.py`) results in the `lightning-whisper-mlx` provider actually being invoked, not a silently ignored/unmatched value.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fix dropdown ids in Dictation_Window_Improved (x2) and Dictation_Window\n2. Check STTS_Window for the same id\n3. Test pinning dropdown values against transcription_service dispatch ids
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed: replaced the misspelled "lightning-whisper" id with the real dispatch
id "lightning-whisper-mlx" in all three known locations -- both branches of
Dictation_Window_Improved.py's _get_provider_options() (lines 351/359,
fixed with one replace_all since both were the identical typo) and
Dictation_Window.py's provider-select options (line 227).

Checked UI/STTS_Window.py per the task's note that a prior review believed
it carries the same id: grep found zero occurrences of "lightning" anywhere
in that file. It mounts ImprovedDictationWindow (the fixed class) as the
Dictation tab and does not keep its own copy of the option list, so there
was nothing to fix there; the prior review's belief does not hold.

Added Tests/Local_Ingestion/test_dictation_window_provider_ids.py
(pytestmark = pytest.mark.unit, outside Tests/UI/ so it runs under CI's
`pytest -m unit`). It pins every id offered by both windows' provider
dropdowns against the real dispatch ids transcription_service.py's
provider-branch chain / Utils/local_stt_providers.py's
LOCAL_PROVIDER_MODULES actually recognizes, plus a documented exception set
("auto" -- a resolved-elsewhere sentinel; "openai-whisper"/"google-speech" --
two options nowhere in transcription_service.py at all, a separate
pre-existing gap unrelated to this typo, deliberately left alone rather than
silently expanding this task's scope). Dictation_Window.py's list is parsed
from its own AST (no Select-literal helper method exists there);
Dictation_Window_Improved.py's is read by calling the real
_get_provider_options() on a bare __new__ instance with both privacy
settings. A third test pins the specific "lightning-whisper" (bare) string
by name as belt-and-suspenders.

Mutation-checked: reverting all three ids back to "lightning-whisper" fails
all 3 new tests; restored byte-identical afterward.

Verification: the new test file (3/3) + the task's suggested Chat/UI/Audio
dictation suites, all green (154 passed). ruff check clean on all three
changed/added files.
<!-- SECTION:NOTES:END -->
