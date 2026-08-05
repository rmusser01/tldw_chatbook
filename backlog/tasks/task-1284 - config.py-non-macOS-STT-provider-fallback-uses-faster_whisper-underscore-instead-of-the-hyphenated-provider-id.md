---
id: TASK-1284
title: >-
  config.py non-macOS STT provider fallback uses "faster_whisper" (underscore)
  instead of the hyphenated provider id
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 15:00'
updated_date: '2026-07-29 14:58'
labels:
  - config
  - dictation
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`config.py` (currently line 920) initializes `default_stt_provider = "faster_whisper"` (underscore) before the macOS-only branches below it override it with correctly hyphenated ids (`"parakeet-mlx"`, `"lightning-whisper-mlx"`). Every provider id actually used for dispatch elsewhere in the codebase -- `console_voice_input.py`'s `LOCAL_PROVIDER_MODULES` (`"faster-whisper"`), and `transcription_service.py`'s provider-branch matching -- is hyphenated. On a non-macOS platform (where neither `if sys.platform == "darwin"` branch runs), `default_stt_provider` keeps the underscored value and gets written into `STT_settings.default_stt_provider`, which downstream code (`console_voice_input.resolve()`'s `STT_settings` fallback path, `transcription_service` dispatch) does not recognize as any installed provider id -- it fails the "is this the configured provider actually installed" check silently rather than matching.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `config.py`'s non-macOS `default_stt_provider` fallback is the hyphenated id `"faster-whisper"`, matching every other provider id in `LOCAL_PROVIDER_MODULES` and `transcription_service.py`.
- [x] #2 On a non-macOS platform with no `[transcription].default_provider` configured, `STT_settings.default_stt_provider` resolves to a value that matches an installed/dispatchable provider id rather than silently falling through to the "not installed" branch.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fix config.py underscore id to faster-whisper\n2. Grep for tests asserting the underscored value as expected\n3. Hermetic test pinning the hyphenated fallback
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed: config.py's non-macOS default_stt_provider fallback was the
underscored "faster_whisper"; every real dispatch id elsewhere
(console_voice_input.py's LOCAL_PROVIDER_MODULES, transcription_service.py)
is hyphenated. Changed the initial assignment to "faster-whisper"; the two
macOS-only branches beneath it already used correct hyphenated ids and are
untouched.

Grepped for tests asserting the old underscored value as expected FIRST, per
the task's warning: found one --
Tests/test_config_stt_provider_probe.py::test_macos_stt_default_probes_packages_without_importing_them
parametrized `((), "faster_whisper")` (macOS, neither optional package
installed, falls through to the pre-branch initial value). Fixed that
expectation to "faster-whisper" in the same commit.

Added a new hermetic test,
test_non_macos_stt_default_fallback_is_hyphenated, pinning the hyphenated
fallback on a non-darwin platform. It does NOT spawn a subprocess with
sys.platform pre-set before importing config (tried that first): a fresh
interpreter with sys.platform forced to "linux" before `import config`
crashes hard during that same import, because loguru's ExceptionFormatter
init and psutil's platform dispatch both probe sysconfig / pick a
platform-specific C extension using the real build's platform tag as soon
as they're imported -- unrelated to this fix, just fatal collateral damage
from spoofing sys.platform process-wide before those imports run. Instead
the test imports config normally (real platform), then
monkeypatch.setattr(config_module.sys, "platform", "linux") only around a
single config_module.load_settings(force_reload=True) call (config.py's
only sys.platform read), with TLDW_CONFIG_PATH pointed at a tmp_path config
and the module-level settings cache cleared first (same pattern as
test_config_private_bootstrap.py's _clear_config_cache()).

Mutation-checked: reverting the config.py fix makes both
test_non_macos_stt_default_fallback_is_hyphenated and the
[installed2-faster-whisper] macOS parametrization fail; restored
byte-identical afterward.

Verification: Tests/test_config_stt_provider_probe.py (4/4) +
Tests/test_config_console_defaults.py + the Chat/UI/Audio dictation suites
from the task's suggested set, all green (171 passed). ruff check clean on
both changed files.
<!-- SECTION:NOTES:END -->
