---
id: TASK-18605
title: >-
  Fix 17 red tests on dev: continuation restore-target mismatch, eager TTS
  import, stale test guards
status: To Do
assignee: []
created_date: '2026-08-18 22:00'
labels:
  - bug
  - console
  - performance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A full `Tests/Chat/` run on dev showed 17 failures, verified pre-existing by
running them against a clean `origin/dev` worktree. They are four unrelated
causes, two of which are production defects rather than test rot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Resuming an interrupted provider continuation no longer fails with a spurious "Pinned provider settings no longer match".
- [x] #2 Importing the Console screen does not pull in the TTS/Audio stack.
- [x] #3 A pending skill-script confirm is released by real process teardown.
- [x] #4 The H3 image-edit guard actually arms instead of erroring on a moved symbol.
- [x] #5 Each fix is verified against the behaviour it protects, not by deleting the assertion.
- [ ] #6 The visual-evaluation renderer-identity guard passes in an environment matching the project's pinned Pillow.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**1. Provider continuation resume was broken in production (11 of the 17).**
`_continuation_restore_target_for_resolution` (`console_chat_controller.py`) built
its target by passing `resolution.base_url` through
`normalize_generic_endpoint_for_compare`, which EXPANDS a base URL to its full
endpoint (`https://api.moonshot.ai/v1` -> `.../v1/chat/completions`). But the
checkpoint pins whatever `ConsoleAgentBridge` recorded, and the bridge records
`resolution.base_url` RAW. `validate_continuation_restore` compares the two
byte-exactly, so recovery compared an expanded URL against a non-expanded one and
every Resume was rejected with "Pinned provider settings no longer match".

Fixed by carrying `api_base_url` through verbatim. Normalizing BOTH sides instead
would have been the wrong repair: the exactness is deliberate --
`test_provider_continuation` pins that `https://api.deepseek.com/v1/` and
`https://api.deepseek.com/v1` are a MISMATCH -- and that comparison is what stops
a private continuation from being replayed against a different endpoint than the
one that produced it. The two writers simply have to agree, and the checkpoint's
writer defines the format.

Introduced by `121146dd6` (the same commit as the two failing
`test_console_chat_controller` cases), which is why all 11 moved together.

**2. The Console screen eagerly imported the TTS/Audio stack (1).**
`Widgets/Console/console_auto_speak_consent.py` imported `ConsoleTTSDestination`
at module scope; `Event_Handlers/TTS_Events/tts_events.py` imports
`Audio/streaming_sink.py` at ITS module scope. Every reference in the consent
module is a type annotation (the file has `from __future__ import annotations`)
except ONE runtime `type(...) is not` check, so the import moved into
`TYPE_CHECKING` plus one function-local import.

Measured: `import tldw_chatbook.UI.Screens.chat_screen` 1301 ms -> 1108 ms
(**193 ms, ~15%**), averaged over 3 cold runs each.

**3. Skill-script confirms were not released by teardown (3).**
task-15860 split teardown into a per-VISIT Event and a headless one; a round armed
with no Console visit open binds the HEADLESS Event, which only
`_cancel_headless_rounds()` sets. Three tests (and a fixture teardown, which is why
the file HUNG rather than merely failing) poked `controller._shutdown_requested`
directly, bypassing that. Switched to `begin_shutdown()`, the real teardown API,
which sets both signals. Production was already correct.

**4. A moved symbol silently disarmed a guard (1).**
`0b8e9e408` extracted the Console video controller out of `chat_screen` into
`UI/Console_Modules/video.py`. `test_console_h3_image_edit` kept
`monkeypatch.setattr`-ing the now-absent `chat_screen.run_video_generation`, so it
raised AttributeError instead of arming its "must not call Video generation"
assertion. Retargeted to the module that now binds it.

**5. Visual-evaluation evidence vs installed Pillow (1) -- NOT fixed, deliberately.**
`renderer_version` embeds the Pillow version, and the checked-in support matrix was
captured under 11.2.1 while this venv has 12.1.1. The page hashes genuinely differ,
so Pillow 12 really does render the visual transcript differently -- the guard is
working. `pyproject.toml` pins `pillow==11.2.1` ("ADR-054 renderer identity;
matches reviewed context-use evidence"), so the VENV has drifted from the declared
contract, not the code.

Not "fixed" in code on purpose: re-stamping the version string would assert that a
model evaluation describes images it never saw. A `uv pip install pillow==11.2.1`
dry-run resolves cleanly (only pillow changes), but that mutates a venv shared with
other concurrent sessions, so it is left as an owner decision.
<!-- SECTION:NOTES:END -->
