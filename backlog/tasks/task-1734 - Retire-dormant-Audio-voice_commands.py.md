---
id: TASK-1734
title: Retire dormant Audio/voice_commands.py
status: Done
assignee: []
created_date: '2026-07-29 16:13'
labels:
  - console
  - cleanup
  - dead-code
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`tldw_chatbook/Audio/voice_commands.py` defines `VoiceCommandProcessor`, a spoken-command
grammar built around `APP_NAVIGATION`-style actions ("switch to chat", "switch to notes",
"pause dictation", punctuation words, a custom-command builder) that predate the
screen-based navigation model. `Chat/console_voice_input.py`'s `ConsoleVoiceInputController`
constructs its service with `enable_commands=False`, so this module's own detection
machinery has never been reachable through the shipping Console dictation path (V1, PR
#1085) — and V2 (this branch) ships its own, unrelated command grammar directly in
`console_voice_input.py` (prefix + whole-segment match against a 7-entry table: `new
paragraph`, `new line`, `stop`, `send`, `discard`, `read that back`, `new session`), so the
two systems will never converge.

The V2 design spec (`Docs/superpowers/specs/2026-07-29-console-voice-control-design.md`,
"Out of scope" / "Follow-ups to file during V2") explicitly calls this out: keep
`voice_commands.py` dormant through V2 and file its retirement as a separate follow-up
rather than doing it as part of V2 work.

Verified with a repo-wide grep (excluding `__pycache__`) that the module has no reachable
production caller:

```
$ grep -rn "voice_commands\|VoiceCommandProcessor" --include="*.py" . | grep -v "__pycache__" | grep -v "/Tests/"
tldw_chatbook/Audio/voice_commands.py:1:# voice_commands.py
tldw_chatbook/Audio/voice_commands.py:41:class VoiceCommandProcessor:
tldw_chatbook/Audio/voice_commands.py:375:def get_command_processor() -> VoiceCommandProcessor:
tldw_chatbook/Audio/voice_commands.py:379:        _command_processor = VoiceCommandProcessor()
tldw_chatbook/Widgets/voice_command_dialog.py:13:from ..Audio.voice_commands import VoiceCommand, CommandType, get_command_processor
```

(The other hits under `Voice_Assistant_Interop/` and `tldw_api/` are an unrelated REST
"voice commands" concept on a different client and do not import this module.)

The one production import is `tldw_chatbook/Widgets/voice_command_dialog.py`
(`VoiceCommandDialog`, a `ModalScreen`) — but that dialog itself is never pushed or
referenced anywhere else in the codebase:

```
$ grep -rn "voice_command_dialog\|VoiceCommandDialog" --include="*.py" . | grep -v "__pycache__"
tldw_chatbook/Widgets/voice_command_dialog.py:1:# voice_command_dialog.py
tldw_chatbook/Widgets/voice_command_dialog.py:16:class VoiceCommandDialog(ModalScreen[bool]):
tldw_chatbook/Widgets/voice_command_dialog.py:28:    VoiceCommandDialog {
```

So the chain `voice_commands.py` → `voice_command_dialog.py` → (nothing) is entirely dead:
no window, screen, or button constructs `VoiceCommandDialog`, and no test file exists for
either module (`grep -rl` over `Tests/` for both module names returns only the unrelated
`Voice_Assistant` interop tests). 380 + 439 = 819 lines of untested, unreachable code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `tldw_chatbook/Audio/voice_commands.py` and `tldw_chatbook/Widgets/voice_command_dialog.py` are either deleted, or explicitly rebound to a real production caller if a maintainer decides app-wide voice navigation (V3+) should reuse this grammar instead of retiring it
- [x] #2 A repo-wide grep for `voice_commands`, `VoiceCommandProcessor`, `voice_command_dialog`, and `VoiceCommandDialog` (excluding `__pycache__` and the unrelated `Voice_Assistant_Interop`/`tldw_api` REST usage) turns up no references once the decision above is carried out
- [x] #3 The full test suite (excluding the two known-hanging real-hardware audio test files) passes with the change in place
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deleted `Audio/voice_commands.py` (380 lines) and its sole importer
`Widgets/voice_command_dialog.py` (439 lines) -- 819 lines total, no production caller and no
tests. Re-verified the dead-code claim against current dev before deleting: the dialog is
imported by nothing, and the module's only other grep hits belong to an unrelated feature
(`Voice_Assistant_Interop`/`tldw_api` server-side voice-assistant commands), which was left
untouched. Console dictation's own grammar lives in `Chat/console_voice_input.py` (V2, PR #1171).
Verified app/Audio/Widgets import cleanly post-deletion and the voice suites stay green (138).
Two historical TTS design docs still name the dialog; left as-is, they are dated records rather
than live documentation.
<!-- SECTION:NOTES:END -->
