---
id: TASK-3023
title: Repoint tests that patch controllers through the chat_screen alias
status: Done
assignee: []
created_date: '2026-08-07 14:58'
updated_date: '2026-08-07 19:39'
labels:
  - tech-debt
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 4 moved every Console controller construction into UI/Console_Modules/wiring.py, leaving five controller imports in chat_screen.py that no code references. They cannot be deleted: 18 test sites across 5 files patch them through the screen module's namespace (chat_screen_module.ConsoleDictationController and friends), which no import-grep can see because the alias hides it. Deleting them turns 28 tests red -- tripped once during the extraction. The imports now carry noqa markers and a block comment so a linter does not harvest them, but the real fix is repointing the patch sites at the modules that own the classes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every test that patches a Console controller patches it on the module that defines it, not through chat_screen
- [x] #2 The five re-export imports and their noqa markers are removed from chat_screen.py
- [x] #3 pyflakes on chat_screen.py returns to its pre-wave-4 count
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repointed all 32 alias-reached sites and deleted the imports they were keeping
alive. `chat_screen.py` now imports no Console controller class at all --
`Console_Modules/wiring.py` is the sole constructor.

**Why repointing was safe here, which was the only real risk.** Patching the
defining module instead of the alias only steers production if production looks
the name up at call time; a symbol captured at import time would keep the old
binding and the test would silently stop testing anything. That failure mode
does not apply, and not by luck: *none* of the 32 sites rebinds a name in
`chat_screen`'s namespace. They are attribute READS off the alias
(`chat_screen_module.ConsoleStreamingDictationSession(...)`, constant
comparisons, `_join_segments(...)`) plus `setattr` on the **class object** --
`monkeypatch.setattr(ConsoleDictationController, "_create_console_dictation_
session", fake)`. All 8 symbols were verified `is`-identical through both
paths before the move, so there is exactly one object to mutate, and
`dictation.py`'s `self._create_console_dictation_session()` resolves it by
normal MRO lookup either way. An AST scan over `Tests/` + `tldw_chatbook/`
confirmed no `setattr(chat_screen_module, "<controller>", ...)` form and no
`from ...chat_screen import <controller>` anywhere -- both invisible to the
attribute-based detector and both would have broken.

Live confirmation, not just reasoning: when the (pre-existing, load-sensitive)
streaming test fails, it fails *after* `assert "Transcribing" in _painted(chip)`
has already passed -- an assertion only the patched fake service can satisfy.
The repointed patch demonstrably still drives the code under test.

**Scope.** 12 imports removed: the 8 alias-reached ones plus the 4 wave-4
controllers (`ConsoleAgentController`, `ConsoleHandsFreeController`,
`ConsolePromptsController`, `ConsoleSessionController`) that nothing reached at
all. pyflakes on `chat_screen.py` 37 -> 25, past the pre-wave-4 baseline of 31
(AC #3). Two test files lost their now-unused `chat_screen_module` import;
`test_console_staged_evidence_strip.py` keeps its own, since it still patches
`capture_console_staged_evidence_for_chat` on that namespace legitimately.

`Tests/Architecture/test_module_alias_reexports.py` was deleted. It asserts it
must not pass on an empty at-risk set and its docstring names repointing as the
condition for its own removal; the set is now empty and the block comment it
policed is gone. The guidance it carried was rewritten into `wiring.py`'s module
docstring, which previously told the reader the opposite (that re-deleting the
imports was a regression).

Ratchet lowered 17,749 -> 17,727 lines, measured with `ast`; methods unchanged
at 593 since only imports went.

**Evidence.** 192 passed before / 190 after across the 5 repointed files plus
the architecture and controller-wiring suites -- exactly the 2 tests in the
deleted guard file. Full `Tests/` collect-only: 32,019 collected, 0 errors.
`Docs/security/production-diagnostic-inventory.json` needed regenerating: the
digest for `chat_screen.py` is content-sensitive and 22 lines moved, but
`call_count` is unchanged at 142, so no diagnostic was added, removed or
re-owned.

`test_the_transcribing_indication_reverts_on_a_mid_capture_stop` failed during
verification. Both arms were measured under matched load before concluding:
19/36 failures with the change, 17/36 with the old indirection restored --
indistinguishable, and mechanically impossible to have caused since both arms
`setattr` the same attribute on the same object with the same factory. It is a
4.0s wall-clock deadline in `_wait_for_mic_label`, filed as task-3400.

**Modified**: `tldw_chatbook/UI/Screens/chat_screen.py`,
`tldw_chatbook/UI/Console_Modules/wiring.py` (docstring),
`Tests/UI/test_console_dictation{,_firstrun,_streaming}.py`,
`Tests/UI/test_console_hands_free_wiring.py`,
`Tests/UI/test_console_staged_evidence_strip.py`,
`Tests/Architecture/test_screen_size_ratchet.py`,
`Docs/security/production-diagnostic-inventory.json`.
**Deleted**: `Tests/Architecture/test_module_alias_reexports.py`.
<!-- SECTION:NOTES:END -->
