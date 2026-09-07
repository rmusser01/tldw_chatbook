# task-3023 — Repoint tests that patch controllers through the `chat_screen` alias

**Status: complete.** All three ACs hold. The guard test was deleted.

## The symbol table

The at-risk set was computed, not grepped: `_imported_but_unreferenced` ∩ `_alias_reached`
from `Tests/Architecture/test_module_alias_reexports.py`, run against the live tree.
It returned 8 symbols / 32 real test sites (the detector's count of 33 for
`ConsoleDictationController` + `ConsoleStreamingDictationSession` included 2 hits inside
the guard test's own docstring).

| Symbol | Sites | How production reaches it | Outcome |
|---|---|---|---|
| `ConsoleDictationController` | 17 | `wiring.build_console_controllers` constructs it from `Console_Modules.dictation`; tests `setattr` the method `_create_console_dictation_session` **on the class object**, and `dictation.py:1560` calls `self._create_console_dictation_session()` — a call-time MRO lookup | **Repointed** → `dictation_module` |
| `ConsoleStreamingDictationSession` | 6 | Constructed inside `dictation.py`'s own `_create_console_dictation_session`; tests construct it directly as a test double | **Repointed** → `dictation_module` |
| `_join_segments` | 4 | Called by `dictation.py:594`; tests call it directly as a pure function | **Repointed** → `dictation_module` |
| `CONSOLE_DICTATION_MAX_BYTES` | 1 | Read-only constant, defaults a `dictation.py` kwarg; test compares against it | **Repointed** → `dictation_module` |
| `CONSOLE_DICTATION_MAX_SECONDS` | 1 | as above | **Repointed** → `dictation_module` |
| `CONSOLE_DICTATION_SAMPLE_RATE` | 1 | as above | **Repointed** → `dictation_module` |
| `CONSOLE_DICTATION_SAMPLE_WIDTH` | 1 | as above | **Repointed** → `dictation_module` |
| `ConsoleWorkspaceController` | 1 | `wiring.py:98` constructs it; test calls `_current_console_workspace_context` on the class directly | **Repointed** → `workspace_module` |

**Nothing was left behind.** The recomputed at-risk set is now `{}` and `chat_screen.py`
contains zero `noqa: F401`.

## Why repointing is genuinely equivalent here

This was the one real risk in the task — a retargeted patch that no longer steers, still
green. It does not apply, and not by luck:

1. **No site rebinds a module attribute.** Every one of the 32 is an attribute *read* off
   the alias, or a `setattr` on the **class object** (`setattr(ConsoleDictationController,
   "_create_console_dictation_session", fake)` — not `setattr(chat_screen_module, "…")`).
   There is exactly one class object, so there is only one thing to mutate.
2. **Verified `is`-identical.** All 8 symbols were checked `chat_screen.X is
   owner_module.X` → `True` *before* the imports were removed. Also verified
   `wiring.ConsoleDictationController is dictation.ConsoleDictationController` → `True`,
   i.e. the class the tests patch is the class production instantiates, and an instance
   sees a patch made on the defining module's class.
3. **Both invisible escape hatches were swept.** An AST scan over `Tests/` +
   `tldw_chatbook/` found no `setattr(chat_screen_module, "<symbol>", ...)` form (which
   *would* have broken, since it rebinds the namespace rather than the object) and no
   `from ...chat_screen import <symbol>` anywhere. The `_alias_reached` detector sees
   neither, so this had to be checked separately.
4. **Confirmed live, not just by reasoning.** When the flaky streaming test fails, it
   fails *after* `assert "Transcribing" in _painted(chip)` has already passed — an
   assertion only the patched fake service can satisfy. The repointed patch demonstrably
   still drives the code under test.

Assertions were not touched; only patch targets moved.

## Guard test: deleted

`Tests/Architecture/test_module_alias_reexports.py` asserts it must **not** pass on an
empty at-risk set, and its docstring names repointing as the condition for its own
removal. The set is empty and the block comment it policed is gone, so it went.

Its guidance was rewritten into `Console_Modules/wiring.py`'s module docstring, which had
been saying the *opposite* — that re-deleting the imports was a regression. Left as-is it
would have invited someone to add them back.

## Numbers

| Measure | Before | After |
|---|---|---|
| pyflakes on `chat_screen.py` | 37 | **25** |
| pre-wave-4 baseline (AC #3 target) | 31 | beaten by 6 |
| ratchet lines | 17,749 | **17,727** (measured via `ast`) |
| ratchet methods | 593 | **593** (unchanged — only imports went) |

12 imports removed: the 8 alias-reached ones plus the 4 wave-4 controllers nothing reached
(`ConsoleAgentController`, `ConsoleHandsFreeController`, `ConsolePromptsController`,
`ConsoleSessionController`). `git diff` of wave 4 confirmed those 6 controllers are exactly
what took pyflakes 31 → 37.

Two test files lost their now-unused `chat_screen_module` import.
`test_console_staged_evidence_strip.py` keeps its own — it still legitimately patches
`capture_console_staged_evidence_for_chat` on that namespace.

## Test evidence

- **Before:** 192 passed (5 repointed files + ratchet + guard + controller-wiring).
- **After:** 190 passed — exactly the 2 tests in the deleted guard file.
- **Full `Tests/` collect-only:** 32,019 collected, **0 errors** (nothing else imported the
  deleted test or the removed symbols).
- **Import smoke:** `chat_screen` imports; all 8 symbols confirmed gone from its namespace.

### The one failure, and why it is not causal

`test_console_dictation_streaming.py::test_the_transcribing_indication_reverts_on_a_mid_capture_stop`.

It first looked causal — 43% failure with the change vs 14% without. Both arms were then
measured **under matched load**, interleaved:

| | fail / runs |
|---|---|
| Arm A (repointed) | 19 / 36 (53%) |
| Arm B (old `chat_screen` indirection restored) | 17 / 36 (47%) |

Indistinguishable. The earlier 14% for arm B was measured on a quieter machine. It is also
mechanically impossible for the change to have caused it: both arms `setattr` the same
attribute on the same object with the same factory, so their runtime effect is identical
byte-for-byte. The real cause is a 4.0s wall-clock deadline in `_wait_for_mic_label` on the
third mic click. Filed as **task-3400**. Arm B was applied and reverted by string
replacement; `git diff` confirmed a clean tree afterwards.

`Docs/security/production-diagnostic-inventory.json` needed regenerating — causal but
benign: the digest for `chat_screen.py` is content-sensitive and 22 lines moved, but
`call_count` is unchanged at **142**, so no diagnostic was added, removed, or re-owned.

Pre-existing and ignored per brief: `test_console_live_work_handoffs.py::test_watchlists_destination_retries_console_follow_after_initial_adapter_failure`.

## Concerns

- **task-3400** (the ~50% flake) will bite CI. It passes reliably when the whole file runs,
  which is why it has gone unnoticed; it is a real test defect, not a product one.
- The ratchet is now exact (17,727 / 17,727). Any rebase that lands another commit touching
  `chat_screen.py` will need it re-measured before merge — the file's own comment warns that
  a budget from a stale base fails on contact with dev.
