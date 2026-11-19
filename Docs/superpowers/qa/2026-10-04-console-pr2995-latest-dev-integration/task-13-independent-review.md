### Spec Compliance

- ✅ Spec compliant for Task13 at `9445ed93ed430ba0e3660ad2008621f6a809bd0e`, against `862c2eaa80bb744b71cb53a68952f8451b3b327c`. No missing integration requirement or unauthorized source change found. This is the scoped newest-dev integration gate; the previously closed whole-branch and Task9/12 gates were not reopened.
- ✅ The 17 incoming paths and six incoming commits are represented in `task-13-safe-evidence/source-union.json`; the final overlay consists solely of controller spacing/call wrapping and the downward controller cap (`authorized-overlay.patch:1`). The cap changes 29367→29301 at `Tests/Architecture/test_module_size_ratchet.py:100`. Other caps and the 50-line slack rule are preserved.
- ✅ Independent reverse application of the supplied diff restores the recorded reviewed body hashes for all four shared methods, while current body hashes match their recorded final values: `ConsoleChatController._submit_draft_body` (`console_chat_controller.py:9728`), `ConsolePromptQueueUIController.presentation_for` (`prompt_queue.py:744`), `build_console_controllers` (`wiring.py:762`), and `ChatScreen._build_console_workbench_state` (`chat_screen.py:14793`). The only shared changes are the paused refusal copy, live recovery reason, named callback, and blocked projection. Prepared Send, accepted Running/Stop, image-edit gates, literal requests and acceptance fences survive.
- ✅ The actual selection is 57 passing cases: 53 behavior cases and four exact ratchets. XML independently contains 7 controller triggers, 13 row-source cases, 12 mounted recovery cases, 13 shelf cases, 1 voice case, 3 runtime-ownership cases, 4 chat-start cases, and 4 ratchets. The additional two cases are the existing `/help` and `@file` parameterizations beside `hello`, not changed selectors. `selected.log:41`, `selected.xml`, `selected-argv.json`, and the preflight selectors agree.
- ⚠️ No fresh loading pass is established or claimed. The 35 passes remain evidence for `a6887fdb8931adb9d1b99a376d63d9b9b66a5bb5`; their historical warnings and zero-headroom measurements remain attached to that source. This is the brief's required carry policy, not a missing Task13 test.

### Strengths

- `console_trace_row_sources.py:58` follows the aggregate's leading-system/memory/history/tool categories; `:92` declines saved revisions when provider rows omit saved images. Real request-aggregate negative controls and real SQLite/controller/mounted tests cover these changes (`Tests/Chat/test_console_trace_row_sources.py:84`, `Tests/UI/test_console_blocked_send_recovery.py:372`).
- `console_runtime.py:392`, `:2487`, `:2549`, and `:2738` transport code-owned refusal reasons through a terminal exception into recovery entries. Exception arguments stay fixed; `_ConsoleTurnCustodyRecord:381` remains unchanged. `prompt_queue.py:286` and `wiring.py:2329` derive the shelf reason from the oldest live recovery without caching a new authority.
- `provider_continuation_recovery.py:113` derives blocked state from the paused preparation rather than the terminal status overwritten by a subsequent refusal. The mounted tests assert the header, chip and Inspector before and after that second send, then exercise the actual recovery buttons.
- Cross-task preservation is supported by exact named guards: `loading-authority-detail.json` pins `UI/Console_Modules/session.py`, `Tests/Chat/test_console_chat_start.py`, `Tests/conftest.py`, app/route/CSS sources, and startup budget sources. It preserves controller creation-source readers/matchers/locked record accessors plus runtime `close_session:4734`, `_close_session_after_voice_drain:4795`, and custody/constructor declarations. No new Session authority, permission, schema, or profile policy appears in the diff. The unchanged Session source and reverse-restored wiring retain named Session recovery callbacks.
- The helper imports are request-local (`console_chat_controller.py:2719`, `:11786`); diagnostics remain failure-local (`:12110`). All nine existing owners keep their eager module sets. The ten-owner worker/diagnostic carry map is scoped and unchanged; the new diagnostic call is tested for keyed, content-free output.

### Issues

#### Critical (Must Fix)

- None found.

#### Important (Should Fix)

- None found.

#### Minor (Nice to Have)

- `task-13-safe-evidence/selected.log:10` / `Tests/conftest.py:609`: the passing run retained an FD warning, start14/end383, growth369 against limit200. It is not a clean resource-usage result. The session-end attribution to the final ratchet does not identify an offending case or prove Task13 introduced a leak. Keep this concern visible; if attribution is required, scope a separate focused investigation before changing fixtures or thresholds. No suppression or speculative repair is warranted by this evidence.

### Verification

- Read the supplied diff in bounded portions after tool-output truncation; no Git commands, suite replays, production edits, or child agents were used. Shared method hunks end mid-function, so source was parsed only to validate the named preservation maps and whole shared-body hashes.
- Recomputed all 48 safe-manifest file hashes, all 8123 recorded execution source/test hashes, and all 518 prior evidence pins: zero mismatches.
- Recomputed Git blob hashes from all 12337 preserved QA files without invoking Git: zero mismatches; independently verified the exceptional file's original7982-byte prefix and4417-byte append SHA256 values. The 12336 whole-file preservation claim is accurate.
- Parsed the ten affected owners with the qualification Python3.12 interpreter: all 1699 final declaration AST hashes match the preservation map. Independently reversed the supplied diff for the four shared methods and matched their complete reviewed body hashes. An initial comparison with the system interpreter produced different AST serialization hashes; the Python3.12 comparison resolves that tooling difference, and body hashes matched in both.
- Inspected complete retained test/static results: 57passed/1warning, fatal Ruff exit0, seven required files formatted, changed-source whitespace exit0. The formatter attribution map has no unattributed final edits and all projections preserve AST. The two incoming controller corrections preserve unrelated inherited formatting.
- Recovery, source-union, loading-carry, closed-process and clean-handoff receipts are internally consistent with the pinned source. Current branch ancestry, publication checks and final merge remain root-owned gates.

### Assessment

**Task quality: Approved, with the retained Minor resource-warning concern.**

**Reasoning:** The incoming repair and reviewed feature blocks coexist without an identified authority or behavioral regression. The evidence maps are reproducible against the fixed source and accurately distinguish fresh 57-case qualification from historical loading/QA evidence.
