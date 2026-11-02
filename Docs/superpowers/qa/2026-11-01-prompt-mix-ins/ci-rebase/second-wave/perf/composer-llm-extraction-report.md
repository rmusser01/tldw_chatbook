# Composer/LLM extraction and warmup completion report

Scope: approved mechanical extraction against ba1cafc55dd5ac9b6b1f066c936206e652ed6b65 (dev88af integration), plus the separately approved transitive idle-voice source witness and app-owned warmup completion. No commits, pushes, manifests or budget edits. Other agents made concurrent disjoint changes; this is not a frozen final combined-tree performance qualification.

## Implementation

- ConsoleComposerBar: 7403 -> 7189 lines (original limit 7195). Three methods retain signatures and delegate to the existing console_speech_controls module. Full original implementations, docstrings and comments moved; all state, CSS and class declarations remain in their owner. Helper dependencies resolve through the live composer module, preserving monkeypatch bindings. The recording-glyph owner import is intentionally retained with a precise F401 explanation because the helper reads that public module binding.
- LLM_API_Calls: 5238 -> 5167 lines (original limit 5176). Responses SSE and usage-line functions delegate to existing openai_protocol, preserving signatures; streaming uses yield from. Time, JSON, ledger, logger and requests resolve through the live API owner. Existing nonstream normalization and transport/payload handling are unchanged.
- A cold-idle optimization previously inspected only composer methods. The two extracted helpers actually traversed by that path now publish definition-time identity/code/defaults/keyword-default items/globals/closure witnesses. Any changed source follows the original complete sync callback. Nonidle width budgeting is excluded from this idle witness because it is not traversed. The exact new body-change test failed before this fix; all 20 idle-voice controls pass after it.
- Chat warmup now receives an explicit zero-argument completion callable and dispatches it with app.call_from_thread. app.py passes its bound _finish_first_send_warmup method. That method lazily calls the new root presentation helper app_idle_warmup.py (18 lines), which resolves shutdown/runtime/current view at completion and requests the same existing idle refreeze owner. No Chat-to-UI import remains, and no eligibility gate moved into the worker. Third-party warmup/pacing logic is unchanged.
- Two old warmup tests directly addressed the deleted Chat presentation helper; they now use the actual app callback. Existing received-custody, current-attachment, shutdown, pacing, import failure and manifest failure assertions remain.

## Evidence

1. /private/tmp/mixins-extraction-ast-proof.json: all five moved function bodies, signatures, return annotations, docstrings and comments equivalent after only helper renaming/local owner qualification/self annotation normalization. Both original owner files have identical remaining declaration/body ASTs against ba1caf.
2. /private/tmp/mixins-extraction-style-comparison.json: all 702 baseline records and 702 after records share the same keys and compare identical for computed styles, region/content geometry and display. Captured at existing mounted pause boundaries across voice/status/reason-width tests. This is mounted computed-style evidence, not a new terminal/font claim.
3. /private/tmp/mixins-extraction-witness-green.log: 20 idle-voice tests passed, including eight new helper-source mutation cases. Two new Responses native request tests in that run failed before their bootstrap-profile marker was added; this was test admission setup, not an HTTP failure. Those exact two passed in the corrected run below.
4. /private/tmp/mixins-extraction-warmup.log: 15 passed, 8 deselected. Exact command selects native_headers or completion or warm_up in test_openai_streaming_usage.py and test_console_first_send_preimport.py. Includes Responses streaming/nonstream payload and native headers at patched real requests.Session.post (no network), current-view/shutdown completion controls, pacing and third-party module-step guard.
5. /private/tmp/mixins-extraction-warmup-mounted.log: 3 passed. Real mounted received custody blocks completion/maintenance, later real idle timer reclaims cycles; two exact failing-warmup controls pass.
6. /private/tmp/mixins-extraction-layer.log: test_no_new_upward_import passed. No allowlist change.
7. /private/tmp/mixins-extraction-lint.json: 12 scoped files default-rule diagnostics 268 -> 268, zero introduced across the aggregate. Existing broad-exception diagnostics moved with the exact SSE body (LLM owner -2/protocol +2). app.py is concurrently root-owned; our six-line wrapper uses ordinary existing local-import style. Scoped git diff --check passes. Range formatting applied only to touched functions/tuple; five small files fully format-clean. Whole console_speech_controls formatting still reports only its pre-existing first 299-line formatting, which was not rewritten.

## Existing failures and limits (not claimed green)

The focused 37-case style cohort was 10 failed/27 passed before extraction and 12 failed/25 passed after. The two extra failing names were replayed using the original three composer bodies in the otherwise current tree: both fail there too. Caller observation shows actual existing control-bar/availability callbacks overwrite a fixture's manually injected error/preparing state with authoritative idle. Current paired replay was 1 failed/1 passed. The recorded mounted style/geometry values nevertheless all match exactly. These runs are not timing-stable all-green suite evidence; no production authority or test assertion was weakened. Artifacts: mixins-extraction-styles-{baseline,after}.{log,json}, mixins-extraction-pair-{original,current}.{log,json} under /private/tmp.

The provider cohort was 3 failed/13 passed. Existing ordinary chat-completion early-close test expects one close and observes two in an AST-unchanged path. Two old TestProviderRequestPayloads tests patch obsolete session factory seams and attempted DNS to reserved api.openai.test; they did not intercept the actual request boundary and were not repeated. The two new native request tests intercept actual requests.Session.post and pass for both streaming and nonstreaming Responses. New direct adapter clock/logger/ledger/JSON monkeypatch and early-close tests passed before and after extraction. Artifacts: /private/tmp/mixins-extraction-providers.log and provider-tests.py.

No final combined-tree five-guard performance run has been started; root will request it after all other extractions stabilize. No active owned test processes remain at report time.

## Owned paths

- tldw_chatbook/Widgets/Console/console_composer_bar.py
- tldw_chatbook/Widgets/Console/console_speech_controls.py
- tldw_chatbook/LLM_Calls/LLM_API_Calls.py
- tldw_chatbook/LLM_Calls/openai_protocol.py
- tldw_chatbook/UI/Console_Modules/composer_run_controls.py
- Tests/Chat/test_openai_streaming_usage.py
- Tests/UI/test_console_voice_chip.py
- Tests/UI/test_console_idle_voice_boot.py
- tldw_chatbook/Chat/console_first_send_modules.py
- tldw_chatbook/app_idle_warmup.py
- Tests/UI/test_console_first_send_preimport.py
- Tests/UI/test_console_idle_heap_refreeze.py
- tldw_chatbook/app.py

The app.py ownership is limited to _preimport_heavy_screens's call and the adjacent _finish_first_send_warmup wrapper. console_first_send_modules/test_console_first_send_preimport also contain earlier/root approved changes; preserve these when reviewing diff. All other listed paths are the specific extraction/witness/test scope.
