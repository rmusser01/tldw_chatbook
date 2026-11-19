# Independent Composer/provider extraction review

Verdict: no actionable Critical, Important, or Minor findings in the reviewed change.

## Scope and qualification

Read-only source review against BASE `ba1cafc55dd5ac9b6b1f066c936206e652ed6b65`, in `/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook`. This verdict qualifies the working-tree bytes listed below, not an assumed subsequent commit. No product edits, runtime tests, performance runs, or broad test sweeps were performed for this review. The prior root-owned extraction review is separately saved at `/private/tmp/mixins-pr3346-root-extraction-review.md`.

Reviewed the three voice presentation extractions from `console_composer_bar.py` to the existing `console_speech_controls.py`; the two stream/usage helpers from `LLM_API_Calls.py` to the existing `openai_protocol.py`; the transitive idle callback witness added to `composer_run_controls.py`; connected caller bodies and the changed-source tests in `Tests/UI/test_console_idle_voice_boot.py`.

## Evidence checked

- Direct source comparison plus a stdlib AST comparison found all five moved executable bodies and all their argument defaults identical after normalizing only `_owner.name` back to the original global name, omitting docstrings, and omitting the added local owner import. The five checks covered `_voice_chip_width_cap`, `set_voice_status`, `_sync_full_width_voice_presentation`, `_responses_stream_to_chat_sse`, and `_openai_stream_line_usage`. All five body/default comparisons returned true. This is source-equivalence evidence, not a runtime test claim.
- Composer facades preserve original method names, signatures, keyword defaults, and synchronous dispatch. The helper reads live defining-module `Content`, `Horizontal`, `Static`, `Button`, `NoMatches`, glyph resolution, lazy voice-state loader/constants, and size/content utility functions. Existing `self` calls remain dynamic, preserving subclass/instance method behavior. Idle and unavailable paths still clear the chip-width cache before the presentation synchronization; sizing, plain-text Content rendering, attachment visibility, and draft/reason-strip sequencing remain unchanged.
- The idle-skip witness still requires exact stock Composer type, idle current/prior state, pre-ready state, and an unloaded voice module. It retains the original method/loader identity checks and adds both newly extracted helpers reachable by the idle path. Definition-time tuples capture function identity, code, positional defaults, keyword-default dictionary identity and value identities, globals identity, and closure identity. The later reader checks the current module binding and those captured identities. Width budgeting is correctly excluded: the idle/unavailable branch returns before reaching it. The existing availability replay and all non-idle dispatch remain unchanged. The new source tests exercise helper code/default/binding replacement before the reader import; these tests were read, not run here. This witness is an optimization qualification, not a new security or data authority boundary.
- The provider facade retains `yield from` rather than returning or manually iterating the nested generator, preserving propagation of `close()` and `throw()` to the original finally/response-close path once started. Generator initialization remains deferred until iteration. The moved helper uses live original-module time, JSON, requests exception type, logger, and session-ledger accessor; response event handling, malformed JSON handling, payload fields, usage recording, and terminal DONE placement are unchanged.
- The connected OpenAI stream caller still yields from the facade for Responses, returns afterward, and retains the existing chat-completions last-usage policy. `_openai_stream_line_usage` retains its broad best-effort exception boundary and nonempty-dictionary requirement. No request construction, credentials, permission, model-selection, retry, or ledger-authority change was introduced by these moves.

## Limits

No assertion is made that all repository tests pass. Runtime/performance qualification is owned by the separately coordinated cohort; this review did not rerun it. No unrelated baseline behavior was reclassified as an extraction regression. The previously reviewed warmup layer hook is covered by the separate root review.

## Reviewed working-tree SHA-256

- `tldw_chatbook/Widgets/Console/console_composer_bar.py`: `24a978b9ecd2ed93bfda0e1de1c3ce7469f2efaa486f54c2cd14ddc06a8fa7df`
- `tldw_chatbook/Widgets/Console/console_speech_controls.py`: `1f6a2b2a6240946c0413e6ea0b840c80cb508b8aa8004688e36af3144f17473e`
- `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`: `49fb84b2e05876c11cd17156593bcfb60b005a220abf029e126d233764038d8b`
- `tldw_chatbook/LLM_Calls/openai_protocol.py`: `11a9e56e4f78c2ef1b83a5fbde01b8e5239b48f7bb52f7991dccd01c9fac4d07`
- `tldw_chatbook/UI/Console_Modules/composer_run_controls.py`: `81407224cf65be388760195a487a7912344f083afe2124ad46eca475c667469b`
