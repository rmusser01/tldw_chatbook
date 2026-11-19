# Finding verdicts

- **I1 — Continue Retry/Regenerate wraps historical input: ADDRESSED.** `_ensure_user_continuation_instruction` now constructs and returns the synthetic current-user anchor (`Chat/console_chat_controller.py:21755`); Retry (`:18509`, `:18537`), Continue (`:18697`), Regenerate (`:18866`, `:18908`) and edit (`:20024`, `:20088`) pass the shared identity. The four adapter tests check Retry versus current Regenerate both live and after capture-Off restart, asserting clean canonical history and composition only around Continue (`Tests/Chat/test_console_prompt_mixin_retry.py:712`). Focused caller search found all four callers updated for the new required parameter.
- **I2 — Marker permits ordinary/edited results to bypass response equality: ADDRESSED.** The validator now requires a nonempty explicit accepted replacement relationship, exact sealed result revision equality, and a bounded cycle-free chain to the exact origin trace call (`Chat/console_trace_service.py:3366–3410`). Ordinary first attempts have no replacement relationship; later edits no longer match their replacement's sealed revision. The first call binds through the exact native checkpoint in the reservation transaction (`Chat/console_trace_runtime.py:756`; `DB/console_prompt_mixins.py:420`), and the replacement result seals inside the existing terminal generation callback (`Chat/chat_persistence_service.py:3695`; `DB/console_prompt_mixins.py:445`). Negative controller tests cover missing/unequal ordinary links, edited first responses and edited replacements (`Tests/Chat/test_console_prompt_mixin_retry.py:831`), alongside existing positive replacement restoration.
- **SIM102 cleanup: ADDRESSED.** The marker condition is a single `if` (`DB/console_prompt_mixins.py:222`). The supplied exact per-file lint differential reports BASE/current 448/448 and no introduced/removed diagnostic multiplicities (`/tmp/task5-fix1-ruff-differential.json`).
- **Cancellation regression coverage: ADDRESSED.** A real persistence worker is held across cancellation; the test asserts no early publication or completion, then durable publication before CancelledError and no second adapter entry (`Tests/Chat/test_console_prompt_mixin_retry.py:778`).

# New breakage in the fix diff

**Important R1 — Current-choice Regenerate now depends on the unavailable original Retry snapshot.** `Chat/console_chat_controller.py:18909–18912` unconditionally passes the selected assistant's old marker as `replaces_attempt_id`. The newly added writer check rejects that replacement unless `read_attempt` can reconstruct the old snapshot (`DB/console_prompt_mixins.py:200–214`). Therefore Send with mix-ins → lose the marked attempt or corrupt its item set → Retry refuses → Regenerate with valid current choices also refuses before adapter entry. This contradicts both current-choice regeneration and the existing recovery instruction, “Regenerate with current choices instead” (`Chat/console_turn_resend.py:285`). This dependency was introduced by this fix; Regenerate previously persisted its current snapshot without requiring the old attempt body.

Keep exact Retry fail-closed, and retain strict replacement witness validation, but make unavailable old replacement proof prevent witness authority rather than prevent a new current-choice generation. Do not fabricate or repair the original snapshot from current library content. Add targeted controller tests for missing marked attempt and corrupt/missing item set → failed exact Retry → successful Regenerate using current choices, with no replacement witness derived from the damaged old record. Cover capture Off at minimum and preserve safe capture-On trace behavior.

# New seal and retention checks

- The authorized unreleased-v82 table amendment has nullable trace/result foreign keys with `ON DELETE SET NULL`; immutable guards permit clearing only after referenced proof disappears (`DB/migrations/chachanotes_prompt_mixins.sql:34–35`, `:69–76`). The current recovery catalog mirrors the table and guards (`DB/recovery_core_schema.py:1042`). No new table or provider-body archive is introduced.
- Sealing uses the existing generation transaction callback; the new forced-failure test verifies both seal and complete lifecycle projection roll back (`Chat/chat_persistence_service.py:3695`; `Tests/Chat/test_console_prompt_mixin_retry.py:958`). The real trace purge test verifies origin call proof clears and exact operational Retry bodies survive (`Tests/Chat/test_console_prompt_mixin_retry.py:875`).
- Fork inserts omit all three witness fields (`DB/console_prompt_mixins.py:365`), and the test checks their null values (`Tests/Chat/test_console_prompt_mixin_retry.py:875`). Promotion retains `replaces_attempt_id` as inert relationship history but has no historical trace/result seals; promoted-history → durable captured send → capture-Off retry/reopen is tested (`Tests/Chat/test_console_prompt_mixin_retry.py:982`). The controller explicitly clarified during re-review that this inert promotion lineage is authorized, superseding the brief's earlier blanket clearing wording; fork must still clear all three. A relationship alone cannot pass the validator: exact result seal and a predecessor chain reaching an actual bound origin are both required. No concrete historical authority or retention defect found in this scoped check.

# Evidence and scope

- Read the 1,541-line fix package once for `a125761794b4e4450b00e089711a221d27e08bea` → `8c551fe41fdff3c637a5cedcbcfc5e81314bf22c`, the appended brief rulings and round-1 implementation report. Inspected cut-off settlement/fork function context and a focused caller search for the changed continuation-helper contract.
- Read `/tmp/task5-fix1-final.log`: **95 passed in 58.33s**. The report names the full targeted command and intermediate covering commands; counts overlap. Read the exact lint differential JSON: **448/448, no introduced diagnostics**. The former 201/11 owner baseline qualification is unchanged, not presented as fresh fix-round evidence.
- No suite reruns, subagents, diff regeneration, product/index/HEAD changes, or live provider/UI verification.

# Out-of-scope observations

None.

# Verdict

**Fix round: Findings remain open — new Important R1 (unavailable original snapshot blocks current Regenerate).** Original I1/I2 and requested cleanup/coverage are addressed, but the new recovery regression must be fixed before approval.

Skipped full-suite and live provider/UI checks. Remaining risk is the newly blocked Regenerate recovery path, not a demonstrated weakness in the new sealed-result proof.
