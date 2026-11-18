# Final extraction performance requalification

Coordinated serial run on settled shared tree at HEAD ba1cafc55dd5ac9b6b1f066c936206e652ed6b65 with uncommitted approved extractions. No simultaneous app tests; parent froze product. Current changed-product hashes are in final-perf-source-inventory.json. This is fresh native/mounted automated evidence, not a new live terminal/server/font claim.

## Final verdict

All five original guards PASS, 33.08 seconds: /private/tmp/mixins-perf-final-extractions.log. The same observation-only driver /private/tmp/mixins-perf-probe.py invokes the original exact five pytest nodes serially and serializes their census returns without changing assertions. No budget constant or snapshot was raised/refreshed.

| Guard | Actual | Existing limit |
|---|---:|---:|
| Actual native UI-ready modules | 1031 | 1031 |
| Screen preimport added modules | 554 | 558 |
| Screen preimport added LOC | 426096 | 426199 |
| Largest route added LOC (Library) | 132405 | 135111 |
| First-send warmup modules | 94 | 94 |
| Ancestor-scoped bare-type CSS rules | guard passed (<=268) | 268 |
| Boot worker/thread census | passes exact existing inventory | unchanged |

UI-ready and warmup module headroom is zero; preimport LOC headroom is 103. Those are measured limits, not promises about future helper additions.

## Causal fixes after the first combined run

The initial combined run (/private/tmp/mixins-perf-post-extractions.log) was 2 failed/3 passed: UI-ready1032 and warmup111. Exact UI-ready delta from previous qualified phase was solely console_memory_banner.

1. Deferred PromptVariablesDialog/Request from the Console prompts owner to the variable/system dialog branch, after the existing variable-free fast path returns. It remains imported by the Library screen at route load, so the preimport census rises by its 309 LOC and remains within the old limit. No dialog code, guards, callback or rendering changed. Native UI-ready now sheds exactly that cold modal.
2. Removed explicit console_agent_bridge and app_served warmup entries because actual native UI-ready already owns their complete import closures. Kept/added native guard assertions that both are resident. No synthetic seeding, hidden imports or weakened count check. Added the actual new console_agent_reply and console_settings_persistence helper owners to the warmup list.
3. Deferred Character_Chat.character_avatar from Tools.character_tool_service until actual avatar approval/save. Two small delegates keep the service-global image_backend_configured and resolve_avatar patch seams, exact parameters/defaults/return annotations, and the existing validation/permission/DB call ordering. The service can be constructed/cataloged without parsing avatar code. This repays the final warmup module, reaching94.

## Focused controls

/private/tmp/mixins-final-cold-controls.log: 16 passed, 1 failed. All actual SQLite generated-avatar/save/failure/invalid-input cases, approval-summary monkeypatch failure control, and new cold-import boundary pass. Console prompt command plain fast path, shared variable dialog/value, original placeholders and stale picker refusal pass. The remaining existing picker test fails only its manually assigned pending-attachment label assertion (None vs photo.png); exact node reproduces on unchanged ba1caf archive, same line356/value, under the same bootstrap contract (/private/tmp/mixins-picker-baseline.log). No assertion or authority was weakened.

The new cold-avatar test was confirmed red against the old eager import (/private/tmp/mixins-avatar-cold-red2.log). The first attempt lacked the private-profile decorator's request fixture and is preserved separately as setup error, not causal evidence (/private/tmp/mixins-avatar-cold-red.log).

Scoped default-rule lint: /private/tmp/mixins-final-cold-lint-comparison.json, five files28->28 with zero added. Exact avatar delegate signature proof: /private/tmp/mixins-final-avatar-signatures.json. Scoped diff whitespace checks pass. No tests/processes remain active. No commits, pushes, manifests or budget-row edits by this agent.

## Final patch ownership

- tldw_chatbook/UI/Console_Modules/prompts.py
- tldw_chatbook/Tools/character_tool_service.py
- Tests/Tools/test_character_tool_service.py
- tldw_chatbook/Chat/console_first_send_modules.py
- Tests/Performance/test_ui_ready_module_census.py

Parent made an independent attribute-access/census-preserving adjustment to app_idle_warmup after my original extraction report; its current26-line source is captured in the final product inventory. The prior18-line report is historical, not the current byte count. Root owns all size enrollments and final documentation.
