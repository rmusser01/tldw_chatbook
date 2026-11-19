## Spec compliance

**Compliant for the supplemental source/test correction.** Reviewed exact range `75456b7a2d3f8ea4f711f41c061d212abdd03add` → `e4a2717b4158d178d2ec1616d9a8ebd983982c08`. Earlier original-feature, Task1 and fix1 reviews remain completed.

The production change removes only the unsupported `collections_service` argument from `_library_provider_for_app` (`tldw_chatbook/Chat/console_runtime.py:661–671`). The real `LocalLibraryToolService.__init__` accepts the remaining keyword arguments and has no Collections parameter (`tldw_chatbook/Library/local_library_tool_service.py:448–474`). The corrected call matches the established UI/MCP composition contracts at `UI/Console_Modules/library_activity.py:145–158` and `MCP/server.py:453–464`. No ignored compatibility argument, service API expansion, Collections implementation or trace/permission relaxation is introduced.

Frozen authority remains the selector: the factory reads `turn_context.library_authority.direct_library_tools`, retains the RAG backend branch and returns None without context (`console_runtime.py:622–637`). The controller still requires Allowed access, exact provider type and issued built-in authority (`console_chat_controller.py:15634–15677`). Production partial binding and named consumers are unchanged (`console_runtime.py:3507–3509`; `console_chat_controller.py:20076,26349`). Supported backend, media-chunk and policy-enforcer forwarding remains intact.

## Quality and evidence

- **Real constructor regression:** `Tests/Chat/test_console_library_runtime_policy.py:112–156` exercises the production partial through the controller with optional services absent/present and a collections attribute present in both cases. It asserts exact provider/service classes, supported backend/enforcer identity, actual media-chunk construction and catalog. This detects the original TypeError without a permissive constructor mock.
- **Frozen mode control:** `test_console_library_runtime_policy.py:159–176` verifies the exact RAG backend/catalog and absent-context refusal. The shared context helper deliberately makes compatibility configuration disagree with frozen authority (`:52–93`), so it checks the intended source of mode selection. Existing Blocked/unavailable policy cases are included in the recorded targeted run.
- **Observed RED:** factory `red.log:86,183,196–198` records the actual unsupported-keyword TypeError for both Direct cases and a passing RAG control:2failed/1passed/10deselected in1.55s. The disposable native Direct log independently names the same constructor failure before this correction; no new live probe was performed during review.
- **Observed GREEN:** the complete `Docs/superpowers/qa/console/2026-09-28-console-direct-library-factory/green.log:1–9` records **15passed in14.92s**, with no warning or teardown/network-guard error:13 policy/factory checks plus2 existing captured native Direct/RAG positives using fake external transport. Final test Ruff/format outputs are clean.
- **Evidence scope:** the factory checks use real constructors with backend sentinels; the existing captured follow-up tests qualify their real native/capture paths with fake transport. This combination is appropriate for the one-line contract repair but does not itself demonstrate the fresh mounted Direct live journey with this shared factory. Root's bounded live verification remains pending; source approval does not spend or extend the four remaining requests under the total10 cap.
- **Static limits:** `static-final.json` records an exact one-argument deletion, matching source/test hashes and equal baseline/current34 Ruff signatures; whole-runtime formatting debt remains. Only test import wrapping/order followed GREEN, without behavior changes. Root independently verified all12 manifested artifact bytes and final hashes. No global pristine lint/format claim is warranted.
- **Raw evidence:** the exact factory `red.log` retains14 original traceback trailing-space lines under the explicit evidence-only exception. Source/tests/authored QA and other staged artifact paths were checked clean; the whole artifact diff remains intentionally non-clean. Historical phases/hashes are preserved.

## Issues

**No actionable findings.** Counts:0 Critical,0 Important,0 Minor. No new breakage identified in the supplemental diff. The implementation is the minimal repair to the established service contract and preserves frozen authority.

## Assessment and scope

**Task quality: Approved. Source-review status: complete; fresh Direct live qualification remains root-owned.**

Read the Task2 brief/report fully, all production/test/authored-QA hunks from the supplied companion, the full package inventory and pertinent raw/static evidence. Named surrounding reads checked the complete shared factory, production binding/controller gate and consumers, real constructor, and established UI/MCP references. No tests were rerun, no apps/models/network were invoked, no subagents or earlier review waves were started, and no source/index/HEAD/task metadata changed. Only this report was written. Existing trace/dictionary/provider/FD/Next Send limitations are neither reopened nor resolved by this repair.
