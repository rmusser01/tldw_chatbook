**Fence merged mix-in updates with explicit `expected_version=None` — ADDRESSED.** `tldw_chatbook/Prompt_Management/prompt_source_capabilities.py:363` replaces missing or explicit-None versions with the version of the merged existing record, preserving supplied non-None versions. `prompt_scope_service.py:931` uses that prepared payload and forwards the fence to `update_prompt_by_id` at line 940; `local_prompt_service.py:188` does the same through its existing database call at line 191. `Tests/Prompt_Management/test_prompt_mixins.py:370` parametrizes both adapters; its second database commits an intervening body edit before the stale write at line 408, then asserts `ExpectedVersionConflictError` at line 417 and preservation of the concurrent compatibility text, canonical definition and version 2 at lines 430–436.

**Share duplicated local mix-in preparation in the existing validation owner — ADDRESSED.** `tldw_chatbook/Prompt_Management/prompt_source_capabilities.py:310` now owns candidate detection, conversion rejection, whitelisted merge, canonical validation, request-size validation, projection and version inference. Both adapters call it for creation (`local_prompt_service.py:127`, `prompt_scope_service.py:908`) and updates (`local_prompt_service.py:188`, `prompt_scope_service.py:931`). Their distinct existing storage calls remain. The non-mixin early return at `prompt_source_capabilities.py:320` preserves ordinary Prompt/Recipe behavior, and the extracted mix-in path retains the original validation and field selection.

**New Critical/Important/Minor breakage in the fix diff — None found.**

**Out-of-scope observations — None.**

**Checks —** Read the supplied fix diff once. Read the original two findings, brief, binding review context and appended fix report. Focused source/test reads checked the helper's normalization output, adapter forwarding and regression assertions. The fix report names the covering regression and records RED (scope failed, direct passed), GREEN (2 passed), and the three-file compatibility run (232 passed in 2.39 s); those claims agree with the changed paths and tests. These are supplied execution results, not independently rerun evidence. No new unresolved doubt required another test.

**Fix round — All findings addressed, no new Critical/Important breakage.**

**Verification limits —** No tests or full suite rerun; no product, index, HEAD or branch mutation. Review was limited to the two findings and fix diff, using supplied targeted test evidence.
