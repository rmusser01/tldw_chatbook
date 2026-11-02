# PR3346 layer correction handoff

Implemented the approved AC17 / ADR286 dependency clarification. Existing operational values and JSON codecs now live in Chat/console_prompt_mixin_models.py; console_prompt_mixins.py re-exports the same objects and retains only request composition / selection reconciliation. DB imports models directly. Three LOCAL_* limits and the existing literal-v3 decoder plus its pure helpers live in prompt_artifact_models.py. Capabilities and codec retain their existing public imports; the general codec keeps v2 compiler dispatch. Prompts_DB retains its transaction, batch/UUID/order rules and checks artifact_type=mixin before calling the pure literal decoder. No schema, serialized JSON, authorizer, receipt or source-currentness changes.

Extracted only _drain_app_unmount_services into root app_shutdown.py; app_lifecycle and the TTS AST ownership test point there. Utils keeps watchdog, signal handling, module state and existing callers. Exact old size pin 776 -> 554; new helper pin 225. No layer allowlist changes.

Owned files:
- tldw_chatbook/Chat/console_prompt_mixin_models.py (new), console_prompt_mixins.py
- tldw_chatbook/Prompt_Management/prompt_artifact_models.py, prompt_artifact_codec.py, prompt_source_capabilities.py
- tldw_chatbook/DB/console_prompt_mixins.py, Prompts_DB.py
- tldw_chatbook/Utils/app_shutdown.py, tldw_chatbook/app_shutdown.py (new), tldw_chatbook/app_lifecycle.py
- Tests/Chat/test_console_prompt_mixin_values.py
- Tests/Prompt_Management/test_prompt_mixins.py
- Tests/TTS/test_tts_app_ownership.py
- Tests/Architecture/test_module_size_ratchet.py (only the two shutdown rows)

Validation:
- Narrow model-boundary regression red before extraction: missing console_prompt_mixin_models import (layer-model-red.log).
- 119 tests passed in 24.33s (layer-fix-tests.log): values/composition, Prompt v3/v2 codecs and local source/coherent batch reads, real SQLite operational ownership, recovery, source worker cancellation, TTS unmount cancellation/outer-finally, both architecture layer guards, both exact shutdown size pins.
- 3 compatibility/export controls passed in 0.49s after export cleanup (layer-export-tests.log).
- layer-move-proof.txt verifies AST identity of all20 operational declarations,12 artifact-codec declarations and16 shutdown declarations across extraction.
- layer-lint.txt compares old/new owned-source lint:269 ->267; no added diagnostic code/message multiplicity. Two import-order warnings removed; moved existing diagnostics remain, not called clean.
- git diff --check clean.

All listed evidence is /private/tmp/mixins-pr3346-<name>. No commits/push/manifests or full suite. No changes to controller/service/runtime (concurrent owners). Performance census must account for the two new modules; the perf owner was informed. Root owns ADR/task and independent review.

Exact main check: PYTHONDONTWRITEBYTECODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=/private/tmp/mixins-layer-fix-tests Tests/Chat/test_console_prompt_mixin_values.py Tests/Chat/test_console_prompt_mixin_composition.py Tests/Prompt_Management/test_prompt_mixins.py Tests/Prompt_Management/test_prompt_artifact_codec.py Tests/DB/test_console_prompt_mixins.py Tests/Backup_Recovery/test_prompt_mixin_restore.py Tests/Chat/test_console_prompt_mixin_admission.py::test_native_source_worker_retires_before_cancel_delivery Tests/TTS/test_tts_app_ownership.py::test_unmount_preserves_service_cancellation_when_tts_cleanup_also_fails Tests/TTS/test_tts_app_ownership.py::test_unmount_closes_owned_tts_resources_from_outer_finally Tests/Architecture/test_layer_direction.py 'Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/Utils/app_shutdown.py]' 'Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/app_shutdown.py]'
