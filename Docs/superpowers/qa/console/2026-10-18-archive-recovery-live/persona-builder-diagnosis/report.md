# Persona first-send boundary diagnosis — 2026-10-18

Status: DIAGNOSED, read-only. Inspected head `20f3f3e24b5c9ba5a58f6793af1cad0ad76034ce`. No production, live fixture, DB or repository changes. No app/model/network calls or test sweep.

The actual pure request builder raises **`TraceProvenanceAlignmentError: trace provenance category mismatch: system`** when given the retained nonempty Persona system prompt and the provenance assigned by `_build_durable_trace_request`. Omitting the system row for an empty prompt builds successfully, with system_count=0 and active_request_count=1. Guarded probe final exit: **0**.

## Exact boundary

- Retained Persona text: `Help the user find archived chats and choose whether to restore one. Present the findings before requesting restoration.`
- Retained user text: `Find recently archived chats across all workspaces.` Saved message `e3aac440-13c3-499e-b99a-fc085608b5d7`, saved revision `d7f9b42c-87d4-4d85-b07f-ed7ee04cc760`, conversation `5b3aafea-0ff2-4202-8956-4f3400e2856c`.
- `console_chat_controller.py:10517`: fallback `descriptors` assigns `ProviderArtifactTraceProvenance(ACTIVE_REQUEST, policy)` to every unowned visible row, including a Persona system row. The current user receives its saved revision descriptor.
- `console_prepared_request.py:1049` constructs `ConsoleRequestProvenance`; its `console_trace_provenance.py:989` system category admits only RENDERED_SYSTEM artifact source. `_validate_category` raises at line855.
- Controller `_continue_prepared_submission`, lines10979–11002, catches that exception after durable acceptance and invokes `_handle_durable_trace_provenance_failure`; that handler pauses TRACE_PROVENANCE and returns provider_started=false. Request publication/resume follows only a successful build. This explains accepted checkpoint, saved user+empty assistant, BLOCKED +2ms after durable_commit, and no call rows/postcommit publication; UI not yet showing a saved active conversation is consistent with publication not running.
- The retained live log did not expose the exception traceback. The exact exception is reproduced offline at the production pure-builder boundary; mapping it to this native failure is code-backed diagnosis, not a captured live traceback.

## Pre-existing status

The AST of the exact fallback descriptors assignment and the AST of the entire ConsoleRequestProvenance validator class are identical at base `75b671c001e84bf2713c380e30de3e587bcc8ae0` and inspected head. The builder assignment lines shifted; its bad classification did not change. Exact sources and both hashes are in result.json. This is outside the completed archive recovery correction.

## Supported fresh fixture correction

A fresh policy-only Persona may use `system_prompt=""`. The Persona profile schema accepts an optional string without a nonempty constraint (`tldw_api/character_persona_schemas.py:642`). `_persona_session_prompt_seed` explicitly returns an empty projection for a blank template (`UI/Console_Modules/session.py:596`); fresh Persona creation replaces canonical settings with `system_prompt=seed.system_prompt or None` (line4273). `_provider_selection_for_session` keeps the settings-derived empty prompt for a template-less named Persona (`console_chat_controller.py:20795`); `_leading_system_message` returns no system row when prompt and greeting are empty (line27720).

Use a fresh session so materialized settings cannot retain the failed Persona's prior prompt, and retain the same narrowing tool policy. No capture, thinking, history, authority or confirmation gate needs changing. Existing General control with native empty default is also supported. No claim is made that this proves the remaining live recovery flow; root owns that native check.

## Evidence and limits

Command: `PYTHONDONTWRITEBYTECODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /private/tmp/archive-recovery-persona-builder-diagnosis-20261018/probe.py > /private/tmp/archive-recovery-persona-builder-diagnosis-20261018/probe.log 2>&1` — exit0. The script uses exact retained system/user text and saved revision ID; synthetic opaque frozen-policy identity/version, PII disabled as in the live fixture. It calls the actual unchanged pure builder, without constructing an app or controller.

`probe.log` and `result.json` retain the A/B result and exact traceback. The final probe blocked all socket/DNS operations (0 attempts), any DB connection, and any file write outside this diagnostic directory. Five live inputs were hashed before and after and were identical: config, Persona control, blocked initial snapshot, blocked checkpoints, and app log. No actual DB was opened. A disposable config read-copy selected import settings; actual live configuration was unchanged.

Two preliminary diagnostic harness failures are retained: phase1 blocked Python tempfile's write outside the diagnostic directory, then TMPDIR was pointed inside; phase2 the successful empty-prompt request was misread through a nonexistent `.semantic` output field, corrected to actual `.system`/`.active_request`. Those are probe errors, not product failures. The final A/B is verified independently by explicit result assertions. Source comparisons use git show only.
