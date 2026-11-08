---
id: TASK-39123
title: "Console trace storage's credentials-v1 projection leaves bare sk_live_, sk_test_, rk_live_ and rk_test_ keys unmasked, and v1 is versioned, so stored traces need a credentials-v2 projection"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 22:47
labels:
- follow-up
- security
- console
- tracing
priority: medium
references:
- tldw_chatbook/Chat/console_trace_redaction.py
- tldw_chatbook/Utils/log_sanitizer.py
- tldw_chatbook/Chat/console_semantic_revision.py
- tldw_chatbook/DB/recovery_core_schema.py
- Tests/Utils/test_log_sanitizer.py
dependencies:
- TASK-39061
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-39061's notes (round-11 branch fu/r11-arch) defer the trace side: "Credentials-v1 traces therefore still do not mask these keys". Line numbers are dev's.

**What the projection masks.**
- The trace projection's `_SECRET_TEXT` (`Chat/console_trace_redaction.py:51-56`) knows `sk-(?:live-)?…` plus a labelled-value rule.
- It then runs `sanitize_trace_credentials_v1` (`:708`). That reads `_STANDALONE_CREDENTIALS` (`Utils/log_sanitizer.py:104-116`), whose key forms are all `sk-`.
- fu/r11-arch adds the underscore forms only for diagnostics (`_DIAGNOSTIC_ONLY_CREDENTIALS`, applied in `_sanitize_diagnostic_text`), not for traces.

**Probe.** A pure probe on dev with synthetic 40-character bodies:

| Input | Masked |
|---|---|
| bare `sk_live_`, `sk_test_`, `rk_live_` or `rk_test_` keys in prose | no |
| a tool-result value `{'stripe': 'sk_live_…'}` | no |
| a nested message value `{"messages":[{"content":"key is sk_test_…"}]}` | no |
| `api_key=sk_live_…` (the label rule) | yes |
| `sk-…`, `sk-live-…` | yes |

**Why v1 cannot simply grow.**
- Each trace policy persists `credential_filter_version` (`DB/recovery_core_schema.py:248`).
- Revisions bind only results whose version equals `CREDENTIAL_FILTER_VERSION` (`console_semantic_revision.py:922`, `:968`).
- `Tests/Utils/test_log_sanitizer.py:87` pins the v1 projection.
- So new patterns added in place would change what an already stored v1 projection means.
- Tool results and provider envelopes are where such keys can appear without a label.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New traces are stored under a credentials-v2 projection that masks bare `sk_live_`, `sk_test_`, `rk_live_` and `rk_test_` keys of realistic length in prose, in a tool-result string and in a nested JSON trace value; a test is red before the change on each case
- [x] #2 Identifiers that only share a prefix (for example `sk_live_mode`) stay readable under v2
- [x] #3 v1 output is byte-identical: the v1 pins stay green, and existing v1 policies and revisions still read
- [x] #4 The owner decides how existing credentials-v1 policies are treated once v2 exists; the decision is recorded
- [x] #5 The new tests run on a PR lane (never the fast PR contract), with three timings of that lane's command recorded
<!-- AC:END -->

## Implementation Plan

1. Record the owner decision (2026-11-06): mask v1 on read; new traces under v2; stored v1 traces keep their tag; no stored data is rewritten.
2. Tests first (in `Tests/Utils/test_log_sanitizer.py`, on a PR lane): bare `sk_`/`rk_` live/test keys are masked by the trace sanitizer in prose, a tool-result string and value, and a nested JSON value; prefix-sharing identifiers stay readable; the v1 projection is byte-identical; a stored credentials-v1 native trace holding a bare key is masked on the Inspector read (both viewer profiles) and on export/share (every export profile; the dialog's file and clipboard destinations write the same text), and its policy tag and artifact bytes are unchanged; a v1 policy's frozen PII spans still land with the v2 key mask on top.
3. `Utils/log_sanitizer.py`: add `sanitize_trace_credentials_v2` = the v1 projection plus TASK-39061's underscore-key patterns (renamed from diagnostic-only, shared, not forked).
4. `Chat/console_trace_redaction.py`: `CREDENTIAL_FILTER_VERSION = "credentials-v2"`; `CredentialSanitizer` takes the projection version (v1 kept only to align a stored v1 policy's frozen PII spans) and reports it.
5. `Chat/console_semantic_revision.py`: revisions keep binding stored v1 policies (readable versions v1 and v2; materialized bytes use the current v2 masks); `project_semantic_revision_trace_message` applies a v1 PII policy's frozen spans to the v1 projection, then the v2 masks.
6. Update pins of the current version string; run the trace suites on head and base.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner decision (2026-11-06), AC#4:** "Mask v1 on read: new traces are written under v2; stored v1 traces keep their version tag, but every read, export and share re-applies the v2 masks, so old keys never surface. No stored data is rewritten."

**Approach.**
- `Utils/log_sanitizer.py`: `sanitize_trace_credentials_v2` = `sanitize_trace_credentials_v1` then TASK-39061's bare `sk_`/`rk_` live/test pattern. The pattern tuple is shared (renamed `_DIAGNOSTIC_ONLY_CREDENTIALS` -> `_UNDERSCORE_KEY_CREDENTIALS`, still used by diagnostics), not copied. v1 is untouched.
- `Chat/console_trace_redaction.py`: `CREDENTIAL_FILTER_VERSION = "credentials-v2"`, so every new policy (controller, voice, legacy import) records v2. `CredentialSanitizer` takes a `version` (default v2, `credentials-v1` accepted) and reports it as `detector_version`. `READABLE_CREDENTIAL_FILTER_VERSIONS` = {v1, v2}.
- Bump rule (`console_semantic_revision.py`): revisions bind only policies whose credential version the code supports (`:922`) and only sanitizer results of the current version (`:968`). So the supported set now holds v1 and v2, or every stored v1 policy would turn into `trace_redaction_policy_unsupported` omissions on the next message edit. What a v1 policy now materializes is masked by the current (v2) sanitizer; nothing already stored is rewritten.
- Reads: the Conversation Inspector (`project_capture_for_viewer`), export to file and share to the clipboard (both write `project_exchange_export(...).json_text`, `Widgets/Console/console_exchange_export_dialog.py`) already re-run `CredentialSanitizer()` over every stored call, native or legacy, so they now apply v2 whatever the trace's version. The one version-sensitive read is `project_semantic_revision_trace_message`: a v1 PII policy's frozen spans index its v1 projection. It now sanitizes with that policy's version, applies the frozen spans, then the v2 masks. A naive v2-first read shifted the spans and made the v1 PII trace's request unreadable (`{'omitted': 'native_request_reconstruction_unavailable'}`), shown by the control below.

**Red first.** In `Tests/Utils/test_log_sanitizer.py` (it runs on a PR lane):
- `test_trace_credentials_v2_masks_bare_underscore_keys[prose|tool-result-string|tool-result-value|nested-json x sk-live|sk-test|rk-live|rk-test]` (AC#1): all 16 red before (`(True, False) == (False, True)`: key present, no marker); green after.
- `test_stored_v1_trace_is_masked_on_read_export_and_share_but_not_rewritten[x4]`: a native trace stored under a `credentials-v1` policy whose user message and provider-response artifact hold a bare key, read through `ConsoleTraceNativeReader`, both viewer profiles and all three export profiles. Before: `read-full`, `export-full_trace`, `export-redacted_diagnostic` exposed the key. After: none does, and the policy row still says `credentials-v1` and the artifact bytes are unchanged before and after the reads.
- `test_stored_v1_pii_masks_still_land_with_the_v2_key_mask`: before, `'key sk_live_... then mail [PII omitted] please'`; after, `'key [credential omitted] then mail [PII omitted] please'`. Control (v2-first read, frozen spans not realigned): `assert None == [...]`, request `{'omitted': 'native_request_reconstruction_unavailable'}`.
- AC#2 `test_trace_credentials_v2_keeps_identifiers_that_share_the_prefix[x5]` and AC#3 `test_trace_credentials_v1_projection_stays_byte_identical`: green before and after (invariants); red under temporary edits (pattern without its 24-character floor: `short-word`, `underscore-identifier` red; the underscore pattern added to v1: the v1 test red), then reverted.
- AC#3, v1 pins: `test_logging_change_preserves_legacy_trace_credential_projection` stays green; `Tests/Chat/test_console_semantic_revision_coordinator.py` (its policies are `credentials-v1`; `test_two_policy_edit_materializes_once_binds_each_policy_and_advances_epoch_once` materializes under them) and `Tests/Chat/test_console_trace_native_reader.py` stay green. `test_sanitizer_never_exposes_secret_through_failure_or_diagnostic_text` pinned the current version string and now reads `credentials-v2`.

**Covering run.** The 107 test files that reference the credential filter version, `CredentialSanitizer`, the trace redaction, semantic revision, exchange export, trace projection or native reader, or the log sanitizer, at `-n 4` on this branch: 4844 passed, 282 failed, 10 errors in 58 min (load 50-160). Every failing node was rerun on base `fu-base-r12`, grouped per file: 272 fail there too. Of the 13 that did not: `test_sanitizer_never_exposes_secret_through_failure_or_diagnostic_text` pinned the old version string (updated, green); the 8 `test_console_voice_factory_composition.py` nodes fail on base as well when run by file (`RecoveryRequired: raw_source_selection_changed`, the known local trap); `test_discovered_tool_schema_changes_preserve_the_captured_run[bare_fence-True-True-False]` and `test_closed_history_is_revalidated_before_dispatch[sibling]` fail on both trees on a rerun; `test_real_gateway_blocking_response_close_is_off_loop_and_owned` and `test_ownerless_segments_reject_calls_and_response_links` pass on both on a rerun (load flakes). `Tests/Architecture/test_persistent_terminal_privacy_boundary.py::test_app_owned_terminal_session_leaves_live_console_boundaries_free` fails serially on both.

**Lane (AC#5).** The new tests are in `Tests/Utils/test_log_sanitizer.py`, already on architecture-guards "Run Architecture guard contracts" (35 targets, serial; it runs `Tests/Architecture/test_backup_owner_inventory.py`, `Tests/Utils/test_log_sanitizer.py`, `Tests/test_persistent_diagnostic_boundary.py` and `Tests/Persona_Buddy/test_controller_failure_logging.py`), with this round's nodes added, through `jobstep.sh` three times: 430 passed each, in 821.6 s, 564.2 s and 522.6 s, at 1-minute load 158, 52 and 33 when each started. Its last CI run took 7.2 min (PR run 42310788537), so the added nodes stay small against the 20-min cap; the local spread is load. Not the fast PR contract.

**Review r12b-rev1 fixes.**
- **Critical: continuing a v1 turn.** After the upgrade, the first capture-on follow-up after a tool-using turn traced under v1 was refused. New per-turn policies are v2, and compound admission required equal credential versions: `completed_tool_turn_policy` -> `TraceCallPersistenceError` -> "Run: Blocked — capture failed", and Retry hit it again. That was not only an interrupted-turn resume, as this note first said.
  - `trace_policies_agree` (`console_trace_redaction.py`) is the one rule: PII settings match exactly, and credential versions agree when they are equal or both are in `READABLE_CREDENTIAL_FILTER_VERSIONS`.
  - It now decides every agreement check: `completed_tool_turn_policy` and the project-context value check (`console_trace_continuation.py`), and the surface-delta checkpoint, the rendered-system slot and `completed_replay_policy` (`console_trace_service.py`).
  - The byte comparisons in that proof (`completed_tool_turn_value`, `completed_replay_*`) compare raw canonical values (saved revisions and continuations), not credential projections, so the version does not reach them. The collapsed tool results are not compared.
  - Test `test_completed_tool_turn_compound_admission_checks_durable_proof[upgraded_policy]` (`Tests/Chat/test_console_trace_runtime.py`): a tool turn stored under `credentials-v1`, with a bare `sk_live_` key in its tool results and in the next user message, continued under the production default. On `e906fb8b72` it was refused (`TraceCallPersistenceError: Provider call trace persistence failed.`). Now it dispatches with the same surface-replacement proof as `valid`. `[wrong_policy]` now uses `credentials-v999`, a version no reader knows, and is still refused.
- **Minor: a frozen span cutting a key.** A frozen PII span applied to the v1 projection could cut a bare key (for example a custom digit rule), and the fragments no longer matched v2.
  - `apply_frozen_pii_masks(..., widen_over_v2_credentials=True)` (used only for a stored older policy) first widens each span that overlaps a v2-only key match on the v1-projected text, so the whole key goes under the mask (the union).
  - The v2 match spans come from `log_sanitizer.trace_credentials_v2_spans`, the same patterns.
  - Test `test_stored_v1_custom_span_inside_a_key_leaves_no_key_fragment`: without the widening it read `'key sk_live_DONOTUSE[PII omitted]EXAMPLEONLYabcdefghij then mail ...'`; with it, `'key [PII omitted] then mail [PII omitted] please'`.
- **Covering rerun after these fixes.** The same 107 files at `-n 4`: 4848 passed, 280 failed, 10 errors in 50.7 min. The failing set is the earlier one (all on base) minus three, plus `test_fallback_trace_recovery_keeps_accepted_owner[True-capture_off-empty]`. That node passed twice on each tree when rerun alone (load flake). After the fixes: preflight passed, size ratchet 201 passed, bootbudget 32 passed, `Tests/CI` 1220 passed.
- **Lint.** `ruff --preview --select E30` findings the range introduced are fixed: `config_adapter.py` E303 and E302, the blank lines after `sanitize_trace_credentials_v2` and `trace_policies_agree`. Default `ruff check` on every Python file this branch touches now reports exactly base's findings (751).

**Not covered.** The trajectory export (`Chat/trajectory_export.py`, the Trace export dialog) has its own credential regex (`_CREDENTIAL_VALUE_RES`, `sk-` only) and does not read console trace storage; listed as a follow-up.

**Files.** `tldw_chatbook/Utils/log_sanitizer.py`, `tldw_chatbook/Chat/console_trace_redaction.py`, `tldw_chatbook/Chat/console_semantic_revision.py`, `tldw_chatbook/Chat/console_trace_continuation.py`, `tldw_chatbook/Chat/console_trace_service.py`, `Tests/Utils/test_log_sanitizer.py`, `Tests/Chat/test_console_trace_credential_filter.py`, `Tests/Chat/test_console_trace_runtime.py`.
<!-- SECTION:NOTES:END -->
