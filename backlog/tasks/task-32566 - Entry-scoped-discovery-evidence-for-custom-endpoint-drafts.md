---
id: TASK-32566
title: Entry-scoped discovery evidence for custom endpoint drafts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-13 19:02'
updated_date: '2026-09-19 02:35'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Custom-ep drafts have no discovery identity (_current_draft_discovery_identity returns None), so entry-scoped model discovery results are always discarded by the evidence fencing: post-create probes report status but the evidence never settles into the connection-evidence store. Surfaced by the CE-fix agent concern and deferred qodo PR-2668 finding 4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Creating a named endpoint from full Conversation settings returns to responsive parent settings and settles success or failure without reverting the saved entry.
- [x] #2 Discovery uses the entry family URL contract and entry credentials while preserving exact entry identity in connection evidence and served-model readiness.
- [x] #3 Switching entry or changing endpoint or credentials rejects stale discovery results; current results settle and restore the discovery action.
- [x] #4 Targeted mounted modal and probe regression tests cover entry creation, credentials, family routing, and stale-result fencing.
- [x] #5 Rapid model-catalog refresh during entry selection releases superseded work without leaking unawaited coroutines.
- [x] #6 Built-in endpoint edits update their bound draft and verification state; read-only named-entry URL echoes cannot cancel discovery.
- [x] #7 Late cancellation-resistant discovery results cannot query or repaint a dismissed modal; real model-selection edits still cancel active probes
- [x] #8 The reported hyphenated llama-local-2 endpoint and Windows GGUF model path survive endpoint creation, quick-picker selection, and Apply through the mounted production controller.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/146-console-custom-endpoint-registry.md
Reason: Restore existing entry ownership, family routing, credential precedence, and discovery contracts.

1. Reproduce parent-modal failure and add regular entry-scoped discovery regressions.
2. Separate registry ownership from family canonicalization in UI and evidence identities; use typed evidence and one cancellable discovery worker group.
3. Carry named-entry credentials through the bounded connection probe and verify exact request routing with controlled transport fixtures.
4. Run scoped discovery/evidence/modal checks and lint; document evidence and any limits.

Crash-report follow-up (2026-09-18):
ADR required: no
ADR path: backlog/decisions/146-console-custom-endpoint-registry.md
Reason: Test-only coverage of the existing registry identity and execution-family contract.
1. Extend the mounted creation/controller regression with the reported hyphenated endpoint and Windows model ID.
2. Exercise quick-picker selection and Apply after creation; prove the regression detects restored legacy normalization.
3. Run targeted endpoint/controller checks and scoped lint/format checks, then publish a test-only PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed named endpoint discovery through the mounted parent modal and the production ChatScreen/controller flow. URL interpretation and probe availability use the entry family while UI/evidence identities retain the exact registry ID. ProviderDraftIdentity now carries an optional validated custom_endpoint_id; evidence/save-lease comparisons reject cross-entry reuse even for matching URLs. Entry credential resolution is shared with the gateway, honors environment then stored-key precedence, never falls through to another provider's credentials, and authenticated model probes retain bounded responses and disabled redirects.

Creation probes start after the provider controls settle, use the correct typed evidence identity and one cancellable worker group, publish terminal success/failure, and restore actions. Credential rotation invalidates completed listings as well as pending results. Source-traced nearby fixes reject queued model adapter echoes before they cancel or overwrite discovery, consolidate duplicate endpoint-change handlers so built-in URL edits update the bound dirty draft, and defer picker/probe coroutine creation until worker admission. The production rebase/raw-ID fix was coordinated with TASK-32707. Its named-entry Make default path also now preserves hyphenated registry IDs through disk reload and next-chat resolution; TASK-32707 AC8 records that evidence.

Files: console_settings_modal.py, provider_test_evidence.py, custom_endpoint_registry.py, settings_endpoint_probe.py, model_search_picker.py; isolated ChatScreen connection/default-intent and gateway credential-wrapper seams; console_settings_defaults.py; new Tests/UI/test_console_endpoint_discovery.py and focused additions to existing transport/evidence/picker/readiness contract tests. Added the incident to backlog/docs/lessons-console-wiring.md. ADR required: no new ADR; this restores backlog/decisions/146-console-custom-endpoint-registry.md.

Verification: 169 passed across the new endpoint suite, provider evidence, picker, and default-writer tests; 272 passed (one controlled loopback test deselected) across settings endpoint transport and readiness; 20 existing modal discovery/connection tests passed; the real controlled loopback HTTP fixture passed separately with required sandbox escalation (462 targeted checks across those selections). Final new endpoint suite after isolating unrelated context metadata: 15 passed. Regression assertions were observed failing first for creation, availability, credential/listing identity, stale callback controls, coroutine admission, bound endpoint edits, registry default reload, and queued model events. New tests/transport/registry files pass Ruff; changed regions were formatted and six legacy touched files show zero added Ruff diagnostics versus HEAD; git diff --check passed. Root owns final formatting/review of shared modal/chat/gateway regions.

Limits: no full-suite run, real account credentials, external provider calls, generation requests, commit, or reporter-machine native-terminal reproduction. An existing requests dependency-version warning remains. One pre-existing TTS MockTransport test used an external hostname for an encoding-only assertion; its fixture now uses a loopback URL so validation no longer depends on external DNS. No unrelated user changes were reverted.

PR integration adds production teardown ownership checks and mounted cancellation-resistant result coverage. Legacy model-change tests now change actual controls instead of dispatching stale adapter events; stale-echo rejection remains covered separately.

PR integration verification: 20 passed after the teardown repair (four real model-change cancellation paths, one late-dismissal regression, and all 15 endpoint regressions). The new dismissal regression reproduced the removed-control query before the fix. Attachment and screen-stack ownership now fence settlement and repaint. Scoped static checks pass with no new Ruff diagnostics. The user has authorized publishing these repairs in a PR against dev; no merge or full-suite run.

2026-09-18 crash-report regression follow-up: current origin/dev already includes the runtime repairs. Extended the mounted production creation/controller test with the exact custom-ep:llama-local-2 endpoint, port 9090, and Windows GGUF model ID, then Cancel, quick-picker selection, and Apply. Both ordinary and reported cases verify the exact committed provider, model, and endpoint. Scoped verification: 22 passed across endpoint discovery, settings retention, and registry options; Ruff lint/format and git diff --check pass. Each legacy discovery/provider-normalization mutation independently makes the reported case fail; production files were restored unchanged. A broader run had 24 passed and one unchanged provider-chip failure (RecoveryRequired: raw_source_selection_changed); the same failure reproduced on unmodified dev with 23 passed, and in the unchanged module alone. Self-review complete; independent review could not run because of an account usage limit. ADR required: no new ADR, test-only coverage of backlog/decisions/146-console-custom-endpoint-registry.md. No full-suite or native Windows verification; no runtime changes.
<!-- SECTION:NOTES:END -->
