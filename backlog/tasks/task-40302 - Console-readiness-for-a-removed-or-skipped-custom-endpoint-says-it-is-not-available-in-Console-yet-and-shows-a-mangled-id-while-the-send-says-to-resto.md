---
id: TASK-40302
title: "Console readiness for a removed or skipped custom endpoint says it is not available in Console yet and shows a mangled id, while the send says to restore the entry"
status: Done
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 19:21
assignee:
- '@claude'
labels:
- follow-up
- console
- providers
- copy
priority: low
references:
- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/Chat/console_provider_support.py
- tldw_chatbook/Chat/console_provider_gateway.py
- Docs/User_Guide/console.md
dependencies:
- TASK-39151
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (security review r13s-rev1 of TASK-39151). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**What the user sees.** After TASK-39151, a schemeless llama.cpp custom entry is skipped. A chat bound to it shows "Provider blocked: 'custom_ep:<slug>' is not available in Console yet." with the label "Unknown". That copy blames a missing feature and shows an internal id; the send path gives the real fix, "this custom endpoint is no longer configured. Restore its entry..." (gateway `:4436`, documented in `Docs/User_Guide/console.md:802-804`).

**Why.** When a `custom-ep:<slug>` entry was skipped or removed, `entry_for` returns None (`console_session_settings.py:1855`). `_canonical_chat_provider_id(settings.provider)` then runs without `app_config`, and `provider_config_key` mangles the id to `custom_ep:<slug>`, which misses the `custom-ep:` check in `console_provider_support.resolve_console_provider_identity`. The provider resolves as unknown, so readiness takes `provider_unsupported` (`:2017-2019`) and its copy (`:2078`). The skip reason is only logged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Readiness for a `custom-ep:` id with no configured entry reads the send's 'no longer configured, restore its entry' copy and never shows the mangled id; a test is red before the fix
- [x] #2 A provider Console truly does not support keeps the 'not available in Console yet' copy
<!-- AC:END -->

## Implementation Plan

1. Re-check the premise on base 26b2e677dd: readiness for `custom-ep:<slug>` with no entry reads provider_unsupported with the mangled `custom_ep:<slug>` id.
2. Red test, then fix in `build_console_settings_readiness`: a `custom-ep:` id with no configured entry blocks with the send's "no longer configured" copy (one shared constant with the gateway), never naming the id.
3. Keep "not available in Console yet" for a provider Console does not support (test).
4. Update `Docs/User_Guide/console.md` where it describes the copy.

## Implementation Notes

Readiness for a `custom-ep:` id with no configured entry now says what the send says.
- **Premise held on base 26b2e677dd**: `build_console_settings_readiness` ran `_canonical_chat_provider_id(settings.provider)` without `app_config` when `entry_for` returned None, so `provider_config_key` turned `custom-ep:gpu` into `custom_ep:gpu`, which missed the `custom-ep:` check in `resolve_console_provider_identity`; readiness took `provider_unsupported` and read "Provider blocked: 'custom_ep:gpu' is not available in Console yet." (label "Unknown").
- **Fix**: when the provider is a `custom-ep:` id and no entry resolves (removed, or skipped at load), readiness blocks with `provider_missing` / `select_provider` (the "Restore its entry or choose another provider" remedy), its detail is `CUSTOM_ENDPOINT_NOT_CONFIGURED_COPY` and its provider name "Custom endpoint", never the id. The copy is one constant in `console_provider_endpoints.py`, which the gateway's send now reads too. The facets carry `configuration="incomplete"`, `configuration_issue="provider_missing"`, which the blocker contract requires. A provider Console does not support keeps its own copy.
- **Tests** (`Tests/Chat/test_local_server_discovery.py`, run whole by the dreams-boundary lane):
  - `test_readiness_for_a_missing_custom_endpoint_says_to_restore_its_entry[removed|skipped]` (skipped = TASK-39151's schemeless remote llama.cpp entry): readiness detail equals the send's `visible_copy`, not ready, `select_provider`, and no label, detail or provider name holds `custom_ep`/`custom-ep`. Red at base: 2 failed, `'Provider blocked: 'custom_ep:gpu' is not available in Console yet. ...' == 'Provider blocked: this custom endpoint is no longer configured. ...'`. Green after.
  - `test_a_provider_console_does_not_support_keeps_its_copy` (AC#2): green before and after.
- **Docs**: `Docs/User_Guide/console.md` says a chat whose custom endpoint was removed reads the same copy, in Chat settings and on send.
- **Runs** (run.sh, `-n 4`, load 57-70): the discovery file, `test_custom_endpoint_registry.py` and `Tests/Agents/test_preset_fallback.py`: 491 passed. The wider covering run is recorded in TASK-40300, which touches the same modules.
- Files: `tldw_chatbook/Chat/console_session_settings.py`, `tldw_chatbook/Chat/console_provider_endpoints.py`, `tldw_chatbook/Chat/console_provider_gateway.py`, `Tests/Chat/test_local_server_discovery.py`, `Docs/User_Guide/console.md`.

**Follow-up (TASK-40300 review round, 2026-11-13).** Two Settings Overview pins in `Tests/UI/test_settings_custom_endpoint_default.py` (a dangling `custom-ep:gone` default and the `CUSTOM-EP:gpu-box` malformed default) read "Status: Not ready · unsupported"; with this task's `provider_missing` blocker they read "Status: Not ready · no provider" and were rewritten on purpose (in TASK-40300's round-2 commit). This round's UI base comparison had run at this task's own commit, so it did not see them.
