---
id: TASK-35190
title: A same-provider pick of an unseen model keeps the chat's session endpoint
status: Done
assignee:
  - '@claude'
created_date: 2026-10-17 11:17
updated_date: '2026-10-18 11:50'
dependencies:
- TASK-33006.19
labels:
- model-config-redesign
- follow-up
priority: medium
references:
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/Widgets/Console/console_model_popover.py
- tldw_chatbook/Widgets/Console/console_settings_field_row.py
- Tests/UI/test_console_endpoint_discovery.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the TASK-33006.19 review, round 2 (2026-10-17). A chat can run on an Endpoint other than the configured URL, for example after a user applies a second local server to one chat (session 127.0.0.1:9200, config :9099). In Chat settings, picking a model on that same provider that the draft has not seen moves the chat to the configured URL: the controller's settings rebaser (`rebase_console_settings_draft`) starts an unseen target from its default chain, and with a clean endpoint draft it takes that chain's base URL, not the chat's session URL. For llama.cpp that base URL is None, and None here means the configured URL, not 'no endpoint': `default_console_session_settings` leaves it unset on purpose, and the configured endpoint is resolved later (the modal shows it through `_initial_base_url_for_provider`). A fix that reads None as 'no endpoint' is wrong. So even a saved-model pick on the same server lands on :9099. Since TASK-33006.19, pick mode offers a model that Test connection listed only while a pick of it stays on the endpoint that listed it, which is correct, but it means a listing at :9200 says '3 models listed' and then Change offers none of them except the current model. The only way to pick them today is to retype the Endpoint. TASK-33006.19's test_a_listing_is_offered_only_while_its_pick_keeps_the_listed_endpoint pins today's reset, so this task changes that test's expectation. The rebaser is not Chat settings' own. Switch model (Alt+M) picks through it too (`ConsoleModelPopover._rebase_to`, on the quick surface, `QUICK_MODEL_DEFAULT_FIELDS`), and a probe at the TASK-33006.19 head (2026-10-17) showed the same reset there: a llama.cpp chat on :9200 with a clean endpoint draft, picking an unseen same-provider model, got base_url None and no endpoint draft on both the full and the quick surface. So the fix belongs in the rebaser, and AC #4 says what Alt+M then does. The rebaser's other two callers do not take this path: the live commit (`ChatScreen._commit_console_settings_submission_live`) rebases a draft onto its own pair, and session adoption (`session.py`) replaces the rebased base_url with the configured one. Account for TASK-33005.12 too (Wave 6 integration review, 2026-10-18): a server's switcher rows are the models the last listing of the readiness's connection returned, and both Alt+M's and pick mode's readiness is the configured connection's (:9099). Once a same-provider pick keeps the session Endpoint (:9200), Alt+M would offer :9099's listing too, with :9099's readiness words, for picks that land on :9200, so the implementer must decide which listing those rows read.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 In Chat settings, picking a saved model on the chat's own provider keeps the chat's session Endpoint (:9200) instead of resetting it to the configured URL (:9099), pinned by a test
- [x] #2 A model that Test connection listed at the chat's session Endpoint (:9200) is offered in pick mode without retyping the Endpoint, and picking it keeps the chat on :9200, pinned by a test
- [x] #3 A pick onto a different provider still takes that provider's own endpoint, so no session URL is carried across providers
- [x] #4 Switch model (Alt+M) picking an unseen model on the chat's own provider also keeps the session Endpoint (:9200), so it and Chat settings land the same pick on the same Endpoint, and the quick surface still saves no Endpoint as a default, pinned by a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
A pick of another model on the chat's own provider now keeps the chat's Endpoint on both surfaces, and Switch model's rows on that provider no longer read the configured server's listing for picks that land elsewhere.

- Rebaser (the root, shared by Chat settings, pick mode's landing check and Alt+M): the endpoint choice moved out of `ConsoleChatController.rebase_console_settings_draft` into `rebased_endpoint` in tldw_chatbook/Chat/console_settings_apply.py, next to `remember_model_draft` (the controller had 2 lines of headroom; it shrinks 29,182 -> 29,150 and its ratchet row is lowered to match). A kept pair (the current or a remembered target) behaves exactly as before. An unseen pair takes the endpoint of the latest draft on its provider: the current draft when it is on that provider, else the last remembered one. So a pick on the chat's own provider keeps the session URL (:9200) even after a pick of another provider in between, which matters for Alt+M: its draft is rebased on every highlight change, and with Find empty PREVIOUS (often another provider) is highlighted first. With the remembered-draft lookup removed, only the three detour cases fail. An unset URL keeps meaning the configured one: the default chain applies, never "no endpoint". A registry entry keeps its own URL (ADR-146). An endpoint edit still carries only on the full surface, so the quick surface's rule is unchanged. The other two rebaser callers are unaffected: the live commit rebases onto its own pair, and session adoption overwrites base_url.
- AC#3: a pick onto another provider has no draft on that provider to read, or only one remembered from this open, which holds that provider's own endpoint. So the session URL is never carried across providers.
- Listing decision (TASK-33005.12, controller note): rows read the listing of the server a pick lands on. `ConsoleModelPopover._listing` (tldw_chatbook/Widgets/Console/console_model_popover.py) feeds `_models_for` and `_catalog_status_rows`. Other providers read the readiness's listing, as before. On the chat's own provider, a Chat settings listing (`served_models`, already kept only where a pick lands, TASK-33006.19) is that provider's listing, so it replaces the configured server's listing and the catalog or saved models it left out (TASK-33005.12 AC#1). Without one, the readiness's listing counts only while the chat's Endpoint is the connection it listed. Otherwise there is no listing, and the rows are the catalog or saved models, as with no listing (TASK-33005.12 AC#2). `_with_known` keeps the 100-model bound rule for both listings. Pick mode now opens on the modal's snapshot (`_open_model_picker` passes `source`, the draft as the controls show it), so a typed Endpoint is known too.
- `ConsoleSettingsReadiness.connection` is set only while evidence for it is current. The first version of `_listing` relied on it, so with no evidence (`test_a_listing_at_this_chats_endpoint_replaces_a_loaded_catalog`) it fell back to catalog plus served. Reading `served_models` first fixed it.
- Not changed, and left for the owner: Switch model's readiness words and its READY/NEEDS SETUP grouping on the chat's own provider still describe the configured connection (:9099), because the readiness resolver and the switcher probe are the configured connection's (UI/Console_Modules/model_switcher.py, outside this task's references). Before this task a pick also landed there, so the words were accurate. Now, with the chat on :9200 and :9099 down, llama.cpp sits under NEEDS SETUP although the chat works. Pick mode cannot pick those rows, and Alt+M's Enter explains them in place, so nothing new is lost, but the words describe another server.

Tests (Tests/UI/test_console_endpoint_discovery.py; CoreFirstHarness config :9099, chat on :9200):
- `test_a_listing_is_offered_only_while_its_pick_keeps_the_listed_endpoint` (AC#2, rewritten on purpose: it pinned TASK-33006.19's reset): served-y listed at :9200 is offered in pick mode without retyping, and picking it keeps :9200.
- `test_a_pick_on_the_chats_own_provider_keeps_its_session_endpoint[direct|after-openai|after-an-entry]` (AC#1, AC#3): model-b keeps :9200, also after an OpenAI or GPU-box pick, and those picks take their own default endpoint.
- `test_an_unseen_pick_keeps_the_chats_url_except_on_a_registry_entry[server|registry-entry]`: the rebaser unit, real implementation.
- `test_switch_model_keeps_the_chats_session_endpoint_and_saves_none` (AC#4, private profile, real Console, real popover, real controller and real default writer; the probe network is stubbed): the configured :9099's listing (model-a) is in the shared evidence, and Alt+M still offers the saved model-b. Picked after a vLLM highlight, it applies with base_url :9200. Make default for new chats then writes chat_defaults.model, leaves api_settings.llama_cpp.api_url at :9099, and writes no :9200 anywhere in the config.
- Rewritten and renamed on purpose (they pinned the configured listing and the catalog beside a :9200 listing): `test_a_listing_is_offered_beside_the_configured_servers_own_listing` -> `test_a_listing_at_this_chats_endpoint_replaces_the_configured_servers[chats-endpoint|typed|typed-unlisted]` and `test_a_listing_is_offered_beside_a_loaded_catalog` -> `test_a_listing_at_this_chats_endpoint_replaces_a_loaded_catalog`; their census lines are renamed in place.
- Red before the fix: with these tests over fc5a596f31's four source files, 10 of the 11 cases fail; the registry-entry unit case passes there by construction. Negative controls at the head: popover `_listing` neutralised -> 5 failed (the listing cases and AC#4); the remembered-draft lookup removed -> 3 failed (the detours and AC#4); pick mode opened on `self._draft` again -> `[typed-unlisted]` failed; the entry exclusion removed -> `[registry-entry]` failed.

Verification: the 37 covering files (switcher, picker, pick mode, Chat settings, Console flow, rebaser unit files; xdist `-n 6 --dist loadfile`, Tests/UI/test_app_instance_warning.py listed first and deselected): head 274 failed / 1708 passed, base fc5a596f31 274 failed / 1700 passed, the same failing names (local RecoveryRequired noise). The six census entries, 3 of 3 serial: 11 passed each run. Tests/Architecture module and screen ratchets serial: only dev's 7 known module rows red. Tests/Performance/test_ui_ready_module_census.py serial: 4 passed (limit unchanged). Tests/CI: 962 passed, 1 skipped. `./scripts/preflight.sh` rc 0 (358 entries, floor 358). ruff on the touched .py files: no new findings against fc5a596f31.

Census: three node ids appended (MINIMUM_FILES 355 -> 358) and two renamed in place. Docs: Docs/User_Guide/console.md (pick mode's Endpoint paragraph and Switch model's listing paragraph). Lesson: backlog/docs/lessons-testing-evidence.md (a pick test that goes straight to its target skips the switcher's rebases).

Modified: tldw_chatbook/Chat/console_settings_apply.py, tldw_chatbook/Chat/console_chat_controller.py, tldw_chatbook/Widgets/Console/console_model_popover.py, tldw_chatbook/Widgets/Console/console_settings_field_row.py, Tests/UI/test_console_endpoint_discovery.py, Tests/Architecture/test_module_size_ratchet.py, scripts/ui_pr_gate_census.txt, scripts/check_ui_pr_gate_census.py, Docs/User_Guide/console.md, backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
