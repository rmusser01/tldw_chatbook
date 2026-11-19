# Latest-dev bounded qualification follow-up

## Outcome and source identities

- Original BASE: `ec8eda1d39a5d8ae8ed043b4270da173b95f6652`.
- Frozen dev: `f0ffcf9e819b577bd38c416f38550969c75fb5a0`.
- Exact dev merged by controller: `9b28ce1479efed6bca687cfbb33d261322f83abe`.
- Initial merged HEAD: `149adb53f1d10b090e31a5f458e8c2ffcd3ca458`.
- Controller docs/identity reconciliation predecessor: `62806f75a1ad5e293e25a00123278c5f1307d601`.
- Owned final HEAD: `cde073ed62d1e8f0d6163bac0b5f667c9fe7670f`.

This follow-up is bounded to the requested baseline21, six telemetry owners, real-profile guard and complete creation owners, plus evidence-backed repairs in six source/test paths. It does not replace the original task report or its 91 command receipts. No app, helper, reviewer, full suite, merge, push or PR was run by this implementer. Controller-owned task/plan/identity updates were committed independently and were not staged here.

## Merge disposition

`latest-merge-source-probe` verifies the exact two parents of 149adb and that chat_screen's only merge delta is `spend.console_rate_limit_line(provider_key)`; removing that exact line restores the source of the first parent. The lessons file preserves every prior line with only disjoint insertions. Fourteen changed upstream Python files match committed bytes and pass scoped fatal Ruff. That initial proof predates the repairs below. Full upstream evidence/doc whitespace is not claimed: controller observed imported QA capture padding. Source-only whitespace covers production, tests and scripts.

## Actual RED and contract repairs

1. Original complete telemetry run: **61 failed, 601 passed**. Keep those 601 successful nodes as a distinct original result, not a manufactured combined green. Actual guarded config source selection refused per-case HOME/config redirects (`RecoveryRequired: raw_source_selection_changed`). Existing native `bootstrap_profile` markers retain the private collection-selected profile in real config/TLS/app consumers. Gateway, egress and cost-screen markers remove that fixture mismatch; a later isolated three-owner run exposed the same UI autouse import in spend projection (**555 passed, 19 setup errors**), so that owner also retains the private bootstrap. Real-profile guard/refusal policy is unchanged.
2. Llama metadata is deliberately asynchronous under TASK33081 AC1. Tests now drain the actual bounded background metadata tasks before asserting exact probe URLs and credentials. Keyless tests retain no-Authorization assertions; stored credentials remain exact on health, both metadata reads and generation. Metadata404 fallback remains tested. No send-path wait or production metadata policy was added.
3. ADR179's default custom endpoint execution is `custom-hosted`; URL/readiness/selected-provider identity remain pinned. Four stale legacy-only execution assertions now name the real default. A paired explicit `custom_endpoints_use_engine=False` control preserves rollback behavior. The all-handler fixture supplies both explicit URL and key for the hosted custom family. Executable `latest_gateway_fixture_probe.py` captures readiness key `custom_hosted` and real resolution; it did not change admission.
4. Mistral's transport moved from LLM_API_Calls to `mistral.py` / `hosted_chat.py`. The old fake intercepted neither the native transport nor native settings owner, producing a 502 response. The fixture now intercepts `hosted_chat.create_default_session` and `mistral.get_runtime_config_snapshot`, returns a strict real requests.Response (index 0, assistant role, finish_reason stop), and retains exact separate alias endpoints and credentials. The intermediate 502 response with incomplete index-less response remains RED evidence; the strict normalizer was not relaxed.
5. Empty spend state still asserts Current $0 and no next-send charge; nonempty tracker failure remains unavailable with independent next-send forecast and cache alert. The incidental Context 11% pin was stale against the shared estimator: actual used 1000/safe 8488 rounds to 12%, including the 512 tool/schema budget. `latest-spend-context-probe` first failed due the scratch import path; the corrected executable probe preserves that loader failure separately from estimator evidence. No price/estimator production change was made.
6. Mounted waiting gateways now inherit the existing cached-context protocol implemented by their sibling double; the production caller remains strict. No fallback was added to runtime code.
7. Governing spend spec `Docs/superpowers/specs/2026-09-04-console-current-and-next-send-spend-design.md:26` excludes unaccepted owners from Context and Current. The predispatch test now asserts positive draft tokens before admission, real VALIDATING state, retained optimistic USER transcript, request tokens 0, and Current $0; accepted/dispatched positive controls remain. The exclusion and original conflicting test were introduced together in dev 144ac2e1083688bf10977d22540acef6943edcb1 and are unchanged at frozen dev. Linked ADR052/095 preserve context/settings ownership.
8. The same spec, line 36, requires active sends to cancel coalesced idle refresh. This was a genuine baseline defect. The canonical dispatch stops the pending timer; observational trace showed that the composer-clear event then rearmed it while runtime custody was already accepted but controller state remained IDLE. Existing ConsoleRuntime.has_custodied_turns now accepts an optional exact session filter, preserving no-argument behavior. Existing Input.Changed run_active projection includes that exact custody. No authority, extra state or new owner was created. Genuine native runtime acceptance before VALIDATING, other-chat idle edits, no-argument/filter truth, terminal custody cleanup, refused-send draft retention/rearm and keyboard callback never fires controls pass. This await-gap mechanism is the AC7 remediation approved by the controller before code.

## Verification stages and regression controls

- Initial merged source: 21/21 literal baseline nodes passed; requested six telemetry owners executed with 61 failures and 601 successes. Other unchanged telemetry owners (rate limits/context controls) remain qualified by that original invocation; no generic repeat was made.
- Intermediate control runs preserve 14 failures/47 passes, then 4 failures/10 passes, then 1 failure/23 passes. Keyboard timer tracing preserves the remaining failure and exact stop/rearm ordering.
- Final custody controls: 3 pass. They catch removing dispatch cancellation, removing exact-session custody admission, using global custody to suppress another chat's idle edits, retaining custody after terminal release, and failing refusal rearm.
- Final complete amended owners: **611 passed, 2 warnings**. Amended-production literal baseline21: **21 passed**. Final-byte guard/creation owners: **115 passed, 2 inherited platform skips**, no warnings. Full results are the literal receipts below. Earlier greens are not represented as executing on later bytes. During the complete amended-owner run only the runtime accessor's wrapping was formatted; `latest-runtime-format-correction` records exact before/after hashes and identical Python AST. Subsequent baseline/creation checks and committed-source proof use final bytes.
- Static qualification is scoped fatal Ruff (E9,F63,F7,F82), source compile/equality and immutable changed-code formatter ratchets, not a claim to remove inherited full-style debt. The original 64 Python files plus four amended test owners are checked against committed HEAD. Fourteen upstream provider/bootstrap files have a separate fatal-lint receipt. Obsolete unshipped migration additions remain absent.
- Controller's 11 derived guards passed on its working tree in `controller-preflight.json/.log`; this implementer did not repeat them and does not relabel that run as committed-HEAD evidence.

## Exclusions, warnings and limits

No skip/xfail marker was added or changed. New upstream guard inherited platform exclusions are `Tests/test_real_profile_guard.py::test_every_write_kind_into_the_profile_is_refused_and_recorded[setxattr]` and `[removexattr]`, exact reasons `no os.setxattr here` / `no os.removexattr here`. These APIs are unavailable on the installed Python, so those two paths remain unqualified. Other 55 guard nodes and all 60 creation nodes passed in the initial guard invocation. Frozen timer XFAIL and historical archaeology SKIP remain unchanged and separately limited in task-1-report.md; neither behavior is qualified by its exclusion.

The 14-failure/47-pass control run emitted an unawaited Textual Timer coroutine warning and FD growth of 403 (start 14, end 417, limit 200). The final complete amended-owner run retained an unawaited Textual Timer coroutine warning and FD growth of 452 (start 14, end 466, limit 200). Every warning is retained verbatim in the receipt output and log; no warning filter or cleanup threshold was changed. No broad FD cleanup was attempted. The first formatter snapshot used the same output filename as its command receipt, so run_check replaced the snapshot with the receipt; it was immediately recaptured into a distinct immutable baseline path. One intermediate hunk-detail file was reused by the next formatter pass; both outer literal invocations/results survive and final per-hunk details use a distinct file. Final checks qualify the corrected paths; no lost intermediate detail is manufactured.

Controller will merge/qualify any later docs/perf-only upstream delta, perform live Console qualification and handle publication. Those remain controller-owned and are not claimed completed here.

## Literal command receipts

Each JSON below stores exact argv, cwd, selected environment, returncode, elapsed time and full untruncated combined output. The companion log is the byte-preserved output. The command shown uses shell quoting solely for readability; argv JSON is authoritative. All tests use the installed 3.12 interpreter and TLDW_TEST_GC_EVERY=1, native private-profile isolation, fresh owned basetemps, no installs or real-profile writes.

### latest-provider-bootstrap-fatal-ruff

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m ruff check --select E9,F63,F7,F82 tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/Chat/provider_rate_limits.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/UI/Console_Modules/console_spend_projection.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Utils/egress.py tldw_chatbook/Widgets/Console/console_context_controls.py Tests/Chat/test_provider_rate_limits.py Tests/UI/app_factory.py Tests/UI/conftest.py Tests/UI/test_console_cost_chip_screen.py Tests/conftest.py Tests/real_profile_guard.py Tests/test_real_profile_guard.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.321s`.

Full literal receipt: [latest-provider-bootstrap-fatal-ruff.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-provider-bootstrap-fatal-ruff.json); stdout/stderr: [latest-provider-bootstrap-fatal-ruff.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-provider-bootstrap-fatal-ruff.log). SHA256 JSON `d32adaf1a5e816dcee285073a1d0362449c9b24d4a75f802dd5940666140e7eb`; log `82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18`.

Literal output excerpt:

```text
All checks passed!
```

### latest-committed-format

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/format-ratchet-baseline.json --head 149adb53f1d10b090e31a5f458e8c2ffcd3ca458
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `32.998s`.

Full literal receipt: [latest-committed-format.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-committed-format.json); stdout/stderr: [latest-committed-format.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-committed-format.log). SHA256 JSON `70ed202231e7b1f50b7f5a2359d1b515709aa7b06257aba560d96beeb4d58930`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-committed-source-static

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest_committed_static.py 149adb53f1d10b090e31a5f458e8c2ffcd3ca458
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `42.709s`.

Full literal receipt: [latest-committed-source-static.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-committed-source-static.json); stdout/stderr: [latest-committed-source-static.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-committed-source-static.log). SHA256 JSON `7f82525582ef8fbafa6744831d1070671f31a086cf356b528fa9b5dc7ad57ec1`; log `d77175859ba2681e0b2314a03e1f6cbbc577edfb3a65152ebece357c02df5ab0`.

Literal output is preserved in the linked log (including empty successful output).

### latest-baseline21

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_prompt_queue_coordinator.py::test_close_tombstones_before_cancel_and_never_starts_next_prompt Tests/UI/test_console_runtime_ownership.py::test_runtime_owned_custody_tracks_only_lifetime_handles Tests/UI/test_console_runtime_ownership.py::test_runtime_tombstones_before_shutdown_and_disposes_via_to_thread Tests/UI/test_console_runtime_ownership.py::test_persistent_attach_sync_failure_has_bounded_backoff_and_resume_retry 'Tests/UI/test_console_runtime_ownership.py::test_reconciled_view_keeps_each_live_poll_reason_and_one_timer[wake]' Tests/UI/test_console_runtime_ownership.py::test_opening_console_during_a_headless_delivery_arms_the_poll Tests/UI/test_console_launch_wake.py::test_a_launch_into_console_delivers_without_stealing_the_active_tab Tests/UI/test_console_launch_wake.py::test_a_launch_with_no_marks_constructs_nothing_and_reads_once 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_invalid_ownership[bad_role]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_invalid_ownership[cross_conversation]' Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_considers_only_checkpoint_owners_on_the_selected_active_lineage 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_malformed_or_mismatched_checkpoint_identity[assistant_message_id]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_state_cas_requires_every_expected_owner_predicate[assistant_state]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[terminal_content]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[sync_intent]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[checkpoint_delete]' Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_is_atomic_and_returns_committed_proof Tests/Chat/test_console_agent_project_instructions.py::test_child_chain_uses_its_own_exact_first_request_budget Tests/Chat/test_console_agent_project_instructions.py::test_primary_token_omission_is_delivery_local_when_child_admits 'Tests/Chat/test_console_chat_fork.py::test_configuration_and_leaf_writers_block_fork_through_live_publication[active_leaf]' Tests/Agents/test_agent_chat_create_tools.py::test_new_chat_schema_shape --basetemp=/private/tmp/console-latest-baseline21-01a0fa6c
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `105.549s`.

Full literal receipt: [latest-baseline21.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-baseline21.json); stdout/stderr: [latest-baseline21.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-baseline21.log). SHA256 JSON `dac1083ba7ddedb0a53d18fca86207d7fca1bdf6a80403ea2aa62aa659b2adf3`; log `096acf71759835969056d7a13794602bee6baa102bece5c495b2b736dfe20287`.

Literal output excerpt:

```text
21 passed in 95.94s (0:01:35)
```

### latest-source-whitespace

```text
/usr/bin/git diff --check ec8eda1d39a5d8ae8ed043b4270da173b95f6652 149adb53f1d10b090e31a5f458e8c2ffcd3ca458 -- tldw_chatbook Tests scripts
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.103s`.

Full literal receipt: [latest-source-whitespace.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-source-whitespace.json); stdout/stderr: [latest-source-whitespace.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-source-whitespace.log). SHA256 JSON `f318ba38cdd8a80716fca691203eba1ca8cea906189370da27c26180801302f6`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-merge-source-probe

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest_merge_source_probe.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `3.871s`.

Full literal receipt: [latest-merge-source-probe.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-merge-source-probe.json); stdout/stderr: [latest-merge-source-probe.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-merge-source-probe.log). SHA256 JSON `f3ba9145ac509ec03366fb1a7102c07184e5bb4cf549a2ee7b8b3615ec8746b3`; log `281e82eeb46e9a58e8306a859b2d653381828b13ddb6b2ee38f540489215da5c`.

Literal output is preserved in the linked log (including empty successful output).

### latest-committed-preview-format

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/format-preview-baseline.json --head 149adb53f1d10b090e31a5f458e8c2ffcd3ca458
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.691s`.

Full literal receipt: [latest-committed-preview-format.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-committed-preview-format.json); stdout/stderr: [latest-committed-preview-format.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-committed-preview-format.log). SHA256 JSON `a960b6b8fd928de6a06c8209096a4bf336961f43ded091d1796a4be39be0745a`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-telemetry-owners

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q --basetemp=/private/tmp/console-latest-telemetry-01a0fa6c Tests/Chat/test_provider_rate_limits.py Tests/Chat/test_console_provider_gateway.py Tests/Utils/test_egress.py Tests/UI/test_console_cost_chip_screen.py Tests/UI/test_console_spend_projection.py Tests/UI/test_console_context_controls.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `1`. Elapsed: `159.341s`.

Full literal receipt: [latest-telemetry-owners.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-telemetry-owners.json); stdout/stderr: [latest-telemetry-owners.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-telemetry-owners.log). SHA256 JSON `61875183a82467f9b4f988caba5ae96eb5087f1a213993f0fdd10e69eb4bef26`; log `db4f1112339c7312a0dd8f1908f2c3894d41f8b9c5f2befc791fd2897cba9fc1`.

Literal output excerpt:

```text
FAILED Tests/UI/test_console_cost_chip_screen.py::test_excluded_historical_media_keeps_next_send_priced[failed]
FAILED Tests/UI/test_console_cost_chip_screen.py::test_excluded_historical_media_keeps_next_send_priced[assistant]
FAILED Tests/UI/test_console_cost_chip_screen.py::test_excluded_historical_media_keeps_next_send_priced[missing-data]
FAILED Tests/UI/test_console_cost_chip_screen.py::test_excluded_historical_media_keeps_next_send_priced[nonvision]
FAILED Tests/UI/test_console_cost_chip_screen.py::test_excluded_historical_media_keeps_next_send_priced[unaccepted]
FAILED Tests/UI/test_console_cost_chip_screen.py::test_context_cost_refresh_does_not_materialize_attachment_payloads
FAILED Tests/UI/test_console_cost_chip_screen.py::test_next_send_estimate_gates_staged_evidence_on_session
FAILED Tests/UI/test_console_cost_chip_screen.py::test_cost_tooltip_ends_its_info_with_the_providers_rate_limit
FAILED Tests/UI/test_console_spend_projection.py::test_nonempty_tracker_failure_is_unavailable_but_true_empty_is_zero
61 failed, 601 passed in 153.55s (0:02:33)
```

### latest-spend-context-probe

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest_spend_context_probe.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `1`. Elapsed: `0.041s`.

Full literal receipt: [latest-spend-context-probe.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-spend-context-probe.json); stdout/stderr: [latest-spend-context-probe.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-spend-context-probe.log). SHA256 JSON `160105b6e8c7a25cc8d336fcd93b515a59ce809ba5c826b070b747ec31ddd9fd`; log `183e6ffd869aed7a384144d8719a883725893841de1646d6aa24a1fab7277eb3`.

Literal output is preserved in the linked log (including empty successful output).

### latest-spend-context-probe-green

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest_spend_context_probe.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `2.442s`.

Full literal receipt: [latest-spend-context-probe-green.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-spend-context-probe-green.json); stdout/stderr: [latest-spend-context-probe-green.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-spend-context-probe-green.log). SHA256 JSON `331a43f50f8f7a619c3fe723f2776675d357d662ca943f567aa7bbb4f1ae8599`; log `73eced3add5148c6ecb491c3c8ad2b1df1e7478098a7e350f997b5f547c6b02d`.

Literal output is preserved in the linked log (including empty successful output).

### latest-telemetry-failure-controls

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q --basetemp=/private/tmp/console-latest-telemetry-controls-01a0fa6c Tests/Chat/test_console_provider_gateway.py::test_resolve_for_send_normalizes_scheme_less_llamacpp_base_url_before_http Tests/Chat/test_console_provider_gateway.py::test_resolve_for_send_all_chat_api_handlers_are_console_supported Tests/Chat/test_console_provider_gateway.py::test_owned_http_client_uses_generous_generation_read_timeout Tests/Chat/test_console_provider_gateway.py::test_owned_http_client_survives_agent_bridge_style_loop_swap Tests/Chat/test_console_provider_gateway.py::test_active_http_client_concurrent_swap_never_leaves_client_bound_to_wrong_loop Tests/Chat/test_console_provider_gateway.py::test_console_send_keeps_each_mistral_credential_on_its_own_endpoint 'Tests/Chat/test_console_provider_gateway.py::test_console_persisted_explicit_keyless_llamacpp_sends_no_authorization[llama_cpp]' 'Tests/Chat/test_console_provider_gateway.py::test_console_persisted_explicit_keyless_llamacpp_sends_no_authorization[local_llamacpp]' Tests/Chat/test_console_provider_gateway.py::test_console_llamacpp_explicit_stored_source_reaches_probe_and_chat Tests/Chat/test_console_provider_gateway.py::test_console_send_honors_configured_anthropic_base_url Tests/Chat/test_console_provider_gateway.py::test_console_send_default_anthropic_url_unchanged_when_unconfigured Tests/Chat/test_console_provider_gateway.py::test_auxiliary_status_less_local_failure_is_a_bad_request_not_an_outage Tests/Chat/test_console_provider_gateway.py::TestGatewayExchangeCapture::test_openai_stop_closes_transport_through_gateway_boundary Tests/Chat/test_console_provider_gateway.py::test_custom_endpoint_openai_compatible_resolves_entry_url Tests/Chat/test_console_provider_gateway.py::test_custom_endpoint_declared_env_key_flows_to_resolution Tests/Chat/test_console_provider_gateway.py::test_custom_endpoint_stored_key_flows_to_resolution Tests/Chat/test_console_provider_gateway.py::test_custom_endpoint_resolution_carries_raw_selected_provider Tests/Utils/test_egress.py::test_create_default_session_returns_a_default_timeout_session Tests/Utils/test_egress.py::test_default_session_applies_default_timeout_when_get_omits_one Tests/Utils/test_egress.py::test_default_session_applies_default_timeout_to_post_too Tests/Utils/test_egress.py::test_explicit_timeout_keyword_wins_over_default Tests/Utils/test_egress.py::test_explicit_timeout_none_is_respected_not_overridden Tests/Utils/test_egress.py::test_explicit_positional_timeout_on_request_wins_over_default Tests/Utils/test_egress.py::test_create_default_session_honours_config_default Tests/Utils/test_egress.py::test_explicit_factory_timeout_overrides_config Tests/UI/test_console_cost_chip_screen.py::test_cost_chip_shows_dollar_figure_after_priced_send Tests/UI/test_console_cost_chip_screen.py::test_editing_earlier_history_alerts_the_warm_cache_chip Tests/UI/test_console_cost_chip_screen.py::test_projected_delta_estimator_skipped_when_warm_without_break_reason Tests/UI/test_console_cost_chip_screen.py::test_reverting_the_edit_clears_the_alert Tests/UI/test_console_cost_chip_screen.py::test_reverting_system_prompt_edit_with_ttl_remaining_returns_to_warm Tests/UI/test_console_cost_chip_screen.py::test_system_prompt_revert_after_genuine_ttl_lapse_still_reports_expired Tests/UI/test_console_cost_chip_screen.py::test_build_console_cost_state_returns_none_without_native_session Tests/UI/test_console_cost_chip_screen.py::test_staged_evidence_changes_next_send_but_not_current_spend Tests/UI/test_console_cost_chip_screen.py::test_build_console_cost_state_includes_fleet_token_spend Tests/UI/test_console_cost_chip_screen.py::test_sync_cost_chip_hides_the_chip_when_state_is_none Tests/UI/test_console_cost_chip_screen.py::test_ttl_timer_expires_the_chip_and_stops_itself Tests/UI/test_console_cost_chip_screen.py::test_fingerprint_recompute_is_skipped_while_streaming Tests/UI/test_console_cost_chip_screen.py::test_cost_chip_press_opens_the_breakdown_modal Tests/UI/test_console_cost_chip_screen.py::test_cost_chip_state_isolated_across_session_tabs Tests/UI/test_console_cost_chip_screen.py::test_build_console_cost_state_includes_a_survivors_post_turn_spend Tests/UI/test_console_cost_chip_screen.py::test_blank_draft_has_exact_zero_current_and_no_next_send Tests/UI/test_console_cost_chip_screen.py::test_first_send_draft_changes_next_send_without_changing_current Tests/UI/test_console_cost_chip_screen.py::test_live_draft_increases_context_fullness Tests/UI/test_console_cost_chip_screen.py::test_followup_draft_does_not_change_completed_current_spend Tests/UI/test_console_cost_chip_screen.py::test_console_next_send_token_estimate_counts_context_not_just_draft Tests/UI/test_console_cost_chip_screen.py::test_historical_text_without_a_draft_is_not_sendable Tests/UI/test_console_cost_chip_screen.py::test_seeded_leading_assistant_greeting_counts_in_context_not_current Tests/UI/test_console_cost_chip_screen.py::test_idle_draft_edit_burst_coalesces_one_cost_refresh Tests/UI/test_console_cost_chip_screen.py::test_active_edit_cancels_an_already_armed_idle_cost_timer Tests/UI/test_console_cost_chip_screen.py::test_keyboard_send_cancels_idle_refresh_before_run_starts Tests/UI/test_console_cost_chip_screen.py::test_predispatch_echo_stays_in_context_but_not_current Tests/UI/test_console_cost_chip_screen.py::test_dispatched_echo_keeps_context_full_and_current_frozen 'Tests/UI/test_console_cost_chip_screen.py::test_excluded_historical_media_keeps_next_send_priced[failed]' 'Tests/UI/test_console_cost_chip_screen.py::test_excluded_historical_media_keeps_next_send_priced[assistant]' 'Tests/UI/test_console_cost_chip_screen.py::test_excluded_historical_media_keeps_next_send_priced[missing-data]' 'Tests/UI/test_console_cost_chip_screen.py::test_excluded_historical_media_keeps_next_send_priced[nonvision]' 'Tests/UI/test_console_cost_chip_screen.py::test_excluded_historical_media_keeps_next_send_priced[unaccepted]' Tests/UI/test_console_cost_chip_screen.py::test_context_cost_refresh_does_not_materialize_attachment_payloads Tests/UI/test_console_cost_chip_screen.py::test_next_send_estimate_gates_staged_evidence_on_session Tests/UI/test_console_cost_chip_screen.py::test_cost_tooltip_ends_its_info_with_the_providers_rate_limit Tests/UI/test_console_spend_projection.py::test_nonempty_tracker_failure_is_unavailable_but_true_empty_is_zero
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `1`. Elapsed: `318.226s`.

Full literal receipt: [latest-telemetry-failure-controls.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-telemetry-failure-controls.json); stdout/stderr: [latest-telemetry-failure-controls.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-telemetry-failure-controls.log). SHA256 JSON `03ca2254e47a2a9bdbc2960196585cca1a7572e7cbaef8d75d39900afae008cd`; log `c0c4748ede9ebe9acbcb6c608fc86b16b5f57c96e222597516ffd6b49236228b`.

Literal output excerpt:

```text
FAILED Tests/Chat/test_console_provider_gateway.py::test_console_llamacpp_explicit_stored_source_reaches_probe_and_chat
FAILED Tests/Chat/test_console_provider_gateway.py::test_custom_endpoint_openai_compatible_resolves_entry_url
FAILED Tests/Chat/test_console_provider_gateway.py::test_custom_endpoint_declared_env_key_flows_to_resolution
FAILED Tests/Chat/test_console_provider_gateway.py::test_custom_endpoint_stored_key_flows_to_resolution
FAILED Tests/Chat/test_console_provider_gateway.py::test_custom_endpoint_resolution_carries_raw_selected_provider
FAILED Tests/UI/test_console_cost_chip_screen.py::test_fingerprint_recompute_is_skipped_while_streaming
FAILED Tests/UI/test_console_cost_chip_screen.py::test_keyboard_send_cancels_idle_refresh_before_run_starts
FAILED Tests/UI/test_console_cost_chip_screen.py::test_predispatch_echo_stays_in_context_but_not_current
FAILED Tests/UI/test_console_cost_chip_screen.py::test_dispatched_echo_keeps_context_full_and_current_frozen
14 failed, 47 passed, 2 warnings in 308.28s (0:05:08)
```

### latest-telemetry-format-baseline

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/format_ratchet.py snapshot --base 149adb53f1d10b090e31a5f458e8c2ffcd3ca458 --output .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-telemetry-format-baseline.json --path Tests/Chat/test_console_provider_gateway.py --path Tests/Utils/test_egress.py --path Tests/UI/test_console_cost_chip_screen.py --path Tests/UI/test_console_spend_projection.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.884s`.

Full literal receipt: [latest-telemetry-format-baseline.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-telemetry-format-baseline.json); stdout/stderr: [latest-telemetry-format-baseline.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-telemetry-format-baseline.log). SHA256 JSON `748b6ed492527e13008b4feb124e01e17d2120c9a7cc4a89263e96b828027767`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-telemetry-remaining-controls

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q --basetemp=/private/tmp/console-latest-telemetry-controls2-01a0fa6c Tests/Chat/test_console_provider_gateway.py::test_resolve_for_send_normalizes_scheme_less_llamacpp_base_url_before_http Tests/Chat/test_console_provider_gateway.py::test_resolve_for_send_all_chat_api_handlers_are_console_supported Tests/Chat/test_console_provider_gateway.py::test_console_send_keeps_each_mistral_credential_on_its_own_endpoint 'Tests/Chat/test_console_provider_gateway.py::test_console_persisted_explicit_keyless_llamacpp_sends_no_authorization[llama_cpp]' 'Tests/Chat/test_console_provider_gateway.py::test_console_persisted_explicit_keyless_llamacpp_sends_no_authorization[local_llamacpp]' Tests/Chat/test_console_provider_gateway.py::test_console_llamacpp_explicit_stored_source_reaches_probe_and_chat Tests/Chat/test_console_provider_gateway.py::test_custom_endpoint_openai_compatible_resolves_entry_url Tests/Chat/test_console_provider_gateway.py::test_custom_endpoint_declared_env_key_flows_to_resolution Tests/Chat/test_console_provider_gateway.py::test_custom_endpoint_stored_key_flows_to_resolution Tests/Chat/test_console_provider_gateway.py::test_custom_endpoint_resolution_carries_raw_selected_provider Tests/UI/test_console_cost_chip_screen.py::test_fingerprint_recompute_is_skipped_while_streaming Tests/UI/test_console_cost_chip_screen.py::test_keyboard_send_cancels_idle_refresh_before_run_starts Tests/UI/test_console_cost_chip_screen.py::test_predispatch_echo_stays_in_context_but_not_current Tests/UI/test_console_cost_chip_screen.py::test_dispatched_echo_keeps_context_full_and_current_frozen
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `1`. Elapsed: `43.298s`.

Full literal receipt: [latest-telemetry-remaining-controls.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-telemetry-remaining-controls.json); stdout/stderr: [latest-telemetry-remaining-controls.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-telemetry-remaining-controls.log). SHA256 JSON `289794dbf7f1cc81ac5db94e1f5a84d3059a8e3b144bd546ef66d25488077978`; log `940a43fe443d68d994e16c40169380ae1120e585a86cbb9f0db00cbf76d7ec48`.

Literal output excerpt:

```text
ERROR    root:app.py:1513 ChaChaNotesDB (CharactersRAGDB) instance not found/assigned in app.__init__.
ERROR    tldw_chatbook.diagnostics.console:persistent_diagnostics.py:256 event=console_send_stage app_version=0.2.3 attempt_id=eed37f0cc98b4ce28ad25d6fe3f04040 component=console duration_ms=784 error_category=internal exception_type=CancelledError phase=provider_resolution python_version=3.12.11 sqlite_version=3.49.1 status=failed
ERROR    root:app.py:1513 ChaChaNotesDB (CharactersRAGDB) instance not found/assigned in app.__init__.
ERROR    tldw_chatbook.diagnostics.console:persistent_diagnostics.py:256 event=console_send_stage app_version=0.2.3 attempt_id=d17b1c3db5354087b6cf245a1fa369cd component=console duration_ms=951 error_category=internal exception_type=CancelledError phase=provider_resolution python_version=3.12.11 sqlite_version=3.49.1 status=failed
FAILED Tests/Chat/test_console_provider_gateway.py::test_resolve_for_send_all_chat_api_handlers_are_console_supported
FAILED Tests/Chat/test_console_provider_gateway.py::test_console_send_keeps_each_mistral_credential_on_its_own_endpoint
FAILED Tests/UI/test_console_cost_chip_screen.py::test_keyboard_send_cancels_idle_refresh_before_run_starts
FAILED Tests/UI/test_console_cost_chip_screen.py::test_predispatch_echo_stays_in_context_but_not_current
4 failed, 10 passed in 35.70s
```

### latest-telemetry-format-baseline-capture

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/format_ratchet.py snapshot --base 149adb53f1d10b090e31a5f458e8c2ffcd3ca458 --output .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/format-latest-telemetry-baseline.json --path Tests/Chat/test_console_provider_gateway.py --path Tests/Utils/test_egress.py --path Tests/UI/test_console_cost_chip_screen.py --path Tests/UI/test_console_spend_projection.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `1.241s`.

Full literal receipt: [latest-telemetry-format-baseline-capture.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-telemetry-format-baseline-capture.json); stdout/stderr: [latest-telemetry-format-baseline-capture.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-telemetry-format-baseline-capture.log). SHA256 JSON `931ada543e267cbc1b1539a4b43d27d516a79dcb24d0babfc84eb3d2d3360ba0`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-gateway-fixture-probe

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest_gateway_fixture_probe.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `2.199s`.

Full literal receipt: [latest-gateway-fixture-probe.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-gateway-fixture-probe.json); stdout/stderr: [latest-gateway-fixture-probe.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-gateway-fixture-probe.log). SHA256 JSON `5c29ceb226100e46a38f230349b503e04c130e85d3f886aaab03c5dc27d2972e`; log `2f36cb67e820f6b59eff6dbef55dbf86e4a9565bca0dfe1b4350b72752551adf`.

Literal output is preserved in the linked log (including empty successful output).

### latest-bootstrap-create-owners

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q --basetemp=/private/tmp/console-latest-bootstrap-create-01a0fa6c Tests/test_real_profile_guard.py Tests/Chat/test_console_chat_create_confirm.py Tests/Chat/test_console_chat_create_integration.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `82.301s`.

Full literal receipt: [latest-bootstrap-create-owners.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-bootstrap-create-owners.json); stdout/stderr: [latest-bootstrap-create-owners.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-bootstrap-create-owners.log). SHA256 JSON `ac2f7340a38105c362be111293522c30bf4e9c73eb6211e5482843071565c92e`; log `9bf7e0fb354c4a67efcc3945da357ffdd69d53a4a12346632130e5d9960a2700`.

Literal output excerpt:

```text
SKIPPED [1] Tests/test_real_profile_guard.py:108: no os.setxattr here
SKIPPED [1] Tests/test_real_profile_guard.py:108: no os.removexattr here
115 passed, 2 skipped in 79.17s (0:01:19)
```

### latest-format-changed-hunks

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest_format_hunks.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `1.814s`.

Full literal receipt: [latest-format-changed-hunks.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-format-changed-hunks.json); stdout/stderr: [latest-format-changed-hunks.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-format-changed-hunks.log). SHA256 JSON `c655c5ee720462b214a6fae907569bfdbf7089518f71bf5f83e515b76098237b`; log `1eaf67fe26c92c191f729ca206cf99fd0bd03bbfa6ccd418da94efcc2d5d4fb1`.

Literal output is preserved in the linked log (including empty successful output).

### latest-amended-gateway-egress-spend-owners

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q --basetemp=/private/tmp/console-latest-amended-gateway-egress-spend-01a0fa6c Tests/Chat/test_console_provider_gateway.py Tests/Utils/test_egress.py Tests/UI/test_console_spend_projection.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `1`. Elapsed: `66.281s`.

Full literal receipt: [latest-amended-gateway-egress-spend-owners.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-amended-gateway-egress-spend-owners.json); stdout/stderr: [latest-amended-gateway-egress-spend-owners.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-amended-gateway-egress-spend-owners.log). SHA256 JSON `359714dd548c1747431e96428a70f085e81e84dde23a3a8e3cb0c7a1f699942d`; log `86474ab71706ad7f6a41e032a6f5cdb48bf22f365964659df843b76c827159d9`.

Literal output excerpt:

```text
ERROR Tests/UI/test_console_spend_projection.py::test_admitted_historical_media_makes_forecast_unavailable
ERROR Tests/UI/test_console_spend_projection.py::test_zero_input_price_is_known_and_empty_draft_is_dash
ERROR Tests/UI/test_console_spend_projection.py::test_invalid_draft_never_promises_a_next_send_charge
ERROR Tests/UI/test_console_spend_projection.py::test_pending_media_or_unknown_forecast_inputs_are_unavailable[True-1000-3.0]
ERROR Tests/UI/test_console_spend_projection.py::test_pending_media_or_unknown_forecast_inputs_are_unavailable[False-None-3.0]
ERROR Tests/UI/test_console_spend_projection.py::test_pending_media_or_unknown_forecast_inputs_are_unavailable[False-1000-None]
ERROR Tests/UI/test_console_spend_projection.py::test_unknown_current_pricing_does_not_hide_known_next_send_forecast
ERROR Tests/UI/test_console_spend_projection.py::test_nonempty_tracker_failure_is_unavailable_but_true_empty_is_zero
ERROR Tests/UI/test_console_spend_projection.py::test_idle_refresh_coalesces_and_uses_late_bound_callbacks
555 passed, 19 errors in 63.10s (0:01:03)
```

### latest-refresh-contract-probe

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest_refresh_contract_probe.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `2.336s`.

Full literal receipt: [latest-refresh-contract-probe.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-refresh-contract-probe.json); stdout/stderr: [latest-refresh-contract-probe.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-refresh-contract-probe.log). SHA256 JSON `fafe485620ad58ad42f43fc2d93003ccc1490342eba5d288ec8f772807fc5ba8`; log `92aec100315c3ea2aae5d9ac65dcedb58e48bd618c3046d8cb353ffc32b8b2de`.

Literal output is preserved in the linked log (including empty successful output).

### latest-format-amended-hunks

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest_format_hunks.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `1.93s`.

Full literal receipt: [latest-format-amended-hunks.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-format-amended-hunks.json); stdout/stderr: [latest-format-amended-hunks.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-format-amended-hunks.log). SHA256 JSON `d5d1abb245bdf78bc6914286ef5775d289b5024465b076ed12b07ad38e9d1670`; log `da0eb3df6ec90115c4eace749466b15b56d8f8e3eacf2cc53ffeafc8743e924f`.

Literal output is preserved in the linked log (including empty successful output).

### latest-cost-spend-focused-controls

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q --basetemp=/private/tmp/console-latest-cost-spend-focused-01a0fa6c Tests/UI/test_console_cost_chip_screen.py::test_keyboard_send_cancels_idle_refresh_before_run_starts Tests/UI/test_console_cost_chip_screen.py::test_refused_send_cancels_idle_refresh_and_next_idle_edit_rearms Tests/UI/test_console_cost_chip_screen.py::test_unaccepted_predispatch_echo_is_visible_but_excluded_from_context_and_current Tests/UI/test_console_cost_chip_screen.py::test_dispatched_echo_keeps_context_full_and_current_frozen Tests/UI/test_console_cost_chip_screen.py::test_active_edit_cancels_an_already_armed_idle_cost_timer Tests/UI/test_console_spend_projection.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `1`. Elapsed: `51.531s`.

Full literal receipt: [latest-cost-spend-focused-controls.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-cost-spend-focused-controls.json); stdout/stderr: [latest-cost-spend-focused-controls.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-cost-spend-focused-controls.log). SHA256 JSON `6e330ecb06d317fed34ae9022754fb2263bd3d6be18cf311d4d59d998e652047`; log `00831386f8fb8ba36ef0398000e43115951d86d5541a8a2bf95aa0ceaa3ed3e5`.

Literal output excerpt:

```text
ERROR    root:app.py:1513 ChaChaNotesDB (CharactersRAGDB) instance not found/assigned in app.__init__.
ERROR    tldw_chatbook.diagnostics.console:persistent_diagnostics.py:256 event=console_send_stage app_version=0.2.3 attempt_id=f81b8748beef4f539cda0be6db98b9c4 component=console duration_ms=1157 error_category=internal exception_type=CancelledError phase=provider_resolution python_version=3.12.11 sqlite_version=3.49.1 status=failed
FAILED Tests/UI/test_console_cost_chip_screen.py::test_keyboard_send_cancels_idle_refresh_before_run_starts
1 failed, 23 passed in 46.63s
```

### latest-working-preview-format

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/format-preview-baseline.json
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.544s`.

Full literal receipt: [latest-working-preview-format.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-working-preview-format.json); stdout/stderr: [latest-working-preview-format.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-working-preview-format.log). SHA256 JSON `7e24d44c9c817481280aa8a842c051baf365a4359dd2490116f0a2a58d804036`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-working-telemetry-format

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/format-latest-telemetry-baseline.json
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `2`. Elapsed: `1.506s`.

Full literal receipt: [latest-working-telemetry-format.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-working-telemetry-format.json); stdout/stderr: [latest-working-telemetry-format.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-working-telemetry-format.log). SHA256 JSON `ea8fc811ee7943b2eb66367f4c96644132077a04485f5f60a2a6ee71f4e83fa6`; log `cc595caa5c2cd0a30c770d9ea4bfbdff76c32895a42827d253a57ba7f9f2d172`.

Literal output is preserved in the linked log (including empty successful output).

### latest-working-main-format

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/format-ratchet-baseline.json
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `12.816s`.

Full literal receipt: [latest-working-main-format.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-working-main-format.json); stdout/stderr: [latest-working-main-format.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-working-main-format.log). SHA256 JSON `fc8aaded62de1b0469b70d96c4147b07b6dc46bc263ec285dbe23c7dce292544`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-format-debt-detail

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m ruff format --diff Tests/Chat/test_console_provider_gateway.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `1`. Elapsed: `0.23s`.

Full literal receipt: [latest-format-debt-detail.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-format-debt-detail.json); stdout/stderr: [latest-format-debt-detail.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-format-debt-detail.log). SHA256 JSON `88c03acb1767a27fb78b197b68c863b0766f238ef4a08a785a920f5714320d11`; log `4ea65e140ff3956483b74f70f54332209f09223dfcbddd5c704c7db025d644c7`.

Literal output is preserved in the linked log (including empty successful output).

### latest-keyboard-timer-trace

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q -p latest_timer_trace_plugin --basetemp=/private/tmp/console-latest-keyboard-trace-01a0fa6c Tests/UI/test_console_cost_chip_screen.py::test_keyboard_send_cancels_idle_refresh_before_run_starts
```

Environment: `{"PYTHONPATH": "/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration:/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook", "TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `1`. Elapsed: `18.101s`.

Full literal receipt: [latest-keyboard-timer-trace.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-keyboard-timer-trace.json); stdout/stderr: [latest-keyboard-timer-trace.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-keyboard-timer-trace.log). SHA256 JSON `cc23774eb2c1319b547ccc97a2d00b33051232d0bd5e6f66e783541079b25c20`; log `52aaf10d748bb19da255dd4365d8b2fd7dfb5539c9d7d46022d08c54d90280ca`.

Literal output excerpt:

```text
ERROR    root:app.py:1513 ChaChaNotesDB (CharactersRAGDB) instance not found/assigned in app.__init__.
ERROR    tldw_chatbook.diagnostics.console:persistent_diagnostics.py:256 event=console_send_stage app_version=0.2.3 attempt_id=65d6a7ddc26d453984067a55698d206e component=console duration_ms=1935 error_category=internal exception_type=CancelledError phase=provider_resolution python_version=3.12.11 sqlite_version=3.49.1 status=failed
FAILED Tests/UI/test_console_cost_chip_screen.py::test_keyboard_send_cancels_idle_refresh_before_run_starts
```

### latest-format-custom-endpoint-hunk

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m ruff format --range 11841-11859 Tests/Chat/test_console_provider_gateway.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.163s`.

Full literal receipt: [latest-format-custom-endpoint-hunk.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-format-custom-endpoint-hunk.json); stdout/stderr: [latest-format-custom-endpoint-hunk.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-format-custom-endpoint-hunk.log). SHA256 JSON `5c87f48dfd181f53e729cc3195801ae3194115a870b4efd575144d9b2946c986`; log `413a709001976657cb743636fd34e48fc41d8d302d665460d3f874b3313da127`.

Literal output is preserved in the linked log (including empty successful output).

### latest-working-telemetry-format-green

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/format-latest-telemetry-baseline.json
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.854s`.

Full literal receipt: [latest-working-telemetry-format-green.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-working-telemetry-format-green.json); stdout/stderr: [latest-working-telemetry-format-green.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-working-telemetry-format-green.log). SHA256 JSON `0b703e0035926b4dd6d66880153767641d6890ad7365f430cc4ff23ee685ea82`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-format-final-amended-hunks

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest_format_final_hunks.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `1.814s`.

Full literal receipt: [latest-format-final-amended-hunks.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-format-final-amended-hunks.json); stdout/stderr: [latest-format-final-amended-hunks.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-format-final-amended-hunks.log). SHA256 JSON `9f1f6f276159eaebeea77d10f9b813d4dec6decf3787bfeaeef243bc376768a6`; log `d9c5e73b5eef467f0729f95f67da0ac4bfb716f53a9d862461c564e12a4c7d51`.

Literal output is preserved in the linked log (including empty successful output).

### latest-keyboard-custody-controls

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q --basetemp=/private/tmp/console-latest-custody-controls-01a0fa6c Tests/UI/test_console_cost_chip_screen.py::test_keyboard_send_cancels_idle_refresh_before_run_starts Tests/UI/test_console_cost_chip_screen.py::test_runtime_custody_cancels_before_validating_and_leaves_other_chat_idle Tests/UI/test_console_cost_chip_screen.py::test_refused_send_cancels_idle_refresh_and_next_idle_edit_rearms
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `23.174s`.

Full literal receipt: [latest-keyboard-custody-controls.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-keyboard-custody-controls.json); stdout/stderr: [latest-keyboard-custody-controls.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-keyboard-custody-controls.log). SHA256 JSON `0d8dd16cfbb3d9b43a8a2c55d58e4e5bbc4b8a270a5e84e227bac4b4591ae434`; log `140184d1bb2a8a9bef67e1b85877e21f6a0557b1e45b7f3989b5b98607050495`.

Literal output excerpt:

```text
3 passed in 20.11s
```

### latest-final-working-fatal-ruff

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m ruff check --select E9,F63,F7,F82 Tests/Chat/test_console_provider_gateway.py Tests/UI/test_console_cost_chip_screen.py Tests/UI/test_console_spend_projection.py Tests/Utils/test_egress.py tldw_chatbook/Chat/console_runtime.py tldw_chatbook/UI/Screens/chat_screen.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.092s`.

Full literal receipt: [latest-final-working-fatal-ruff.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-working-fatal-ruff.json); stdout/stderr: [latest-final-working-fatal-ruff.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-working-fatal-ruff.log). SHA256 JSON `2daf1d87b0d6de173c6e9f2a37dcca058fc3c30501ec7f7fb2d3f430a2e10442`; log `82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18`.

Literal output excerpt:

```text
All checks passed!
```

### latest-final-working-source-whitespace

```text
git diff --check -- Tests tldw_chatbook
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.049s`.

Full literal receipt: [latest-final-working-source-whitespace.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-working-source-whitespace.json); stdout/stderr: [latest-final-working-source-whitespace.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-working-source-whitespace.log). SHA256 JSON `408baefe9c0f1de0b578f3e69cae106e65604121c31ee838fccfc9f173925214`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-final-working-telemetry-format

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/format-latest-telemetry-baseline.json
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.783s`.

Full literal receipt: [latest-final-working-telemetry-format.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-working-telemetry-format.json); stdout/stderr: [latest-final-working-telemetry-format.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-working-telemetry-format.log). SHA256 JSON `6a2602641908c4ee2718dcb631e34f0b120a9a90ac8dcbb51bfdb5f82e153d8e`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-final-working-main-format

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/format-ratchet-baseline.json
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `2`. Elapsed: `8.039s`.

Full literal receipt: [latest-final-working-main-format.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-working-main-format.json); stdout/stderr: [latest-final-working-main-format.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-working-main-format.log). SHA256 JSON `f100d5a1616dab7ca6c5f6da193296fdf46289d16ec14385dd479df6ba318824`; log `5c7ee617b4551a9b577c3a17061f6241e808e1b33073a0e977dda49abd7be8ac`.

Literal output is preserved in the linked log (including empty successful output).

### latest-runtime-format-correction

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest_runtime_format.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.182s`.

Full literal receipt: [latest-runtime-format-correction.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-runtime-format-correction.json); stdout/stderr: [latest-runtime-format-correction.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-runtime-format-correction.log). SHA256 JSON `14ffca4f73056a7bab20ae545b3b1d497a872ad2f8ba51885efc5cd91d4226c3`; log `5157c8d15ec8f3250c3ec2e6d39d57ec1597da3335348614cbb5584d0c5d6454`.

Literal output is preserved in the linked log (including empty successful output).

### latest-final-working-main-format-green

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/format-ratchet-baseline.json
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `7.81s`.

Full literal receipt: [latest-final-working-main-format-green.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-working-main-format-green.json); stdout/stderr: [latest-final-working-main-format-green.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-working-main-format-green.log). SHA256 JSON `fe12a7b3e911ff2e92c1db20068620c084b2fd87825abdac4ce77bc9247b852e`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-final-amended-telemetry-owners

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q --basetemp=/private/tmp/console-latest-final-amended-telemetry-01a0fa6c Tests/Chat/test_console_provider_gateway.py Tests/Utils/test_egress.py Tests/UI/test_console_cost_chip_screen.py Tests/UI/test_console_spend_projection.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `276.188s`.

Full literal receipt: [latest-final-amended-telemetry-owners.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-amended-telemetry-owners.json); stdout/stderr: [latest-final-amended-telemetry-owners.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-amended-telemetry-owners.log). SHA256 JSON `7bf3233ff0eae9af7bf0e8c47f83d366beb1591ba2c5fc0ffec6df88098fcba7`; log `d46d926450ca32777f0d4fe31ecf188e53ebb7784bc9707cea1c8ffe1b7b3794`.

Literal output excerpt:

```text
    warnings.warn(message, UserWarning, stacklevel=0)
611 passed, 2 warnings in 267.35s (0:04:27)
```

### latest-final-baseline21

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_prompt_queue_coordinator.py::test_close_tombstones_before_cancel_and_never_starts_next_prompt Tests/UI/test_console_runtime_ownership.py::test_runtime_owned_custody_tracks_only_lifetime_handles Tests/UI/test_console_runtime_ownership.py::test_runtime_tombstones_before_shutdown_and_disposes_via_to_thread Tests/UI/test_console_runtime_ownership.py::test_persistent_attach_sync_failure_has_bounded_backoff_and_resume_retry 'Tests/UI/test_console_runtime_ownership.py::test_reconciled_view_keeps_each_live_poll_reason_and_one_timer[wake]' Tests/UI/test_console_runtime_ownership.py::test_opening_console_during_a_headless_delivery_arms_the_poll Tests/UI/test_console_launch_wake.py::test_a_launch_into_console_delivers_without_stealing_the_active_tab Tests/UI/test_console_launch_wake.py::test_a_launch_with_no_marks_constructs_nothing_and_reads_once 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_invalid_ownership[bad_role]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_invalid_ownership[cross_conversation]' Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_considers_only_checkpoint_owners_on_the_selected_active_lineage 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_read_quarantines_malformed_or_mismatched_checkpoint_identity[assistant_message_id]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_state_cas_requires_every_expected_owner_predicate[assistant_state]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[terminal_content]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[sync_intent]' 'Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_failure_at_each_write_boundary_rolls_back[checkpoint_delete]' Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_terminal_settlement_is_atomic_and_returns_committed_proof Tests/Chat/test_console_agent_project_instructions.py::test_child_chain_uses_its_own_exact_first_request_budget Tests/Chat/test_console_agent_project_instructions.py::test_primary_token_omission_is_delivery_local_when_child_admits 'Tests/Chat/test_console_chat_fork.py::test_configuration_and_leaf_writers_block_fork_through_live_publication[active_leaf]' Tests/Agents/test_agent_chat_create_tools.py::test_new_chat_schema_shape --basetemp=/private/tmp/console-latest-final-baseline21-01a0fa6c
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `68.973s`.

Full literal receipt: [latest-final-baseline21.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-baseline21.json); stdout/stderr: [latest-final-baseline21.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-baseline21.log). SHA256 JSON `70d16793c026f32777a3b831c2568c65bb1ee4baca80a183b176ae7afc08d861`; log `2b34292b035e617cee09436f12d21fe1f89576dfa3d8b5c793723735fa6a99dc`.

Literal output excerpt:

```text
21 passed in 64.30s (0:01:04)
```

### latest-final-bootstrap-create-owners

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q --basetemp=/private/tmp/console-latest-final-bootstrap-create-01a0fa6c Tests/test_real_profile_guard.py Tests/Chat/test_console_chat_create_confirm.py Tests/Chat/test_console_chat_create_integration.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `107.17s`.

Full literal receipt: [latest-final-bootstrap-create-owners.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-bootstrap-create-owners.json); stdout/stderr: [latest-final-bootstrap-create-owners.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-bootstrap-create-owners.log). SHA256 JSON `60ffd648b6de0b0f235218947c649f8f7c15c930ae3d52d0ff2d78b5887f1f9f`; log `8c6944b3701404d6877bdd6922bac0a0621f7a43559521e0ceab2ccf2709a858`.

Literal output excerpt:

```text
SKIPPED [1] Tests/test_real_profile_guard.py:108: no os.setxattr here
SKIPPED [1] Tests/test_real_profile_guard.py:108: no os.removexattr here
115 passed, 2 skipped in 103.52s (0:01:43)
```

### latest-precommit-head

```text
git rev-parse HEAD
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.037s`.

Full literal receipt: [latest-precommit-head.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-precommit-head.json); stdout/stderr: [latest-precommit-head.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-precommit-head.log). SHA256 JSON `a02d5d1629e8ceed939526af7d7472691fae744f7d834dbc4586a0f3dc1571fd`; log `1cd0c4d27c8fe8d063ec65354638ce015227621eb7b3013bd4c611ad459a2ff5`.

Literal output is preserved in the linked log (including empty successful output).

### latest-precommit-index

```text
git diff --cached --name-only
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.048s`.

Full literal receipt: [latest-precommit-index.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-precommit-index.json); stdout/stderr: [latest-precommit-index.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-precommit-index.log). SHA256 JSON `79a0c8a6ce5b54631a3576690b1da81be3941f469a32eff4a98c2b1037168db8`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-precommit-owned-diff

```text
git diff --name-only
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.567s`.

Full literal receipt: [latest-precommit-owned-diff.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-precommit-owned-diff.json); stdout/stderr: [latest-precommit-owned-diff.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-precommit-owned-diff.log). SHA256 JSON `3865d896fdf07ce52c9cd3d4ca8dad6a49e6f2bdd42a62afc904f21287ce2be7`; log `d442f7c44e726146a811920d38dd7d60be7833666d8b3858c3f0e8a4e45c062c`.

Literal output is preserved in the linked log (including empty successful output).

### latest-owned-add

```text
git -c gc.auto=0 add -- Tests/Chat/test_console_provider_gateway.py Tests/UI/test_console_cost_chip_screen.py Tests/UI/test_console_spend_projection.py Tests/Utils/test_egress.py tldw_chatbook/Chat/console_runtime.py tldw_chatbook/UI/Screens/chat_screen.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.278s`.

Full literal receipt: [latest-owned-add.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-owned-add.json); stdout/stderr: [latest-owned-add.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-owned-add.log). SHA256 JSON `6c047161d17859d6fc7274f5c6e24b4f94c7408361b0fdfd104d6b4ae1eac1ac`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-owned-stage-proof

```text
git diff --cached --name-only
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.046s`.

Full literal receipt: [latest-owned-stage-proof.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-owned-stage-proof.json); stdout/stderr: [latest-owned-stage-proof.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-owned-stage-proof.log). SHA256 JSON `297e914cc8e175bb4408e9cee24a76f796d93eba4514b298a42296fb38a9181f`; log `d442f7c44e726146a811920d38dd7d60be7833666d8b3858c3f0e8a4e45c062c`.

Literal output is preserved in the linked log (including empty successful output).

### latest-owned-stage-whitespace

```text
git diff --cached --check -- Tests/Chat/test_console_provider_gateway.py Tests/UI/test_console_cost_chip_screen.py Tests/UI/test_console_spend_projection.py Tests/Utils/test_egress.py tldw_chatbook/Chat/console_runtime.py tldw_chatbook/UI/Screens/chat_screen.py
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.064s`.

Full literal receipt: [latest-owned-stage-whitespace.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-owned-stage-whitespace.json); stdout/stderr: [latest-owned-stage-whitespace.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-owned-stage-whitespace.log). SHA256 JSON `ee74cd82d1e1200a3dd54c7c8063486a687194480e702579ae3d89ab743fea77`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-owned-commit

```text
git -c gc.auto=0 commit -m 'fix(console): cancel idle spend refresh under turn custody'
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.186s`.

Full literal receipt: [latest-owned-commit.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-owned-commit.json); stdout/stderr: [latest-owned-commit.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-owned-commit.log). SHA256 JSON `72e03d8768c277275959eb254ee0a9c35d301f1e8c9ff1ec1b159b654a4c0ef7`; log `d823741a0ce67d9976007ddb4747abc407551394cccb13f2dc7500718b943f05`.

Literal output is preserved in the linked log (including empty successful output).

### latest-owned-commit-head

```text
git rev-parse HEAD
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.044s`.

Full literal receipt: [latest-owned-commit-head.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-owned-commit-head.json); stdout/stderr: [latest-owned-commit-head.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-owned-commit-head.log). SHA256 JSON `6fc25e217d829367800999ea13d1e2f77e4f5ec3cbb766003539a41f658af6c6`; log `d7c4dec4aa96970297dc344d86598a6da238edd7d6ec573732f449e497a45959`.

Literal output is preserved in the linked log (including empty successful output).

### latest-owned-final-status

```text
git status --short
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.274s`.

Full literal receipt: [latest-owned-final-status.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-owned-final-status.json); stdout/stderr: [latest-owned-final-status.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-owned-final-status.log). SHA256 JSON `34c5e9c5acaa8fdd1c240a65dd0d64d1f18fe8350b4f4684dab0006efc893408`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-final-committed-preview-format

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/format-preview-baseline.json --head cde073ed62d1e8f0d6163bac0b5f667c9fe7670f
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.711s`.

Full literal receipt: [latest-final-committed-preview-format.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-committed-preview-format.json); stdout/stderr: [latest-final-committed-preview-format.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-committed-preview-format.log). SHA256 JSON `17845a785ec852a3de030179b3c36d432decb6539a2a334fbf95e65979c56723`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-final-committed-source-whitespace

```text
git diff --check ec8eda1d39a5d8ae8ed043b4270da173b95f6652 cde073ed62d1e8f0d6163bac0b5f667c9fe7670f -- tldw_chatbook Tests scripts
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `0.186s`.

Full literal receipt: [latest-final-committed-source-whitespace.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-committed-source-whitespace.json); stdout/stderr: [latest-final-committed-source-whitespace.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-committed-source-whitespace.log). SHA256 JSON `e84f476dcd557187e6c670b75b91893120efaae9d69fdb3edaa94b86b2c2b9cd`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-final-committed-telemetry-format

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/format-latest-telemetry-baseline.json --head cde073ed62d1e8f0d6163bac0b5f667c9fe7670f
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `2.218s`.

Full literal receipt: [latest-final-committed-telemetry-format.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-committed-telemetry-format.json); stdout/stderr: [latest-final-committed-telemetry-format.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-committed-telemetry-format.log). SHA256 JSON `07d6202ff8420a37fe720b99843a103d85ef539f05b4da8d4b930ea96707fd4a`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).

### latest-final-committed-source-static

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest_final_static.py cde073ed62d1e8f0d6163bac0b5f667c9fe7670f
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `15.741s`.

Full literal receipt: [latest-final-committed-source-static.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-committed-source-static.json); stdout/stderr: [latest-final-committed-source-static.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-committed-source-static.log). SHA256 JSON `154b3195ba36d5e9fd14356386e15af430afc237fb6d2d47ad0e4d7dadf34bb7`; log `93997c957432de4d231681500a5a4124a21d18adad40287be01d8c2283bdfcac`.

Literal output is preserved in the linked log (including empty successful output).

### latest-final-committed-main-format

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/format-ratchet-baseline.json --head cde073ed62d1e8f0d6163bac0b5f667c9fe7670f
```

Environment: `{"TLDW_TEST_GC_EVERY": "1"}`. Cwd: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Exit: `0`. Elapsed: `22.666s`.

Full literal receipt: [latest-final-committed-main-format.json](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-committed-main-format.json); stdout/stderr: [latest-final-committed-main-format.log](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/latest-final-committed-main-format.log). SHA256 JSON `eeb1107299ed25b3e3cae570400cc32731d2e8cb3e9630feca905589bc1a75fa`; log `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Literal output is preserved in the linked log (including empty successful output).


Recorded 53 owned follow-up command receipts. Executable capture/repair/trace/static/formatter sources and all detail JSON files are direct children of this same plan scratch directory, so the controller byte-exact archiver can retain them.
