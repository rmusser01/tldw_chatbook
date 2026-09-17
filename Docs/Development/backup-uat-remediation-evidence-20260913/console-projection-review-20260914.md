# Console synchronous projection review

APPROVED. Independent read-only review of the working-tree projection delta; no reviewer edits.

Frozen sources verified:
- tldw_chatbook/UI/Screens/chat_screen.py: ceefda4d3139d09c31cb6544e1368745765cecacbfaf40528ee110dff87892d7
- Tests/Backup_Recovery/test_console_projection_config_lifetime.py: 9d7a1b4ed9a6d92df1d4282b2b5bd33ac3d78fabe970e0c8f829fa1abac27bd6
- Full tested composition hashes, including separately reviewed config/Canvas lock corrections: /private/tmp/uat-console-projection-independent-hashes.json.

The async tick still performs the same contiguous rail derivation, settings summary, and control update, in that order. Only these synchronous calls share the checked config operation. The next settings-recovery/readiness/mode calls and all awaits remain outside that scope. The post-await rail derivation remains fresh, independently observing a real intervening config write.

AST comparison confirms the extracted _run_console_config_sync body is identical to the prior _sync_console_control_bar operation/error/retry body after substituting the callback for its one rendering call. The original _sync_console_control_bar_under_config implementation is also AST-identical. Direct control refresh delegates to the same helper; no guarded call is replaced with unchecked data or cached authority.

Native entry pause still returns False before any of the three projection calls. The existing whole-sync replay flag and one coalesced timer recompute current state after resume/canceled intent; the async pass exits before its later native reads. UI-only failures leave config drain usable, while genuine nested native writes retain sticky failure; exact body error identity and changed-selector refusal remain enforced. There is no new worker join, await, lock, deadline, or runtime authority in this projection change.

Independent verification: all five new native tests passed in 6.85s, /private/tmp/uat-console-projection-independent.log. They exercise real checked owner operations, same-owner rail/summary/control reads, operation retirement before await, ungrouped mode-bar reads, fresh post-await values, UI-error versus native-write-error drain behavior, selector refusal, and actual native maintenance entry followed by fresh replay.

Author initial combined run: 46 passed / 1 new-test expectation failure. The failure omitted the existing later mode-bar control derivation; only the assertion was corrected to include that unchanged call and explicitly require its two owner observations to be None. Independent final five-case run covers that correction. The author's remaining 42 cases were already green; no claim is made here of a new complete 47-case rerun or Windows performance acceptance.

No actionable findings remain in this bounded projection delta. Separate installed/mounted and platform performance measurements remain the parent's acceptance work.
