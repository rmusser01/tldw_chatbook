# TASK-35582 demonstrated CI guard repair

Delivery tree `/Users/macbook-dev/.codex/worktrees/task-35582-delivery/tldw_chatbook`, base candidate `cee6fdcf26`. Source is frozen for root's focused review. Changes remain unstaged; root owns final commit/push/rebinding and the independent modal-inventory correction.

## RED and causal import evidence

- Exact three original CI nodes reproduced locally: **3 failed in 24.04s** (`ci-guards-red.log`). UI-ready **1049 > 1031**, ancestor-scoped bare-type selectors **271 > 270**, registry preimport **559 > 558**. No ratchet, pinned snapshot, gate or baseline was edited.
- First mount edge: `ConsoleChatController.__init__` eagerly constructed `ConsoleCodeReviewCoordinator`, which imported the review source/service/store graph. Deferring this dropped UI-ready **1049 → 1035**.
- The next causal mount edge was `_register_fleet_wake`: merely reading `runs_db.automatic_work` loaded `DB.automatic_work`, `Agents.automatic_work_budget` and `Agents.code_review_models`. Registering only once the lazy review owner exists dropped **1035 → 1032**. The actual authority methods remain the same bound methods.
- The final new resident was `Chat.console_trace_continuation`, pulled into already-resident `console_trace_final_values` only for annotations and two non-null witness type checks. TYPE_CHECKING annotations and branch-local imports dropped **1032 → 1031**.
- One immutable `git archive e3d07afa65927b9d477fe9f13af619550da9e005` control (no new worktree) passed the exact census at **1031/1031** in 11.75s (`ci-census-base-control.log`), establishing that no unrelated baseline cleanup was needed.
- Registry preimport's first edge was the module-scope `code_review_models` import in `Widgets/settings_agents_panel.py`; moved to the panel's actual construction/load/edit/save use sites.
- The extra CSS selector was `#code-review-actions Button`. All five existing buttons now carry `code-review-action`; the rule subject is `Button.code-review-action`. Its token values, layout, handlers and permission behavior are unchanged. Normal CSS build changed only its generated counterpart (`ci-guard-css-build.log`).

## Exact diff scope

1. `Chat/console_chat_controller.py`: lazily cache the same native review coordinator; initialize its same owner/callbacks on first use; defer review-ledger lookup/registration until that owner exists; attach it through the existing fleet-wake registration seam.
2. `Chat/console_trace_final_values.py`: move custody/witness annotation imports behind TYPE_CHECKING and witness runtime imports into the two existing non-null validation branches. Every exact-type check remains.
3. `Widgets/settings_agents_panel.py`: move review-model imports off module import into existing interaction methods/constructor.
4. `Widgets/Console/console_code_review_modal.py`: explicit selector class on the same five buttons.
5. `css/widget_defaults_scoped.tcss`: normal generated selector counterpart.
6. `Tests/Chat/test_console_code_review_authority.py`: one direct lazy-owner regression. It failed before the fix (`ci-lazy-owner-red.log`, 1 failed in 1.19s) and passed afterwards.

No source, role, permission, continuation-budget or two-child contract changed. No new wrapper framework, provider/model request, credential read, new worktree, commit or push.

## GREEN evidence

The first exact-gates/direct-controls command was **11 passed, 1 failed in 39.43s**, with only the remaining UI-ready census red (`ci-guards-green.log`). Both CSS and preimport gates were green; all nine direct owner/team/modal/continuation controls passed. After the ledger edge was deferred, the two owner/continuation controls again passed; the sole census remainder was 1032 (`ci-census-ledger-green.log`). After deferring that final witness import, the census passed **1031/1031, 1 passed in 8.42s** (`ci-census-final-green.log`). One direct zero-network Capture On two-round witness control passed **1 passed in 2.08s** (`ci-witness-control.log`).

Exact green gate nodes:

- `Tests/Performance/test_ui_ready_module_census.py::test_ui_ready_module_census_stays_at_the_pinned_size`
- `Tests/Performance/test_textual_css_fastpath.py::test_ancestor_scoped_bare_type_rule_count_is_a_ratchet`
- `Tests/Performance/test_screen_preimport_payload_budget.py::test_preimport_pass_payload_stays_within_budget`

Exact green direct controls:

- `Tests/Chat/test_console_code_review_authority.py::test_controller_defers_review_owner_until_first_use`
- `Tests/Chat/test_console_code_review_continuation.py::test_default_two_child_schedule[2-True-expected1]`
- `Tests/Widgets/test_settings_code_review_team.py::test_load_rename_disable_delete_preserves_order_ids_and_diagnostics`
- `Tests/Widgets/test_settings_code_review_team.py::test_team_add_focus_reorder_remove_save_cancel_and_preserve_settings`
- `Tests/Widgets/test_settings_code_review_team.py::test_keyboard_team_controls_at_narrow_width[size0]`
- `Tests/Widgets/test_settings_code_review_team.py::test_keyboard_team_controls_at_narrow_width[size1]`
- `Tests/Widgets/test_settings_code_review_team.py::test_contextless_team_load_and_cancel_do_not_read_or_write_profile`
- `Tests/UI/test_console_code_review_reports.py::test_native_modal_literal_pages_focus_and_safe_close[size0]`
- `Tests/UI/test_console_code_review_reports.py::test_native_modal_literal_pages_focus_and_safe_close[size1]`
- `Tests/Chat/test_console_dispatch_continuation_handoff.py::test_normal_console_multiple_native_rounds[capture-on-2-1]`

All commands borrowed the existing original worktree venv with explicit delivery-tree PYTHONPATH and PYTHONDONTWRITEBYTECODE; pytest used `-q -p no:cacheprovider`. No full suite or prior 99-node integration rerun. `git diff --check` passed. Test processes emitted pytest cleanup warnings for old temporary garbage-* directories; no cleanup attempted. The UI-ready headroom warning is unchanged versus base (1031/1031).

## Frozen SHA-256

- `Tests/Chat/test_console_code_review_authority.py`: `511238e9015baffcbb87ffbbd4d398d68d1e685425536c0f0a79eb352026d61e`
- `tldw_chatbook/Chat/console_chat_controller.py`: `ea051c4a88cd9c5bbff5299b0eb9100151b9e646e1c1b50300a92a19bdec56c8`
- `tldw_chatbook/Chat/console_trace_final_values.py`: `789514d097315ad84ff4b06266aa4a70728aa77897f0086576320394b93d274b`
- `tldw_chatbook/Widgets/Console/console_code_review_modal.py`: `5e23bfa2e553c346cb32313715a65acdba81ca7a33a9308332d6a802b4af8d8e`
- `tldw_chatbook/Widgets/settings_agents_panel.py`: `b6deb6d9be355e8c4b44f74380b4e33d45dcc3e4ce031b06d173f81405400df0`
- `tldw_chatbook/css/widget_defaults_scoped.tcss`: `c5b518c3c7c20359d5c9e665ca59f34c2c40e564878736b3587b9209b297cde1`

## Remaining work / risk

No remaining demonstrated blocker in the assigned three gates. Root owns the separate modal-inventory test correction, focused review, final staging/commit/push and private QA source-pin rebind. No paid qualification was run; the one allowance remains user-pending. Broad requalification and baseline cleanup were skipped.

## Single scoped review finding: concurrent first use

The focused reviewer identified one new lazy-initialization race: two actual worker callers could construct different native review owners, while `_review_ledger` caused only the earlier owner's bound methods to remain registered. A deterministic constructor-overlap regression reproduced the exact split before the fix: cache/first caller owner A; second caller and all three ledger callbacks owner B. **RED: 1 failed in 1.12s**, `ci-owner-race-red.log`.

Smallest correction: one existing stdlib `threading.RLock` per controller protects construction, callback setup and first attachment as one critical section, plus the connected review-ledger lookup/unregister/reset/register block in `_register_fleet_wake`. Reentrancy preserves the same getter/registration calls and exact bound authority methods. No alternative publication path, new module/import, permission change or allowance change.

**GREEN: 3 passed in 6.24s**, `ci-owner-race-green.log`:

- `Tests/Chat/test_console_code_review_authority.py::test_concurrent_first_review_owner_matches_registered_ledger`
- `Tests/Chat/test_console_code_review_authority.py::test_controller_defers_review_owner_until_first_use`
- `Tests/Chat/test_console_code_review_continuation.py::test_default_two_child_schedule[2-True-expected1]`

The new control asserts one constructor, identical returned/cache owner identity, and all registered callback `__self__` identities equal that owner. `git diff --check` passed. No census rerun: threading/RLock were already imported and used in this controller; the added instance lock adds no resident module. No other completed check/cohort/review was rerun. The frozen hashes above now include this correction. Source frozen again for root's single-finding closure review.
