# PR 3166 historical CI inventory

Machine-readable exact nodes and job links: inventory.json. Historical head9f07ca956d; current source basecc6006fcf5. This agent ran no tests/apps. Current pass reports below are supplied by root.

Current root-selected static run:124cases,123pass/1fail. Historical app string lookup/layer/CSS/sheet/theme selections passed. Only new PluginRestoreAdoptionModal lacked classification; root added its existing90%-viewport shell reason, rerun pending. Historical runtime/preimport/UI behaviors await fresh selected runs. Current after-close control fails before gate_show_tabs; root investigating. CI43743285529 pending.

Next bounded boot/preimport batch, keeping subprocess/UI profiling exclusive:

```
python -m pytest Tests/Performance/test_ui_ready_module_census.py::test_ui_ready_module_census_stays_at_the_pinned_size Tests/Performance/test_boot_worker_census.py::test_boot_worker_and_thread_starts_stay_within_the_allowlist Tests/UI/test_console_first_send_preimport.py::test_the_idle_pass_imports_the_first_send_modules Tests/UI/test_console_first_send_preimport.py::test_the_idle_pass_warms_every_screen_route_before_the_first_send_modules Tests/UI/test_console_first_send_preimport.py::test_the_send_guard_counts_a_module_resident_at_ui_ready_as_warm Tests/UI/test_console_first_send_preimport.py::test_a_harness_send_imports_nothing_lazy_beyond_the_list Tests/UI/test_console_runtime_ownership.py::test_runtime_owned_custody_tracks_only_lifetime_handles
```

Historical boot-ready failure is in test_ui_ready_module_census.py, not test_app_import_weight.py. It asserted1033resident modules against1031. Boot worker failures were(_load_recovered_images,console-recovered-images) and(_reconcile_console_after_attach,console-attach-reconciliation). PerfGuard additionally failed CSS271>270; no historical latency budget assertion failed.

Standalone markup census contract:

```
python scripts/check_markup_interpolation.py
```

AST scan compares source occurrences to committed census; exit0 clean,1 new/grown occurrence,2 malformed census. No app/DB execution. Do not use --write for validation. Historical offending Kanban error notifications now explicitly set markup=False; fresh check pending.

Historical modal inventory omission included ConsoleCodeReviewModal, NewCardModal, PluginAuthorityDialog, PluginAuthorityResetModal, PluginReviewModal, PluginStatusModal, PluginVariableDialog; all seven now have documented source classifications. Current plugin focus border uses ds-active-fg instead of historical ds-focus-accent. Preimport source now treats received_dispatch as resident and includes console_plugin_context/hook_permissions. Custody expected fields now include private composer witnesses. These source changes alone do not establish passing behavior.
