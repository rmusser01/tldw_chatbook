---
id: TASK-38949
title: "Console: a cold-config deferral entry outlives the config fix until another owner reads warm config"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-06 22:29
labels:
- follow-up
- console
- performance
priority: low
references:
- tldw_chatbook/UI/Console_Modules/console_spend_projection.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
- Tests/UI/test_console_projection_reads_off_loop.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-36511.4's notes list it as "Not changed (proposed follow-up)". The helper is unchanged since 983f363c8c (#3385).

- `defer_cold_live_read` keeps its per-owner episode set, `_console_cold_read_deferrals`, in `tldw_chatbook/UI/Console_Modules/console_spend_projection.py:1170-1188`. It clears the set (`:1183`) only when some owner calls it while config is warm, checks for re-entry at `:1185`, and adds the owner at `:1187`.
- The credential poll returns before its locked section unless its pre-gate changed (`tldw_chatbook/UI/Screens/chat_screen.py:10779`; the call is at `:10819`). The other callers are `config_sync.py:289` and `console_spend_projection.py:1220`.
- So once a broken config is fixed, an owner's entry can stay until another owner happens to read warm config. If config goes cold again before that, the owner's first cold read reads live on the UI loop instead of deferring. TASK-36511.4 and TASK-36511 measured that read at 147-222 ms for the poll and about 80 ms for the identity refresh.
- The existing test covers only the path where a warm call clears the set (`Tests/UI/test_console_projection_reads_off_loop.py:491`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Once config is warm again, whichever owner or path warmed it, a new cold episode makes each owner's first cold read defer even when no owner was called while config was warm; a unit test in Tests/UI/test_console_projection_reads_off_loop.py is red on dev
- [x] #2 The bound still holds: with an unparseable config.toml each owner defers at most once per episode, and the existing deferral tests pass, including test_an_unparseable_config_leaves_an_idle_console_idle and test_the_credential_poll_defers_a_cold_read_once_and_then_stays_idle
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the gap on the base: an owner's entry in `_console_cold_read_deferrals` survives a warm period that no owner observed, so its first read in the next cold episode reads live.
2. Give `config.py` a count of successful cache installs (the bootstrap cache's two install sites and the settings cache's one). A failed load installs nothing, so the count stands still for as long as an unparseable config.toml stays cold.
3. Key each screen's deferral set on that count in `defer_cold_live_read`: a set recorded under an older count belongs to a past episode and is cleared. The existing warm-read clear stays.
4. Add a unit test to `Tests/UI/test_console_projection_reads_off_loop.py` that warms config through another path between two cold episodes and checks that each owner defers again; show it red on the base.
5. Run the existing deferral tests, the covering tests, the size ratchets and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** A cold episode now ends when config loads warm, whoever loads it, not only when an owner happens to be called while config is warm.
- `config.py` keeps `_CONFIG_CACHE_INSTALLS`, which takes a fresh number at every successful cache install. The installs happen at the bootstrap cache's two sites (`_load_cli_config_bootstrap_unlocked`, `_install_bootstrap_cache_from_raw`) and at the settings cache's one (`_load_settings_uncached`). Each of those runs only after a successful load, so a failed load takes no number.
- Review r11c-rev1 pointed out that the three sites run under different locks: the settings site under `_SETTINGS_CACHE_LOCK`, the other two under the config-file lock. So the numbers come from `next()` on one module-level `itertools.count`, which is atomic. Two installs on different threads never share a number. If they assign out of order, the value can only change, which at worst starts one extra episode.
- `defer_cold_live_read` records the count beside each screen's deferral set (`_console_cold_read_installs`). A set recorded under an older count belongs to a past episode, so it is cleared before the check. The existing clear on a warm call stays.
- Why a counter rather than an identity token: a broken config clears both caches to `None` on each failed reload, so a token built from the cache objects reads the same before and after a fix-then-break cycle, and the bug stays.

**Evidence.**
- New test: `Tests/UI/test_console_projection_reads_off_loop.py::test_a_warm_load_no_owner_saw_still_ends_the_cold_episode`. Three owners defer once and then read live within one episode. Then a load warms config while no owner is called, and config goes cold again. On the base it was red on the assertion: `an owner read cold config live on its first call of a new episode: [False, False, False]`. With the fix it is green, and the bound holds in the new episode as well.
- Real loader, through run.sh on a scratch profile (a probe, not committed): a good load moved the count 4 -> 6, two loads of an unparseable config.toml left it at 6, and the fixed file moved it to 8. So an unparseable file keeps one episode for as long as it stays broken.
- Review r11c-rev1 found the `config.py` half untested: deleting the three increments kept the suite green. New test: `test_only_a_successful_config_load_counts_as_a_cache_install`, in the same file under `@private_profile_test`. It runs real loads of its child's own profile: a good load must move the count, an unparseable file must leave it unchanged over two loads, and the fixed file must move it again.
  - With the three increments deleted it was red: `a good load installed nothing: 0 -> 0`.
  - With the `itertools.count` source it is green.
- AC#2: the whole file passed at `-n 4` (18 passed, load 42-44), including `test_an_unparseable_config_leaves_an_idle_console_idle` (the mounted, real-file bound) and `test_the_credential_poll_defers_a_cold_read_once_and_then_stays_idle`.
- Covering files (`test_console_response_rules.py`, `test_console_cost_chip_screen.py`, `test_console_credential_poll_gate.py`, `test_console_poll_tab_source_boundaries.py`, `Tests/test_config_read_fastpath_task21124.py`, `Tests/test_config_warm_read_no_handshake_task32804.py`), `-n 4`: 79 passed and 2 failed on head at load 44. The same command failed 9 on the base at load 48-50, including both of head's nodes. Those are load flakes in the response-rule and cost-chip files.

**Files:** `tldw_chatbook/config.py`, `tldw_chatbook/UI/Console_Modules/console_spend_projection.py`, `Tests/UI/test_console_projection_reads_off_loop.py`.
<!-- SECTION:NOTES:END -->
