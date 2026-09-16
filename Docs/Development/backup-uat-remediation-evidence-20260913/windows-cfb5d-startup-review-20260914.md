# Windows cfb5 startup: independent causal review

**The startup criterion still fails. The final candidate stack identifies a real config writer holding the locks needed by composition; it does not establish a deadlock or justify a guard/policy change.** No code edits, app boots or new runs.

Run34919030019/job104222787333 tested exact cfb5d35e7354b171215e17db261228f48832463a. Saved log SHA256 independently matches `e8362cdf4c5bd0d42644ea4c81715c125edfbe878ed3d2ed68061087a86ae6ce`.

Verified outcomes: **31 non-UI passed10.46s; four separately executed primary full-app cases hit their unchanged60s timeouts** (claim/recompose, external-copy geometry, supported-width llama.cpp and llamafile). The paired timeout banners are not eight failures. The separate same-node profiled comparison on pinned dev4631b60f **passed54.51s with2warnings**; current candidate timed out. All8 profile records (4dev/4candidate) have source_matches=true and observer_errors empty. These are source-selection observations, not installed-wheel receipts. No observer error does not imply no app error.

## Exact current lock owner and pending work

Final candidate timeout has11 threads (log2133–2398); every stack was inspected:

- **MainThread6800**: CompactModelBar.compose46 → config.get_cli_providers_and_models8719 → decorated load_settings → config_participants.operation343, waiting in the existing timed acquisition loop. That loop acquires REBUILD then FILE; the stack alone does not say which iteration is pending.
- **asyncio_3/2856**: actual `_save_console_rail_preferences:12997` → save_setting8407/save_settings8352 → apply mutation7894 → literal transaction7781 → `_publish_runtime_config_unlocked:6722` → forced `load_settings:1806` → `_load_settings_uncached:3414` → Utils.paths.get_user_data_dir130 → config.get_user_data_dir9141/config_data operation → raw scope713 → acquire_storage/_acquire_storage907 → admission_authority121 → qualified_for259 → pinned-directory context cleanup → native Windows open/open_handle545.
- Six executor workers (asyncio0,1,2,4,5,6) are idle in thread-pool queue waits.
- ui-stall-persist6688 waits for queue data; chatbook-storage-admission5428 waits on its Event; backup-startup-profile9928 waits on its sampling Event. None of these stacks identifies another active config/native owner.

Source establishes the ownership: `_apply_literal_settings_transaction_locked` enters `_config_write_lock` at config7649. `_config_write_lock:6383` owns REBUILD, FILE and the config interprocess lock throughout its yield; the inner runtime reload has not exited that scope. Therefore thread2856 holds both Python config locks while MainThread waits. The stack is past raw config write and into runtime cache publication. This call uses publish_noop=False; unchanged mutations return before this publication, so the observed call took the changed-settings route. It is not a failure merely to defer UI work behind a live writer.

The worker is performing real native directory qualification/open work, not shown waiting for MainThread or another worker. This one sample proves current progress stage, not progress rate, how long the native call or lock has been held, completion, or absence of all possible OS contention. There is no reciprocal wait/cycle evidence. It also does not identify the particular rail preference key or originating UI event; the same worker entry serves default/fallback seeding and explicit preference updates (screen13294/13896). Do not label this an unnecessary/default/no-op save without the missing origin and changed-field evidence.

## Causal limits and next step

This differs from05e's terminal main-thread summary/Canvas admission stack: cfb5 is waiting during CompactModelBar composition behind a rail writer reloading config. It does not show redundant same-view binding remains or idle Canvas polling as the lock holder. The last bounded profile includes272 acquire_storage calls/35.853s inclusive and1392 registry calls/17.842s inclusive, while the constructor completed in22.253s inclusive. These totals overlap and represent incomplete candidate work; they cannot be added, compared as whole-work ratios with completed dev, or used to assign a measured writer duration.

**No product patch is justified from this artifact alone.** In particular do not remove the required idle watcher, bypass config_data/native checks, cache policy, move atomic publication outside its locks, or change deadlines. If further evidence is authorized, the smallest useful probe is the actual rail-save→runtime-publication path with fixed entry/exit timestamps and a fixed origin category (initial seed/fallback migration/explicit update), changed boolean and config-lock wait/hold durations; retain native operations and record no preference values or paths. That would decide whether avoidable caller work or an unnecessarily broad held operation is actually present. Another full startup aggregate without writer origin/hold timing would not settle it.

Evidence: `/private/tmp/uat-cfb5d-windows-startup-job.log`, supplied summary, and extracted bounded11-thread/profile/accounting JSON `/private/tmp/uat-cfb5d-startup-independent-summary.json`. Current native36 success is separate verification and does not turn this startup failure into a pass.
