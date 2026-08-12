---
id: TASK-15270
title: >-
  Console test apps mount with a config that silently defaults every
  turn-context setting
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 09:00'
updated_date: '2026-08-11 18:55'
labels:
  - tests
  - console
  - test-infrastructure
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while triaging task-15210, and raised rather than absorbed because the blast radius (90+ modules) is far larger than that task's scope.

`_build_test_app` patches `load_settings` to a small synthetic dict that carries no `[console]` or `[chat_defaults]` section, and production *correctly* refuses to refresh a settings snapshot that lacks the disk-load markers. So every mounted Console test sees a `ConsoleTurnExecutionContext` whose `rag_defaults` are frozen at their defaults, no matter what the test believes it configured. Measured directly during 15210: `get_cli_setting=True` while the app_config key was `MISSING` and the context read `auto_retrieve_on_send: False`.

The live app is unaffected — `app.py` does `self.app_config = load_settings()`, whose result carries both the toggle and the markers (verified) — so this is a test-harness defect, not a product one.

**Why it is worth its own task: it can hollow out a passing test.** `test_send_proceeds_when_auto_retrieve_fails` was green **vacuously** for exactly this reason — auto-retrieval never fired, so the deliberately-exploding backend was never called, and the test only ever asserted that an ordinary send works. It was repaired in 15210 (`exploding_search.await_count == 1`), but any other test whose subject reads through the turn-context snapshot has the same exposure and would look green while asserting nothing.

The fix is presumably to make `_build_test_app` produce a config the snapshot will accept (markers included), so a test that sets a `[chat_defaults]` value gets that value. That change will likely flip some currently-green tests to red — those are the ones that were never really testing their subject, and each needs the 15210 treatment rather than a revert.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Console test that configures a `[chat_defaults]`/`[console]` setting sees that value through the turn-context snapshot, instead of a silent default
- [x] #2 Tests that turn red once the config is honoured are triaged individually (real regression vs assertion that was never exercised), not reverted wholesale
- [x] #3 A guard makes the vacuous-pass shape detectable: a test whose subject is "X still works when Y fails" asserts that Y was actually attempted
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate the mechanism: which marker sections production requires (ChatScreen._CONSOLE_LIVE_CONFIG_MARKER_SECTIONS), where the turn-context snapshot is built (ConsoleSessionController._build_console_turn_execution_context, off _provider_readiness_app_config), and what _build_test_app patches.
2. TDD: add Tests/UI/test_console_harness_config_honesty.py pinning (a) markers present, (b) a persisted [chat_defaults] value in app_config, (c) that value reaching the mounted turn context. Confirm all three RED first.
3. Fix at the seam: source the factory's app_config from the real load_settings() -- hermetic because the root conftest sandboxes TLDW_CONFIG_PATH/HOME per test -- keeping only the deliberate synthetic overrides (tldw_api.base_url, first_run.setup_completed). One config file, one truth, for the snapshot and every later refresh.
4. Measure the blast radius: run all 241 modules that use _build_test_app, diff against a pre-fix baseline, and triage every newly-red test individually (real product bug vs test that never ran its own scenario).
5. AC#3: make the vacuous-pass shape checkable -- a registry-backed helper that fails a test whose 'Y fails' double was never called -- and apply it to the high-value cases.
6. Mutation-check the harness fix; run the keep-green set with READ counts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`_build_test_app` now boots the app on the **real sandbox config** instead of a
three-key synthetic dict: `build_test_app_config()` calls `load_settings()` (the
same call `app.py` makes) and merges only the two deliberate test overrides
(`tldw_api.base_url`, `first_run.setup_completed`). That is hermetic because the
root conftest re-points `TLDW_CONFIG_PATH`/`HOME`/`XDG_*` at a per-test sandbox
before anything imports the app, so `load_settings()` reads the *test's* config
file -- the same file `save_setting_to_cli_config` writes and `get_cli_setting`
reads.

**Mechanism.** `ChatScreen._provider_readiness_app_config` re-sources from
`load_settings()` only when the snapshot it was handed looks disk-loaded --
`_CONSOLE_LIVE_CONFIG_MARKER_SECTIONS = ("general", "logging")` -- and otherwise
honours it verbatim, deliberately, so an injected test config is never
overwritten by the developer's real one. The synthetic dict carried neither
marker and no `[chat_defaults]`/`[console]` section, so
`ConsoleSessionController._build_console_turn_execution_context` read defaults
forever. Sourcing the snapshot from `load_settings()` satisfies the markers AND
makes the refresh return the very same dict object, so snapshot and refresh can
no longer disagree.

**Blast radius, measured.** All 241 modules that use the factory (6,016 tests)
were run with the fix, then the 51 modules containing failures were re-run with
the legacy config behind a temporary env switch. 31 tests were genuinely
newly-red across that whole sweep. Two tests went from red to green:
`test_happy_path_stages_then_send_consumes` and
`test_send_proceeds_when_auto_retrieve_fails` -- the vacuous pair from
task-15210, which now actually fire retrieval.

**Per-test outcome, re-measured at the rebase.** The 10 modules this branch
touches were run whole against the branch and again against `origin/dev`, so
"newly-red" here means red on the branch AND green on dev -- 9 tests (the wider
sweep's 31 covers 241 modules, most of them outside this branch). Five are fixed
outright: the two Ollama ones were opening a real socket to 127.0.0.1:11434
because a URL-based provider now has an endpoint and readiness takes the
task-191 live-probe branch (stubbed at the seam the sibling probe tests already
stub, rather than marked `allow_network`); the two privacy ones asserted
absolute sensitive-field counts over the WHOLE config, so they now scrub the
config first and count only what they arranged; and
`..._tolerates_missing_provider_values` was patching `_provider_setting_values`,
a seam this path does not read -- it calls `_provider_setting_values_mapping()`
-- so the stub was dead and the "missing values" premise had been supplied for
free by the empty config.

The remaining four are xfail(strict=True) against two filed product bugs, not
silenced: **task-15673** (navigation preselect applies the provider, which is
itself what marks the category dirty, so the deferred apply hits its
unsaved-changes guard and drops the model -- measured: dirty=False before,
dirty=True and model unchanged after) and **task-15511** (a completed
non-streaming send leaves `run_state` at IDLE, stable across ~0.8s so not a
race; and an inline image row vanishes on the pixels -> graphics toggle).

Note for anyone re-running this: five further failures in
`test_settings_configuration_hub.py` and `test_console_workbench_contract.py`
reproduce on `origin/dev` untouched and are NOT from this change.

Recurring cause among the newly-red: a test arranged state through a seam the
shipping app does not read, and only the empty config let it look effective.
`_chat_images_config` prefers `COMPREHENSIVE_CONFIG_RAW`, so
`app_config["chat"] = {...}` never reached production; the llama.cpp URL tests
relied on `provider_config_key(...) or "llama_cpp"`, a fallback the template's
`provider = "OpenAI"` removes; two "cli config fallback" tests asserted the
absence of a `[library]` section rather than arranging it.

**AC#3.** `Tests/fixtures/required_doubles.py` + an autouse root-conftest fixture:
a double built through `exploding_double(...)` is registered, and a test that
never calls it FAILS. A fully automatic guard is impractical and the module says
why -- `Mock(side_effect=SomeError)` carries two opposite intents in this suite
(the failure a test claims to survive vs a tripwire that must never fire, e.g.
`test_console_native_chat_flow.py`'s "server identity must not use local cards"),
and nothing on the object distinguishes them. Detection of unregistered ones is
therefore a separate, reporting-only audit (`TLDW_AUDIT_UNCALLED_DOUBLES`).

Modified: `Tests/UI/app_factory.py`, `Tests/conftest.py`,
`Tests/UI/test_console_auto_rag_on_send.py`,
`Tests/UI/test_console_native_chat_flow.py`,
`Tests/UI/test_console_character_avatar.py`, `Tests/UI/test_library_shell.py`,
`Tests/UI/test_settings_configuration_hub.py`,
`Tests/UI/test_settings_footer_hints.py`,
`Tests/UI/test_console_workbench_contract.py`,
`Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`.
Added: `Tests/fixtures/required_doubles.py`,
`Tests/test_required_doubles_guard.py`,
`Tests/UI/test_console_harness_config_honesty.py`.
<!-- SECTION:NOTES:END -->
