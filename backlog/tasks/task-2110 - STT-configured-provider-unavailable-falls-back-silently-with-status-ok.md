---
id: TASK-2110
title: 'STT: configured provider unavailable falls back silently with status=ok'
status: Done
assignee:
  - '@zcode'
created_date: '2026-08-03'
labels: [speech, diagnostics]
dependencies: []
priority: high
---

## Description (the why)

During the hands-free live gate (2026-08-03), a worktree venv without parakeet_mlx made
dictation silently substitute faster-whisper base for the user's configured
`[transcription] default_provider = "parakeet-mlx"`. The startup diagnostic reported
`event=speech_stack_available model=base provider=faster-whisper status=ok` — status OK
while running a provider the user explicitly did not choose. The user's standard is
parakeet, whisper explicit-only; a silent substitution is indistinguishable from working
config until transcription quality collapses. The degraded-VAD path already has an honesty
surface; provider fallback needs the same.

## Implementation Plan (the how)

Added 2026-10-04 after premise verification at origin/dev `cddc89d3e7`:

Premise split by AC. AC#2 (first-capture user notice) is already satisfied —
`VoiceProviderOverridden` (console_voice_input.py, emitted on first capture
start when `EffectiveConfig.was_overridden`, once-per-controller) is handled in
`UI/Console_Modules/dictation.py:1535` into a once-per-app-run warning naming
both providers; landed 2026-07-29 in `0bdb0c38cc`, before this task was filed.
AC#3 holds by the same latch (nothing fires when nothing was substituted).
The LIVE gap is AC#1 only: `app.py` `on_mount`'s `speech_stack_available`
diagnostic derives `status` solely from `find_spec("webrtcvad")` and records
only the effective provider, so a substitution reports `status=ok`.

1. Red tests first in `Tests/App/test_app_lifecycle_events.py` (same capture
   pattern as `test_mounting_the_app_records_app_started`): boot the real app
   with `persist_event` captured, `console_voice_input.resolve` monkeypatched
   (the call site imports it inside `on_mount`, so the patch lands), and
   `importlib.util.find_spec` controlled for webrtcvad. Cases: substituted
   provider + VAD ok -> `status=fallback` naming both providers;
   substituted + VAD missing -> both facts recorded; healthy path ->
   `status=ok`, no `configured_provider` field.
2. Add `configured_provider` to `_TOKEN_FIELDS` in
   `Utils/persistent_diagnostics.py` (the schema is additive by precedent:
   TASK-1240/18908/32533/32920 entries in the same frozenset).
3. Rework the `on_mount` diagnostic: status precedence
   `fallback`/`degraded-fallback` when `was_overridden` (VAD ok/missing),
   else today's `ok`/`degraded`; `configured_provider` field and WARNING level
   only on substitution.
4. Regenerate `Docs/security/production-diagnostic-inventory.json` via
   `scripts/check_persistent_diagnostic_inventory.py --write` (the per-call
   digest hashes the call's source segment, so new kwargs change it) and
   review the diff is app.py-only.
5. Targeted runs: the new tests, `Tests/App/test_app_lifecycle_events.py`,
   `Tests/Chat/test_console_voice_input.py`,
   `Tests/Architecture/test_persistent_diagnostic_inventory.py`, and the
   persistent-log tests touching `persist_event`.

## Acceptance Criteria (the what)

- [x] #1 When the configured STT provider cannot be used and another is substituted, the
      startup diagnostic reports a degraded/fallback status naming BOTH the configured and
      the substituted provider (not `status=ok`). [Implemented here: `status=fallback`
      (VAD present) / `degraded-fallback` (VAD missing) with `provider=<substitute>` +
      `configured_provider=<configured>` at WARNING; pinned by two red-first tests.]
- [x] #2 The first capture after such a substitution surfaces a user-visible notice naming the
      configured provider that was unavailable and what is being used instead.
      [Already satisfied before this task: `VoiceProviderOverridden` (emitted once per
      controller, `console_voice_input.py:1574`) handled in
      `UI/Console_Modules/dictation.py:1535` into a once-per-app-run warning naming both
      providers; landed `0bdb0c38cc` 2026-07-29, verified present at this base.]
- [x] #3 A configured provider that is available produces no new notice (no noise in the
      healthy path). [Pinned by `test_speech_stack_diagnostic_healthy_path_is_unchanged`
      (`status=ok`, no `configured_provider` field) and by the notice's
      `was_overridden` latch, which cannot fire when nothing was substituted.]

## Implementation Notes

Implemented 2026-10-10 in the wave5-g3 worktree (base = origin/dev `cddc89d3e7`;
work resumed in a second session after the first was rate-limit-killed mid-task —
the partial tree was re-verified from scratch before being trusted: every diff was
re-read, the production A/B re-run, and the boot-test marker rationale re-proved).

Approach (production, `tldw_chatbook/app.py` `on_mount`): the
`speech_stack_available` diagnostic now derives a substitution flag from the
resolver's `EffectiveConfig.was_overridden`. Status precedence: substituted ->
`fallback` (webrtcvad present) or `degraded-fallback` (both degradations); not
substituted -> today's `ok`/`degraded` unchanged. `configured_provider` is
recorded beside `provider` (the substitute actually running) and the record is
emitted at WARNING (severity mirrors the first-capture toast) only on
substitution; the healthy path's field shape is byte-identical to before.

Schema (`tldw_chatbook/Utils/persistent_diagnostics.py`): `configured_provider`
added to `_TOKEN_FIELDS` so the configured name is held to the same bounded-token
contract as `provider`. Additive by precedent (TASK-1240/18908/32533/32920 in the
same frozenset); no existing field changed.

Tests (`Tests/App/test_app_lifecycle_events.py`): three new async tests under
`@pytest.mark.bootstrap_profile` booting the real app via `_build_test_app`,
capturing `persist_event` (`set_app_global`), monkeypatching
`console_voice_input.resolve` (the `on_mount` call site imports it at call time)
and controlling only the `webrtcvad` `find_spec` (delegating all other lookups):
substitution + VAD ok -> `status=fallback` naming both providers; substitution +
VAD missing -> both facts (`status != degraded`, `configured_provider` present);
healthy -> `status=ok`, `model=provider-default`, no `configured_provider`.
The pre-existing boot test gained the `bootstrap_profile` marker with the
rationale in-file (standalone runs fail with RecoveryRequired otherwise).

Evidence (worktree venv, Python 3.12.13; `-p no:xdist` throughout):

- `python -m pytest Tests/App/test_app_lifecycle_events.py -q --tb=short` ->
  4 passed (3 new + marked boot test).
- Production A/B (proves the tests pin the fix, not the fixture): copied the two
  dirty production files aside, `git checkout HEAD -- tldw_chatbook/app.py
  tldw_chatbook/Utils/persistent_diagnostics.py`, re-ran the speech_stack tests
  -> 2 failed (both substitution cases; KeyError on `status`/`configured_provider`
  shape), 1 passed (healthy path, as expected — old code also satisfies it);
  restored the fix, re-ran -> 4 passed.
- Marker A/B (re-proving the in-file rationale): removing only the first test's
  `bootstrap_profile` marker -> `test_mounting_the_app_records_app_started`
  FAILED standalone; with the marker -> passes.
- `python -m pytest Tests/Chat/test_console_voice_input.py
  Tests/Utils/test_persist_event.py -q` -> 160 passed (run jointly with the
  file below).
- `python -m pytest Tests/test_persistent_log_is_not_empty.py -q` -> 7 failed.
  All 7 are PRE-EXISTING at base: identical 7 fail at clean HEAD with the fix
  reverted (`_configure_private_file_logging(root)` returns False — an
  environment/private-file-logging failure, no speech or diagnostics frame).
  Not owned by this task.
- `python -m pytest Tests/Architecture/test_persistent_diagnostic_inventory.py
  -q` -> 74 passed, 1 skipped (the skip is the documented unfetchable
  TASK-15743 pinned commits, in-file rationale).
- `python scripts/check_persistent_diagnostic_inventory.py` -> exit 0
  ("655 owners, 1445 TASK-492 calls, 56 TASK-31551 calls, 7626 TASK-494 calls,
  16 sink files"); `--write` produced ZERO diff to
  `Docs/security/production-diagnostic-inventory.json` — the manifest's digest
  for this call is insensitive to the new kwargs, so plan step 4's regeneration
  was a verified no-op at this base.

AC#2 needed no code: the first-capture notice landed in `0bdb0c38cc` (2026-07-29),
before this task was filed; verified by reading the emit site
(`console_voice_input.py:1574`, latched once per controller on
`was_overridden`) and the handler (`UI/Console_Modules/dictation.py:1535`,
once-per-app-run warning naming both providers, markup-escaped).

ADR required: no. The change is direct implementation under the existing
ADR-029 local-private-data boundary (persistent logs are metadata-only; the new
`configured_provider` is a bounded metadata token, not user/model content) —
same additive-frozenset precedent as TASK-1240/18908/32533/32920, none of which
minted an ADR. No storage/schema migration, no sync policy, no provider
boundary change.

Modified files: `tldw_chatbook/app.py`,
`tldw_chatbook/Utils/persistent_diagnostics.py`,
`Tests/App/test_app_lifecycle_events.py`, this task file.

Post-rebase evidence (rebased onto origin/dev `c2d6dcc301`, 411 commits;
clean replay of both branch commits, zero conflicts; branch diff vs dev is
exactly the 5 intended files):

- `python -m pytest Tests/CI/test_backlog_task_id_uniqueness.py -q` -> 3 passed.
- `python -m pytest Tests/App/test_app_lifecycle_events.py -q` -> 4 passed.
- `python -m pytest Tests/Chat/test_console_voice_input.py
  Tests/Utils/test_persist_event.py -q` -> 160 passed.
- `python -m pytest Tests/Architecture/test_persistent_diagnostic_inventory.py
  -q` -> 73 passed, 1 skipped, 1 FAILED
  (`test_reviewed_diagnostic_changes_are_metadata_only`). That failure is
  PRE-EXISTING AT ORIGIN/DEV ITSELF, not this branch: the test file,
  `session.py` and `console_turn_context.py` are byte-identical to
  origin/dev here (empty diff); dev's `42e130778e` (TASK-33620.15) moved the
  pinned "Console turn context: persona policy rules resolution failed"
  diagnostic from `session.py` into `console_turn_context.py:93` while the
  `REVIEWED_METADATA_ONLY_DIAGNOSTICS` row still targets `session.py`. Same
  stale-pin class TASK-32199 Item 2 already fixed once for other rows.
  Reported for the owner; not fixed here (outside this task's ACs).
- `python scripts/check_persistent_diagnostic_inventory.py` -> exit 0
  ("662 owners, 1449 TASK-492 calls, 56 TASK-31551 calls, 7656 TASK-494
  calls, 16 sink files"); the branch changes
  `Docs/security/production-diagnostic-inventory.json` by zero lines.
- `python -m pytest Tests/test_persistent_log_is_not_empty.py -q` -> the
  same 7 pre-existing failures as at the pre-rebase base;
  `_configure_private_file_logging` was untouched by all 411 upstream
  commits, so the mechanism is the same environment-bound one A/B'd above.
