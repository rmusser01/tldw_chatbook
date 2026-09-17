# Providers & Models edit, Revert, Save and return — TASK-32724

2026-09-17 UTC, `feat/component-pattern-library`, based on `cb263af0d5`.

The existing staged form passes this bounded review. No production code, visual
token, stylesheet or service boundary needed changing. Four new production-CSS
journey cases and a private native runner establish keyboard access, draft
retention, Revert cancellation/confirmation, Save and same-process return.

ADR required: no. Existing ADR-012 (provider credentials), ADR-033 (session
ownership), ADR-031 (keys), ADR-150 and ADR-161 (design language/components) apply.

## Automated evidence

The [targeted selection](targeted-tests.txt) passed **228 tests**, with 301
unrelated cases deselected, in 285.02 seconds. Coverage includes provider
switching, staged save models, validation, credential masking, atomic-save
failures, endpoint-test draft identity and navigation echoes.

Independent review found that directly focusing Model and the dialog buttons
did not prove keyboard reachability. The four new cases were strengthened to
reach those controls with **Tab**, then edit/activate them with keys and assert
compositor visibility. Save now checks the complete mutation and exact permitted
credential deletions. All [four final cases](keyboard-final-tests.txt) pass in
80.80 seconds. These overlap the 228-case selection; they are not 232 unique tests.

The [first harness failure](keyboard-harness-correction.txt) incorrectly used
Ctrl+A as select-all. Textual uses it to move to the start; Home, Shift+End and
Backspace correctly replace the field. A later
[fixture expectation failure](mocked-credential-expectation-failure.txt) assumed
the minimal mounted fixture had the native catalog's environment credential
source. It has no source, so the exact mutation records `none` and deletes the
inactive key/environment-name entries as tuples. These were test assumptions,
not product defects. [Static checks](static-checks.json) and
[review resolutions](review.json) are recorded separately.

## Native evidence

The [runner](native_check.py) uses real TldwCli, production CSS, terminal-backed
rendering, an acquired instance lock and private SQLite/configuration. Its seed
must select `llama_cpp`, model `model-a`, endpoint `http://127.0.0.1:9099`, with
all data paths contained in the private profile. Model-catalog startup refresh
and hooks were disabled. The endpoint-test boundary has a fail-on-call sentinel;
the persistence path is real.

All [four final journeys](result.json) pass in dark/light at 170×48 and 80×24:

1. F4 and the category filter open Providers & Models; Tab reaches Model and
   Endpoint, and typed values produce an unsaved draft without writing the file.
2. Ctrl+2 visits Console; F4 returns with the draft intact.
3. Esc, r opens confirmation. Tab/Enter chooses Keep editing, then a second
   invocation chooses Discard changes. Cancellation retains the draft; discard
   restores saved values. Neither changes the file.
4. Editing again and pressing Esc, s writes the new model and endpoint. The
   entire parsed file is compared with the expected mutation, including the
   existing provider confirmation and credential-source bookkeeping; unrelated
   settings remain identical. A second Console round trip returns cleanly.

The native catalog supplies `LLAMA_CPP_API_KEY` as the configured environment
name, even though this keyless fixture needs no credential. An
[intermediate strict assertion](native-credential-expectation-failure.json)
incorrectly expected source `none`; the final expectation follows the existing
environment-source contract. The earlier
[direct-focus run](initial-native-direct-focus.json) passed but did not establish
Tab reachability. Both earlier processes exited and their terminals were closed.

Eight final captures were rendered and inspected in one batch. Fields and focus
remain visible in both layouts; the compact unsaved banner wraps. Save feedback
is readable in the notification (which temporarily covers the lower compact
form) and the wide persistent result line. No visual repair was needed. Only
trailing SVG source whitespace was normalized; [hashes](capture-hashes.json)
record raw and stored bytes.

| Theme / size | Draft | Saved |
| --- | --- | --- |
| Dark / 170×48 | [Capture](textual-dark-170-draft.svg) | [Capture](textual-dark-170-saved.svg) |
| Dark / 80×24 | [Capture](textual-dark-80-draft.svg) | [Capture](textual-dark-80-saved.svg) |
| Light / 170×48 | [Capture](textual-light-170-draft.svg) | [Capture](textual-light-170-saved.svg) |
| Light / 80×24 | [Capture](textual-light-80-draft.svg) | [Capture](textual-light-80-saved.svg) |

[Lifecycle evidence](lifecycle.json) records exit 0, normal app stopping, app-run
return, exact-PID absence before terminal closure, eleven healthy private
databases, zero conversations/messages, unchanged default config/UI-state/policy
fingerprints, no error/critical/traceback log entries and empty faulthandler logs.
There were zero endpoint-test calls. No server readiness or model generation is
qualified here. Return means navigation in the same process; process restart was
not tested, though real file contents were read after each Save.

## Task hygiene and limits

The initial [allocation sweep](id-allocation.json) found maximum 32720. A
concurrent workstream claimed 32721–32723 while this review ran. The
[closeout collision check](id-collision.json) caught the shared 32721; the CLI's
next offer, 32722, was also taken. This workstream alone moved to the freshly
verified **32724** ([allocation](id-check.json)); the other tasks were untouched.
Private run paths preserve the original number as provenance.

The guide and ongoing audit now document the verified journey. Next bounded
review: provider switching and saved-default selection, including the existing
TASK-214 report. This pass does not close that report, qualify all Settings
features, or test live provider connectivity. No full suite, push or merge ran.

## Reproduce the targeted selection

```sh
.venv/bin/python -m pytest -q \
  Tests/UI/test_settings_provider_keyboard_journeys.py \
  Tests/UI/test_settings_provider_switch_atomic.py \
  Tests/UI/test_settings_provider_test_draft.py \
  Tests/UI/test_settings_save_commit_models.py \
  Tests/UI/test_settings_provider_view_model.py \
  Tests/UI/test_settings_provider_select_out_of_options.py \
  Tests/UI/test_settings_configuration_hub.py \
  -k 'provider or save_revert or save_commit' --tb=short --show-capture=no
```
