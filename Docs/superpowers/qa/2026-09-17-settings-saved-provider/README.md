# Saved provider defaults and draft ownership — TASK-214

2026-09-17 UTC, `feat/component-pattern-library`, based on `4f9e7ffc21`.

TASK-648 had already removed the boot provider cache behind the original July
report. Fresh Settings showed saved defaults correctly. This review reproduced
two remaining problems: a retained clean form stayed on the previous provider,
and a sparse unsaved draft could restore under a newly saved default provider.

The repair refreshes clean retained provider/model/endpoint projections through
the existing category rebuild path. A changed selection invalidates old test and
model-discovery evidence; an unchanged selection keeps its widgets and verdict.
Drafts pin their provider/model as equal original/value entries, preserving
ownership without marking those fields dirty. Dependent values and later edit
originals follow that selection. API-mode-only edits use the same pinning path.

ADR required: no. Existing ADR-006 (provider-aware settings), ADR-012 (credential
boundary), ADR-033 (session ownership), ADR-031 (keys), ADR-150 and ADR-161
(design language/components) apply. No new service, storage or visual-token
boundary was introduced.

## Automated evidence

The final [targeted selection](targeted-tests.txt) passed **320 tests**, with
297 unrelated cases deselected, in 534.49 seconds. It includes all 26 new cases,
provider save/switch/test/mode checks and real-app restart/session-ownership
coverage. The 21 additional Console fixture/privacy checks below also pass.

[Initial reproductions](initial-red.txt) produced six failures and eight passes:
four retained clean forms stayed stale and two restored/retained model drafts
changed providers. The final regression file contains 26 production-CSS cases:

- Fresh, restored and retained clean visits across dark/light at 170×48 and 80×24.
- Model, endpoint, credential-environment-name and generation-profile drafts
  across restored and retained visits after defaults change.
- API-mode-only drafts, second-edit originals followed by Save or Revert, and
  unchanged returns with catalog aliases and custom provider names.
- Current visible provider/model/endpoint, picker selection, readiness/test
  inputs, dirty status and stale-verdict behavior.

[Independent review](review.json) identified the API-mode path and second-edit
originals. All four [new cases failed before those corrections](review-red.txt).
The no-op checks separately reproduced unnecessary rebuilds for a
[saved catalog alias](alias-noop-red.txt) and a
[manual provider name](custom-noop-red.txt); both sides of the comparison now
use the canonical provider key.

The [initial broader run](initial-targeted.txt) passed 272 cases and hit one
obsolete Console serializer fixture, before the privacy assertion could run.
Its fake runtime lacked the staged-evidence methods now owned by ConsoleRuntime.
[Unchanged-source hashes](fixture-baseline.json) establish that the fixture and
Console screen matched HEAD. Using the real runtime restores that privacy check;
[20 related fixture tests](fixture-tests.txt) and the
[exact mounted-credential privacy test](privacy-and-api-save-tests.txt) pass.
The fixture's other direct consumer, the
[secret-free credential handoff test](credential-handoff-test.txt), also passes.
These 21 additional Console cases are distinct from the main selection.

Two Qwen tests also [failed against the unchanged HEAD implementation](qwen-baseline.txt).
The broad query `mode` matches the category title Models, whose established
TASK-23109 priority keeps category focus; specific API-mode queries still focus
the selector. The other test clicked an off-screen button and now uses the
implemented Esc, t / Esc, s actions. No production search or layout change was
made for these fixture corrections. A [66-case focused pass](focused-tests.txt)
preceded the final custom-provider no-op correction; it overlaps the final run.

[Static checks](static-checks.json) cover syntax, changed-range formatting,
new-file lint/format and diff whitespace. Existing lint diagnostics remain at
118 for the large Settings screen and six for the existing Console test file;
none were added. The Qwen test file and both new Python files are lint-clean.

## Native evidence

The [runner](native_check.py) uses real TldwCli, production CSS, terminal-backed
rendering, an acquired instance lock and an isolated configuration/database
profile. Real SettingsConfigAdapter writes publish external default changes,
then real load_settings reloads the app configuration. This models another
configuration producer; it does not replay the removed July local-connect UI.

All [four final journeys](result.json) pass in dark/light at 170×48 and 80×24:

1. Open Providers & Models, visit Console, write llama.cpp and a new model as
   saved defaults, then return through F4. Provider, model, endpoint, picker and
   readiness/test inputs agree without reselection, and the form is clean.
2. The real Console new-chat settings constructor reads the same default pair.
   Existing explicit-session ownership is covered by the production-app tests.
3. Tab reaches Model. Esc, t performs actual HTTP against a task-owned loopback
   `/v1/models` fixture. Each result confirms its selected model and explicitly
   says generation was not tested.
4. Typing a model draft, visiting Console, saving OpenAI defaults and returning
   preserves the llama.cpp draft and endpoint. Esc, r and Tab/Enter discard it;
   the clean form then follows the latest OpenAI defaults.

The native destination router replaces Settings on these visits; the result
records both screen-retained flags as false. Mounted push/pop tests separately
cover the retained-screen defect. Native evidence therefore establishes actual
restoration, config persistence and HTTP dispatch, without pretending that the
native route retains the same screen object.

Eight final captures were rendered and inspected in one batch. Provider, Model,
Endpoint and input focus remain readable; the compact unsaved banner wraps.
Wide catalog-result captures include the readable completion notification; narrow
captures show the focused fields, with the result below the scroll viewport.
No visual repair was needed. Only trailing SVG source whitespace was normalized;
[hashes](capture-hashes.json) preserve raw and stored provenance.

| Theme / size | Saved default | Owned draft after another default change |
| --- | --- | --- |
| Dark / 170×48 | [Capture](textual-dark-170-saved-provider.svg) | [Capture](textual-dark-170-owned-draft.svg) |
| Dark / 80×24 | [Capture](textual-dark-80-saved-provider.svg) | [Capture](textual-dark-80-owned-draft.svg) |
| Light / 170×48 | [Capture](textual-light-170-saved-provider.svg) | [Capture](textual-light-170-owned-draft.svg) |
| Light / 80×24 | [Capture](textual-light-80-saved-provider.svg) | [Capture](textual-light-80-owned-draft.svg) |

[Lifecycle evidence](lifecycle.json) records normal app stopping, exit 0, app-run
return, loopback server closure, exact-PID absence before owned-terminal closure,
eleven healthy private databases, zero conversations/messages, unchanged default
config/UI-state/policy fingerprints, no error/critical/traceback log entries and
empty faulthandler logs. Two earlier native passes were also closed; the final
result hashes both the runner and the exact final Settings source.

There were four catalog GETs and zero generation requests. This qualifies the
loopback HTTP probe, not external-provider availability or model generation.
The native journey did not restart the process; separate production-app tests
cover saved defaults on restart and explicit Console-session preservation.
No full suite, push or merge ran.

## Reproduce the final targeted selection

```sh
.venv/bin/python -m pytest -q \
  Tests/UI/test_settings_saved_provider_return.py \
  Tests/UI/test_settings_provider_keyboard_journeys.py \
  Tests/UI/test_settings_provider_switch_atomic.py \
  Tests/UI/test_settings_provider_test_draft.py \
  Tests/UI/test_settings_save_commit_models.py \
  Tests/UI/test_settings_provider_view_model.py \
  Tests/UI/test_settings_provider_select_out_of_options.py \
  Tests/UI/test_settings_configuration_hub.py \
  Tests/UI/test_settings_qwencloud_api_mode.py \
  Tests/ProductionApp/test_provider_selection_ownership.py \
  Tests/Provider/test_provider_model_resolution.py \
  -k 'provider or save_revert or save_commit or mount or qwencloud' \
  --tb=short --show-capture=no
```

Next bounded review: model discovery, selection, Save selected and failure/retry
feedback. This pass closes TASK-214, not the whole Settings review.
