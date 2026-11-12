# Task18 report — frozen at unexpected BASE precondition

Status: BLOCKED. No implementation or commit began. Source/index/HEAD remain clean at `1a9b777bdbe018181e6ea7fabd17d98174e912e8`; no next node was run after this unexpected failure. Root has been notified.

## Requirements and investigation

Read updated task-18-brief.md and task-18-ruling75.md, Task17 interrupt/Stop preflights, Qodo comments4179836579/6584/6586/6590, the frozen Stop argv/result and CI failure receipt, ADR094/219/220, relevant testing/live-profile/backlog lessons, both actual stale test owners, selected binding-control bodies, native Stop body and its existing selector helper, current host constructor/wiring/caller census and actual size governance. ADR required: no new ADR; retain existing ADR219/220 and ADR094 custody. The actual getters are `self.read_global_Any/Mapping`, matching the corrected brief.

The clock owner retains five positional constructions and Buddy setup retains one. Existing adapter `make_interrupt_host` already provides nullable invocation-time fake and global reads. Exactly15 Any/two Mapping host annotation uses are enumerated by preflight, with separate review-hook and compaction APIs excluded. Stop projection publication wait is authorized using the existing2second helper; unchanged CI/local receipts remain separate sources. No proposed source repair was applied.

## Actual RED receipt

Source: integrated BASE `1a9b777bdbe018181e6ea7fabd17d98174e912e8`. Python `Python 3.12.11`, cwd `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`, sole environment override `PYTHONPATH=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. Unmodified canonical Tests/conftest.py/Tests/UI/conftest.py private profile bootstrap, original markers; `-p no:randomly`; fresh private basetemp `/private/tmp/pr2995-task18-red-i7i1vjg_`; unchanged pytest timeout300s and subprocess bound300s. No warning suppression, extra plugin, retry or marker alteration.

Exact argv:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:randomly --basetemp=/private/tmp/pr2995-task18-red-i7i1vjg_/pytest Tests/Chat/test_console_decision_clock.py Tests/UI/test_buddy_speech.py --junitxml=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-18-safe-evidence/red.xml
```

Subprocess exit `1`, elapsed `11.274s`, closed PID `91625`, timed_out `False`. Actual JUnit44 collected cases:30 FAIL,14 setup ERROR,0 PASS/skip. The full IDs/statuses live in `red-result.json`; complete output and traceback in `red.log`; JUnit in `red.xml`; exact argv/source hashes in `red-argv.json`.

Expected clock RED: all30 cases reach the five original `InterruptRoundHost(seams)` sites and raise `TypeError: InterruptRoundHost.__init__() takes 1 positional argument but 2 were given`.

Unexpected Buddy mask: all14 fail before their body/setup helper at `Tests/UI/conftest.py:192` `_disable_model_catalog_refresh`; loading app.APP_CONFIG calls load_settings -> guarded config participant -> `_RawParticipant` state, which raises `RecoveryRequired("raw_source_selection_changed")` in raw_participants.py:132. These are NOT Buddy constructor RED and are not claimed as such. No repair to recovery/config/fixtures, bypass or extra execution was attempted.

Pytest warning summary: none. Captured stderr includes optional-dependency warning `python-frontmatter not installed. Markdown import will not be available.` and informational unavailable HuggingFace datasets log; full output is preserved verbatim. Pytest terminal summary: `30 failed, 14 errors in 8.26s` (JUnit8.244s).

## Frozen carry and incomplete work

`frozen-manifest.json` verifies all30536 tracked files unchanged and all20171 existing own-SDD historical/private receipt hashes unchanged, including exact prior RED, CI and local passes/warnings. Tracked source/QA/ZIP files are part of the unchanged tracked map. No historical log was bulk copied. No AST reversal applies because no source change began. No GREEN/binding/Stop run, Ruff/formatter/whitespace or new size measurement occurred; no completion or current passing claim is made.

Historical source-specific Stop evidence remains `06cfe2f6a30236dcd8f81893ebf49bc3d0b78036`: CI run37244353688/job111559166890 had1failed/221passed/1existing xfail/5warnings at hidden Stop despite STREAMING; unchanged original one-node local run exit0/1passed30.06s (subprocess35.959s) is preserved and is not called a fix. The authorized Task18 Stop wait has not yet been added or run.

## Self-review and concern

Scope maintained: no dependencies, profiles, source, assertions, deadlines, caps, historical evidence, external PR/check/thread actions or Git objects changed. No child/reviewer dispatched. Process closed and hash stability confirmed. Need root's explicit ruling on the canonical Buddy UI setup masking precondition before changing source or running any next node. Source/index/HEAD ownership can return immediately.

## Root-requested read-only Buddy profile diagnosis

Every actual Buddy node (all14 cases from eight functions) shares `Tests/UI/conftest.py::_disable_model_catalog_refresh`, an autouse fixture explicitly dependent on `isolate_test_environment`. Its final monkeypatch string imports `tldw_chatbook.app` at line192; app.APP_CONFIG loads the real guarded config. None of these functions uses the private-profile-child decorator, so the fixture's existing child-parent early return does not apply. Buddy contains no profile/environment re-selection, no config.toml write, no install_config_source or config-loader monkeypatch. Both owners import real controller/store code at collection; the guarded config source is bound to the canonical collection profile. The root fixture's unmarked branch later selects tmp_path/test_data and changes HOME/XDG/TLDW_CONFIG_PATH before the UI autouse import. Raw config participant admission compares this selected path with its actual binding and fails closed with `raw_source_selection_changed` before the Buddy body.

The registered marker description in Tests/conftest.py:1040 is `bootstrap_profile: keep the collection-time profile instead of the per-test sandbox (config-participant admission)`. `isolate_test_environment` checks `request.node.get_closest_marker("bootstrap_profile")` at1188; choosing it retains `_BOOTSTRAP_CONFIG_ROOT` at1433, its data/config/home paths, the normal native/raw admission, real-profile guard and singleton cleanup. This is a marker, not a fixture or bypass. Task16's frozen profile-attribution lesson proposes this same per-owner/node route; the actual Buddy qualification owner Tests/UI/test_buddy_v1_qualification_capture.py:24 already has `pytestmark = pytest.mark.bootstrap_profile`. Other Buddy UI owners without this marker were inspected as source only; their absence is not green evidence and no sibling was run.

Candidate only: add `pytestmark = pytest.mark.bootstrap_profile` after existing imports and before `DESTINATION` in Tests/UI/test_buddy_speech.py (currently line15 constant). A module marker is scoped to this owner and justified here because all14 share the exact failing source-bound autouse fixture and none reselects the profile. Preserve every existing function/statement/asyncio marker/parameterization/wait/assertion. No shared fixture change, blanket plugin, fake config, skip, native/raw admission alteration or extra test is proposed. No candidate overlay or execution has occurred; root must record scope before any mutation.
