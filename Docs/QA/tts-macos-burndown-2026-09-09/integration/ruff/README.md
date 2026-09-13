# Bounded Ruff delta audit

Ruff 0.16.6 finds no newly introduced diagnostics in the 12 changed existing production/UI/test files audited. The current files still have 152 pre-existing diagnostics; origin/dev has 160. Eight baseline diagnostics were removed. This is a lint-delta result, not a clean legacy-file or full-repository claim.

| Existing changed file | origin/dev | Current | New |
|---|---:|---:|---:|
| Tests/TTS/test_kokoro_pytorch_runtime.py | 0 | 0 | 0 |
| Tests/TTS/test_kokoro_validation.py | 3 | 2 | 0 |
| Tests/TTS/test_legacy_bridge.py | 2 | 2 | 0 |
| Tests/UI/test_speech_playground_pane_lifecycle.py | 2 | 2 | 0 |
| tldw_chatbook/TTS/backends/alltalk.py | 11 | 10 | 0 |
| tldw_chatbook/TTS/backends/higgs.py | 44 | 44 | 0 |
| tldw_chatbook/TTS/backends/kokoro.py | 52 | 46 | 0 |
| tldw_chatbook/TTS/kokoro_pytorch.py | 0 | 0 | 0 |
| tldw_chatbook/UI/Speech/speech_playground_model.py | 0 | 0 | 0 |
| tldw_chatbook/UI/Speech/speech_playground_pane.py | 17 | 17 | 0 |
| tldw_chatbook/UI/Speech/speech_settings_mixin.py | 29 | 29 | 0 |
| tldw_chatbook/UI/Speech/speech_settings_model.py | 0 | 0 | 0 |

Baseline: `a36fc6133c69f77b8b14596a59918de34261f8ef`. Audit HEAD: `1edcbb1aee9bacfd7573c53f70b22d51b27f6300` plus the three separately recorded working-tree edits. Both sides were fed to the same Ruff executable with `check --no-cache --output-format json --stdin-filename RELATIVE_PATH -`, from the actual worktree, so path-sensitive rule selection/configuration is identical. The resolved settings select Python 3.12. Current and baseline pyproject.toml hashes match.

New modules and the already-qualified legacy_catalogs.py were excluded per the requested scope. No models, servers, audio, formatter mutations, package edits, or evidence-script edits were performed.

Four changed-line matches were checked manually: the same pre-existing Dict/List/Tuple deprecated imports remain after import sorting in backends/kokoro.py; the same unused phonemes target remains where the awaited helper was renamed. All other current findings match unchanged or exactly moved baseline text.

The eight resolved diagnostics are one I001 in test_kokoro_validation.py, one BLE001 in backends/alltalk.py, and six in backends/kokoro.py (two I001, one UP035 for AsyncGenerator, one BLE001, one UP012, one RUF010).

The root-announced changes were already present in the lint snapshot: two missing-runtime messages specify the new Python 3.12 application baseline and the stale AllTalk fresh-default test expects alloy. They are retained separately in concurrent-root-edits.patch; no diagnostic delta comes from these three edits. All file hashes still matched at the final check.

Artifacts: receipt.json contains concise counts/hashes; audit.json contains all raw diagnostics plus line mapping/manual classifications; baseline/ and current/ retain exact source snapshots, with resolved-settings.txt and per-side diagnostic JSON.
