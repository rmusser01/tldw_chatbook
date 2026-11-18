---
id: TASK-41525
title: "A keyless settings save followed by quitting may write the shipped api_key_env_var back into config.toml"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-18 03:21'
created_date: 2026-11-17 05:27
labels:
- follow-up
- config
- settings
priority: high
references:
- tldw_chatbook/config.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the docs sweep (TASK-41101, round 15) in TASK-33001.7's fix round 1 notes: quitting after a keyless save wrote the shipped `api_key_env_var` back into `config.toml`. Not reproduced or traced; `config.py:1697` and `:7740` read `api_key_env_var`. Writing a default back is a silent config change and can mask the user's own setting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The path is reproduced in a scratch profile (never the real one) or shown absent, with the notes recording which
- [x] #2 If reproduced, a keyless save then quit leaves `config.toml` without a shipped `api_key_env_var` the user did not set, with a test red before the fix
<!-- AC:END -->

## Implementation Plan

1. Read the history: TASK-33001.7 fix round 1 saw the quit re-insert the template's `api_key_env_var`; TASK-35835 later stopped `persist_cli_config_for_shutdown` from writing the template merge back.
2. Reproduce on the wrapper's scratch profile only: a probe that builds the real `TldwCli`, makes a keyless Settings save of a provider whose template names an `api_key_env_var`, then quits through the app's own quit, and reports the provider table after the save and after the quit.
3. If it reproduces, fix with a red-before test; if not, record the probe and its output here and leave AC#2 to these notes.

## Implementation Notes

**Shown absent at base 45b1949ca3 (AC#1).** AC#2 is conditional on a reproduction; with none, no fix or test is owed, and it is ticked on that basis (these notes are its record, as the brief asks).

History: TASK-33001.7's fix round 1 saw a live quit, 12 s after a keyless save, write the shipped `api_key_env_var` back next to `credential_source = "none"`, because `persist_cli_config_for_shutdown` wrote the template-merged config back. TASK-35835 (Done, 2026-10-22) removed that rewrite: the quit now writes only encryption work on the file's own contents, and `test_settings_keyless_provider_save_keeps_template_env_name_out` (a real Settings save, then the real shutdown persist) and `test_real_app_quit_persists_only_the_effective_config` pin it. The two `config.py` reads the filing named (`:1697` and `:7740` at base) only read the name; neither writes.

Reproduction attempt, through `run.sh` only (scratch HOME and `TLDW_CONFIG_PATH` under `$FU/home-<label>`; the probe asserts the path is there before it runs). The probe (`$FU/r16ky_quit_probe.py`, not committed) loads a fresh profile from the template, builds the real `TldwCli` on the Settings screen, opens Providers & Models, selects the provider, types a model, clicks **Save**, then quits with the app's own `_confirm_and_quit()`, and prints the provider's table at each step. The variable the template names is unset.

Runs (macOS, the run.sh scratch profile; load average 60-100):

| Provider | Sources | Shipped table | After the save | After the quit | File after the quit |
| --- | --- | --- | --- | --- | --- |
| `llama_cpp` | base 45b1949ca3 | `api_key_env_var = "LLAMA_CPP_API_KEY"`, `model = ""` | `credential_source = "none"`, `model = "probe-model"` | same | byte-identical to the save's |
| `custom_2` | base 45b1949ca3 | `api_key_env_var = "CUSTOM_2_API_KEY"`, `model = "custom-model-gamma"` | `credential_source = "none"`, `model = "probe-model"` | same | byte-identical to the save's |
| `llama_cpp` | this branch | as at base | as at base | as at base | byte-identical |

Settings prefilled the shipped name in the Env var field (`ENV_FIELD 'LLAMA_CPP_API_KEY'` / `'CUSTOM_2_API_KEY'`), the save removed it and recorded the keyless decision, and the quit wrote nothing.

So a keyless save followed by a quit leaves no shipped `api_key_env_var` in the file. The name still comes back in the in-memory merge at the next load (the template is merged under the file), which is why the save records `credential_source = "none"`, as TASK-35835 and the User Guide's Credentials row already say. No code change and no new test: the existing pins above already fail if the quit writes the template back (TASK-35835's red evidence: "At index 26452 diff: b'k' != b'u'").

Existing pins, run with this branch (through the wrapper, `-n 4`, load 60-100): `Tests/UI/test_settings_configuration_hub.py::test_settings_keyless_provider_save_keeps_template_env_name_out` (3 cases), `Tests/ProductionApp/test_config_profile_isolation.py` and `Tests/test_config_persistence_owner.py` all passed, in one run of 33 nodes with the Settings save-scope tests (32 passed; the one failure, `Tests/UI/test_console_saved_defaults_flow.py::test_a_blank_top_p_applies_summarises_and_sends`, "the Custom OpenAI 2 handler was never called", fails the same way with base sources).
