# PR review fix report — first-run setup wizard

11 verified code-review findings fixed on `feature/first-run-setup-wizard`
(rebased on `dev`). Behavioral bugs 1-5 fixed test-first (failing test
confirmed red against the pre-fix source, then green after the fix);
compliance items C1-C6 are mechanical.

## Bug 1 — key reused across providers

**Root cause:** `ProviderStep.select_provider()` never cleared the shared
`#setup-provider-key-input` value when the selected provider changed, only
resetting `_clear_requested`. A key typed for provider A therefore survived
a switch to provider B and would commit under B's `api_settings` section.

**Change:** `select_provider()` now tracks whether the provider actually
changed (`provider_key != self.selected_provider_key`, computed before the
attribute is overwritten) and clears `key_input.value` only on a real
change — a redundant re-selection of the same provider leaves an
in-progress typed key alone.

**Covering tests** (`Tests/Wizards/test_first_run_setup_wizard.py`):
- `test_provider_step_switching_provider_clears_key_input` — type a key
  under `openai`, switch to `anthropic`, commit → input is blank and
  neither provider's `api_settings` section is written.
- `test_provider_step_reselecting_same_provider_keeps_typed_key` — guards
  the boundary: re-selecting the *same* provider does not clear the input.

## Bug 2 — theme clobbered on rerun + random un-resettable

**Root cause:** `AppearanceStep.selected_theme` was only ever assigned by
the `RadioSet.Changed` handler, never initialized from the persisted
config, so a rerun that touched only the splash card left it `""` and
`commit()` fell back to a hardcoded `"textual-dark"`, silently overwriting
the persisted theme. Separately, "Surprise me (random)" always mapped to
`splash_card=None`, and the old `build_appearance_commit` only ever wrote
`splash_screen` when `splash_card` was truthy — so a previously persisted
specific card could never be reset back to random.

**Change (three parts, `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py` +
`first_run_setup_state.py`):**
- (a) `AppearanceStep.compose()` now sets
  `self.selected_theme = prefill.default_theme` where the theme
  `RadioSet` is pre-selected from the same prefill.
- (b) `commit()` is delta-aware: it computes `chosen_theme` and only passes
  a non-`None` `default_theme` to `build_appearance_commit` when it
  differs from `prefill.default_theme` (empty persisted → always
  include, matching `build_appearance_commit`'s new falsy-omits contract).
- (c) `_on_card()` now also tracks `_picked_surprise_me` (True only when
  the user explicitly re-presses "Surprise me", not on the RadioSet's own
  mount-time pre-selection, which never fires `Changed`). `commit()` passes
  `reset_splash_to_random=True` to `build_appearance_commit` when
  `_picked_surprise_me` is set and the persisted `card_selection` (now
  exposed via a new `WizardPrefill.card_selection` field /
  `read_wizard_prefill`) names a real, non-random card.
  `build_appearance_commit` gained the `reset_splash_to_random: bool = False`
  keyword and made `default_theme`/`splash_card` independently optional
  (falsy → omit that section).

**Covering tests:**
- `Tests/Wizards/test_first_run_setup_wizard.py::test_appearance_step_rerun_change_only_splash_card_leaves_theme_untouched`
- `Tests/Wizards/test_first_run_setup_wizard.py::test_appearance_step_surprise_me_over_persisted_card_writes_random`
- `Tests/Wizards/test_first_run_setup_wizard.py::test_appearance_step_fresh_run_untouched_commits_nothing`
- `Tests/Wizards/test_first_run_setup_state.py`: additive
  `TestCommitBuilders` cases for `build_appearance_commit`
  (`test_appearance_commit_omits_theme_when_falsy`,
  `test_appearance_commit_reset_splash_to_random`,
  `test_appearance_commit_specific_card_wins_over_reset_flag`,
  `test_appearance_commit_nothing_changed_is_empty`) plus a
  `card_selection` assertion added to `TestWizardPrefill`.

## Bug 3 — provider choice not persisted on first selection

**Root cause:** `invalidate_model_for_provider_change()` only wrote
`chat_defaults` when `if previous_provider_value and previous_provider_value
!= new_provider_value:` — a falsy `previous_provider_value` (`None` on the
very first commit this session) was treated as "nothing to compare",
so a first-ever provider selection with the Model step skipped left
`chat_defaults.provider` at whatever the template/previous run had, even
though credentials had just landed under `api_settings`.

**Change:** `invalidate_model_for_provider_change()` now computes
`effective_previous = previous_provider_value or ""` and compares with
`!=` unconditionally, so an empty/absent previous is treated as "differs
from any real provider" rather than "skip". `ProviderStep.commit()`
resolves the previous value as the in-session
`_last_committed_provider_value` when this step has already committed once
this run, else falls back to the **persisted**
`read_wizard_prefill(app_config).provider_value`.

**Covering tests:**
- `Tests/Wizards/test_first_run_setup_state.py::TestDependencyInvalidation::test_empty_previous_still_writes_on_first_selection` / `test_none_previous_is_treated_like_empty` (pure-function level).
- `Tests/Wizards/test_first_run_setup_wizard.py::test_provider_step_first_selection_persists_chat_defaults_provider` — empty config, type a key, commit → `chat_defaults == {"provider": "openai", "model": ""}`.
- `test_provider_step_rerun_same_provider_leaves_chat_defaults_untouched` — persisted provider re-selected + Keep → no `chat_defaults` key at all.
- `test_provider_step_rerun_different_provider_blanks_model` — persisted `openai`/`gpt-4o`, switch to `anthropic` → `chat_defaults == {"provider": "anthropic", "model": ""}`.

**Side effect on 3 pre-existing tests:** `test_provider_step_one_click_connect_adopts_discovered_server`,
`test_provider_step_keep_preserves_existing_key_without_note`, and
`test_provider_step_clear_persists_empty_key_without_note` all select a
provider against an `app_config` with no persisted `chat_defaults`, so
under the fix they now *correctly* also emit a `chat_defaults` sync
alongside their existing (unchanged) credential-handling assertions. Their
`committed == {...}` expectations were updated to include the new
`chat_defaults` entry; the credential-value and `note_key_entered`
assertions that were the actual point of each test are untouched.

## Bug 4 — Protect-keys unreachable on rerun with existing plaintext keys

**Root cause:** `SetupWizardContainer.active_ids` gated `STEP_PROTECT`
behind `self.key_entered`, which only flips `True` when a secret is typed
*this run* (`note_key_entered()`). A rerun over a config that already has
a plaintext key on disk (hand-edited `config.toml`, or a previously
completed run) could never reach Protect Keys without retyping a
credential, even though the feature's intent is config-derived (mirroring
`check_encryption_needed`).

**Change:** New pure helper
`first_run_setup_state.stored_plaintext_key_present(app_config) -> bool`
— True when any `api_settings.<provider>.api_key` is a real secret (via
the existing `_is_real_secret`) **and** `encryption.enabled` is not set.
`SetupWizardContainer` gained `_effective_key_entered()` (used both at
`__init__` and in `_refresh_active_ids()`) that ORs `self.key_entered`
with `stored_plaintext_key_present(app_config)`.

**Covering tests:**
- `Tests/Wizards/test_first_run_setup_state.py::TestStoredPlaintextKeyPresent` — truth table: inline key + no encryption → True; encryption enabled → False; env-var-only → False; placeholder key → False; empty config → False.
- `Tests/Wizards/test_first_run_setup_wizard.py::test_rerun_with_stored_plaintext_key_activates_protect_step_without_typing` — a rerun wizard mounted over an `app_config` with a plaintext key includes `STEP_PROTECT` in `active_ids` with nothing typed.
- `test_fresh_config_without_stored_key_omits_protect_step` — regression guard for the untouched no-stored-key path.

## Bug 5 — custom model input can't clear

**Root cause:** `ModelStep._on_custom_model()` only assigned
`self.selected_model_id` when `event.value.strip()` was truthy, so
clearing a previously-typed custom model left the stale value in place —
a "skip-safe" commit would then silently persist a model the input no
longer showed.

**Change:** `ModelStep` gained `_model_id_from_custom_input: bool` to
track whether the current selection came from the free-text input (set in
`_on_custom_model`, cleared in `set_selected_model()` and the
provider-change reset in `on_show()`). `_on_custom_model()` now, on an
empty value, only resets the selection when the *current* selection came
from the custom input — falling back to whatever `RadioSet` button is
pressed (or `""` if none).

**Covering tests:**
- `Tests/Wizards/test_first_run_setup_wizard.py::test_model_step_clearing_custom_input_clears_stale_selection` — type then clear the custom input → `selected_model_id == ""`, commit writes nothing.
- `test_model_step_clearing_custom_input_falls_back_to_radio_selection` — pick a radio model, type a custom model, clear the input → selection reverts to the radio pick, not blank.

## Compliance sweep (mechanical)

- **C1** — `Tests/Wizards/test_first_run_setup_integration.py`: renamed
  every token-lookalike literal (`sk-integration` → `wizard-test-key-alpha`,
  `sk-secret` → `wizard-test-key-beta`, `sk-upgrader` → `wizard-test-key-gamma`,
  `sk-to-encrypt` → `wizard-test-key-delta`); every assertion (including the
  `repr()`-absence checks) now asserts the same renamed value.
- **C2** — same file: `TestEncryptionAtRest.test_enable_encryption_encrypts_stored_key`
  now reads `temp_config.read_text()` (the fixture's own yielded `Path`)
  instead of re-deriving the path via `Path(os.environ["TLDW_CONFIG_PATH"])`;
  the now-unused `os`/`Path` imports were dropped.
- **C3** — Added Google-style `Args:`/`Returns:` docstrings to
  `build_model_commit`, `build_rag_commit`, `build_tools_commit`,
  `build_notes_commit` (reformatted its existing prose docstring to add
  the headers), `build_appearance_commit` (bundled with its Bug-2 signature
  change), `build_wizard_state_commit`, and the new
  `stored_plaintext_key_present` helper (bundled with Bug 4) — all in
  `tldw_chatbook/UI/Wizards/first_run_setup_state.py`.
- **C4** — `ProviderStep.__init__` in `FirstRunSetupWizard.py` is now fully
  type-annotated: `wizard: Optional["SetupWizardContainer"] = None`,
  `config: Optional[WizardStepConfig] = None`,
  `discover`/`probe: Optional[Callable[..., Any]] = None`,
  `environ: Optional[Mapping[str, str]] = None`, `-> None`.
- **C5** — `ProviderStep._run_probe`'s bare local-probe branch now imports
  and uses `DISCOVERY_PROBE_TIMEOUT_SECONDS` from
  `tldw_chatbook.Chat.local_server_discovery` instead of the literal `2.5`
  (same value, no duplicate constant minted).
- **C6** — `tldw_chatbook/Chat/provider_readiness.default_api_key_env_var`
  gained `Args:`/`Returns:` sections.

## Verification

```
PYTHONPATH=<worktree> <worktree>/.venv/bin/pytest Tests/Wizards/ \
  Tests/UI/test_first_run_wizard_live_contract.py \
  Tests/UI/test_product_maturity_phase1_first_run.py \
  Tests/Chat/test_provider_readiness.py -q
```
→ 165 passed.

Each of the 5 behavioral fixes was also verified red-then-green: the
fix commit's source changes were stashed (tests kept), the new/regression
tests were re-run and confirmed to fail against the pre-fix code, then the
stash was restored and the full suite re-confirmed green.

## Commits

1. `fix: 5 first-run wizard behavioral bugs from code review (TDD)` —
   behavioral fixes + their tests (Bugs 1-5), including the 3 adjusted
   pre-existing `ProviderStep` commit tests.
2. Compliance sweep (C1-C6) — this file included.
