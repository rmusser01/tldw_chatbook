# UAT guard fix: first-run wizard never auto-offered on a fresh install

## Root cause

`tldw_chatbook/UI/Wizards/first_run_setup_state.py::any_provider_configured()`
treated any real (non-placeholder) value under a set of endpoint-URL keys
(`_ENDPOINT_KEYS = ("api_url", "api_base_url", "api_base", "base_url",
"api_endpoint", "endpoint")`) as proof a provider was "configured".

The shipped config template (`tldw_chatbook/config.py`'s
`CONFIG_TOML_CONTENT`) pre-populates roughly a dozen `[api_settings.*]`
blocks with default local-server endpoint URLs on **every** fresh install,
none of them ever touched by the user — e.g.:

- `[api_settings.llama_cpp]` → `api_url = "http://localhost:8080"`
- `[api_settings.ollama]` → `api_url = "http://localhost:11434/v1/chat/completions"`
- `[api_settings.vllm]` → `api_url = "http://localhost:8000/v1/chat/completions"`
- `[api_settings.huggingface]` → `api_base_url = "https://router.huggingface.co/v1"`
- plus oobabooga, koboldcpp, aphrodite, tabbyapi, custom, custom_2,
  local-llm, local_llamafile, local_llamacpp, etc.

Because `any_provider_configured` counted these, a brand-new install with
zero user-entered credentials was misclassified as "configured", so
`should_offer_wizard()` (the exact function `app.py::_maybe_offer_first_run_wizard`
calls) returned `False` and the wizard never auto-offered — confirmed live
via tmux UAT against the real app. Every pre-existing test built its
`app_config` as a synthetic Python dict and never reproduced the template's
baked-in endpoint defaults, so none of them caught it.

## Fix

`any_provider_configured` now only counts:
- a real (non-placeholder) inline `api_key`, or
- a declared `api_key_env_var` whose environment variable is actually set.

Endpoint-URL keys are no longer inspected at all; `_ENDPOINT_KEYS` was
deleted (nothing else in the codebase imported it — `settings_screen.py`
and `local_llm_provider_catalog_service.py` each define their own unrelated
`_ENDPOINT_KEYS`/`PROVIDER_ENDPOINT_KEYS`). The module docstring and the
function's own docstring now record the incident and the accepted
consequence: a hand-configured local-endpoint-only user (no keys at all)
will get exactly one auto-offer; skipping it persists `setup_completed`,
so it never re-offers — mirroring the already-approved upgrader behavior.

## TDD evidence

1. Flipped `Tests/Wizards/test_first_run_setup_state.py::TestAnyProviderConfigured::test_local_endpoint_url_counts`
   → renamed `test_template_default_endpoint_urls_do_not_count`, now asserts
   an endpoint-only config does **not** count. Ran red against the
   pre-fix code:
   ```
   AssertionError: assert True is False
   where True = any_provider_configured({'api_settings': {'llama_cpp': {'api_url': '...'}}}, {})
   ```
2. Added `Tests/Wizards/test_first_run_setup_integration.py::TestFreshTemplateOfferGuard`
   — the regression pin — loading the **real** generated template via the
   existing `temp_config` fixture (`load_cli_config_and_ensure_existence`)
   and asserting `should_offer_wizard(...) is True`; a sibling test asserts
   the same template with one real inline key does NOT offer. Ran red
   against pre-fix code (`test_fresh_template_offers_wizard` failed).
3. Implemented the fix (see above). Both new/flipped tests green.
4. Re-ran the full requested wizard set:
   ```
   PYTHONPATH=.../first-run-wizard .venv/bin/pytest \
     Tests/Wizards/ Tests/UI/test_first_run_wizard_live_contract.py \
     Tests/UI/test_product_maturity_phase1_first_run.py -q
   → 149 passed, 1 pre-existing unrelated warning (coroutine never awaited
     in test_model_step_provider_change_resets_selection)
   ```
   No fixture needed adjustment — none of the tests in this set relied on
   endpoint-only configs to mean "configured" (verified via
   `test_upgrader_config_never_auto_offers`, `TestAppOfferGating`, and
   `_build_test_app`'s synthetic `fake_app_config`, none of which set an
   `api_url`).

## Files changed

- `tldw_chatbook/UI/Wizards/first_run_setup_state.py` — removed
  `_ENDPOINT_KEYS` and the endpoint-checking loop from
  `any_provider_configured`; added incident-recording docstrings.
- `Tests/Wizards/test_first_run_setup_state.py` — flipped/renamed the
  endpoint-counts test.
- `Tests/Wizards/test_first_run_setup_integration.py` — added
  `TestFreshTemplateOfferGuard` (the real-template regression pin).

`BaseWizard.py` was not touched; no test wrote to a real user config
(`temp_config` uses `tmp_path` + `TLDW_CONFIG_PATH`).
