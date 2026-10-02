# TASK-33005.4 live captures (2026-10-01)

Plain-text `tmux capture-pane -p` dumps of the real app, launched from this
worktree in an isolated tmux server at 211x44 (capture 10 after resizing the
window to 235x52). The profile was disposable: `HOME`, `XDG_CONFIG_HOME`,
`XDG_DATA_HOME`, `XDG_CACHE_HOME` and `TLDW_CONFIG_PATH` pointed at a scratch
directory, `users_name = "verify_t33005_4"`, the keyring backend was the null
one, and the real `~/.config/tldw_cli/config.toml` hashed the same before and
after (as did the `~/.local/share/tldw_cli` listing). The startup "Check model
lists online?" prompt was answered **Don't check**, so no ADR-020 consent was
recorded.

The scratch config saved fake keys only (`api_settings.openai`,
`api_settings.openrouter`, `api_settings.google`) and llama.cpp at
`http://127.0.0.1:9202`. No valid cloud key was available, so the "verified"
views use a throwaway local server standing in for the provider: on port 9201
it answered `GET /v1/models` with two models only for `Authorization: Bearer
<the saved fake key>` (and 401 otherwise), and logged whether each request
carried that key. On 9202 it answered keyless. Neither server recorded a
`POST` (no generation). The driver and server lived in the scratchpad and are
not committed.

1. `1-settings-t-openai-shipped-endpoint-key-rejected-211x44.txt`: Settings
   (F4) ▸ Providers & Models, OpenAI on its shipped endpoint, `t`. This was a
   real request to `https://api.openai.com/v1/models` with the fake key, which
   OpenAI answered 401: "Readiness   Not ready · key rejected", Endpoint
   "… · model listing failed (unauthorized) — check the API key".
2. `2-console-key-rejected-after-settings-t-211x44.txt`: the Console: the
   header badge and the Model section read "Not ready · key rejected", and the
   composer says "Send blocked — add an API key to continue".
3. `3-settings-t-draft-endpoint-verified-211x44.txt` (and `3a-…-toast-…`, the
   same moment with its toast): Settings, Endpoint typed (unsaved) as
   `http://127.0.0.1:9201/v1`, `t`: "Readiness   Ready · verified 22:47",
   "Key         saved in config · key accepted (2 models listed)", Generation
   "not tested"; the toast reads "Ready · verified 22:47 — key accepted (2
   models listed) · generation not tested." Server log: one `GET /v1/models
   auth=match`.
4. `4-settings-after-save-verified-carried-211x44.txt`: `s` (Save): the
   result for exactly the saved values carries over; the Readiness row still
   reads "Ready · verified 22:47".
5. `5-console-verified-after-save-211x44.txt`: the Console: header badge and
   Model section read "Ready · verified 22:47"; Send is no longer blocked.
6. `6-switcher-openai-verified-211x44.txt`: Switch model (Alt+M): the OpenAI
   row reads "Ready · verified 22:47"; every other row "Ready · not tested".
   The server log still held exactly one request: saving, the Console and
   the switcher sent nothing.
7. `7-settings-t-openrouter-public-listing-211x44.txt`: Settings, provider
   OpenRouter, `t`: a real request to OpenRouter's public catalog. "Readiness
   Ready · not tested", "Key … · models listed; key not checked"; the toast
   reads "Ready · not tested — 100+ models listed; key not checked".
8. `8-settings-t-google-no-key-check-211x44.txt`: provider Google Gemini,
   `t`: nothing is sent; "Key check   No non-billable key check is available
   for Google Gemini at this endpoint; the configuration was checked locally."
9. `9-settings-t-llamacpp-no-model-listed-211x44.txt`: provider llama.cpp
   with the Model field cleared, `t`: the listing still runs ("Endpoint …
   · model listing reached", server log `GET /v1/models auth=none`), the
   Readiness row says "Not ready · no model" and the toast "Model listing
   reached (2 models listed); choose a default model."
10. `10-console-verified-235x52.txt`: the Console at 235x52, same word.

Reading notes for reviewers:

- "22:47" is the local time the listing answered, not a provider time.
- "100+ models listed": evidence keeps at most 100 model ids (the existing
  evidence bound), so a longer list is counted as "100+".
- "Agent blocked" in the status strip is the agent's Library access policy
  chip, unrelated to provider readiness.
- The strings these captures rely on are asserted against the real widgets in
  `Tests/UI/test_settings_provider_key_check.py` (real
  `LocalLLMProviderCatalogService.discover_models` and discovery client over
  an `httpx.MockTransport`).
