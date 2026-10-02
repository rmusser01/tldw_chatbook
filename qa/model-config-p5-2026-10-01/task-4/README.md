# TASK-33005.4 live captures (2026-10-01, fix round 1)

Each capture is a `tmux capture-pane` dump of the real app, launched from this
worktree at the fix-round-1 code in an isolated tmux server. The `.txt` file is
the plain text (`capture-pane -p`). The `.ansi.txt` file is the same moment
with colour and style escapes (`capture-pane -p -e`); view it with `cat` in a
truecolor terminal. Captures 01–10 are at 211x44. Captures 11–12 were taken
after the window was resized to 235x52.

Setup:

- The profile was disposable. `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`,
  `XDG_CACHE_HOME` and `TLDW_CONFIG_PATH` pointed at a scratch directory, with
  `users_name = "verify_t33005_4fix"` and the null keyring backend.
- The real `~/.config/tldw_cli/config.toml` had the same sha256 before and
  after (`15c6cb22…`), and the `~/.local/share/tldw_cli` listing did not change.
- The startup "Check model lists online?" prompt was answered **Don't check**,
  so no ADR-020 consent was recorded.
- The scratch config saved fake keys only (`api_settings.openai`,
  `openrouter`, `google`, `huggingface`) and llama.cpp at
  `http://127.0.0.1:9202`.

No valid cloud key was available, so the "verified" views use a throwaway
local server as a stand-in for the provider:

- On port 9201 it answered `GET /v1/models` with two models (`gpt-4o`,
  `gpt-4o-mini`) only for `Authorization: Bearer <the saved fake OpenAI key>`,
  and returned 401 for any other request.
- On 9202 it answered without a key.
- Both servers logged every request.
- After the whole run the 9201 log held exactly one line, `GET /v1/models
  auth=match` (capture 03). The 9202 log held exactly one line, `GET
  /v1/models auth=none` (capture 09). Neither server ever logged a `POST`.

The driver and the servers lived in the scratchpad and are not committed.

1. `01-settings-t-openai-shipped-endpoint-key-rejected-211x44`: Settings (F4) ▸
   Providers & Models, OpenAI on its shipped endpoint, `t`. This was a real
   request to `https://api.openai.com/v1/models` with the fake key, and OpenAI
   answered 401. The rows read "Readiness   Not ready · key rejected" and
   "Endpoint … · model listing failed (unauthorized) — check the API key".
2. `02-console-key-rejected-after-settings-t-211x44`: the Console. With no
   conversation yet it shows its Get started card. Step 1 reads "Reconnect
   the provider credential", with "Not ready · key rejected" under it.
3. `03-settings-t-draft-endpoint-verified-211x44` (and `03a-…-toast-…`, the
   same moment while the toast was showing): Settings with the Endpoint typed,
   and not saved, as `http://127.0.0.1:9201/v1`, then `t`.
   - The rows read "Readiness   Ready · verified 23:50" and "Key   saved in
     config · key accepted (2 models listed)".
   - Endpoint reads "… (draft) · model listing reached" and Generation "not
     tested".
   - The toast reads "Ready · verified 23:50 — key accepted (2 models listed) ·
     generation not tested."
4. `04-settings-revisit-after-save-verified-carried-211x44`: `s` (Save), then
   the Console, then back to Settings, so the rows are rebuilt from the stored
   evidence.
   - Endpoint now reads `http://127.0.0.1:9201/v1 · model listing reached`,
     with no "(draft)", because the value is saved.
   - Readiness still reads "Ready · verified 23:50": the result for exactly
     the saved values carried over.
   - Immediately after `s`, before leaving, the rows on screen are the ones
     from the `t` run and still say "(draft)". Save does not rebuild them,
     which is why this capture is taken on a return visit.
   - "Provider settings have not been saved this session" counts saves in this
     visit to Settings, so it shows on any return visit.
5. `05-console-verified-after-save-211x44`: the Console header badge reads
   "Ready · verified 23:50", and the conversation pane reads "Ready — type a
   message to begin." instead of the Get started card.
6. `06-switcher-openai-verified-211x44`: Switch model (Alt+M). The OpenAI rows
   read "Ready · verified 23:50" and the other providers read "Ready · not
   tested". Saving, the Console and the switcher sent nothing: the 9201 log
   still held one line.
7. `07-settings-t-openrouter-public-listing-211x44` (and `07a-…-toast-…`):
   provider OpenRouter, `t`, which sent a real request to OpenRouter's public
   catalog.
   - The rows read "Readiness   Ready · not tested" and "Key   saved in config
     · models listed; key not checked".
   - The toast reads "Ready · not tested — 100+ models listed; key not
     checked".
8. `08-settings-t-google-no-key-check-211x44`: provider Google Gemini, `t`.
   Nothing is sent. The "Key check" row reads "No non-billable key check is
   available for Google Gemini at this endpoint; the configuration was checked
   locally."
9. `09-settings-t-llamacpp-no-model-listed-211x44`: provider llama.cpp with no
   model set, `t`.
   - The listing still runs: Endpoint reads "… · model listing reached", and
     the 9202 log shows `GET /v1/models auth=none`.
   - The Readiness row says "Not ready · no model".
   - The toast ("Model listing reached (2 models listed); choose a default
     model.") had closed before this capture. The same copy is asserted in
     `test_settings_provider_test_lists_endpoint_before_a_model_is_chosen`.
10. `10-settings-t-huggingface-send-ignores-endpoint-211x44` (and
    `10a-…-toast-…`): **new in fix round 1.** Provider Hugging Face with the
    Endpoint typed as `http://127.0.0.1:9201/v1`, `t`.
    - A Hugging Face send still takes its URL from the legacy `[API]` table
      (TASK-2117), so a listing at this endpoint would check a place sends
      never go.
    - Nothing is sent: the 9201 log did not grow.
    - The "Key check" row reads "No non-billable key check is available for
      Hugging Face at this endpoint; the configuration was checked locally."
    - Readiness stays "Ready · not tested".
11. `11-console-verified-235x52`: the Console at 235x52. The header badge and
    the Model section both read "Ready · verified 23:50".
12. `12-settings-revisit-verified-235x52`: Settings at 235x52 on a return
    visit, with the same "Ready · verified 23:50" and the saved endpoint.

Reading notes for reviewers:

- "23:50" is the local time the listing answered, not a provider time.
- "100+ models listed": evidence keeps at most 100 model ids (the existing
  evidence bound), so a longer list is counted as "100+".
- "Agent blocked" in the Console status strip is the agent's Library access
  policy chip. It is unrelated to provider readiness.
- The strings these captures rely on are asserted against the real widgets in
  `Tests/UI/test_settings_provider_key_check.py`. Those tests use the real
  `LocalLLMProviderCatalogService.discover_models` and discovery client over an
  `httpx.MockTransport`.
- The server-mode refusal ("Key not checked: the model listing could not run
  for …") is covered there with the real runtime-policy enforcer, not live:
  reaching server mode needs a configured tldw server.
