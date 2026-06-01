# Provider CDP UAT

This folder contains the manual CDP/Textual-web user acceptance test harness for Chatbook Console provider coverage.

## Scope

- Exercise provider setup and Console send behavior through the rendered Textual-web app.
- Use CDP only for browser control, screenshots, short text entry, key presses, and app-log polling.
- Treat a provider as accepted only after the rendered app receives two assistant replies in the same Console session.
- Do not use adapter-only calls as UAT pass evidence.

## Safety

- Never place raw API keys in screenshots, logs, reports, commits, command output, or issue comments.
- Inspect screenshots before keeping them as evidence when any provider settings screen was visible.
- Prefer credential source labels such as `env_file:OPENAI_API_KEY` or `process_env:ANTHROPIC_API_KEY` over values.
- Log excerpts must be redacted before being pasted into reports.
- Do not run the UAT flow against the normal user profile. Use the isolated HOME/XDG paths from the launch helper.

## Source Of Truth

The runtime inventory JSON is the source of truth for provider rows, readiness keys, execution keys, model choices, key-source labels, and endpoint reachability:

`Docs/superpowers/qa/provider-cdp-uat/provider-inventory.json`

Regenerate the inventory when provider code, config defaults, model defaults, or reachable local endpoints change. The Markdown inventory is a readable derivative; update report tables from the JSON-backed inventory.

## Skip And Fail Classification

- `pass`: two Console turns completed through the rendered app and the second assistant reply is visible.
- `fixed-then-pass`: a Chatbook defect was captured, fixed, rerun through CDP, and then met the pass criteria.
- `skip`: the provider was not attempted because it lacked a usable key, required an unreachable local/custom endpoint, or was otherwise not testable in the isolated profile.
- `fail_external`: the provider was attempted but an external condition blocked acceptance, such as authentication, quota, provider outage, endpoint reachability, or model availability.
- `fail_chatbook`: the provider was attempted and evidence points to Chatbook request construction, response normalization, streaming fallback, or Console UI behavior.

Use these failure categories in the report:

- `missing_key`: no usable key exists in the inventory source or isolated settings.
- `auth`: the provider rejects the key.
- `quota_or_rate_limit`: quota, billing, or rate limiting blocks completion.
- `model_unavailable`: the selected model is invalid, retired, unavailable to the key, or incompatible.
- `request_shape`: Chatbook sends an invalid provider-specific request.
- `response_shape`: Chatbook receives a response but fails to normalize or display it.
- `streaming`: the streaming path fails when non-streaming should work or fallback should occur.
- `endpoint_unreachable`: a local/custom endpoint is unavailable.
- `console_ui`: the rendered app prevents setup, selection, send, or verification.
- `unknown`: available evidence is insufficient for a tighter category.

Only `request_shape`, `response_shape`, `streaming`, and `console_ui` are default candidates for fixes in this UAT pass. External provider and endpoint issues are recorded as skips or failures with evidence.

## CDP Helper Sequence

Use the helper without printing raw keys or raw environment values:

The Node helpers load `playwright` from normal Node resolution by default. If Playwright is supplied by a bundled runtime instead of local `node_modules`, set `TLDW_QA_PLAYWRIGHT_BUNDLE_PATH` to that runtime's importable Playwright package path before running the helper.

1. Capture the pre-send log offset:

   ```bash
   TLDW_QA_APP_LOG=/path/to/tldw_cli_app.log node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs readLogTail
   ```

   Copy the returned `nextOffset`. If the log is large, `readLogTail` may clip displayed text and will report `truncated`, `truncatedBytes`, and `maxTailBytes`.

2. Focus and send through Textual-web:

   ```bash
   node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs focusTerminal
   node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs typeText "Reply with one short sentence: provider UAT turn one."
   node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs press Enter
   ```

3. Wait for app-log evidence from the captured offset:

   ```bash
   TLDW_QA_APP_LOG=/path/to/tldw_cli_app.log node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs waitForLog "Updated message ID" 45000 <nextOffset>
   ```

4. Capture the rendered state:

   ```bash
   node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs screenshot provider-turn-1
   ```

## Evidence Folder Conventions

- Screenshots live under `Docs/superpowers/qa/provider-cdp-uat/screenshots/`.
- Use names that start with a provider or flow label, for example `openai-turn-2.png` or `settings-openrouter-model.png`.
- Keep generated inventory files in this folder and commit them only during inventory/report tasks.
- Keep isolated HOME, XDG config, XDG data, runtime logs, and temporary app state outside the repository.
- Report command output should include screenshot paths, numeric log offsets, and redacted summaries only.
