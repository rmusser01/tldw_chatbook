# PR3346 second-wave runtime CI repair

Source baseline: `022e3e54353ada2eb5e271247144df6527df7dcb`, with this report's uncommitted changes. Paired target: archive of `3412c2b6a3c362cb710cdffdc8293a617e1bdd55`. Newer fetched dev has not been used for these results. No full suite, Qodo, push, rebase or commit was performed by this worker. Parent owns final integration and manifests. TASK38544 AC16 and its approved implementation plan cover this work; existing ADR286 is amended for the accepted operational identity boundary.

## Diagnosis and fixes

1. **Accepted recovery receipt identity (real feature regression).** Both agent/non-agent capture-off continuation failures were absent on target dev. A real native commit accepts attempt A. Optional trace fallback recovery creates diagnostic/provider admission B. The stream owner incorrectly submitted B to the mandatory operational receipt writer; the immutable native checkpoint correctly refused it (`mixin_checkpoint_ownership`). `save_probe.py` observed the actual save boundary, recording IDs/counts only. The validated native resume now passes its checkpoint attempt A explicitly through the stream owner. Default fresh/Retry call sites continue using their existing authority IDs; diagnostic admission B remains independent. No checkpoint validation, DB immutability, source fence or receipt authority was relaxed.

   The actual recovery test now verifies durable original A snapshot/marker, exactly one receipt, no new optional trace capture on capture-off, no current-source reads, literal bodies only in provider envelopes and not transcript. Nonempty source replacement plus unavailable library still succeeds from the accepted snapshot. Missing receipt and incomplete child payload with retained marker both refuse before another provider request. Foreign continuation ownership still refuses. An initial corrupt-test setup attempted to modify immutable receipt fields and correctly failed the DB trigger; it was replaced with child-payload removal, without disabling triggers. Existing operational-inspection/display-count nonauthority control also passes.

2. **Wizard environment selector (real upstream bug).** A configured but unset environment-name hint was passed as `credential_env_var` even when the final credential source was `none`. The strict ProviderSetupDraft rejected this contradictory input. `setup_probe.py` observed the real builder's validation failure, not a UI timeout hypothesis. The wizard now carries a selector only when the final source is environment. Keyless, stored-key and explicit-environment controls exercise the real mutation builder and keep strict validation. The new keyless control failed before the fix; the other two already passed.

3. **Streaming response fixture.** Six native local provider cases used a Mock response whose headers were not a mapping. Existing provider Content-Type inspection (introduced by fc31874bdd for servers ignoring streaming) raised on this fixture. Response headers are now an actual SSE Content-Type mapping. Provider, model, native schema, reasoning and credential/capture assertions remain.

4. **Conditional error-copy fixture.** Five old assertions required the removed generic model-picker wording. They now assert the actual HTTP error/refusal and current Chat settings/config.toml recovery destinations. Conditional streaming guidance, no-cloud-retry behavior, request counts and error/usage checks remain intact.

5. **Trace usage fixture.** Two tests expected raw provider usage in a field now storing normalized per-call ProviderUsage. They now require exact provider/model and all normalized counters, including first-leg 21 input/0 output and retry-leg 21 input/5 output. Project guidance, both calls and lineage remain asserted.

6. **Light-meter representation fixture.** Two tests inspected only native `tools=`, although local fenced tool schemas correctly live in prepared system messages. They now capture the real capacity preparation's full messages and tools and compare both exactly with the actual gateway request. Preview intentionally uses the latest composed workspace/profile context; following the existing PR3295 representation control, a real first send establishes that context before the second-send comparison. The cold comparison probe isolated the only difference to the default-workspace/no-bound-folder notice. The final assertions require all and exactly three light tool names in the fenced prompt. Small unverified-window behavior still requires no tools and the exact plain system prompt. No production preview logic was changed.

## Exact targeted evidence

Each `<label>.json` records argv, cwd where applicable, exit code and runner elapsed time; corresponding logs/XML preserve results. `run.py` is the exact runner. Its fixed 300-second timeout was not raised. Timings below are pytest summaries, not runner overhead.

| Evidence | Result | Meaning |
| --- | --- | --- |
| red | 20 failed, 38.98s | Exact 18 runtime failures plus P0 and UI31 on HEAD022; excludes intentional QA negative control. |
| baseline | 18 failed, 2 passed, 37.13s | Same exact nodes against target archive; only capture-off recovery pair is feature-added. |
| probe + receipt-save-observations | Reproduced failure | Actual operational save uses B against checkpoint A. |
| wizard-probe + wizard-observations | Reproduced failure | Actual contradictory credential-selector validation. |
| wizard-red | 1 failed, 2 passed, 5.06s | New source controls before product fix; keyless fails. |
| fix-first | 24 passed, 2 failed, 35.32s | All six provider fixtures, five copy fixtures, two normalized usage cases, ten recovery cases and existing authority control passed; only corrupt-test setup hit immutable trigger. |
| fix-second | 6 passed, 2 failed, 15.21s | Corrected two corrupt-receipt cases and all four wizard cases passed; only then-incomplete meter comparison failed. |
| meter-probe | Reproduced mismatch | Exact planned/actual request differs only by latest workspace context notice. |
| meter-final | 2 passed, 6.27s | Full representation checks with real composed-context setup. |
| meter-bounded-final | 2 passed, 5.99s | Added exact bounded light-set assertions. Final test logic. |

These runs overlap and are not summed into a new monolithic run. All 18 originally assigned runtime failures are covered by passing cases across fix-first, fix-second and meter-bounded-final. Final whitespace-only formatting was checked AST-equivalent; completed tests were not repeated for formatting.

P0 natural Preparing source-await is handed to the parent's performance worker. UI31's old quiet-gap node is superseded by upstream PR3338's per-send start-window ownership and replacement tests; parent will qualify after rebase. Neither is claimed fixed or tested on newer dev here.

## Scope and static checks

Changed owned paths: console_chat_controller.py; UI/Wizards/first_run_setup_state.py; Tests/Chat/test_local_reasoning_gateway.py, test_console_refused_stream_retry.py, test_self_hosted_retry_accounting.py, test_console_light_surface_context_meter.py; Tests/Wizards/test_first_run_setup_integration.py; ADR286. `source-changes.patch` contains only these eight paths. Other workers' changes are excluded.

`lint-final.json` compares each of the seven Python files against HEAD022 under the same Ruff configuration: diagnostic code/message multisets unchanged and no newly introduced formatter changes. This is a delta claim, not a claim that inherited lint debt is clean. Source SHA256 values are recorded there. `duplicate-methods.json` confirms controller AST duplicate-method inventory unchanged (property setters excluded). `diff-check.txt` is exit 0. No module-size, performance, architecture counter or inventory ratchet was raised by this worker.

The helper's new `mixin_attempt_id` is populated only from the validated native checkpoint at resume and passed through its wrapper; all other callers retain default identity behavior. No service/DB operational ownership code changed. Product and tests remain uncommitted for parent review.
