# ADR-200: Explicit preset fallback targets before tool activity

Status: Accepted for the user-authorized orchestration burn-down
Date: 2026-09-29
Task: TASK-32508
Amends: ADR-147; extends ADR-110 with a separate preset policy

## Decision

A named preset may author an ordered, bounded `fallback_models` list of explicit provider/model pairs. Empty is off. No URLs or model-authored parameters are accepted. Resolve and freeze each candidate's registry or effective configured/default endpoint URL, execution keys (including the custom engine choice), and child-owned sampling parameters through the existing ADR-147 resolver before admitting the child. Invalid authoring fails validation; unavailable candidates produce visible skips rather than silently shortening the chain.

Reuse the existing fallback runtime and retry classifier. Preset fallback applies only to a fresh child before the first proposed/attempted tool batch, including runtime and refused tool calls. It is disabled for retained/private continuation. Only typed retryable rate-limit, overload, transport/provider timeout and explicitly classified unavailable-model failures authorize a switch; arbitrary 400/404, auth errors or exception text do not. Existing provider-only `fallback_providers` retains ADR-110's mid-run behavior and is not merged into a preset chain.

Persist authored fallback pairs on the definition and full resolved targets plus active index on the run. Keep original `resolved_*` as audit identity. Persist the selected index before dispatch; a failed write stops. Continuation uses the active frozen target without enabling additional switches. Configuration edits cannot retarget it. Each candidate rebuilds model, raw selection identity, URL, sampling params, native protocol and cache state from its own target; a primary endpoint or params cannot leak. Execution-family capability checks and dispatch use the frozen execution key, distinct from raw registry identity. Credentials are never persisted in a target snapshot. Routed gateway resolution requires the raw registry entry to exist before consulting its current credentials and readiness; a deleted entry refuses before any probe and cannot borrow family credentials. Existing owned Console resolutions retain their normal call-local credential semantics. Live and resumed fleet metadata follows the active target while the original audit columns remain unchanged.

Each model attempt consumes existing model/token/wall/automatic-work allowances. Switching does not reserve another fleet slot, reset a deadline, grant new permissions or replay tools. Cancellation and budget exhaustion stop the same child. Fallback steps identify provider/model/reason without URL or credentials.

Schema changes include normal migrations, installed backup schema and exact recovery authorization. Settings uses the existing preset editing surface and tokens; expose the explicit provider/model pairs without a second route registry.

## Alternatives

A second fallback engine would duplicate retry and budgeting. Re-resolving after failure would let config edits redirect a running child. Inferring unavailable-model semantics from text or generic HTTP status could hide auth/validation failures. Retrying after tools under this policy is deliberately excluded; ADR-110 remains the explicit legacy mid-run contract.

## Verification

Validate authoring and storage round trips, same-provider alternate models, native/fence and custom endpoint transport, own URL/params, pre/post-tool failures including refused tools, frozen active-target continuation, cancellation and automatic accounting. Use typed unavailable-model errors with provider mappings, not arbitrary exception strings.
