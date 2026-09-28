# ADR-165: Keep llama.cpp tuning profiles separate from source and connection authority

Status: Accepted
Date: 2026-09-17
Extends: ADR-025, ADR-114, ADR-119
Related: ADR-117, ADR-150

## Decision

Implement the first milestone accepted after the Catapult review: verified llama.cpp readiness and Console adoption, tuning-only named profiles, and bounded startup diagnostics.

ADR-114 continues to own verified connection descriptors, process versus API readiness, the reserved `chatbook-llamacpp` alias, absent-value port 8080, session-only Console adoption, and explicit Settings persistence. ADR-025 continues to own GGUF sources and process-held artifact leases. ADR-119 continues to own snapshot compatibility and slot options.

Profiles contain a name, stable opaque identifier, and typed tuning: context size, GPU layer count, CPU threads, parallel slots, flash attention, K/V cache types, and logical/physical batch sizes. An unset value means omit its flag and use the installed runtime default. Profiles contain no executable, model, projector, external path, endpoint, credential, raw arguments, or Console sampling values. Selecting or saving a profile never starts/restarts a server or changes provider defaults. Source and expert argument edits remain session-local.

The device-local profile document is `llamacpp_launch_profiles.json` under the active user-data directory, version 1, capped at 32 profiles and 256 KiB. Writes use compare-and-swap revisions, interprocess locking, and atomic replacement. Duplicate keys, unsupported versions, corrupt data, duplicate case-insensitive names, and stale writes fail with recoverable bounded messages; corrupt files are never silently replaced. Existing vLLM ownership patterns are precedent, without changing vLLM storage or extracting a generalized profile framework.

Structured values and expert arguments have no silent precedence: reject an expert option when its corresponding structured field is set. Always reserve model-source, alias, host, and port options for their existing owners. Slot ownership remains conditional on snapshots being enabled. Preserve quote-aware argv parsing; never invoke a shell. An explicit unsupported runtime option produces a visible launch failure rather than silently dropping it.

Diagnostics remain process-local and Lab-local. Drain stdout/stderr in bounded chunks; keep at most 128 classified entries with no raw payload stored in app state. The default diagnostic output uses an allowlist of fixed categories and numeric exit status rather than attempting to make arbitrary model/prompt text safe with regex redaction. Recognize loading, backend/library failures, unsupported options, out-of-memory failures, bind failures and model-load failures without echoing the source line. Unknown lines are drained and suppressed. No diagnostic line establishes readiness; only current HTTP/model evidence can do that. A diagnostic buffer belongs to one exact launch claim and stale callbacks cannot replace its successor.

## Alternatives

- Persist complete launch commands or path-keyed model presets: rejected because these conflict with ADR-025's external-path ownership and mix credentials/source authority into reusable tuning.
- Automatically apply model sampling recommendations: outside this milestone; Console retains request-setting ownership.
- Raw log tail with regex scrubbing: rejected for the initial milestone because prompts and arbitrary paths have no reliably exhaustive redaction grammar. Fixed diagnostic messages deliver useful recovery without retaining arbitrary output.
- Replace lifecycle or reuse vLLM types by changing their provider identity: rejected; share established patterns while keeping exact provider contracts and existing process claims.
- Install runtimes, import publisher presets, or automate snapshots now: deferred to separately scoped work after this milestone.

## Verification

Targeted tests cover profile persistence/revision conflicts, malformed documents, managed/raw flag conflicts, HTTP health/model readiness, stale generations, exact process ownership, and session-only adoption. Real child-process fixtures cover pipe draining and cancellation. Mounted Textual tests cover the user actions and preserve manual snapshot/source controls. Live llama.cpp qualification is recorded only when an existing suitable binary and model are available; deterministic loopback fixtures are not represented as model-inference evidence.

## References

- [Catapult comparison](../../Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md)
- [Milestone design](../../Docs/superpowers/specs/2026-09-17-llamacpp-management-milestone.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-17-llamacpp-management-milestone.md)
