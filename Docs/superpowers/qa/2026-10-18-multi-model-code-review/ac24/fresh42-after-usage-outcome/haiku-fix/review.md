# Scoped Haiku 5.5 review

What this change does: Commit `d87d165c741e9df68935833693a976c48893ce0d` adds exactly the (`haiku`, 5, 5) family to the existing Anthropic request-capability set. Existing request construction then omits sampling and maps effort to adaptive thinking; Console fixed-budget controls use the same fact.

No actionable source findings. Verdict: **Ship the scoped source correction.**

I read the four-file commit against its exact parent `3fb3c9b478c9c50bf6609c7da8f57e9be26b8c0d`, the predicate parser/table, every production predicate caller, real dispatcher payload tests, Console control tests, and the existing bootstrap-profile marker implementation. Production guards remain intact. Haiku 4.5 keeps sampling and fixed budgets; no provider boundary, owner, configuration key, schema, or future Haiku family is added. The existing family parser handles prefixed/suffixed variants; the new table row matches minor 5 only. No new ADR is required for this routine validity correction under the existing provider-policy owner.

All **31 indexed files** independently match their recorded byte lengths and SHA-256 hashes. All four working source files match the committed blobs. The commit parent and current HEAD match the supplied base and implementation commit. The root-owned dirty plan is excluded.

Retained causal RED is **9 failed, 1 passed, 661 deselected**; failures expose leaked sampling, legacy budget/token-cap expansion, wrong adaptive mapping, and incorrect Console control support. Retained covering GREEN is **671 passed**; final focused GREEN is **10 passed, 661 deselected**. Tests exercise the real dispatcher/request builder while faking HTTP POST. Retained static comparison reports zero new Ruff diagnostics and no new formatting debt against the exact base; inherited I001 findings and Console format debt remain.

## Separate private probe

No actionable probe findings. Verdict: **Ready for exactly one authorized synthetic request**, invoked with `--source-head d87d165c741e9df68935833693a976c48893ce0d` in the normal, unoptimized interpreter. This approves readiness only; this reviewer made no network call.

Reviewed script `/private/tmp/task35582-ac24/haiku-handler-probe.py` SHA-256 `2b5cee2362a6a3fc84aa8339ecac9df3846bf1ac79a9016bad467d14aebaa5f5` matches its preparation receipt. The retained pure self-check log passes; the invocation claim is absent at review time. Default execution returns before profile/key/import/claim/network work. Run mode checks source bytes, verifies the historical 211-attempt receipt, acquires an exclusive claim, uses the existing isolated profile helper with environment credentials, and installs the real-profile write guard. The secret is not written into the profile or manifest.

The live path uses the real dispatcher and Anthropic handler, exact `claude-haiku-5-5`, streaming enabled, a tiny synthetic prompt, 512 output tokens, and the two retained verifier tool schemas. It deliberately supplies sampling arguments so production omission is checked before transport. Its guard permits only the Anthropic POST endpoint, requires zero adapter retries, increments and saves the physical counter before the send, rejects every subsequent send, and blocks both synchronous and asynchronous httpx clients. Thus even an attempted handler fallback cannot spend a second request. One request would move the aggregate from **211/215 to 212/215**, leaving three; it creates no child, review session, or checkpoint. Error capture is bounded and scrubs the credential. A 200 result supports payload compatibility only, never completion of AC24 or a successful verifier journey.

## Evidence limits retained

The historical Haiku request was HTTP 400 with known-invalid sampling fields, but its error body was not retained; the precise historical rejection reason remains unknown. Neighbor coverage remains **54 failed, 39 passed** at profile admission. Only one representative node was reproduced against exact-base production bytes (525 imported modules); the whole-neighbor baseline attempt was inconclusive. These observations support the inherited failure category, not universal attribution of all failures or a green neighbor suite.

Skipped: no tests, network, source/index/HEAD edits, config/key access, new subagents, broad suite, push, or merge during this review. Risk: fixture/resource/static debt and incomplete live AC24 evidence remain; the one-call probe cannot close AC24.
