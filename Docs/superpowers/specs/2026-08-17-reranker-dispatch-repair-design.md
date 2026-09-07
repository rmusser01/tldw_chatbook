# Reranker dispatch repair: delete the bespoke path, use the app's (TASK-17065)

Date: 2026-08-17
Status: draft-pending-user-review
Programme: RAG server-port (fifteen merged; last: 17165 endpoint redaction #1759)
Worktree: `.worktrees/rag-17065-reranker-dispatch`, branch
`fix/task-17065-reranker-dispatch`, off dev `1c328e1a7`.

## The decision, pre-registered: REMOVE the reranker's credential code

TASK-17065 offers two arms for AC#1. Spec review found a third option that
subsumes the better one, and it is a **deletion**:

**Every `chat_with_<provider>` handler already resolves its own credential
when the caller passes `api_key=None`** — `final_api_key = api_key or
<provider>_config.get("api_key")` (`LLM_Calls/LLM_API_Calls.py:590`,
`:1226`, `:2191`, `:2902`, …), with `resolve_provider_api_key` imported at
`:55` and the precedence CLAUDE.md documents (explicit
`api_settings.<provider>.api_key` → env → legacy `[API]`, every source
validity-checked) applied there.

So the reranker should not resolve credentials at all. Delete
`_call_llm_impl`'s hand-rolled `if/elif` (`RAG_Search/reranker.py:183-206`)
and the `self._settings = load_settings()` read it exists for
(`:128`), and call `chat_api_call` with KEYWORD arguments, passing
`api_key=None`. That single change satisfies:

- **AC#1/#2** — credentials come from the normalised path; the
  `self._settings["API"]` read that `load_settings()` never builds is gone
  rather than repaired.
- **AC#3** — keyword arguments; no positional list can misroute a key into
  `api_endpoint` again.
- **AC#5** — local providers (`ollama`/`llama_cpp`/`vllm`/`koboldcpp`/
  `mlx_lm`) are never rejected for a key they do not need, because the
  reranker stops judging keys entirely; each handler decides.
- **AC#6** — the documented precedence is obeyed by construction, because
  it is the same code every other call in the app uses.
- **AC#9** — the picker's enumeration stays derived (untouched).

Writing a second resolver inside the reranker would re-create the divergence
that caused this defect. The lesson this arc records is that one: a bespoke
credential path in a feature module is how a feature ends up calling zero of
twenty-nine providers while its tests stay green.

## What the arc must NOT assume (plan-phase verification)

1. **Do all 29 handlers self-resolve?** Verified for openai/anthropic/
   cohere/deepseek. The plan must enumerate `API_CALL_HANDLERS` and check
   each handler for the `api_key or <config>` fallback, and record the list
   of any that do NOT — those either need `api_key` passed explicitly from
   the shared resolver, or are honestly named as not-yet-callable in the
   arc's report (no silent gap).
2. **Does `chat_api_call` forward `api_key=None` cleanly** through
   `PROVIDER_PARAM_MAP` for each handler, or does any provider require a
   non-None value at the dispatcher level?
3. **What `messages_payload` shape does the reranker build**, and does the
   real signature's `temp`/`system_message`/`streaming`/`model` mapping
   accept it as keywords with the reranker's config values?
4. Whether any other caller in the repo copies the reranker's positional
   pattern (grep for `chat_api_call` call sites; the seam guard covers only
   the reranker's).

## Scope

- **In:** `RAG_Search/reranker.py`'s credential + dispatch path; the seam
  guard rewrite (below); the fakes that mirror the caller's wrong order;
  the spend release note (AC#10).
- **Out:** the picker (untouched — AC#9 holds by not touching it), the
  reranker's strategies/scoring, `cross_encoder` (TASK-16965), anything
  retrieval-affecting. **AC#8 is ALREADY SATISFIED** on dev — the picker's
  repairability shipped in TASK-3502's Qodo remediation; the arc verifies
  and cites it rather than re-doing it.

## The seam guard MUST flip (AC#7)

`test_reranker_dispatch_binding_against_the_real_chat_api_call_signature`
was written deliberately RED-ON-REPAIR: it asserts today's wrong landing
(`api_endpoint`←the key, …). This arc MUST rewrite it to assert the CORRECT
binding, and that rewrite is the proof the repair happened. Any fake at this
seam must match the real signature — the existing ones
(`Tests/RAG_Search/test_reranker_degraded_paths.py:75`) mirror the caller's
wrong assumption and are exactly why ~2,500 green tests never saw this.
Every fake this arc touches or adds binds through
`inspect.signature(chat_api_call).bind(...)` so a future mis-order cannot
pass.

## AC#4 — the acceptance evidence

A provider with a valid credential configured completes a scoring call. No
live provider call is permitted in tests; AC#4 is demonstrated by a fake
that (a) binds against the real signature and (b) asserts the reranker
handed it the endpoint/model/temperature it was configured with — i.e. the
call is well-formed at the seam. Whether the network round-trip succeeds is
the provider's business, not this arc's.

## AC#10 — the spend consequence, stated

This repair converts a silent no-op into real provider calls: one call per
candidate, up to the configured rerank top-k, on the first search of any
reranking-enabled profile. It lands on a build that already carries
TASK-3502's pre-enable cost disclosure and the skipped/degraded notice. The
arc's close-out and the release note must say plainly that reranking-enabled
profiles begin spending.

## Testing
- Credential seam: no credential logic left in the reranker to test —
  instead, a test asserting the reranker passes `api_key=None` and does not
  read a settings table (structural: the `load_settings` import/read is
  gone).
- Dispatch seam: the rewritten guard + signature-bound fakes.
- Batteries: `Tests/RAG_Search`, `Tests/Chat`, the settings-RAG files.
- The gated suite is **vacuous for the reranker** (no gated cell runs it —
  established in TASK-3502); it is still run, and the close-out repeats that
  caveat rather than implying coverage.
