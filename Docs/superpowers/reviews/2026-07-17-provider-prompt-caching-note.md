# Provider-side prompt caching: what task-245 sets up, and what is follow-up

task-245 memoizes `_make_call_model`'s fence-protocol render per active-set
change instead of per turn (`AgentService._make_call_model`,
`tldw_chatbook/Agents/agent_service.py`). The payload text was already
byte-identical turn to turn before this change — the memo only removes
redundant recomputation, it does not alter what gets sent. That byte-
stability is the *precondition* for provider-side prompt caching, not the
caching itself. None of the below is implemented by task-245; it is
explicit follow-up.

## (a) Anthropic: `cache_control` breakpoints

Anthropic's prompt caching needs an explicit `cache_control: {"type":
"ephemeral"}` marker on a content block, which requires the system
prompt as structured content blocks (a list of `{"type": "text", "text":
...}` dicts) rather than the plain string built today. That change
belongs in `chat_with_anthropic`'s payload build, not in
`agent_service.py`. It is moot for the agent loop specifically until
task-263 makes Anthropic tool-capable (fence mode is provider-agnostic;
native mode is Anthropic-shaped tool use) — but equally applicable to
plain, non-agent chats through the same provider call today.

## (b) OpenAI/compatible: automatic prefix caching

Needs no API-side markers — it keys off a stable prompt *prefix*. This
task's byte-stable system prompt plus the loop's already-stable
`[system, ...history]` message ordering already satisfies half of what
prefix caching wants. The other half — history growing by **append
only**, never rewriting/reordering earlier messages — the loop already
satisfies unchanged; no further work needed there.

## (c) Native-mode symmetry

`schemas_to_openai_tools(schemas)` (native branch, same closure) is
likewise recomputed every turn and byte-stable per unchanged active set,
exactly like the fence branch was before this task. A follow-up could
memoize it identically. Provider-side `tools` arrays only participate in
OpenAI's prefix caching if serialized stably; dict insertion order is
already deterministic here, so that precondition is likely already met —
worth confirming, not assuming, when that follow-up lands.

## (d) Measurement before further investment

Before spending more effort on caching plumbing, log provider-reported
cache-hit usage fields (e.g. `cached_tokens` / `cache_read_input_tokens`,
naming is provider-dependent) from the response `usage` block. That
tells us whether this task's byte-stability is already earning cache
hits on providers needing no explicit markers (OpenAI-compatible) before
deciding whether (a)/(c) are worth building.
