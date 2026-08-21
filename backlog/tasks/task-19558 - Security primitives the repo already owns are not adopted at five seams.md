---
id: TASK-19558
title: >-
  Security primitives the repo already owns are not adopted at five seams
  (FTS5 quoting, defusedxml, log_sanitizer, path redaction, risk tags)
status: To Do
assignee: []
created_date: '2026-08-21 20:08'
labels:
  - security
  - db
  - tech-debt
priority: medium
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 2 (security & privacy) — Tier 3 items
**#7, #8, #9, #10, #11**. Grouped because they share one disposition: *the
correct primitive exists in this repo and these call sites do not use it.*
All re-verified at this branch base.

**1. FTS5 quoting — including three sites where the protection is a dead
store.** This is the sharpest of the group. Several `ChaChaNotes_DB.py` search
methods compute a quoted term and then **bind the raw one**:

```
11322:        safe_search_term = f'"{content_query}"'
   ...
11335:        params_list: List[Any] = [content_query]      # ← raw, not safe_

 7112:        safe_search_term = f'"{search_term}"'
   ...
 7122:        cursor = self.execute_query(query, (search_term, limit))   # ← raw
```

`safe_search_term` is used only in the error message. It **reads as protection
in code review and is not protection at all** — the same shape also at
`:9253`. Separately, the quoting that *is* applied never doubles embedded `"`
characters (`:13293`, `:12045`, `:12189`), permitting column-filter injection
into the MATCH expression and raising on ordinary input. Consequences already
visible: the live Library notes filter **swallows the exception and returns
silently wrong results**, and the Study flashcard search box raises a bare
`OperationalError`. Equivalent sites exist in the Evals, Media and RAG
searches.
The repo already owns `Utils/fts5_match_forms.quote_fts5_token` — currently
used only by `RAG_Search/simplified/rag_service.py`.

**2. OPML import uses stdlib etree.**
`Subscriptions/watchlist_opml_service.py:3` — `import xml.etree.ElementTree as
ET`, while **every sibling parser uses `defusedxml`**, which is a core
dependency. The lane probed this: the exposure is billion-laughs entity
expansion, **not** XXE. One-line fix. CONFIRMED live.

**3. `log_sanitizer._is_sensitive_log_key` computes a normalized key and never
uses it for the suffix rules.** `Utils/log_sanitizer.py:31-32` builds
`normalized` (lowercased, hyphens→underscores) but passes the **raw** key to
`is_sensitive_config_key`, using `normalized` only for an exact-membership
check. Proven by probe to return False for `Ocp-Apim-Subscription-Key`,
`X-Subscription-Token`, and bare `key`. This is **LATENT** — the module has
exactly one call site in the whole package
(`Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py:82`) —
but it is a defect in a primitive that TASK-19555 may want to lean on.

**4. Agent-supplied paths are echoed at WARNING.** `validate_path` prints the
full path unless `redact_paths=True`, and only 9 of 30 call sites pass it.
`Tools/local_tool_impls.py:53` omits it, so model-supplied paths are echoed.

**5. Local-tool risk-tag asymmetry.** `fs_read`/`fs_list`/`fs_glob`/`fs_grep`,
`web_*` and `watchlists_*` declare `tags=()` in
`Agents/local_tool_provider.py` (lines 1054, 1084, 1202, 1234, 1257, 1300…)
while the equivalent builtins declare `("reads",)`. The documented rationale
for the split is untrusted MCP-supplied tags — which **does not apply to local
tools, whose tags we author ourselves**. Untagged tools are not floored to
`ask`.

## Acceptance Criteria

- [ ] Every FTS5 search path binds the value it actually quoted — no call site
      computes a `safe_*` term and then binds the raw one
- [ ] FTS5 term quoting doubles embedded `"` characters, via the existing
      `Utils/fts5_match_forms.quote_fts5_token` rather than a new local helper
- [ ] A search term containing `"` and a column-filter expression returns
      correct results (or a clean user-facing error) instead of silently wrong
      results in the Library notes filter or a bare `OperationalError` in the
      Study flashcard box
- [ ] The Evals, Media and RAG search sites are fixed in the same sweep
- [ ] A test fails if a `safe_*`-style dead store is reintroduced at any FTS5
      call site
- [ ] `watchlist_opml_service.py` parses with `defusedxml`, and a test pins
      that an entity-expansion OPML payload is rejected
- [ ] `_is_sensitive_log_key` applies its suffix rules to the normalized key;
      `Ocp-Apim-Subscription-Key`, `X-Subscription-Token` and bare `key` are
      recognised as sensitive, pinned by test
- [ ] `Tools/local_tool_impls.py` passes `redact_paths=True`; the remaining
      `validate_path` call sites that handle untrusted paths are swept
- [ ] The local-tool risk-tag decision is made explicitly: either the read-only
      local tools carry `("reads",)` like their builtin equivalents, or the
      rationale for the asymmetry is written down where the next reader will
      find it
