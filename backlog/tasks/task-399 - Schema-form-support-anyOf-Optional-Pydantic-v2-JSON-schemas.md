---
id: TASK-399
title: 'Schema form: support anyOf/Optional (Pydantic v2) JSON schemas'
status: Done
assignee: []
created_date: '2026-07-21 02:16'
labels:
  - mcp
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ATHF UAT found ~half of a real third-party server's 21 tools fall back to raw-JSON in the Test Tool because parse_schema() (mcp_schema_form.py) treats any property it can't render faithfully as unrenderable and fails the WHOLE schema to raw — and Pydantic v2 emits Optional params as anyOf:[{type:X},{type:null}] which parse_schema doesn't handle. This is the intended honest-fallback (partial forms lie), but anyOf-with-null is the single most common third-party pattern, so most real servers get raw-JSON forms. Handle the anyOf:[T, null] Optional idiom: render as the non-null type's field, not-required. Keep whole-schema-None for genuinely unrenderable shapes (nested objects, real oneOf, arrays).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An anyOf:[{type:string},{type:null}] property renders as an optional string field, not raw fallback,A tool schema mixing renderable + anyOf-Optional properties renders a form (not raw),Genuinely unrenderable shapes still trigger the honest raw fallback
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

Commit c4c7d221a7 'feat(mcp-hub): render Pydantic v2 Optional[T] schemas as forms' implements the anyOf:[T,null] handling, hardened by 012499e018 'harden path-literal secret guard; respect required list for nullable fields'; UI/MCP_Modules/mcp_schema_form.py documents the Optional[T] idiom at lines 8/78/123/145.
