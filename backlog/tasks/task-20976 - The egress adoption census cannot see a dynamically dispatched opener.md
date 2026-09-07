---
id: TASK-20976
title: >-
  The egress adoption census cannot see a dynamically dispatched opener
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - testing
  - test-integrity
  - egress
  - security
  - hygiene
priority: low
dependencies:
  - TASK-19556
---

## Description

Source: disclosed by **TASK-19556**'s implementer when Qodo required the
adoption census be rewritten on the AST. Re-verified at `684c6aba4`.

`Tests/Utils/test_egress_adoption_census.py` is the guard that stops the *next*
network seam from silently skipping the egress policy. It decides, per module,
"does this open URLs" and "does it consult the policy". Round one of TASK-19556
made that decision by regex over raw source text, which was shown to be
bypassable in both directions — most sharply, `Local_Ingestion/
video_processing.py`'s own docstring quotes `guarded_fetch_requests(...)` in
prose, and the regex read that as "still consults the policy" **even with the
actual call deleted**, so the guard would not have caught its own subject
regressing. It is now an AST scan (`_analyze_source` → `_scan(ast.parse(source))`,
`Tests/Utils/test_egress_adoption_census.py:192-193`), with three
non-hollowness proofs.

The AST version is strictly better and has one honest, known limit that should
be recorded rather than discovered later. A statically-resolvable call like
`urllib.request.urlopen(u)` is a `Name`/`Attribute` chain the scanner can trace.
A dynamically dispatched one — `getattr(urllib.request, "urlopen")(u)`, or the
same shape through a dict lookup or a module alias — is not, so an unguarded
opener written that way would not be flagged. The old regex would have matched
the string literal.

No opener in the codebase uses that pattern today; a search for
`getattr(<module>, "<opener>")` across `tldw_chatbook/` returns nothing. So this
is a documented boundary of a guard, not a defect in it, and it should not be
re-reported later as an unguarded seam.

The value in doing something about it is that the limit is currently known only
to the people who were in that review. A test that states the boundary makes it
survive them, and makes it visible to whoever next widens the census.

## Acceptance Criteria

- [ ] The census's inability to resolve a dynamically dispatched opener is
      recorded in the guard itself, not only in a task or a review thread
- [ ] A test documents the limit by example — a dynamically dispatched opener is
      shown to be invisible to the scanner — so the boundary is executable rather
      than prose that can go stale
- [ ] The three existing non-hollowness proofs still pass, including Qodo's
      bypass case (a comment naming the real egress function beside a genuine
      unguarded opener)
- [ ] It is re-confirmed at implementation time that no module in
      `tldw_chatbook/` currently opens URLs through a dynamically resolved
      reference, and the check used is recorded so it can be re-run
- [ ] If the limit is closed rather than documented, the closure is shown not to
      reintroduce the regex-era false greens the AST rewrite removed

## Notes

Filed explicitly as a known limit rather than a defect, and filed low. Over-rating
it would invite a fix that trades the AST's precision back for text matching,
which is the shape this programme has repeatedly found hollow — the regex version
of this very guard failed to notice its own subject regressing.
