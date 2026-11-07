# Lessons: hook teardown verification

## Exercise trusted projection beside a real standalone hook (TASK-33648, 2026-10-06)

**Incident.** The initial SessionEnd disposal fix passed its standalone close,
disposal, grant-change and cancellation controls. Independent source review then
found that installing a native plugin enables the shared event projector, which
returns a copied event even for a standalone handler. Comparing the authority
callback with the original event by object identity consequently suppressed a
valid granted SessionEnd. A real installed/trusted/activated native plugin plus
a saved standalone grant reproduced both disposal paths as missing process
markers (two behavioral failures, no errors).

**What to do.** Cover supported mixed owners through their real projector. Keep
original host-event/execution provenance, but authenticate the exact trusted
projected callback through its existing live delivery; copied labels alone grant
nothing. Replace inherited per-delivery context and reset it on every exit, and
retain public replay, current-grant and physical-cleanup checks. The corrected
62-case selection passes; its broader mounted-executor teardown timeout remains
non-green and separately reproduces on the complete unchanged base. Evidence:
`Docs/superpowers/qa/2026-10-06-task33648-session-end/`.
