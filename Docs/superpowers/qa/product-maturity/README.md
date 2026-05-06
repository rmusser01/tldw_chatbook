# Product Maturity QA Evidence

This directory stores durable QA evidence for the product-maturity roadmap.

Canonical tracker:

`Docs/superpowers/trackers/product-maturity-roadmap.md`

Canonical spec:

`Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`

Rules:

- Verify running-app behavior, not only rendered widgets.
- Record whether each workflow completed, was honestly blocked with recovery, or failed.
- Use one defect taxonomy label: `blocker`, `workflow-degradation`, `recoverability`, or `polish`.
- Record P0/P1/P2/P3 only when release or phase-exit decisions need that mapping.
- Store one markdown QA summary per gate.
