# Product Maturity Roadmap

The canonical product-maturity design spec lives at:

`Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`

The canonical product-maturity tracker lives at:

`Docs/superpowers/trackers/product-maturity-roadmap.md`

The post-UX roadmap design lives at:

`Docs/superpowers/specs/2026-05-06-post-ux-product-roadmap-design.md`

The canonical product-maturity tracker remains the execution source of truth. Do not create a parallel phase tree from the post-UX roadmap; map new child tasks to the existing product-maturity parent tasks unless the tracker explicitly supersedes them.

Durable QA evidence lives at:

`Docs/superpowers/qa/product-maturity/`

Backlog tasks in `backlog/tasks/` are PR-sized execution units. Parent tasks represent product-maturity phases; child tasks represent implementation or QA gates.

Do not mark a product-maturity task or phase complete because UI renders or a button is clickable. Completion requires automated evidence, running-app QA evidence where product behavior is in scope, and a repo-tracked QA summary.
