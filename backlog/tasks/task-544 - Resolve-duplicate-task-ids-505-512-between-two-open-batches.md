---
id: TASK-544
title: >-
  Resolve duplicate task ids 505-512 between two open batches
status: To Do
assignee: []
created_date: '2026-07-24 07:15'
updated_date: '2026-07-24 07:15'
labels:
  - backlog-hygiene
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every id 505-512 exists TWICE on dev (the backlog CLI resolves by id → all eight are ambiguous). Two batches collided: (a) a web-scraping/egress batch created '2026-07-23 12:00' (Confluence sync, scrape_from_sitemap, recursive_scrape, guarded_fetch, Subscriptions validator, redirect credentials — all To Do), and (b) a model-artifact/STT batch created '2026-07-24 01:01-01:03' (artifact leases/descriptors/downloads/browser, GGUF/ONNX import, STT contracts — task-505 of this batch is **In Progress**).

NOT resolved unilaterally because batch (b) appears to belong to a live session (In Progress task); renumbering under an active branch would just re-introduce dupes on its next merge, and per the standing rule the mover should be the not-started side with its owner aware. Whichever session finishes (or a coordinated cleanup) should renumber ONE side (rule: In Progress/older keeps; per-pair — batch (a) is older for all pairs but batch (b) has the In Progress 505) to the next free ids, updating frontmatter `id:` + any cross-references, then re-run the two-namespace dup-check (python os.listdir scanner, not git-ls-tree|uniq).

Note: this session already resolved its own pairs (503 RAG-SP3 kept / MCP-nav → 542; 519 get_user_data_dir kept / console-branching → 543) in the same PR that files this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Each id 505-512 identifies exactly one task on dev (filename prefix AND frontmatter id namespaces).
- [ ] All cross-references (dependencies, prose) updated to the surviving/renumbered ids.
<!-- AC:END -->
