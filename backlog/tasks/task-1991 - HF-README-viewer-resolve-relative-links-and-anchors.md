---
id: TASK-1991
title: 'HF README viewer: resolve relative links and #anchors (Frogmouth-style tiers)'
status: To Do
assignee: []
created_date: '2026-08-02 22:30'
labels:
  - huggingface
  - markdown
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Widgets/HuggingFace/model_card_viewer.py` renders model READMEs in a default-configured `Markdown` widget. Real READMEs are full of relative links (`./docs/usage.md`, images) and heading anchors (`#quickstart`) — today those either do nothing useful or fail silently; only absolute URLs behave.

Frogmouth (MIT — port freely) solved exactly this in `screens/main.py` `on_markdown_link_clicked` with tiered resolution: absolute URL → open; relative href while viewing a remote doc → join against the document's base URL; `#anchor` → `Markdown.goto_anchor()` (present in installed Textual 8.2.7); anything unresolvable → an explicit "can't handle this link" message instead of silence. For the model-card viewer the base URL is the model repo's canonical blob root (e.g. `https://huggingface.co/<repo>/blob/main/`).

Scope note: opening resolved links in the system browser is sufficient; in-app remote markdown browsing is out of scope. Any future in-app fetch must go through the `Utils/egress.py` SSRF policy, not a raw client.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Clicking an absolute URL in a rendered README opens it in the browser (current behavior preserved)
- [ ] #2 Clicking a relative link resolves it against the model repo's base URL and opens the resolved URL in the browser
- [ ] #3 Clicking a `#anchor` link scrolls the README pane to that heading via goto_anchor
- [ ] #4 An unresolvable href produces a visible notify with the href — never a silent no-op
- [ ] #5 No fetch path is added that bypasses the egress policy
<!-- AC:END -->
