---
id: TASK-1991
title: 'HF README viewer: resolve relative links and #anchors (Frogmouth-style tiers)'
status: Done
assignee:
  - '@claude'
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
- [x] #1 Clicking an absolute URL in a rendered README opens it in the browser (current behavior preserved)
- [x] #2 Clicking a relative link resolves it against the model repo's base URL and opens the resolved URL in the browser
- [x] #3 Clicking a `#anchor` link scrolls the README pane to that heading via goto_anchor
- [x] #4 An unresolvable href produces a visible notify with the href — never a silent no-op
- [x] #5 No fetch path is added that bypasses the egress policy
<!-- AC:END -->

## Implementation Plan (the how)

1. Pure resolver `resolve_readme_href(href, repo_id)` returning ("anchor"|"browser"|"unresolvable", value): anchors strip `#`; http(s)/mailto pass through; `//host` gets https; any other explicit scheme (javascript:, file:, data:…) is unresolvable; relative paths urljoin against `https://huggingface.co/<repo>/blob/main/`; relative with no known repo is unresolvable.
2. `open_links=False` on `#readme-display`; `@on(Markdown.LinkClicked, "#readme-display")` handler applies the tiers — `goto_anchor()` for anchors (warning notify when the section does not exist), `webbrowser.open` + notify for browser, warning notify otherwise.
3. Unit tests over the tiers + a mounted-widget test driving real `Markdown.LinkClicked` events.

## Implementation Notes

`Widgets/HuggingFace/model_card_viewer.py`: resolver + handler as planned; no fetch path added anywhere (browser-only), so the egress policy is untouched (AC#5). Repo id comes from the already-displayed `model_info`. Tests: `Tests/UI/test_hf_readme_links_1991.py` (resolver tiers incl. traversal-adjacent cases, browser/anchor/missing-anchor/unsupported-scheme flows with monkeypatched `webbrowser.open`). Consuming suite `test_non_obscuring_focus_contract.py` runs 95/96 — the 1 failure is pre-existing on dev BEFORE this branch (verified at `054cdbd92`; CSS focus-contract assertion unrelated to this change).
