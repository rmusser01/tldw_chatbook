# UI Screenshot Approval Workflow Handoff

Date: 2026-05-08
Status: binding for future UI work
Current branch: `codex/visual-ui-correction`
Current base at creation: `origin/dev` = `037d7b3e` (`Correct destination visual parity (#283)`)

## Purpose

This handoff adapts the Console visual correction workflow into the main UX thread and the original paused Phase 3.9 workflow chain.

The durable rule is:

**No UI screen or UI state is approved until the user reviews and approves an actual rendered screenshot of the running app.**

## Approval Rules

- Actual screenshots only: PNG screenshots from the running Textual app or `textual-web`/`textual-serve` browser surface.
- Not approval evidence: SVG captures, ASCII diagrams, mockups, code layouts, or Textual geometry dumps.
- Geometry dumps and ASCII are still useful for diagnosis and planning, but they cannot replace user screenshot approval.
- Every changed screen/state needs its own screenshot approval before the work is called visually approved.
- If the user rejects a screenshot, fix the UI, recapture the actual rendered screen, and ask again.
- Archive approved screenshots in the relevant QA evidence folder. For current Phase 3 UI work, use:

`Docs/superpowers/qa/product-maturity/phase-3/actual-visual-captures/`

## Recommended UI QA Loop

1. Run the real app, or a focused harness that mounts the production screen/widget with the production TCSS bundle.
2. Capture a PNG screenshot from the rendered app surface.
3. Inspect the screenshot before presenting it. If it shows a loader, crash, blank state, or wrong screen, debug and recapture.
4. Present the screenshot path to the user and wait for explicit approval.
5. After approval, record the screenshot path, verification commands, and residual risks in a QA note or PR body.
6. Keep temporary automation output such as `.playwright-cli/` out of commits unless explicitly required.

## Troubleshooting Checklist

- If `textual-serve` shows only a loading screen, inspect server stdout/stderr before accepting the screenshot.
- If a temp harness is used, make `CSS_PATH` absolute; otherwise Textual may resolve TCSS relative to `/private/tmp`.
- If local port binding fails under sandboxing, rerun the local server/browser capture command with escalation.
- If browser automation is flaky, use mounted Textual tests to put the UI into the needed state, then still capture an actual rendered screenshot.
- If the composer or focused control is hard to see, type/paste real content and capture the resulting rendered state.
- Use `file <path>.png` to verify screenshot artifacts are PNGs before citing them as evidence.

## Current Console Paste-Collapse Evidence

User-approved actual rendered screenshots:

- `Docs/superpowers/qa/product-maturity/phase-3/actual-visual-captures/console-large-paste-collapsed-2026-05-08.png`
- `Docs/superpowers/qa/product-maturity/phase-3/actual-visual-captures/console-large-paste-unfurl-prompt-2026-05-08.png`
- `Docs/superpowers/qa/product-maturity/phase-3/actual-visual-captures/console-large-paste-unfurled-2026-05-08.png`

## Main UX Thread Adoption

For future Home, Console, Library, Artifacts, Personas, Watchlists, Schedules, Workflows, MCP, ACP, Skills, and Settings work:

- Start with the current `dev` branch after the latest UX PR merge.
- Capture the actual rendered screen before judging visual quality.
- Compare screenshots against the approved Textual-native ASCII/design contracts and user feedback.
- Do not claim visual parity or screen approval from tests alone.
- Preserve power-user speed while validating that first-time users can see the primary action, current state, and recovery path.

## Original Phase 3.9 Workflow Chain Update

When the original paused workflow resumes:

- Treat Phase 3.9 Library Collections as merged and verified on `dev`.
- Treat destination visual parity correction as merged and verified on `dev`.
- Do not resume from stale temporary worktrees or old concept-page artifacts.
- Fetch `origin/dev` and continue from a clean worktree after the current Console paste-collapse PR merges.
- Preserve the product-model decisions already made:
  - `Watchlists` is the top-level monitored-source destination.
  - `Collections` lives inside `Library`.
  - Citations/snippets and citation carry-through remain later-stage Library/Search/RAG work.
  - Console remains the primary agentic control surface.

## Resume Prompt For The Other Thread

Use this as the continuation packet:

```text
Resume from current origin/dev after the Console paste-collapse PR merges. Phase 3.9 Library Collections and destination visual parity correction are already merged. Do not use stale temporary worktrees or generated concept pages as approval evidence.

New binding UI rule: no screen/state is approved without an actual rendered PNG screenshot from the running Textual app or textual-web surface and explicit user approval. SVGs, ASCII diagrams, geometry dumps, and code layouts are diagnostic/planning artifacts only.

For future UI work, capture screenshot -> inspect it -> show the user -> wait for approval -> archive under the relevant QA evidence folder -> record verification and residual risks.

Keep Watchlists top-level, Collections inside Library, Console as the primary agentic control surface, and citations/snippets as later-stage Library/Search/RAG items.
```
