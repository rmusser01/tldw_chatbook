# Console Setup Card Phase 2 — QA evidence (2026-07-04)

Branch: claude/console-setup-card-phase2 (base 9e09b475).
Captured from textual-serve (real app CSS) in headless bundled chromium,
two isolated HOMEs; live llama.cpp at 127.0.0.1:9099 for the real-send flow.

- setup-card-blocked-fresh-2026-07-04.png — virgin HOME, OpenAI default, no
  key: "Get started" card (1. ● Add an API key / 2. ✓ Pick a model /
  3. ○ Send your first message  Type below, Enter to send) with the action
  row; NO top blocker banner (removed this phase, including the workbench
  recovery callout duplication found during QA, fixed in 72e6864d); draft
  typed in composer.
- setup-card-ready-line-2026-07-04.png — HOME pre-seeded with a working
  llama_cpp config: single line "Ready — type a message to begin.", no card,
  no action buttons.
- setup-card-quiet-new-tab-after-send-2026-07-04.png — after a real accepted
  send (auto-titled "Reply with one…" tab/rail row), a NEW tab shows only
  "No messages yet." — the card never returns once first_send_completed.
- setup-card-quiet-persisted-relaunch-2026-07-04.png — fresh app instance,
  same HOME: still quiet; prior conversation listed with live age label.
  On-disk proof: [console.onboarding] first_send_completed = true written to
  the HOME's config.toml.

Blocked-send behavior (observed during capture): pressing Enter while blocked
posts durable System guidance in the transcript ("Add API key in Settings >
Providers & Models before sending.") plus a composer-side callout; the card
yields to that transcript feedback (has_messages -> quiet per the approved
state builder).

Verification: full affected suites green except the one pre-existing base
failure test_console_left_rail_prioritizes_attach_and_active_conversation
(confirmed failing at 9e09b475).
