# Console Visual Hierarchy Phase 4 — QA evidence (2026-07-05)

Branch: claude/console-visual-phase4 (base f370954f). Captured from
textual-serve (real app CSS) in headless bundled chromium, ready-seeded
llama_cpp HOME (post-onboarding: `console.onboarding.first_send_completed`
set), live llama.cpp server — real streamed assistant responses, no fixtures.

- console-visual-ready-overview-2026-07-05.png — ready Console at rest:
  exactly one frame per region (rail/transcript/composer; no doubled rail
  rules), glyphed section toggles (`▾` open / `▸` collapsed on Details),
  directional collapse handles (`◂` Context/Inspector), `▸` active-row
  marker on the current chat, `✕` tab close, the three zero counters
  (Sources/Tools/Approvals) rendered dim next to the full-contrast
  Provider/Model/Assistant/RAG chips, one blank line between the tab strip
  and transcript content, and Stop ABSENT from the composer (Send/Attach/
  Save only — Send sole primary).
- console-visual-mid-stream-stop-visible-2026-07-05.png — mid-stream during
  a real send: Stop visible beside Send only while the run is active; dim
  `User`/`Assistant` role labels with full-contrast message body;
  `[streaming]` tail marker; tab auto-titled from the prompt.
- console-visual-after-send-dim-roles-2026-07-05.png — stream complete:
  Stop hidden again, dim role labels vs full-contrast six-paragraph body,
  single transcript frame, breathing room under the tab strip.
- console-visual-selected-message-2026-07-05.png — resumed the persisted
  conversation (fresh app process), selected a message: selected treatment
  is the repo's non-obscuring-focus contract (focus background +
  bold-underline text + inline action row/guide). NOTE: the plan's
  `border-left: thick $ds-accent` was intentionally NOT applied — it is
  forbidden by the actively-enforced contract
  (Tests/UI/test_non_obscuring_focus_contract.py rejects `border-left` and
  `$accent` on `.console-transcript-message-selected`). Decision
  (keep contract treatment vs amend contract) is an approval-gate item.

Chips-bright (`console-chip-alert`) state: not staged in the live capture
(needs seeded staged sources/approvals); covered by pilot test
`test_console_workbench_contract.py` chip-emphasis assertions (dim at 0,
alert when > 0) and pure-state tests in `test_console_display_state.py`.

Verification (2026-07-05, HEAD 14410be9): nine affected suites = 639
passed, 1 failed — `test_console_details_toggle_expands_and_persists`,
a load flake in the 9.5-minute combined run only (passes isolated 5.5s and
file-scoped 35/35; rail-persistence path untouched by this branch). No
expected-failure baseline remains: the long-stale left-rail priority test
was realigned to the approved Session→Context order
(`test_console_left_rail_orders_session_then_staged_context`), and the
tab-reach traversal test was realigned to the post-onboarding state (stale
since the Phase 2 blocking setup modal, verified failing at branch base).

Accuracy note for reviewers: the `.console-region` CSS border removal is
inert cleanup, not a visual fix — Textual inline styles always beat CSS, so
`_frame_console_region` was already the single drawn frame; the CSS rule was
a redundant second source.
