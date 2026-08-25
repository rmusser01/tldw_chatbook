---
id: TASK-22204
title: >-
  Resolve the Console expression state once per tick and stop re-copying the transcript
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - console
  - streaming
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22204).

New with PR #2020 (streaming emotes), default-ON (`resolve_show_character_avatar` defaults
True, `Chat/console_image_view.py:137-152`). `UI/Console_Modules/character.py:284-295`:
`_current_request` calls `resolve_console_expression_selection` and then — whenever
`selection.source` is `idle`/`operational`, the common case — re-runs
`resolve_console_expression_state`, and `_request_is_current` (`:353-354`) re-enters
`_current_request`. Every resolution funnels into
`store.messages_for_session(...)` (`Chat/console_expression_state.py:71`), which
materializes stream buffers and returns a `dataclasses.replace` copy of every message
(`Chat/console_chat_store.py:5227-5234`). At the pin this was one copy per tick; now a
repainting tick pays 4-6. The 0.2 s tick runs for the whole duration of a run, so this is
10-30 whole-transcript copies per second while streaming. Context: the tick already pays
~3 other full copies (native transcript, cost chip, setup-card guidance) — a shared
per-tick snapshot seam would collapse all of them, but the minimum fix is restoring 1 copy
for the avatar path.

## Acceptance Criteria

- [ ] One `messages_for_session` copy at most per avatar refresh per tick (reuse `selection.state`; make `_request_is_current` compare against the already-built request) — proven by a call-count probe during a simulated streaming tick
- [ ] Emote/idle/operational selection behavior unchanged (existing expression tests green)
- [ ] Stretch (may split to a follow-up): a single shared per-tick transcript snapshot consumed by the avatar, guidance, and cost-chip paths, with the seam documented
- [ ] Per-tick copy count during streaming measured before/after
