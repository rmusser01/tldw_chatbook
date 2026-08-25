---
id: TASK-22227
title: 'Character emote pipeline: bound the per-chunk and per-send constants'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-25 23:17'
labels:
  - performance
  - chat
  - personas
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22227).

New with PR #2020, character sessions only (armed only when the assistant is a character).
(a) `Character_Chat/emote_directives.py:251-263`, `:333-337`, `:434-438`: the streaming
parser consumes per CHARACTER with a per-character `str.encode('utf-16-le')`
(`:92-95`) and list append — O(len(chunk)) with a high constant (~16k encodes per reply)
plus a clone + `safe_copy` per chunk. (b) `Chat/console_chat_store.py:7905-7947`:
`detect_character_mood` runs 14 compiled regex passes + 2 more over the full reply at the
terminal seam, on the loop. (c) `Chat/console_chat_controller.py:16091-16150`:
`_build_character_emote_snapshot` is O(assets^2) in regex evaluations per send (~1600 for
a 40-asset pack).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The parser publishes visible text in runs, not per character (or its per-chunk cost is measured and shown acceptable at 16k-char replies)
- [x] #2 The snapshot projection is O(assets) (normalize each asset once)
- [x] #3 Mood detection cost per turn is measured and bounded (or moved off-loop)
- [x] #4 Emote semantics unchanged: existing directive/mood tests green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red-first probes (tee'd): (a) count utf16 encodes per 16k-char reply streamed in 64-char chunks (~16k expected today); (c) count normalize-regex calls per 40-asset snapshot (~1600 expected today); (b) time detect_character_mood on a 16k-char turn.\n2. emote_directives.py: rewrite CharacterEmoteStreamParser consumption to publish visible text in RUNS (scan to next newline in ordinary/fence/directive modes; bounded per-char prefix scan with combined prefix+run publish); utf16_length computed per run. Keep push clone + safe_copy checkpoint contract, EMOTE_EVENT_LIMIT, partial-prefix fencing.\n3. console_chat_controller.py: replace the per-state inner scan (singleton project per asset) with a single-pass slug->first-source projection shared with project_character_emote_states (new project_character_emote_assets in emote_directives.py).\n4. character_mood.py: measure per-turn cost at 16k chars; decide bounded-on-loop vs off-loop honestly; document.\n5. Equivalence tests (parametrized boundary shapes: directive split mid-prefix, chunk ending mid-marker, back-to-back directives, 1-char chunk stream) asserting same visible text + events + clean_length as one-shot parse; count-bound regression tests for (a) and (c).\n6. Targeted suites (emote/mood/expression/controller) + --collect-only sweep + preflight + mutation tests (break run-splitting at a directive boundary; un-normalize the asset lookup).\n7. Wall-time before/after for all three items; notes; Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Bounded all three constants of the character emote pipeline (probes tee'd; before/after counts read from tees).

(a) Streaming parser (Character_Chat/emote_directives.py): CharacterEmoteStreamParser now consumes chunks with newline-bounded RUN publishing (_consume_chunk + _consume_prefix_run) instead of per-character _consume. utf16_length is computed once per run (additive over concatenation, so event at_char offsets are bit-identical). Probe at a 16k-char reply streamed in 64-char chunks: 16,000 -> 329 utf16 encode calls; wall 3.85 -> 0.25 ms (15x). Semantics preserved exactly: partial Emote:/fence prefixes across chunk boundaries (bounded per-char scan only inside the <=65-char prefix window), EMOTE_EVENT_LIMIT=5, directive dedup, the push clone and safe_copy checkpoint contract, cancel/flush. Equivalence proven three ways: new parametrized boundary-shape tests (directive split mid-prefix, chunk ending mid-emote-marker, chunk ending mid-fence-marker, back-to-back directives, astral UTF-16 offsets, 1-char chunk stream) assert identical visible text + events + clean_length against char-by-char AND one-shot parsing; the frozen cross-language vector corpus with adversarial chunkings stayed green; new count-bound regression test (runs <= chunks + 2*newlines + 8).

(c) Snapshot projection (Chat/console_chat_controller.py + emote_directives.py): _build_character_emote_snapshot dropped the per-state singleton re-projection (O(assets^2)) for a new single-pass project_character_emote_assets (slug -> first round-tripping source asset, same order/membership as project_character_emote_states, which is now a thin wrapper -- one implementation, no drift). Probe at 40 assets: 821 -> 0 singleton projections, 1,720 -> 80 regex-bearing normalize calls (exactly 2 per asset); wall 1.02 -> 0.06 ms per send. Count-bound regression tests at both the emote_directives and controller levels.

(b) Mood detection (Character_Chat/character_mood.py): measured, decision = bounded, stays on-loop. 2.3 ms median per 16k-char turn, ~9 ms at a degenerate 64k, and detect_character_mood runs at most ONCE per completed character turn at the terminal seam (only when no directive fired). Off-loading a one-shot ~2 ms call was judged worse than keeping it (thread-hop cost + ordering complexity at the finalize seam); the heuristic input is pinned to the server corpus so no truncation. Documented in the docstring and at the call seam.

Verification: targeted suites green (97 emote/mood/expression + 195 controller + 419 store/metadata/avatar + 997 Character_Chat); --collect-only sweep: 59,469 collected, 28 pre-existing optional-dep (numpy etc.) errors in untouched areas; preflight all green. Mutation tests: (1) ordinary run overshooting the directive boundary -> 6 reds incl. the frozen-vector equivalence harness and the new count test; (2) re-projecting the pack per state -> controller count probe red at 3280 > 88. Both restored via Edit; post-restore suites green.

Pre-existing, not mine (A/B-proven at base 050913498 with all changes reverted): Tests/Character_Chat/test_character_persona_scope_service.py::test_app_wires_character_persona_services fails in full-directory runs (ActorPackActivationError in app wiring) and the file cannot be collected standalone (circular import via Chat/__init__ -> server_chat_conversation_service -> runtime_policy.bootstrap). Also one flake: Tests/UI/test_console_character_avatar.py::test_oversized_character_controls_use_local_scroll_and_keep_offset failed once in a 3-file run, passed alone, whole-file, and on identical rerun.

Files: tldw_chatbook/Character_Chat/emote_directives.py, tldw_chatbook/Chat/console_chat_controller.py, tldw_chatbook/Character_Chat/character_mood.py (doc), tldw_chatbook/Chat/console_chat_store.py (comment), Tests/Character_Chat/test_emote_directives.py, Tests/Chat/test_console_chat_controller.py.
<!-- SECTION:NOTES:END -->
