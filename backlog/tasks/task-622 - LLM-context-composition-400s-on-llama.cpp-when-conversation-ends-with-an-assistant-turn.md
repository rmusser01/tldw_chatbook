---
id: TASK-622
title: >-
  LLM context composition 400s on llama.cpp when conversation ends with an
  assistant turn
status: Done
assignee: []
created_date: '2026-07-25 10:15'
updated_date: '2026-07-25 17:04'
labels:
  - image-generation
  - bug
  - uat
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live UAT 2026-07-25: no-prompt `/generate-image` on a llama.cpp session silently used the shallow keyword extractor instead of the task-559 LLM-composed prompt. Reproduced outside the app: `compose_llm_context_prompt` passes the conversation turns as the payload, and when the LAST turn is an assistant message (the normal case — user asks, assistant answers, user runs `/generate-image`), `chat_with_llama` treats the trailing assistant message as a response PREFILL; llama.cpp rejects it with `400: "Assistant response prefill is incompatible with enable_thinking."` The graceful fallback masks the failure by design (debug log only), so the feature quietly degrades to the keyword prompt on the app's most common local provider.

Repro: `compose_llm_context_prompt([("user", ...), ("assistant", ...)], LLMContextOptions(api_endpoint="llama_cpp", model=..., ...))` against a live llama.cpp server → 400 → returns None.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The composition payload never ends on an assistant turn: append the compose instruction as a final USER-role message (provider-agnostic fix — better prompt construction and avoids every prefill-detecting handler), or an equivalent approach that works across providers.
- [x] #2 With a live llama.cpp session whose last turn is an assistant message, no-prompt `/generate-image` produces a genuinely LLM-composed prompt (integration-style test with a faked chat_call asserting the payload shape; the trailing-role invariant pinned).
- [x] #3 Fallback behavior on real failures is unchanged (still silent, still keyword extractor).
- [x] #4 Existing task-559 unit-3 tests stay green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a module-level trailing compose-instruction constant (user-role text: 'Compose the image-generation prompt for the scene above now. Reply with the prompt text only.') and trim the now-duplicated 'Respond with the prompt text only' clause from _CONTEXT_LLM_SYSTEM_PROMPT so the system+trailing-user pair reads coherently, not redundantly.\n2. In compose_llm_context_prompt, after _shape_llm_context_payload(messages, options.turns) produces a non-empty payload, unconditionally append {'role': 'user', 'content': <instruction>} as the new last message before invoking chat_call -- this happens AFTER truncation so the instruction is never truncated away, and applies regardless of whether the truncated window's last turn was user or assistant. Empty-payload short-circuit (turns<=0 / no messages) is unchanged -- no LLM call is made either way, so it can't hit the prefill bug.\n3. Update test_compose_llm_context_prompt_success_returns_cleaned_text (the one test that pins the old payload shape) to expect the appended instruction message.\n4. Add new unit tests: trailing-assistant payload ends with the instruction as a user message; trailing-user payload also gets the instruction appended as a separate final user message (pinned decision: always append, never merge into the prior user turn); single-turn and empty-after-truncation shapes behave correctly.\n5. Red/green against the live llama.cpp server at 127.0.0.1:9099 via a throwaway script: reproduce the pre-fix 400 with a trailing-assistant conversation, then rerun post-fix and capture the real composed prompt text.\n6. Run the full existing test files + ruff + app import check; write Implementation Notes; flip ACs; set task Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Root cause confirmed live** (`Docs`/scratch script against the live llama.cpp server at 127.0.0.1:9099): the pre-fix `messages_payload` for a trailing-assistant conversation ended with `{"role": "assistant", ...}`; sending that to `chat_api_call(api_endpoint="llama_cpp", ...)` raised `ChatBadRequestError('Bad request to llama_cpp (Status 400). Detail: {"error":{"code":400,"message":"Assistant response prefill is incompatible with enable_thinking.",...}}')`, and `compose_llm_context_prompt` swallowed it (by design) and returned `None`.

**Fix** (`tldw_chatbook/Chat/console_generate_image.py`, `compose_llm_context_prompt`): after `_shape_llm_context_payload(messages, options.turns)` truncates to the last N turns, a new final USER-role message is unconditionally appended — `_CONTEXT_LLM_COMPOSE_INSTRUCTION` ("Compose the image-generation prompt for the scene above now. Reply with the prompt text only."). This happens strictly after truncation, so the instruction is never truncated away, and it is appended regardless of whether the truncated window's last turn is `user` or `assistant` — the payload handed to `chat_call` now never ends on `assistant`. Nothing in `chat_with_llama`/`LLM_API_Calls_Local.py` was touched — the fix is entirely in payload composition, so it protects every provider, not just llama.cpp. `_CONTEXT_LLM_SYSTEM_PROMPT`'s trailing "Respond with the prompt text only" clause was removed (now redundant with the new trailing user instruction) so the system+final-user pair reads coherently instead of repeating itself. The empty-after-truncation short-circuit (`turns<=0` / no messages) is untouched — no LLM call is made in that case either way, so it was never exposed to the prefill bug.

**Live red/green** (throwaway script, not a permanent test; live llama.cpp at 127.0.0.1:9099, `LLMContextOptions(api_endpoint="llama_cpp", model="local", api_key=None, provider_ready=True)`, conversation `[("user", "Let's write a scene where a lone astronaut discovers a glowing forest on Mars."), ("assistant", "The astronaut knelt beside the bioluminescent ferns, visor lit blue-green, dust settling around her boots as the twin moons rose.")]`):
- RED (pre-fix `compose_llm_context_prompt`): returned `None`; the underlying raw call raised `ChatBadRequestError(... "Assistant response prefill is incompatible with enable_thinking.")`.
- GREEN (post-fix `compose_llm_context_prompt`): returned `"Cinematic shot of a lone astronaut kneeling beside bioluminescent blue-green ferns on the dusty surface of Mars, the astronaut's visor reflecting a soft eerie glow, twin moons rising in a dark, star-filled sky, atmospheric, mysterious, and highly detailed."`

**Tests** (`Tests/Chat/test_console_generate_image.py`): added `test_compose_llm_context_prompt_trailing_assistant_appends_user_instruction`, `..._trailing_user_still_appends_separate_instruction` (instruction is always a separate final message, never merged into an existing trailing user turn), `..._single_turn_appends_instruction`, `..._instruction_survives_turns_truncation` (turns=1 still keeps the instruction as message 2 of 2), and `..._empty_after_truncation_still_returns_none` (unaffected no-op case). Updated the one pre-existing test that pinned the old payload shape, `test_compose_llm_context_prompt_success_returns_cleaned_text`, to expect the appended instruction as the payload's new final entry — this update to a task-559 test IS the fix, not a regression; every other task-559 test in the fallback matrix (disabled/not-ready/no-endpoint/empty-messages/call-raises/timeout/empty-response/garbage-response/saturation/recovery) is unmodified and still green. Full suite: `Tests/Chat/test_console_generate_image.py` (98 passed) + `Tests/Chat/test_console_generation_actions.py` (32 passed) = 130 passed; `ruff check` clean; `python -c "import tldw_chatbook.app"` succeeds.

**Files modified**: `tldw_chatbook/Chat/console_generate_image.py`, `Tests/Chat/test_console_generate_image.py`.

**Self-review / concerns**: The only behavioral risk is providers whose chat templates treat "system + ...history... + trailing user 'compose now' instruction" oddly for context composition — none observed across the fallback-matrix tests or the live llama.cpp verification. The debug-log turn count on the exception path now reports `len(context_turns)` (the real conversation-turn count) rather than `len(payload)` (which would have included the instruction message and over-counted by one) — a minor accuracy fix to the log line, not a behavior change visible to callers.
<!-- SECTION:NOTES:END -->
