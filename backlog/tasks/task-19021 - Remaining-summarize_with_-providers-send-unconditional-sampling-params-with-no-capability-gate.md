---
id: TASK-19021
title: >-
  Remaining summarize_with_* providers send unconditional sampling params with
  no capability gate
status: Done
assignee: []
created_date: '2026-08-20 15:16'
updated_date: '2026-08-20 16:08'
labels:
  - llm
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-18802 fixed the Anthropic and OpenAI halves of the summarization path and swept the rest of Summarization_General_Lib.py for the same unconditional-params shape. CODE-READ, not probe-verified -- each needs a reproduction before being fixed. Sites (line numbers as of the 18802 branch): summarize_with_cohere :1307 temperature; summarize_with_groq :1555 temperature; summarize_with_openrouter :1779 and :1868 temperature (openrouter proxies OpenAI/Anthropic models, so a routed gpt-5 or claude-sonnet-5 may inherit exactly the 400s 18802 fixed -- worth probing first); summarize_with_huggingface :1967-1969 max_tokens + temperature; summarize_with_deepseek :2159 temperature; summarize_with_mistral :2342-2344 temperature AND top_p together plus max_tokens (the both-at-once shape that 400s on Anthropic Claude 4.x per TASK-19020 -- check whether Mistral restricts the combination). summarize_with_google sends none (commented out) and is unaffected. None of these providers' modern-model behavior has been probed; if a provider rejects nothing, record that and close. Follow the model_capabilities predicate design from TASK-18414/18802 for any that reproduce.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each provider's modern-model behavior toward its unconditional params is probed and recorded before any code change
- [x] #2 Any provider that reproduces a rejection consults a model_capabilities predicate instead of sending the param unconditionally
- [x] #3 Providers whose models accept the params are left unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Build honest key matrix (ls repo-root key files): cohere YES, openrouter YES-but-verify (known 401), groq/huggingface/deepseek/mistral NO KEY\n2. Probe cohere with the exact summarize_with_cohere payload shape (system+user messages, temperature, stream:false) against the current flagship (ask the API for the model list first); capture verbatim bodies\n3. Verify the openrouter key; if working, probe routed openai/gpt-5 and anthropic/claude-sonnet-5 with summarize_with_openrouter's exact shape (temperature only)\n4. Record groq/huggingface/deepseek/mistral as unprobed (no credential), leave their functions unchanged\n5. For each reproduced rejection only: immutable predicate in model_capabilities.py (18414/18802 design), red-first payload pins, legacy control pins, mutation tests, live analyze() seam call + clean origin/dev control\n6. No-regression: 18802/19020 pins stay green untouched; one live modern-Anthropic + one live modern-OpenAI summarization on the branch\n7. Report, task close-out, PR against dev
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Probe-first close-out with ZERO code changes -- the honest outcome the task allows. Key matrix (repo-root key files + env sweep): cohere YES; openrouter key file present but DEAD (401 'User not found' on both GET /api/v1/key and POST /chat/completions -- matches the prior session's caveat); groq/huggingface/deepseek/mistral NO credential anywhere. Cohere probes (standalone curl, exact summarize_with_cohere v2 /chat payload shape: system+user messages, temperature 0.3, stream false): current flagship command-a-plus-05-2026 HTTP 200, function default command-a-03-2025 HTTP 200, command-a-reasoning-08-2025 HTTP 200 -- no rejection anywhere, including the reasoning family; the thinking-part responses still parse (the builder joins only 'text' parts). AC #1 satisfied by the recorded matrix (probed where a credential exists, 'no credential, unprobed, left unchanged' recorded for the rest); AC #2 vacuously satisfied (no provider reproduced a rejection, so no predicate was owed); AC #3 satisfied by leaving all six functions byte-untouched, with live seam evidence for cohere: production analyze() -> _dispatch_to_api -> summarize_with_cohere against command-a-plus-05-2026 with a passthrough wire spy -- wire sampling keys ['temperature'], HTTP 200, real summary. No-regression: the 18802/19020 pin suite (Tests/test_probe_import_provenance.py + Tests/LLM_Calls/test_summarization_model_capabilities.py) green at 81 passed untouched, plus live modern-Anthropic (claude-sonnet-5) and modern-OpenAI (gpt-5) summaries through the production analyze() seam on this branch. Bonus credential-free finding: api-inference.huggingface.co is NXDOMAIN (control hosts resolve), so summarize_with_huggingface fails at connect regardless of params. Filed TASK-19051 (single follow-up) for the five unprobeable providers, carrying the openrouter prefix-stripping design guidance and the HF endpoint migration. Full evidence: Docs/superpowers/plans/2026-08-20-task-19021-report.md
<!-- SECTION:NOTES:END -->
