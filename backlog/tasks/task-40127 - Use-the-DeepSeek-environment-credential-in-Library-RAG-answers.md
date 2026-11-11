---
id: TASK-40127
title: Use the DeepSeek environment credential in Library RAG answers
status: Done
assignee: []
created_date: 2026-11-11 11:51
labels:
- uat
- rag
- providers
dependencies: []
priority: medium
updated_date: 2026-11-11 20:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library RAG Answer admits a request when provider readiness recognizes `DEEPSEEK_API_KEY`, then fails at generation with **DeepSeek API Key required**. Console can send successfully with the same provider, model, and environment credential. Retrieval completed before the answer failed.

### Reproduction

1. On local `origin/dev` `6bd97b112f` on Windows, launch Chatbook with a disposable profile and a valid `DEEPSEEK_API_KEY` in the process environment. Leave the DeepSeek API key field empty and keep its default **Env var** setting.
2. In **Settings → Providers & Models**, save DeepSeek as the default provider and select a supported model. The UAT run used `deepseek-v4-flash`.
3. Open a new Console chat and send a synthetic prompt. Confirm the hosted reply arrives.
4. With the embeddings extra and MiniLM available, import a text file containing **The rehearsal uses blue lanterns.** In **Settings → Domain Defaults → RAG**, use **Backfill** on **Hybrid Basic**.
5. In **Library → Search / RAG**, switch to **RAG Answer** and run **What colored lights were used during the practice run?**.
6. Observe that the source is retrieved and Answer reports **DeepSeek API Key required**. The expected result is an answer grounded in the retrieved source using the configured environment credential.

### Evidence

- [Successful Console reply with the same credential source](../../qa/uat-core-workflows-2026-11-11/deepseek-console-final.txt)
- [Retrieved document and recorded vector score](../../qa/uat-core-workflows-2026-11-11/rag-media-retrieval.txt)
- [RAG Answer failure](../../qa/uat-core-workflows-2026-11-11/rag-paraphrase-result.txt)

At the tested revision, `Library/library_rag_answer_service.py` checks provider readiness before invoking generation. Its answer call does not carry a resolved API key. `LLM_Calls/deepseek.py` falls back to the provider's `api_key` field when no key is supplied.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 With a valid environment-only DeepSeek credential, Library RAG Answer uses the same credential source as Console and produces a cited answer for retrieved evidence.
- [x] #2 With the credential absent, RAG Answer shows the credential setup guidance before attempting generation.
- [x] #3 The environment key value stays out of saved configuration, diagnostic output, and UI evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: No
ADR path: N/A
Reason: Reuse the existing provider readiness credential boundary for Library answer dispatch.

Test an environment-only DeepSeek credential through the real answer adapter, forward the resolved credential after retrieval, and verify the cited answer with live DeepSeek.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Library RAG Answer uses Console's credential resolver for DeepSeek. It resolves the credential off the UI loop after selecting citable evidence and passes it to the existing provider dispatcher. Missing credentials use provider setup guidance. The focused answer and provider suite passed all 81 cases, including authentication-error redaction.

The live rerun used the environment-only DeepSeek credential with `deepseek-v4-flash`. The reopened disposable profile initially showed zero Library sources. After refreshing the Library source list, the media was visible, but Backfill skipped its missing vector entry as already tracked. Reindexing the single fixture through the production indexing service let the Library Run action retrieve `juniper-source` with a vector score of 0.4168. DeepSeek answered that the rehearsal used blue lanterns, and the UI validated citation `[S1]`. The saved TOML and walkthrough evidence contained no key value. [Walkthrough result](../../qa/uat-core-workflows-2026-11-11/deepseek-rag-fixed.txt) · [Library screen](../../qa/uat-core-workflows-2026-11-11/deepseek-rag-ui-fixed.txt)
<!-- SECTION:NOTES:END -->
