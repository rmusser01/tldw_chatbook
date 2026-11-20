---
id: TASK-42002
title: "Migrate provider base URLs that an earlier version saved with /v1 wrongly appended"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:58
labels:
- follow-up
- providers
- settings
priority: medium
references:
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/config.py
- Docs/User_Guide/settings.md
- Docs/Architecture/llm-providers.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41704 (round 17). Before TASK-41704, Settings appended `/v1` to a first-party cloud base with no `v1`, so configs may hold known-broken values such as Google `.../v1beta/v1` or `https://api.cohere.com/v1`; TASK-41704 only stops new saves. Owner decision, 2026-11-19: "Migrate on load": when the app loads one of these known-broken shapes for a first-party provider, it rewrites it to the documented base. This changes those configs' project-instruction keys once.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On the provider's own host only, these saved shapes are rewritten to the documented base when config loads, and the rewrite is saved once: Google `…/v1beta/v1` to `…/v1beta`; Cohere `https://api.cohere.com/v1` to the bare host; Groq, OpenRouter and QwenCloud `<own bare host>/v1` to `/openai/v1`, `/api/v1` and `/compatible-mode/v1`
- [ ] #2 A value that is not one of those shapes is never rewritten, including OpenAI's correct `<host>/v1`, any foreign or gateway host, and any custom base. Google's `https://generativelanguage.googleapis.com/v1` is kept too: v1 is a real Gemini API version, and whether the adapter's requests need v1beta cannot be checked without contacting Google
- [ ] #3 The rewrite is logged without the URL's credentials or query, and the notes list the shapes covered
- [ ] #4 Tests cover each shape and a non-matching control, red before the change
- [ ] #5 `Docs/User_Guide/settings.md` (the paragraph that says such values stay as they are) and `Docs/Architecture/llm-providers.md` ("it is not rewritten") describe the migration
<!-- AC:END -->
