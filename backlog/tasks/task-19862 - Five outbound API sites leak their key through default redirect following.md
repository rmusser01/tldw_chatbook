---
id: TASK-19862
title: >-
  Five outbound API sites leak their key through default redirect following
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - security
  - credentials
  - websearch
  - llm-providers
priority: medium
dependencies:
  - TASK-19557
  - TASK-19733
---

## Description

Source: unfiled residue in **TASK-19557**'s implementation notes, with exact
lines and per-site severities confirmed by **TASK-19733**'s reviewer.
Re-verified at `3605bd52d`: neither
`Web_Scraping/WebSearch_APIs.py` nor `LLM_Calls/LLM_API_Calls_Local.py`
contains the string `allow_redirects` anywhere, so all five sites take the
`requests` default of `allow_redirects=True`.

Five outbound call sites send an API key in a **custom header** and follow
redirects automatically. `requests` strips only `Authorization` and `Cookie`
when a redirect crosses origins; a vendor-specific header such as
`X-Api-Key`, `Ocp-Apim-Subscription-Key`, `X-Subscription-Token` or
`X-API-KEY` survives the hop and is delivered to whatever host the redirect
names.

None of these sites route through `Utils/egress.py`, so the per-hop
re-validation and cross-origin allowlist that **TASK-19733** built cannot reach
them — that fix hardened the shared primitive, and these five bypass it
entirely.

Sites, with the severities the reviewer assigned:

| Site | Header | Request | Severity |
| --- | --- | --- | --- |
| KoboldAI | `LLM_API_Calls_Local.py:1011` (`X-Api-Key`) | `:1061` `session.post` | **medium** |
| Bing | `WebSearch_APIs.py:2711` (`Ocp-Apim-Subscription-Key`) | `:2728` `session.get` | **medium** |
| Brave | `:2964` (`X-Subscription-Token`) | `:2980` `requests.get` | low |
| Serper | `:3985` (`X-API-KEY`) | `:3992` `requests.post` | low |
| Exa | `:4055` (`x-api-key`) | `:4062` `requests.post` | low |

Kobold is the worst of the five on two counts. Its destination,
`current_api_base_url`, is **user-configured** — a redirect-serving endpoint is
reachable by ordinary misconfiguration or by a hostile local-server URL, not
only by a compromised vendor. And it is a POST: on a 302 or 303 `requests`
converts the method to GET and re-issues to the new host, still carrying the
key header.

Bing is medium because `bing_search_api_url` is likewise read from config
(`WebSearch_APIs.py:2672`) and the credential is a paid Azure subscription key.
Brave, Serper and Exa are low because their endpoints are hard-coded vendor
URLs — only a vendor-side compromise or DNS attack reaches them.

Kagi and Yandex were checked and are genuinely clean.

Minimal fix is `allow_redirects=False` at each site: none of these APIs
legitimately redirects, so refusing a redirect is a behaviour-preserving change
that turns a silent credential disclosure into a visible error.

## Acceptance Criteria

- [ ] None of the five sites follows a redirect while carrying its API key
      header
- [ ] A redirect response at each site produces a clear, non-silent failure
      rather than a second request to the redirect target
- [ ] A test per site drives a redirect from a stubbed server and asserts the
      key header is never sent to the second host, and each is mutation-checked
      (restoring default redirect following makes it red)
- [ ] The Kobold POST case is covered specifically for a 302/303, where the
      method converts to GET — a test that only exercises a 307 does not
      demonstrate the fix
- [ ] Every redirect response that is refused is closed, so the connection is
      not leaked (the defect TASK-19557's Qodo round found on its own refusal
      paths)
- [ ] The refusal message does not echo the raw attacker-controlled `Location`
      header (consistent with the exception-text rule applied in TASK-19321 /
      TASK-19552 / TASK-19557)
- [ ] The remaining `requests` call sites in both modules are swept for the same
      shape and the result recorded, so this residue does not have to be
      rediscovered a third time

## Notes

This residue has now been reported by two separate reviewers (TASK-19557's
notes, then TASK-19733's) without ever being filed. That is the reason for the
sweep criterion: the point is to close the class, not the five known members.
