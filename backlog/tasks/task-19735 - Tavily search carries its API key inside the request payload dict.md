---
id: TASK-19735
title: >-
  Tavily search carries its API key inside the request payload dict
status: To Do
assignee: []
created_date: '2026-08-21 23:40'
labels:
  - security
  - credentials
  - hardening
  - websearch
priority: low
dependencies: []
---

## Description

Source: surfaced while fixing **TASK-19552** (Google Search API key logged at
INFO on the default search engine) and confirmed **latent** by that task's
reviewer. Re-verified at this branch base (`839819e1a`).

`Web_Scraping/WebSearch_APIs.py:4105-4110`, in `search_web_tavily`:

```python
tavily_api_key = loaded_config_data["search_engines"]["tavily_search_api_key"]

payload = {
    "api_key": tavily_api_key,
    "query": search_query,
    "max_results": result_count,
}
```

The credential lives inside a plain dict alongside ordinary, log-worthy request
parameters — the same smell as the Google leak TASK-19552 fixed, where a key
sitting in a `params` dict was eventually rendered into a log line.

**This one is latent, and the task should be sized accordingly:**

1. Nothing in the current code logs `payload`. Grep at this base finds no log
   statement in `search_web_tavily` at all.
2. The key travels in the POST **body** (`requests.post(tavily_api_url,
   headers=headers, data=json.dumps(payload), ...)`, `:4127-4129`), not in the
   URL. So the exception path is clean too: the handler at `:4131-4132` returns
   `f"There was an error searching for content. {str(e)}"`, and `str(e)` on a
   `requests` exception carries the URL, not the body — unlike a query-string
   credential, which would land in that string and then in the returned error
   text.

The risk is entirely prospective: a future maintainer adding
`logger.debug(f"Tavily payload: {payload}")` while debugging — a one-line change
with no local signal that it is dangerous — writes the key to disk. Every other
credential in this module is at least kept in a `headers` dict, which reads as
"do not log this"; this one does not.

Per the owner's standing ruling (durable/pragmatic over clever), the preferred
shape is the one that makes the hazard structurally impossible rather than one
that relies on future reviewers noticing: keep the credential out of the
loggable parameter object entirely (Tavily accepts `Authorization: Bearer
<key>`, so it can move to `headers` like every sibling backend in this file), or
— if it must stay in the body — assemble it only at the call and never bind it
to a name that a debug log would naturally reach for.

## Acceptance Criteria

- [ ] The Tavily API key is not a member of any dict that also holds ordinary
      request parameters, so a debug log of the request parameters cannot
      disclose it
- [ ] Tavily search still authenticates and returns results (the credential
      reaches the service by whichever transport the fix chooses)
- [ ] A test asserts that the object holding the request parameters contains no
      credential value, and is mutation-checked (restoring the old payload
      shape makes it red)
- [ ] The already-clean properties are pinned, not just assumed: a test asserts
      the error string returned on a request exception contains no credential
- [ ] The other backends in `Web_Scraping/WebSearch_APIs.py` are checked for the
      same shape and the result recorded (headers-based ones — Bing, Brave,
      Serper, Exa, Kagi, Yandex — are expected clean; Google's `params["key"]`
      at `:3428` is the one TASK-19552 addressed)

## Notes

Low priority on purpose. This is hardening against a plausible future edit, not
a live disclosure — filing it as a live leak would misrepresent the evidence.
The value is that the next person to debug this function finds the task instead
of adding the log line.
