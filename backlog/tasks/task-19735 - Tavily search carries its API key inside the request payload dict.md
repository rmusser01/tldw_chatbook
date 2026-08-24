---
id: TASK-19735
title: >-
  Tavily search carries its API key inside the request payload dict
status: Done
assignee:
  - '@claude'
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

- [x] The Tavily API key is not a member of any dict that also holds ordinary
      request parameters, so a debug log of the request parameters cannot
      disclose it
- [x] Tavily search still authenticates and returns results (the credential
      reaches the service by whichever transport the fix chooses)
- [x] A test asserts that the object holding the request parameters contains no
      credential value, and is mutation-checked (restoring the old payload
      shape makes it red)
- [x] The already-clean properties are pinned, not just assumed: a test asserts
      the error string returned on a request exception contains no credential
- [x] The other backends in `Web_Scraping/WebSearch_APIs.py` are checked for the
      same shape and the result recorded (headers-based ones — Bing, Brave,
      Serper, Exa, Kagi, Yandex — are expected clean; Google's `params["key"]`
      at `:3428` is the one TASK-19552 addressed)

## Notes

Low priority on purpose. This is hardening against a plausible future edit, not
a live disclosure — filing it as a live leak would misrepresent the evidence.
The value is that the next person to debug this function finds the task instead
of adding the log line.

## Implementation Plan

1. Confirm the latency claim at this base: grep `search_web_tavily` for any
   log statement, and confirm the key travels in the POST body rather than
   the URL (so the exception path is already clean).
2. Write the born-red test first: drive `search_web_tavily` with a sentinel
   key through an in-process fake `requests`, and assert the decoded request
   body holds no credential.
3. Move the credential to `Authorization: Bearer <key>`, matching every
   sibling backend in the module.
4. Pin the already-clean properties rather than assume them: the returned
   error string on a request exception.
5. Turn the sibling-backend census into an executable AST check instead of
   prose that can go stale, with `search_web_google`'s `params["key"]` as
   the one recorded, referenced exception.

## Implementation Notes

The Tavily credential moved from the `payload` dict into `headers` as
`Authorization: Bearer <key>`, which Tavily accepts and which bing, brave,
kagi, serper, exa and yandex in the same module already use. It is read at
the point of use and never bound to a local name, so there is no
`tavily_api_key` variable for a debug log to reach for either.

**This stayed sized as latent.** Nothing logged `payload` at the base and the
key travelled in the body, not the URL, so the error path was clean too. The
value of the change is that the one-line edit a future maintainer would make
while debugging (`logger.debug(f"payload: {payload}")`) can no longer
disclose the key. No claim of a live leak was added anywhere.

**Born-red evidence** (all at base `f12bb21ad`, before any source change --
which is also the mutation check the third AC asks for, since the base *is*
the old payload shape):

```
test_tavily_request_parameters_hold_no_credential
E  AssertionError: the Tavily key is inside the request-parameter object:
   {'api_key': 'tvly-TASK19735-SENTINEL-NOT-A-REAL-KEY', 'query': 'cherry cake', 'max_results': 3}
test_tavily_credential_travels_in_the_headers
E  AssertionError: the key reaches no transport at all:
   {'Content-Type': ..., 'User-Agent': ...}
test_no_search_backend_hides_a_credential_in_its_parameter_dict
E  Extra items in the left set: ('search_web_tavily', 'payload')
```

Three sibling tests were **green at base and stayed green**, which is the
point of the fourth AC: the returned error string carries no credential, the
optional domain filters still reach the request, and the census detector
demonstrably detects (a synthetic offending module is found, so the empty
census result is not vacuous).

**Sibling-backend census, recorded and now executable.** Clean (credential in
`headers`): bing `Ocp-Apim-Subscription-Key`, brave `X-Subscription-Token`,
serper `X-API-KEY`, exa `x-api-key`, kagi `Authorization: Bot ...`, yandex
`Authorization: Api-Key ...`. searx / duckduckgo / baidu carry no credential
at all. The one parameter-dict credential left is `search_web_google`'s
`params["key"]` -- the engine's own API contract, whose two disclosure points
TASK-19552 fixed -- and it is the single allowlisted entry in
`_KNOWN_PARAMETER_CREDENTIALS`, so a new offender fails the census.

Modified: `tldw_chatbook/Web_Scraping/WebSearch_APIs.py`.
Added: `Tests/Web_Scraping/test_tavily_credential_transport.py` (6 tests).
