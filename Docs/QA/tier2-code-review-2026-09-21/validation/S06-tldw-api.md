# S06 — `tldw_api/` validation

Zero commits touched `tldw_chatbook/tldw_api/` between the review commit `3722a85748` and
`HEAD` (`git log --oneline 3722a85748..HEAD -- tldw_chatbook/tldw_api/` → empty; `client.py`
is still exactly 16,661 lines). Every finding below was checked directly against the current
file, not inferred from the review text.

## 1. P1 [D1] — SSE `ResponseNotRead` escapes the exception family
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/tldw_api/client.py:1668-1686` (unchanged)
- Proof: `python -c "import httpx; print(issubclass(httpx.ResponseNotRead, ValueError))"` → `False`
  (MRO: `ResponseNotRead → StreamError → RuntimeError`). Read confirms `_sse_request`'s except
  block does `response_data = e.response.json()` guarded only by `except ValueError: pass`
  (`:1671-1677`), so a genuinely-unread streaming response's `.json()` call raises
  `ResponseNotRead` uncaught. Sibling `_stream_request` (`:1585-1587`) does `await
  e.response.aread()` first — the correct shape.

## 2. P1 [D1] — `tldw_api` is an unenumerated strict-JSON family (NaN / dup-keys / RecursionError)
- Verdict: CONFIRMED
- Site now: `client.py:1357` (`return response.json()`) inside `_request`, whose `except` clauses
  (`:1408-1420`) are only `httpx.HTTPStatusError`, `httpx.RequestError`, `json.JSONDecodeError`
- Proof: `grep -rn strict_json_loads tldw_chatbook/tldw_api/` → no match; `grep -n
  'tldw_api\|tldw_server' backlog/decisions/175-one-strict-json-acceptance-contract.md` → no
  match (ADR-175, Status: Accepted, enumerates hosted-chat/QwenCloud/continuation boundaries
  only). Repro: `json.loads('{"score": NaN}')` → `{'score': nan}`; `json.loads('{"id":1,"id":2}')`
  → `{'id': 2}`; `json.loads('['*200000+']'*200000)` → `RecursionError`.
- Note: TASK-32805.5 is Done and ADR-175 is Accepted (both confirmed on disk) — the finding's
  "insufficient enumeration, not a restatement" framing holds.

## 3. P2 [D4] — five hand-rolled HTTP-error handlers; two streaming copies silently drop detail
- Verdict: CONFIRMED
- Site now: `client.py:1358-1406` (`_request`), `:1440-1489` (`_binary_request`),
  `:1512-1555` (`_headers_request`), `:1558-1605` (`_stream_request`), `:1608-1688` (`_sse_request`)
- Proof: read all five — only `_request` has the structured `{"detail": {...}}` branch
  (`message`/`code` fallback with the "schedules task 6 round 2, D9" comment); the other four have
  a two-line `isinstance(response_data.get("detail"), str)` branch only. Decisive repro for the
  `_stream_request` ordering bug: built a genuinely-unread streaming `httpx.MockTransport` 401
  (async-generator body, not a pre-buffered `Response(json=...)`) — `raise_for_status()` inside
  `async with client.stream(...)` fires, `__aexit__` closes the stream, and the `except` block's
  `await e.response.aread()` raises `StreamClosed: Attempted to read or stream content, but the
  stream has been closed.` Exactly reproduces the review's claim (an earlier attempt using a
  pre-buffered mock response gave a false negative — `is_stream_consumed=True` before
  `raise_for_status()` even ran, because `httpx.Response(json=...)` precomputes content).

## 4. P2 [D4] — `server_notifications_schemas.py` is a 163-line duplicate, 16/17 classes unreachable
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/tldw_api/server_notifications_schemas.py` (163 lines, unchanged)
- Proof: `grep -rn server_notifications_schemas tldw_chatbook/ Tests/` → exactly 2 hits
  (`__init__.py:2012` map row for `ServerNotificationStreamEvent`, `client.py:980` import).
  `__init__.py:1709` maps `NotificationResponse` to `notifications_reminders_schemas` instead.
  `NotificationResponse.kind` is `str` at `server_notifications_schemas.py:88` vs
  `NotificationKind` (a `Literal`) at `notifications_reminders_schemas.py:114`.

## 5. P2 [D1] — two client methods hit the same SSE endpoint with incompatible event models
- Verdict: CONFIRMED
- Site now: `client.py:7968` (`stream_notification_events` → `NotificationStreamEvent`),
  `client.py:8036` (`stream_server_notifications` → `ServerNotificationStreamEvent`), both against
  `/api/v1/notifications/stream`
- Proof: `NotificationStreamEvent` declares `event_id: str | None` (`notifications_reminders_
  schemas.py:191`, required `data: dict`); `ServerNotificationStreamEvent` declares `id: str |
  None` (`server_notifications_schemas.py:161`, `data: dict[str, Any] | str | None`). `_sse_
  request._flush_event` emits `event_id`, so the `id`-named model never fills it. Confirmed
  `Tests/tldw_api/test_server_notifications_client.py:205-230` mocks `_stream_sse_request`
  wholesale and hand-constructs `ServerNotificationStreamEvent(id="11", …)` — proves nothing about
  the SSE→id path.

## 6. P2 [D4] — two live, drifted MCP client surfaces (16 overlapping endpoints, quoting drift)
- Verdict: CONFIRMED
- Site now: `client.py:15891-16140` vs `mcp_unified_client.py:475-560,712-898`
- Proof: `client.py:15948/15959/15966/15976` all interpolate `quote(server_id, safe='')`;
  `mcp_unified_client.py:523/531/539/…` interpolate raw `{server_id}` (grep confirms zero `quote(`
  in the external-servers block). `stream_mcp_governance_events` (`client.py:15891`) and
  `stream_governance_events` (`mcp_unified_client.py:475`) are duplicate names for the same
  `/api/v1/mcp/hub/events/stream` endpoint.

## 7. P2 [D1] — 334 `str`-typed path segments interpolated unquoted next to a validating helper
- Verdict: CONFIRMED
- Site now: `client.py` — `_workspace_source_path_id` helper at `:1083-1096`; cited examples all
  present verbatim: `f"/api/v1/sharing/public/{token}"` (`:9006/9016/9022`),
  `f"/api/v1/chat/shared/conversations/{share_token}"` (`:13122`),
  `f"/api/v1/consent/preferences/{purpose}"` (`:3814/3819`),
  `f"/api/v1/llm/providers/{provider_name}"` (`:4149`) — none wrapped in `quote(...)`.
- Proof: `grep -c "quote("` → 17 (matches the review's "17 inline copies"). A crude single-line
  grep for `f"/api/v1/...{var}..."` finds 657 interpolations with 16 already-quoted — same order of
  magnitude as the review's AST-derived 705/17/334 split; the gap is methodology (single-line grep
  vs AST `FormattedValue` resolution across multi-line f-strings), not a refutation. Confidence
  matches the review's own framing: counts verified, end-to-end exploitability inferred (callers
  outside this slice).

## 8. P2 [D3] — `client.py` (16,661 lines) has no size-ratchet row
- Verdict: CONFIRMED
- Site now: `Tests/Architecture/test_module_size_ratchet.py:_BUDGETS` (7 rows, unchanged since
  the review — file not touched in the 25 commits)
- Proof: `_BUDGETS` dict has 7 entries (`app.py`, `console_chat_controller.py`,
  `console_chat_store.py`, `personas_screen.py`, `console_transcript.py`,
  `console_settings_modal.py`, `mcp_workbench.py`) — no `tldw_api/client.py` row.
  `backlog/tasks/task-32809.2` status is `In Progress` (confirms "Already covered: insufficient").

## 9. P2 [D2] — `media_reading_schemas.py` pulls in the whole STT subsystem for two validators
- Verdict: CONFIRMED
- Site now: `media_reading_schemas.py:11-16` (module-scope import of 4 `STT.persistence` names),
  used only inside `_normalize_transcription_provenance`/`_normalize_failed_attempt` (`:74-83`)
- Proof: `python -c "import time; t0=time.time(); import tldw_chatbook.STT.persistence; ..."` →
  99.4 ms (review measured 94.3 ms — same order, environment variance).

## 10. P2 [D1] — ADR-173 timestamp guard matches calls only, misses a bare `default_factory` ref
- Verdict: CONFIRMED
- Site now: `chat_loop_schemas.py:37` — `ts: datetime = Field(default_factory=datetime.utcnow)`
- Proof: `python scripts/check_timestamp_writers.py` → `"0 datetime.utcnow() site(s) ... OK"`
  despite this live occurrence — the guard's `visit_Call` only fires on `ast.Call` nodes, and a
  bare `datetime.utcnow` reference (no `()`) is an `ast.Attribute`, never visited as a call.

## 11. P2 [D1] — six data-path handlers swallow a validation/decode failure into a different shape
- Verdict: CONFIRMED
- Site now: `flashcards_schemas.py:143-145` (`_populate_tags`), `:642-646`
  (`_populate_json_fields`); `client.py:13622-13624/13635-13637/13647-13649/13671-13673`
  (`create/list/get/update_prompt_collection`)
- Proof: read all six — each is `try: ... / except Exception: <fallback shape>` with the fallback
  discarded silently; the four client.py methods declare `PromptCollectionResponse | Dict[str,
  Any]` and return the raw dict on any `ValidationError`, with no log line.

## 12. P3 [D3] — two unreachable pydantic-v1 fallbacks, both below unguarded v2-only imports
- Verdict: CONFIRMED
- Site now: `flashcards_schemas.py:8-13`, `rag_admin_schemas.py:8-13`
- Proof: both files import `ConfigDict`/`field_validator` (v2-only) two lines above their
  `try: from pydantic import model_validator / except Exception: from pydantic import
  root_validator as model_validator` guard. `pyproject.toml:61` pins `pydantic>=2.4,<3`;
  installed is 2.12.5 (`python -c "import pydantic; print(pydantic.VERSION)"`).

## 13. P3 [D3] — two production branches exist only for monkeypatched tests
- Verdict: CONFIRMED
- Site now: `client.py:2992-2996` (`download_media_file`), `:6521-6526` (`tts_reading_item`) —
  both `self.__dict__.get("_request_bytes")` overrides
- Proof: `grep -rn '_request_bytes' Tests/tldw_api/ | grep monkeypatch` → exactly 4 hits across
  `test_watchlists_client.py` (×2) and `test_media_reading_client.py` (×2), matching "four
  monkeypatch.setattr calls" exactly.

## 14. P3 [D1] — malformed NDJSON line dropped via `print()`, reaching no log surface
- Verdict: CONFIRMED
- Site now: `client.py:1581`
- Proof: `grep -rn "print(" tldw_chatbook/tldw_api/*.py` → exactly one hit, this line.

## 15. P3 [D3] — `server_notifications_schemas.py` has no module docstring
- Verdict: CONFIRMED
- Site now: `server_notifications_schemas.py:1-3` — `from __future__ import annotations` precedes
  the string literal, so it is a dead expression, not a docstring
- Proof: `python -c "import tldw_chatbook.tldw_api.server_notifications_schemas as m;
  print(repr(m.__doc__))"` → `None`.

## 16. P3 [D3] — `extra=` has no stated rule across ~470 configured models; one file inverts it
- Verdict: CONFIRMED
- Site now: `chat_loop_schemas.py:33` (`ChatLoopEvent`, `extra="forbid"`), `:45`
  (`ChatLoopStartRequest`, `extra="allow"`)
- Proof: direct read confirms the inversion. A crude single-line grep census (`grep -rho
  'extra="[a-z]*"'`) gives `145 allow / 138 forbid / 105 ignore` vs the review's AST count of
  `145/140/104/81(none)` — same order of magnitude; the small delta is grep-vs-AST methodology,
  not a refutation of "no stated policy, real inversion exists."

## 17. P3 [D1] — `base_url` crosses a credential boundary with no validation
- Verdict: CONFIRMED
- Site now: `client.py:1175` — `self.base_url = base_url.rstrip("/")`, no scheme/host check
- Proof: read confirms `_validate_timeout` (`:1116-1147`, 20-line docstring, called at `:1178`)
  validates the timeout exhaustively in the same constructor while `base_url` gets none.

TOTALS: confirmed=17 fixed=0 wrong=0 demoted=0 promoted=0
