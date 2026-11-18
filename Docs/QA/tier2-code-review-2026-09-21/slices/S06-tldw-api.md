# S06 — API client (`tldw_api/`)

**Coverage:** files read in full: 14 | sampled: 21 | mechanical only: 26 (of 61).
`client.py` (16,661 lines) was read as: lines 1–1720 in full (every request/stream primitive) **plus all 131
non-passthrough method bodies extracted by AST and read in full**; the other 1,086 methods are one identical
`_request(...)`→`Model.model_validate` shape, covered by an AST census of endpoint strings, arg annotations and
return types rather than line-read. `mcp_unified_client.py`: 1–560 line-read + all 8 non-passthrough methods after
560 + an endpoint census of all 92. `__init__.py`: head/tail plus programmatic resolution of all 980 exports.
Mechanical only: 26 schema modules, covered by five package-wide AST censuses (`model_config`/`extra` policy,
endpoint interpolation + `quote()`, `json.loads`/`.json()` sites, `utcnow` refs, function-size + class-name duplication).

## Findings

### P1 [D1] — every non-2xx response on an SSE endpoint escapes the package's exception family as a raw `httpx.ResponseNotRead`
- Where: `tldw_api/client.py:1668-1686` (`_sse_request`'s `except httpx.HTTPStatusError`, specifically
  `response_data = e.response.json()` at `:1672` guarded only by `except ValueError` at `:1677`). Reached by 9 public
  methods: `stream_prompt_studio_events` (4760), `stream_meeting_session_events` (5063),
  `stream_audio_job_progress` (5525), `stream_media_ingest_job_events` (5942), `stream_notification_events` (7968),
  `stream_server_notifications` (8036), `stream_research_run_events` (8610), `stream_mcp_governance_events` (15891),
  `mcp_unified_client.py:475 stream_governance_events`.
- Evidence: `issubclass(httpx.ResponseNotRead, ValueError)` → `False` (its MRO is
  `ResponseNotRead → StreamError → RuntimeError`). Repro driving `_sse_request` against a MockTransport returning an
  *unread* streaming 401 — what a real transport produces:
  `_sse_request: LEAKED -> httpx.ResponseNotRead: Attempted to access streaming response content, without having
  called read().` The sibling `_stream_request` under the identical body returns `OK -> AuthenticationError`, because
  it calls `await e.response.aread()` first (`:1587`).
- Why it matters: an expired token on the notifications/governance/ingest streams never becomes
  `AuthenticationError`. `Notifications/event_observer.py:236` classifies it under the generic `except Exception:`
  reconnect arm, so a 401 is retried with exponential backoff exactly like a dropped socket instead of prompting
  re-auth; `Notifications/server_notification_events.py:145` records the user-facing status as
  `reason="ResponseNotRead"`. With `max_reconnects == 0` it propagates out of the worker as a non-`TLDWAPIError`
  that no caller catches.
- Recommended correction: inside the `async with`, when `response.status_code >= 400`, `await response.aread()`
  before `raise_for_status()` — the shape `_stream_request` already uses. Do it once in a shared
  `_raise_api_error_from(response)` and call it from all five primitives (see the D4 finding below).
- Size: M · ADR: no · Confidence: **verified**
- Pinning test: none. `Tests/tldw_api/test_client_error_classification.py` pins only `_request` (3 tests).
- Already covered: **no.** TASK-32805.1 is a *different* shape (`yield` inside `finally`) filed against `LLM_Calls/`.
  `tldw_api/` has **zero** `yield`-in-`finally` sites — all 25 `finally:` hits are `finally: cleanup_file_objects(...)`
  after the generator body, which is correct. **`tldw_api/` does not carry the TASK-32805.1 shape; it carries this
  different one, and no filed task covers it.**

### P1 [D1] — `tldw_api` is a fourth, unenumerated strict-JSON family: the tldw_server wire boundary accepts `NaN`, silently last-wins duplicate keys, and lets a deep body escape as a bare `RecursionError`
- Where: `client.py:1357` (`return response.json()` — every non-streaming API response), `:1363`, `:1471`, `:1537`,
  `:1672` (error bodies), `:1578` (`json.loads(line)` per NDJSON line), `:1588`, `:1630` (per SSE event).
- Evidence: driving `_request` through a MockTransport:
  `NaN: ACCEPTED -> {'score': nan}` · `dup-key: ACCEPTED -> {'id': 2}` · `deep-nest (200k): ESCAPED as
  builtins.RecursionError`. `grep -rn strict_json_loads tldw_chatbook/` → importers are
  `Chat/provider_continuation.py`, `Chat/thinking_blocks.py`, `LLM_Calls/hosted_chat.py`, `LLM_Calls/qwencloud.py`,
  `LLM_Calls/qwencloud_streaming.py`. **No `tldw_api` module imports it.**
  `grep -n 'tldw_api\|tldw_server' backlog/decisions/175-one-strict-json-acceptance-contract.md` → **no match**:
  ADR-175 (Accepted 2026-09-21) enumerates the adopters *and* the deliberate exceptions, and this boundary appears
  in neither list.
- Why it matters: `_request` catches only `HTTPStatusError`, `RequestError` and `JSONDecodeError` (`:1408-1420`), so
  `RecursionError` from a hostile or buggy server crosses the package boundary untyped — in a Textual worker
  (`exit_on_error=True` default) that ends the app. `NaN` floats reach pydantic float fields and then storage/UI.
  Duplicate keys resolve last-wins, the exact drift ADR-175 exists to remove.
- Recommended correction: route the five `.json()` sites and three `json.loads` sites through
  `Utils.input_validation.strict_json_loads` (a package-local wrapper translating `StrictJSONError` →
  `APIResponseError`), and add a `tldw_api` row to ADR-175's boundary list. Pair with a `max_bytes` read cap —
  `Utils/egress.py` already defines `MAX_FETCH_BYTES_*` for exactly this; `response.json()` currently buffers an
  unbounded body.
- Size: M · ADR: **yes** (`175-one-strict-json-acceptance-contract.md` — amend its enumeration) · Confidence: verified
- Already covered: **TASK-32805.5 is Done and ADR-175 is Accepted, but their enumeration is incomplete.** This is
  the insufficiency, not a restatement.

### P2 [D4] — five hand-rolled copies of the HTTP-error handler; the structured-detail fix landed in one and the two streaming copies silently drop the server's explanation
- Where: `client.py:1358-1406` (`_request`), `:1466-1489` (`_binary_request`), `:1532-1555` (`_headers_request`),
  `:1582-1605` (`_stream_request`), `:1668-1688` (`_sse_request`).
- Evidence: `_request` alone has the structured `{"detail": {...}}` branch (`:1370-1396`, with a 20-line comment
  citing "schedules task 6 round 2, D9" — the incident where dropping it surfaced a 409 as "this action requires a
  server connection"). The four siblings have a two-line `detail`-as-string branch only. Worse, in `_stream_request`
  the `await e.response.aread()` at `:1587` **always fails**: `raise_for_status()` fires inside
  `async with client.stream(...)`, so `__aexit__` closes the response before the `except` runs. Repro:
  `aread RAISED: StreamClosed`, and the live result is
  `AuthenticationError: Client error '401 Unauthorized' for url …` with `response_data == {"raw_text": ""}` — the
  server's `{"detail":"token expired"}` never reaches the message.
- Why it matters: the D9 regression is still live on both streaming paths.
- Recommended correction: one module-level `_raise_api_error_from(response)` in `client.py` carrying the full
  `:1370-1396` detail logic plus the read-before-`raise_for_status` ordering; the five call sites become one line
  each. Module-level, not a method — `mcp_unified_client.py` needs it too.
- Size: M · Confidence: verified
- Pinning test: `Tests/tldw_api/test_client_error_classification.py::test_request_409_dict_detail_surfaces_the_server_message`
  states the behaviour as a requirement — **for `_request` only**. Extending it to the other four is the pin.

### P2 [D4] — `server_notifications_schemas.py` is a 163-line duplicate of `notifications_reminders_schemas.py`; 16 of its 17 classes are unreachable, and the one field that drifted lost its `Literal`
- Evidence: AST class-by-class comparison — 16 shared class names, **15 byte-identical ignoring docstrings**, one
  drifted: `NotificationResponse.kind` is `str` in `server_notifications_schemas.py` and `NotificationKind` (a
  `Literal`) in `notifications_reminders_schemas.py`. `grep -rn server_notifications_schemas tldw_chatbook/ Tests/`
  → exactly **2 hits**: the `__init__.py` map row for `ServerNotificationStreamEvent` and `client.py:980` importing
  that one name. `__init__.py` maps all 16 other names to `notifications_reminders_schemas`, so the copies are not
  even reachable by name.
- Why it matters: 16 unreachable model definitions on the wire boundary, and the dead copy is the one where an
  unknown notification `kind` would be accepted. A maintainer editing the wrong file changes nothing.
- Size: S · Confidence: verified
- Already covered: fits TASK-32807's family but is in none of `.1`-`.6`.

### P2 [D1] — two client methods hit `/api/v1/notifications/stream` with two different event models; the server-notifications one can never resolve an SSE event id, and has no production caller
- Where: `client.py:7968-7981` (`stream_notification_events` → `NotificationStreamEvent`) and `:8036-8046`
  (`stream_server_notifications` → `ServerNotificationStreamEvent`) — same endpoint string.
- Evidence: `_sse_request._flush_event` (`:1626-1636`) emits `{"event", "data", "event_id"}`.
  `NotificationStreamEvent` declares `event_id`; `ServerNotificationStreamEvent`
  (`server_notifications_schemas.py:160`) declares **`id`** and defaults `extra` to ignore, so the real `event_id`
  key is silently dropped and `.id` is always `None`. `stream_media_ingest_job_events` (`:5947`) does the
  `event_id`→`id` rename explicitly; `stream_server_notifications` does not. Zero production callers, one test —
  and that test (`Tests/tldw_api/test_server_notifications_client.py:205-230`) mocks `_stream_sse_request` wholesale
  and *constructs* `ServerNotificationStreamEvent(id="11", …)` by hand, so it proves the model can hold an id, never
  that the SSE path fills it. The two models also disagree on `data`: required `dict` vs `dict | str | None = None`.
- Why it matters: `Last-Event-ID` resumption is structurally impossible on this path; and a `data:` line whose JSON
  is a scalar kills `stream_notification_events` with a `ValidationError` while the other accepts it.
- Size: S · Confidence: verified
- Pinning test: the mocked-seam test above would stay green through the fix — the brief's "mocked tests never catch
  these" case.

### P2 [D4] — two live, drifted client surfaces for the same 16 MCP-hub endpoints; one quotes path segments and one does not
- Where: `client.py:15823-16140` (`list_mcp_*`, `create_mcp_*`, `stream_mcp_governance_events`, …) vs
  `mcp_unified_client.py:475-560, 712-898`.
- Evidence: AST endpoint extraction over both — `client.py` declares 19 `/api/v1/mcp/*` endpoints,
  `mcp_unified_client.py` declares 80, **16 overlap** with different method names (e.g.
  `/api/v1/mcp/hub/events/stream` → `stream_mcp_governance_events` vs `stream_governance_events`, whose bodies are
  line-for-line identical). **Drift:** `client.py:15948/15959/15966/15976` interpolate `quote(server_id, safe='')`;
  `mcp_unified_client.py:517/527/533` interpolate `{server_id}` raw. Both families have live callers
  (`get_effective_policy`: 5 refs vs 4; `list_external_servers`: 5 vs 8).
- Why it matters: a `server_id` containing `/` or `?` changes the request path through one family and not the other;
  which one a caller picked decides whether the guard applies.
- Recommended correction: keep `MCPUnifiedClient` as the single MCP surface and delete the 16 `*_mcp_*` methods from
  `client.py` — but adopt the quoting side first.
- Size: M · Confidence: verified

### P2 [D1] — 334 `str`-typed path segments are interpolated into request URLs unquoted, next to a validating-and-quoting helper used on 29 sites
- Where: `client.py` — 705 interpolated `/api/v1/…` segments: 355 `int`-annotated (safe), 16 resolved by
  `_workspace_source_path_id` (`:1083-1096`, which validates type/blank/length then `quote(…, safe="")`), 17 wrapped
  in an inline `quote(…, safe='')`, and **334 `str`-annotated with neither**. User-reachable examples:
  `f"/api/v1/sharing/public/{token}"` (`:9006, 9016, 9022` — a token the user pastes),
  `f"/api/v1/chat/shared/conversations/{share_token}"` (`:13122`), `f"/api/v1/consent/preferences/{purpose}"`
  (`:3814/3819`), `f"/api/v1/llm/providers/{provider_name}"` (`:4149`),
  `mcp_unified_client.py:1706/1713`.
- Evidence: AST scan resolving each `FormattedValue` to its enclosing function's parameter annotation →
  `TOTAL 705 · str-annotated 334 · int-annotated 355 · unresolved 16` (the 16 unresolved are the locals *produced*
  by the helper — i.e. the guarded ones). httpx resolves a relative path via `base_url.join()`, which normalises
  `../` and treats `?`/`#` as delimiters.
- Why it matters: a pasted share token containing `?` appends attacker-chosen query parameters to an authenticated
  request; one containing `../` walks out of the API namespace with the `X-API-KEY` header attached. The repo already
  decided this matters (the helper, the 17 inline `quote`s, `_normalize_api_namespace_endpoint`'s `..` refusal at
  `:1790`) — it was applied to 8% of the sites.
- Recommended correction: make `_workspace_source_path_id` a general `_path_segment(value, field_name)` and route
  every `str`-typed segment through it; an AST guard in `Tests/Architecture/` keeps new methods honest.
- Size: L · Confidence: verified (counts) / **inferred (exploitability — callers are outside this slice)**

### P2 [D3] — `tldw_api/client.py` is the largest module in the repo with no size-ratchet row: 16,661 lines, **1,217 methods on one class**
- Evidence: AST → `TLDWAPIClient methods: 1217 lines: 15557`, of which 1,086 are a single
  `_request(...)`→`Model.model_validate` shape and 131 carry any control flow. `wc -l | sort -rn` ranks it **8th
  repo-wide**; `Tests/Architecture/test_module_size_ratchet.py:_BUDGETS` has 7 rows and includes
  `personas_screen.py` (16,449, rank 9) but **not** this file (16,661, rank 8). The only other unbudgeted top-10
  module is `DB/ChaChaNotes_DB.py` (24,424), outside this slice.
- Recommended correction: add a `"tldw_chatbook/tldw_api/client.py": 16661` row. The honest decomposition is by API
  namespace, with `TLDWAPIClient` composed of per-namespace mixins or delegates — `MCPUnifiedClient` is the
  precedent already in the package.
- Size: S (the row) / L (the split) · Confidence: verified
- Already covered: **TASK-32809.2 is In Progress and its docstring says the rows are "hand-picked god modules, not a
  directory family, so there is no glob that auto-adds new files." This file was missed by the hand-picking.**

### P2 [D2] — `media_reading_schemas.py` pulls the whole STT subsystem into `client.py`'s module scope for two field validators: 94 ms of a 475 ms import
- Where: `tldw_api/media_reading_schemas.py:11-16` (four `STT.persistence` names), used only at `:74-75` and `:82`,
  inside `_normalize_transcription_provenance` / `_normalize_failed_attempt`.
- Evidence: `import tldw_chatbook.STT.persistence` → **94.3 ms, 9 STT submodules**;
  `import tldw_chatbook.tldw_api.client` → **503 / 473 / 472 ms**, with `STT modules pulled in: 9` every time.
  `client.py:646` imports `media_reading_schemas` at module scope; `tldw_api/__init__.py`'s own docstring records
  that ~44 `Server*Service` modules import `TLDWAPIClient` at module level, several from `app.py` module level.
- Why it matters: ~20% of the client's import cost, on the boot path, for two validators that fire only when a media
  record carries transcription provenance. These four names are the **only** cross-package import in all 61 files
  besides the documented `notes_workspace_limits` re-export.
- Recommended correction: move the import into the two validator bodies. The repo's precedent for exactly this is
  `notes_workspace_limits.py`, created by TASK-23023 to stop `Research_Workspace/server_adapter.py` paying a 782-LOC
  pydantic module's import cost "for one integer", guarded by
  `Tests/Packaging/test_research_workspace_import_closure.py`.
- Size: S · Confidence: **verified (measured)**

### P2 [D1] — the ADR-173 timestamp guard matches calls only, reports "0 sites", and misses the one live `datetime.utcnow` in the repo
- Where: guard `scripts/check_timestamp_writers.py:78-81` (`visit_Call` → `func.attr == "utcnow"`); occurrence
  `tldw_api/chat_loop_schemas.py:37` — `ts: datetime = Field(default_factory=datetime.utcnow)` (a bare reference,
  never an `ast.Call`).
- Evidence: `python scripts/check_timestamp_writers.py` → `0 datetime.utcnow() site(s) … OK` (exit 0), and
  `scripts/timestamp_writer_census.tsv` contains only its 4 comment lines. An AST sweep for `ast.Attribute` nodes
  named `utcnow` that are *not* the `func` of a `Call` → **1 hit repo-wide, this one**. The guard's own docstring:
  "`datetime.utcnow()` — deprecated in 3.12 and always **naive**; it is never correct here. Forbidden outright."
- Why it matters: the guard is the artifact the repo trusts to know this is clean, and it reports clean while it
  isn't. The occurrence itself: a server chat-loop event with no `ts` gets a naive datetime that
  `model_dump(mode="json")` serialises with no offset — ADR-173's stated latent bug.
- Size: S · Confidence: verified
- Already covered: **TASK-32803.1 and .5 are both Done; this is the hole they left, with a live occurrence.**
  *(Fourth independent sighting of a `check_timestamp_writers.py` blind spot — see S01-P3, S11-P3, S05-P3.)*

### P2 [D1] — six data-path handlers swallow a validation/decode failure and return a shape the caller reads as success
- Where: `flashcards_schemas.py:143-145` (`_populate_tags`: `except Exception: parsed = []` → the card reports zero
  tags), `:642-646` (`_populate_json_fields`: `except Exception: parsed = {}` → `structured_payload`/
  `context_snapshot` silently empty); `client.py:13622-13624, 13635-13637, 13647-13649, 13671-13673`
  (`create/list/get/update_prompt_collection`: `try: return Model.model_validate(response) / except Exception:
  return response` — declared return type `PromptCollectionResponse | Dict[str, Any]`, so every caller must
  type-test, and none of the four logs the discarded `ValidationError`).
- Why it matters: a malformed `tags_json` becomes "this card has no tags" rather than an error anyone can see; the
  four collection methods turn a schema mismatch into a raw dict that a caller indexing `.name` will
  `AttributeError` on, far from the cause.
- Size: S · Confidence: verified (read)

### P3 [D3] — two unreachable pydantic-v1 fallbacks, the only ones in the repo, and both sit below unguarded v2-only imports
- Where: `flashcards_schemas.py:10-13`, `rag_admin_schemas.py:10-13`.
- Evidence: `pyproject.toml:61` pins `pydantic>=2.4,<3`; installed 2.12.5. Both files import v2-only names
  *unguarded two lines above* (`ConfigDict, Field, field_validator`), so under v1 the module would already have
  failed before reaching the guard.
- Why it matters: dead, and actively misleading — `root_validator` and `model_validator` have incompatible
  signatures, so if the branch ever fired every `@model_validator(mode="after")` in the file would break at import.
- Size: S · Confidence: verified

### P3 [D3] — two production branches exist only so monkeypatched tests work, and make the public return type depend on whether a test patched an attribute
- Where: `client.py:2992-2996` (`download_media_file`) and `:6521-6526` (`tts_reading_item`):
  `request_bytes_override = self.__dict__.get("_request_bytes")` / `if … is not None: return await …`.
- Evidence: the only writers of that instance attribute are four `monkeypatch.setattr(client, "_request_bytes", …)`
  calls in `Tests/tldw_api/`. The dict lookup is redundant even for them (an instance attribute already shadows the
  method); its sole effect is to make the declared `ReadingExportResponse | bytes` return `bytes` under test and
  `ReadingExportResponse` in production.
- Size: S · Confidence: verified

### P3 [D1] — a malformed line in an NDJSON ingest stream is dropped with a `print()` that reaches no log surface
- Where: `client.py:1579-1581` — the only `print(` in all 61 files; `tldw_api/` has no logger at all except stdlib
  `logging` in `utils.py`.
- Evidence: Textual 8.2.8 `App` runs under `with redirect_stdout(self._capture_stdout)`, so this does **not** corrupt
  the terminal (that reading was retired) — it goes to the app's stdout capture and never reaches the loguru sinks
  or the in-app Logs window. `process_mediawiki_dump` / `ingest_mediawiki_dump` therefore skip pages silently.
- Size: S · Confidence: verified

### P3 [D3] — `server_notifications_schemas.py` has no module docstring: the string sits below `from __future__ import annotations`
- Evidence: `m.__doc__` → `None`. The misplacement is why lines 5 and 7 carry `# noqa: E402`. Moot if the module is
  deleted per the duplication finding. · Size: S · Confidence: verified

### P3 [D3] — `extra=` has no stated rule across 470 configured models, and 21 response-shaped models hard-fail on a server field addition
- Evidence: package-wide AST census — `extra="allow"` 145 · `extra="forbid"` 140 · `extra="ignore"` 104 · no `extra`
  key 81. Response-shaped models with `extra="forbid"`: 21, in `chat_loop_schemas.py` (4), `meetings_schemas.py` (5),
  `research_runs_schemas.py` (4), `web_clipper_schemas.py` (3), `sync_schemas.py` (4), `mcp_unified_schemas.py` (1).
  The clearest inversion is inside one file: `chat_loop_schemas.py:42 ChatLoopStartRequest` is `extra="allow"`
  (unknown client fields go **to** the server) while `:33 ChatLoopEvent` is `extra="forbid"` (a new server field
  breaks the **client**) — the policy is backwards on the pair.
- Why it matters: a server minor upgrade adding one field to `MeetingSessionResponse` makes `get_meeting_session()`
  raise `pydantic.ValidationError`, which is **not** a `TLDWAPIError`, from outside any `try` in the 1,086
  passthrough methods.
- **Explicitly excluded:** the `sync_schemas.py` `forbid` + `strict` models (`SyncPersonalContextBootstrap*`,
  `:620-800`) are deliberate — the "strictly validated, content-free" first-link security boundary, with cross-field
  binding validators. **Do not loosen them.**
- Size: M · ADR: yes (new) · Confidence: verified (census)

### P3 [D1] — `base_url` crosses a credential boundary with no validation, in a constructor that validates timeouts exhaustively
- Where: `client.py:1178` — `self.base_url = base_url.rstrip("/")`, versus `_validate_timeout` (`:1116-1147`) with a
  20-line docstring on why an invalid timeout must be rejected at the boundary.
- Evidence: `runtime_policy/bootstrap.py:198-233` passes whatever the config's `base_url`/`api_url`/`url` string is,
  stripped, with no scheme check. `_raise_if_redirected`'s own docstring (`:1240`) names "a MITM on an `http://` base
  URL" as part of its threat model, yet nothing warns on or refuses a plaintext base URL carrying `X-API-KEY`.
- Recommended correction: validate scheme ∈ {http, https} and a parseable host in `__init__`; warn once (not refuse)
  on `http://` to a non-loopback host — local servers are the legitimate `http://` case.
- Size: S · Confidence: verified (read)

## Candidate triage
**RETIRED:** `except_exception_pass` `utils.py:93` (inner `file_obj.close()` in a cleanup path).
`function_body_import` `__init__.py:2347` — PEP 562 lazy re-export; verified sound, all 980 `__all__` names resolve,
0 unmapped. `legacy_markers` ×11 — all protocol *values* (`Literal["legacy","structured"]`) or server fields named
`deprecated: bool`. `mutable_class_attr` `character_persona_schemas.py:218/306/904` — `model_config = {...}` is a
dict literal read once at class creation, not shared mutable state (it *is* inconsistent with the package's
`ConfigDict(...)` convention — P3 drift, not a bug). `mutable_class_attr` `schemas.py:325 chunks: List[...] = []` —
pydantic v2 deep-copies field defaults; verified `a.chunks is b.chunks` → `False`. `raw_1024x1024`
`skills_schemas.py:14,15` — named module constants used by `_validate_supporting_files`.
`DUP_SHAPE 57efb65ad3a7` — the three `media_reading_schemas` copies are thin `None`-guard wrappers delegating to a
*single* shared normalizer; the sharing already happened. Five more `DUP_SHAPE` rows — 3-to-5-line pydantic
validators whose only shared shape is "raise on empty", each with a different field name, bound and message.
**CONFIRMED:** `except_exception_return` `client.py:13623/13636/13648/13672` (filed); `try_import_guard`
`flashcards_schemas.py:10`, `rag_admin_schemas.py:10` (filed);
`DUP_VERBATIM 47b24285afb5`/`DUP_SHAPE 3a244703eaf4` (`_validate_schedule_fields` ×2) — **escalated**: not two
helpers, two whole duplicate modules; `DUP_VERBATIM 6f2d7b8db08d` and `78673adb6435` (the `str(v).strip() or None`
family) and `DUP_SHAPE 12df524000c5` — confirmed, cross-slice, handed to Phase 3.

## D4 observations for repo-wide Phase 3
1. **The five-copy HTTP-error handler is the slice's biggest D4.** No shared helper exists; canonical home is a
   module-level function in `tldw_api/client.py`. Drift is behavioural on three of five copies and has already
   re-opened a fixed bug (D9).
2. **Two whole duplicate modules**, not two helpers. The repo-wide question: how many other `*_schemas.py` pairs
   overlap? My class-name census found no other pair, but it compared names only within `tldw_api/`.
3. **Two live client families for one wire surface**, drifted on path quoting. Home: `mcp_unified_client.py`.
4. **`quote(…, safe='')` is the re-rolled helper**: 17 inline copies + 1 validating-and-quoting function (16 uses)
   against 334 sites that skip both. Repo-wide, check whether `Utils/` already owns a URL-segment encoder — I found
   none inside `tldw_api/`.
5. **The `str(v).strip() or None` family** has 3 members here with **no drift**; they are 3-line pydantic validators.
   My read: **low-value to consolidate** — the cost is an import edge from ~8 packages to `Utils/` for three lines
   each. Recommend the lead weigh it as P3, not P2.
6. **Boilerplate scale, for the decomposition brief:** 1,086 of `client.py`'s 1,217 methods and 84 of
   `mcp_unified_client.py`'s 92 are a single `await self._request(m, ep, …)` → `Model.model_validate(response)`
   shape. That is ~13,000 lines expressing ~1,100 `(method, verb, endpoint, request-model, response-model)` tuples.
   A declarative table plus one dispatcher would eliminate it — and would fix observations 1, 2 and 4 **structurally
   rather than site-by-site**. ADR-sized, so recorded as an observation, not a recommendation.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| The unquoted-path-segment defect is reachable with attacker-influenced text end to end | callers live outside S06; mechanism and 334 sites verified, not a full path from a UI input to `_request` | `rg -n 'import_public_share\|get_public_share\|verify_share_password' tldw_chatbook/UI tldw_chatbook/Sharing`, read `UI/Sharing_Panel.py:_public_token`, then `pytest Tests/tldw_api/test_sharing_client.py -q` with a token containing `?a=b` |
| The SSE `ResponseNotRead` reaches a Textual worker and ends the app (vs only being mis-logged) | brief forbids running the app; depends on each caller's `max_reconnects` | `rg -n 'max_reconnects' tldw_chatbook/ --include='*.py'` then `pytest Tests/Notifications/ -q` with a fault-injecting transport returning an unread 401 |
| Whether the 21 `extra="forbid"` response models were deliberate (as the `sync_schemas` ones demonstrably were) | no comment or ADR on the meetings/chat_loop/web_clipper/research_runs ones | `git log -L 40,46:tldw_chatbook/tldw_api/meetings_schemas.py` and `git log -S 'extra="forbid"' --oneline -- tldw_chatbook/tldw_api/chat_loop_schemas.py` |
| Exact boot-path cost of the STT import in a real app start (94 ms of 475 ms measured in isolation) | did not run the boot budget | `pytest Tests/Performance/ -q` and `python -X importtime -c 'import tldw_chatbook.app' 2>&1 \| grep -E 'STT\|tldw_api'` |
| Whether `RecursionError` from a deep body is caught upstream of `_request` | verified it escapes `_request`; the ~44 `Server*Service` wrappers not audited | `rg -n 'except (Exception\|RecursionError)' tldw_chatbook/*_Interop/ tldw_chatbook/Media/server_media_reading_service.py` |
