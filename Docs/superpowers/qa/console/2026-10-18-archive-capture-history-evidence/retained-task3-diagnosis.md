# Task 3 read-only diagnosis — historical thinking-owner projection

The remaining Direct follow-up refusal is faithfully reproduced at the original reseal guard: `ValueError("archive_trace_admitted_rows")`, controller10607, becomes the preserved content-free `TraceCallPersistenceError(reservation_status="not_established")` at10654. A real saved historical assistant has a thinking-owner marker in the admitted request, while the agent's transcript-shaped rows retain only their native-owner annotation. The comparison mixes these two existing representations.

Baseline/unchanged HEAD: `75b671c001e84bf2713c380e30de3e587bcc8ae0`. Source/index/HEAD were not edited. Only root's existing plan/task changes remain in tracked status; index is empty. No app, real provider/server/network request, subagent or full suite was invoked. Root owns live cleanup and the15/20 request accounting; its native Cancel/Quit/PTYexit0 is separate evidence.

## Actual source evidence

- Scratch: `/private/tmp/archive-capture-native-_yuzjq8a/direct-complete`. The actual file is `source-provenance.json` at that directory's top level (the initially requested nested path does not exist).
- `after-search.json` and `blocked.json` have exactly equal five COMPLETE call records and reconstruction records. Two saved source users and their revisions exist; the blocked current user's revision is `3ec752f6-4fa3-426b-a93c-1ce9b3ced091`, owned by message `b6f0e740-5dad-44cd-9939-17cbef3cc5d8` in source conversation `b4ad3b53-6ca4-4900-b1e0-3bcb60e9df8f`. No successor call was created.
- A read-only connection to the actual scratch chat DB confirms historical search assistant `4bff0158-5f8a-423a-9c1b-d4c6ff8a196e` has five displayable complete thinking blocks for llama_cpp/Gemma/chat_completions and no provider continuation. The current user has neither sidecar. No actual DB write occurred.
- The actual log has `capture_provider_failure` on the successful initial send as well as the blocked follow-up. It comes from optional staged-RAG capture and is not established as the blocker. There is no new Library constructor TypeError. The actual blocked send ends before trace reservation/provider entry.

## Producer/consumer mismatch

`_build_durable_trace_request` strips native/persisted locator annotations for its provider rows, associates saved revisions and real thinking/continuation groups, then adds `THINKING_OWNER_KEY` / `CONTINUATION_OWNER_KEY` to the selected saved owners at10394–10401. Request construction validates attachment/group ownership. `_run_agent_reply` still receives transcript-shaped rows with `_native_message_id` and strips its leading system row at26234–26240.

The reseal strips only its three native/persisted locator annotations from those before rows, then compares them with raw `request.flattened_messages()` at10603–10607. That raw admitted historical assistant also contains `_tldw_thinking_owner`; before does not. The current row and historical role/content otherwise match.

Existing `_flatten_preflight_messages` at1814–1834 is the canonical local projection: it removes typed continuation/thinking owner markers, rejects conflicting owner markers, and restores their matching `_native_message_id` association. Existing preflight consumers use it. Applying that existing pure projection to the admitted request, then the same existing locator exclusion on both sides, makes the complete unchanged row comparison equal in the probe. No arbitrary field exclusion was tried.

## One bounded offline production-boundary reproduction

Temporary script: `/private/tmp/test_archive_capture_sidecar_diagnosis.py`, SHA256 `d632494d9a666a6fd088b6376758a96ba4da5b45b04cccf0bfcf219433f1e5d9`. It imports the existing real SQLite/native/captured fixtures, uses actual native admission/controller/agent bridge/capture gateway/factory/trace persistence and the corrected production Library factory. It pins a valid local llama_cpp/Gemma resolution with native tools, displayable version1 thinking and no preservation support. Only the documented external `chat_api_call_fn` transport returns synthetic responses. Supported start-anchored thinking parsing/persistence creates three real historical assistant thinking blocks; native source producers and guard are not stubbed. Transparent thread-local exception observation calls the original reseal unchanged. DNS and IPv4/IPv6 socket guards remain enabled.

Command prefix, cwd isolated worktree:

```sh
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -p Tests.conftest -p no:cacheprovider -c pyproject.toml /private/tmp/test_archive_capture_sidecar_diagnosis.py -q
```

| Phase | Additional arguments / raw output | Exit | Result |
| --- | --- | --- | --- |
| Setup | `--basetemp=/private/tmp/archive-capture-task3-offline-stores`; `/private/tmp/archive-capture-task3-offline.log` | 1 | 1 failed,4.77s. Invalid harness resolution omitted required thinking round-trip version; rejected before any send. Initial script preserved as `/private/tmp/test_archive_capture_sidecar_diagnosis-setup.py`. |
| Faithful reproduction | `--basetemp=/private/tmp/archive-capture-task3-offline2-stores`; `/private/tmp/archive-capture-task3-offline2.log` | 0 | **1 passed,3.65s**, no teardown/network-guard errors. Required version1 was supplied; no permissive fallback or guard modification. |

Exact observation: `/private/tmp/archive-capture-task3-offline.json`, SHA256 `d9f12c2e9c86ba75ad9486aa1916f4721b64fb39ad31b30da9e7c1a5bf7211b5`:

- Search completes. All three before rows have `_native_message_id`, `content`, `role`. Admitted rows have `content`, `role`, with `_tldw_thinking_owner` additionally on the historical assistant.
- Exactly one historical thinking group, zero continuation groups. Current descriptor is a real `SavedRevisionTraceProvenance`, revision `984f59ba-4702-4a31-acc2-b2035b579cc4`, associated with current native/persisted user `d5b6b620-89ff-45b1-80cd-95ecad39554f` and unchanged text `Restore the second one.`
- Raw comparison false; canonical preflight projection comparison true. The original guard emits the precise exceptions above and returns blocked before any successor capture factory/transport/restore.
- Predecessor call rows/reconstructions remain exactly unchanged; pending recovery authorities0; DNS attempts empty. The exact native paused-preparation Cancel clears the probe, and fixture cleanup completes. The pytest command finishes; no held worker or app remains from this probe.

## Smallest supported direction and limits

Use the existing canonical preflight value projection for the admitted side of this local comparison, preserving its validated owner/group association, marker-conflict rejection, historical fields and all current-source/non-text checks. Retain the original request's sidecar groups and frozen fields; do not drop history markers indiscriminately, widen generic provider-factory equality, replay TOOL history, or weaken trace/runtime/service gates. This is a source-backed direction, not an applied fix.

The actual mounted process did not expose its suppressed internal exception before root canceled it. Its persisted thinking shape and no-successor boundary match the reproduced mismatch; attribution is strongly supported, not a claim that an observation hook captured the original live exception. The offline synthetic parser uses supported start-anchored thinking rather than the actual server's reasoning_content format, and three fake search calls rather than the actual five. Both produce the same validated historical thinking-group/owner representation at the failing boundary. Continuation-bearing and contradictory-marker cases were not separately executed; follow-up implementation requires a new brief and focused regression controls.

Final source checks exited0 and preserve controller SHA256 `0652d3fe258d974c06680889f558a2a993de75ed8da7300cba970d471d932a7a`, shared runtime `3cd6e642f5b8e9bec25329af6d7cac2d9d0d2fc26a2d02be2efce2beb9ec8cf2`, and existing follow-up test `c2f69ace650aa9d2d8c2efce697dac223de54fb41f29963b166bd258cbcdffe5`. No implementation or extra live qualification was performed.
