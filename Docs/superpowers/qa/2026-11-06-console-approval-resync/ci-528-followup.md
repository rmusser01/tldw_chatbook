# Hosted528 approval integration follow-up

Hosted run42377423452 tested `528520e750d005b8eebc42ffc732b973630df4a7`. The Dreams/Guardian job128419825253 failed the real mobile approval round-trip because `ApprovalBatchView` reached JSON encoding. Architecture job128419825248 reported two missing class-only logging census entries, one unreviewed Details memory-open owner and two screen-size refusals. The affected tests were unchanged from dev `a0ae30b7ed9c313fe19cb3a0da5d95ebde182385`.

Mobile events and pending approvals now use an explicit wire projection. Its four core call fields reuse the existing relay projection, including intersection of producer options with captured legal decisions. Existing mobile detail fields and desktop relay ownership remain present. Arguments still come from the original calls; the captured view and UI presentation state remain internal. The owner retains the original pending payload.

The screen keeps both approval message decorators and delegates routing to its existing wiring module. Feedback uses the originating chat/run; decisions retain the original `ApprovalDecisions` object, denial text and captured round after a chat switch. The screen is24608 lines under the unchanged24617 ceiling. The logging census adds only `RAG_Search/state_paths.py::_explicit_rag_setting` and its single class-name keyword. The owner inventory records `ChatApprovalCard._open_details` as memory-only page selection, with no independent file.

Local RED evidence includes the original real-engine JSON failure, two captured-view event/pending failures with two legacy controls passing, both logging census failures and the missing owner row. The two local size controls ran after delegation and passed; their original RED evidence is the hosted run above.

All final invocations used the reviewed protected launcher, the existing Python3.12 environment and isolated original-home/origin/startup guards:

| Target | Passing cases |
| --- | ---: |
| Mobile service: real approval round-trip, captured/legacy event and pending projections, desktop relay owner preservation | 6 |
| Mobile HTTP/WS API | 32 |
| Served relay publisher | 21 |
| Class-only failure logging | 7 |
| Backup owner inventory | 18 |
| Realtime controller boundary | 6 |
| Review-selection controller boundary | 7 |
| Approval feedback, including actual MessagePump routing | 16 |
| Approval action ownership | 25 |
| Screen-size ratchet | 4 |
| **Total** | **142** |

Static comparison proves the screen is unchanged outside its reviewed handler and wiring is unchanged outside the added delegate. Ruff diagnostic fingerprints are identical before/after:19 inherited diagnostics. Only changed code ranges were formatted; existing whole-file formatter differences remain outside scope. The owned diff passes whitespace checks. Independent source review found no material issue. Existing ADR235/236/355 govern this repair.

These product SHA256 bindings use UTF8 text with CRLF converted to LF, matching portable Git text. They are separate from raw private runtime/freezing receipts:

| Product file | UTF8/LF SHA256 |
| --- | --- |
| `tldw_chatbook/Web_Server/mobile_service.py` | `1a3f45128ad145707f696e50cd6e8bf495c10100bc1fbb6341364c01dd26f06f` |
| `tldw_chatbook/UI/Console_Modules/wiring.py` | `0bcd001f8eaf45ab10f5d0dd2caf88e295572ed40cfce0672f0225f1b8a9af95` |
| `tldw_chatbook/UI/Screens/chat_screen.py` | `9b94335791ff74b47bd0ab5f9ae8f95b14155292e26cb1afb6ba638b64569e7a` |

Commands, stdout/stderr, exit records, original-home/origin/startup receipts and the raw static proof remain in the private `ci-integration-fixes-20261107T142500Z` evidence directory. No full suite, App or physical timing run occurred in this correction. Current-head hosted acceptance and native/browser presentation qualification remain open; no performance result is inferred from these checks. Task34569 stays In Progress.
