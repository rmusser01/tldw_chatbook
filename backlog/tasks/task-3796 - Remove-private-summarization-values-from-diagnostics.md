---
id: TASK-3796
title: Remove private summarization values from diagnostics
status: Done
assignee:
  - '@codex'
created_date: '2026-08-09 14:49'
updated_date: '2026-08-11 06:02'
labels:
  - llm-calls
  - observability
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-2118's final review found that its identifier-filtered candidate list had been incorrectly promoted into a complete summarization follow-up. A fresh review of every logger call in `Local_Summarization_Lib.py` and `Summarization_General_Lib.py` verified a broader boundary: raw/processed/extracted input, custom/system/combined prompts, response and generated-output content, credential fragments, private endpoint or local-path values, and exception/error details are written to diagnostics. These sites are not raw provider request-payload dictionaries or tool definitions, so they remain outside TASK-2118 acceptance criterion 4. The current persistent sink rejects these ordinary records, but their private values are still formatted and submitted to logging, remain observable by non-persistent handlers, and could be admitted by a later sink change. Applying ADR-029's metadata-only allowlist at these call sites requires one atomic containment repair.
<!-- SECTION:DESCRIPTION:END -->

## Design Reference

- [TASK-3796 summarization diagnostic privacy design](../../Docs/superpowers/specs/2026-08-10-task-3796-summarization-diagnostic-privacy-design.md)

## Verified Diagnostic Inventory

The stable owner is the module plus enclosing function plus diagnostic label/category; current line numbers are navigation aids only. The complete all-call audit reviewed 523 logger calls and classified 200 direct private diagnostics. Safe constant, type, length, status, count, and provider/model metadata calls are intentionally absent. Exception messages are included because ADR-029 permits exception type metadata but expressly excludes exception messages.

### `Local_Summarization_Lib.py` — 100 sites

| Enclosing function | Category | Diagnostic labels (current lines) |
| --- | --- | --- |
| `summarize_with_local_llm` | raw/processed/extracted input | `Loaded data` (48) |
| `summarize_with_local_llm` | exception/error detail | `Error in processing` (136) |
| `summarize_with_local_llm.stream_generator` | response/output content | `Error decoding JSON from line` (113) |
| `summarize_with_llama` | raw/processed/extracted input | `Loaded data` (175) |
| `summarize_with_llama` | prompt content | `System Prompt being sent` (205); `Prompt being sent` (212) |
| `summarize_with_llama` | private endpoint/path | `Using API URL` (169) |
| `summarize_with_llama` | response/output content | unlabelled `response_data` (306); `request failed ... response.text` (315) |
| `summarize_with_llama` | exception/error detail | `Error in processing` (321) |
| `summarize_with_llama.stream_generator` | response/output content | `Error decoding JSON from line` (296) |
| `summarize_with_kobold` | raw/processed/extracted input | `Loaded data` (364) |
| `summarize_with_kobold` | prompt content | `Prompt being sent` (390) |
| `summarize_with_kobold` | credential fragment | `Using API Key` (357) |
| `summarize_with_kobold` | response/output content | `Received streamed data` (449); `Ignoring line` (479); `request failed ... response.text` (481, 538); `API Response Data` (521) |
| `summarize_with_kobold` | exception/error detail | `Error decoding streamed JSON` (474); `Error in processing` (486, 543, 546); `Error parsing JSON response` (535) |
| `summarize_with_oobabooga` | prompt content | `Prompt being sent` (644) |
| `summarize_with_oobabooga` | credential fragment | `Using API Key` (596) |
| `summarize_with_oobabooga` | private endpoint/path | `Using API URL from config` (583); `Invalid API URL configured` (588) |
| `summarize_with_oobabooga` | response/output content | unlabelled `response_data` (753); `Summary` (757); response-text `error_msg` (763); `Error response` (769) |
| `summarize_with_oobabooga` | exception/error detail | `Error parsing JSON string` (609); `Error streaming summary` (724); `Error decoding JSON` (772); `Error making API request` (775); `Unexpected error` (778) |
| `summarize_with_oobabooga.stream_generator` | exception/error detail | `JSON decode error` (719) |
| `summarize_with_tabbyapi` | raw/processed/extracted input | `Loaded data` (825) |
| `summarize_with_tabbyapi` | credential fragment | `Using API Key` (817) |
| `summarize_with_tabbyapi` | response/output content | `Received non-data line` (920) |
| `summarize_with_tabbyapi` | exception/error detail | `Failed to parse JSON streamed data` (916); `Error summarizing` (924, 976); `Unexpected error` (927, 984, 988) |
| `summarize_with_vllm` | raw/processed/extracted input | `Raw input data` (1024); `Processed data` (1050); `Extracted text` (1072) |
| `summarize_with_vllm` | prompt content | `Custom prompt` (1075) |
| `summarize_with_vllm` | credential fragment | `Using API key from config` (1010); `Using API Key` (1020); `vLLM API Key` (1086) |
| `summarize_with_vllm` | response/output content | `Summary` (1198); `Error response` (1211) |
| `summarize_with_vllm` | exception/error detail | `Error parsing JSON string` (1037); `Error decoding JSON` (1214); `Error making API request` (1219); `Unexpected error` (1224) |
| `summarize_with_vllm.stream_generator` | response/output content | `Error decoding JSON from line` (1162) |
| `summarize_with_ollama` | prompt content | `Summarization prompt` (1338) |
| `summarize_with_ollama` | response/output content | `Full JSON response` (1469) |
| `summarize_with_ollama` | exception/error detail | `Error loading config` (1257); `HTTP error occurred` (1403); `Request exception` (1406); `Unexpected error` (1409); `Exception` (1488) |
| `summarize_with_ollama.stream_generator` | response/output content | `JSON decode error on line` (1435) |
| `summarize_with_custom_openai` | raw/processed/extracted input | `Raw input data` (1521); `Processed data` (1548); `Extracted text` (1572) |
| `summarize_with_custom_openai` | prompt content | `Custom prompt` (1575) |
| `summarize_with_custom_openai` | credential fragment | `Using API Key` (1515) |
| `summarize_with_custom_openai` | private endpoint/path | `Using API URL` (1608) |
| `summarize_with_custom_openai` | response/output content | `full API response data` (1705); unlabelled `response_data` (1708); `Chat response` (1714); `Error response` (1725) |
| `summarize_with_custom_openai` | exception/error detail | `Error parsing JSON string` (1534); `Error decoding JSON` (1728); `Error making API request` (1733); `Unexpected error` (1738) |
| `summarize_with_custom_openai.stream_generator` | response/output content | `Error decoding JSON from line` (1675) |
| `summarize_with_custom_openai_2` | raw/processed/extracted input | `Raw input data` (1771); `Processed data` (1798); `Extracted text` (1822) |
| `summarize_with_custom_openai_2` | prompt content | `Custom prompt` (1825) |
| `summarize_with_custom_openai_2` | credential fragment | `Using API Key` (1765) |
| `summarize_with_custom_openai_2` | private endpoint/path | `Using API URL` (1858) |
| `summarize_with_custom_openai_2` | response/output content | `full API response data` (1955); unlabelled `response_data` (1958); `Chat response` (1964); `Error response` (1977) |
| `summarize_with_custom_openai_2` | exception/error detail | `Error parsing JSON string` (1784); `Error decoding JSON` (1980); `Error making API request` (1985); `Unexpected error` (1990) |
| `summarize_with_custom_openai_2.stream_generator` | response/output content | `Error decoding JSON from line` (1925) |
| `save_summary_to_file` | private endpoint/path | `Summary saved to file` (2004) |

Category totals: 13 raw/processed/extracted input, 8 prompt content, 8 credential fragments, 6 private endpoint/path values, 29 response/output-content sites, and 36 exception/error-detail sites.

### `Summarization_General_Lib.py` — 100 sites

| Enclosing function | Category | Diagnostic labels (current lines) |
| --- | --- | --- |
| `log_debug_data` | raw/processed/extracted input | `Loaded data` (82) |
| `extract_text_from_segments` | raw/processed/extracted input | `Segments received` (91); `Skipping segment` (105) |
| `extract_text_from_input` | private endpoint/path | `Input is a file path` (207); `Error reading file` (220) |
| `recursive_summarize_chunks` | response/output content | `Error during recursive step` (175) |
| `recursive_summarize_chunks` | exception/error detail | `Unexpected error calling summarize_func` (191) |
| `_dispatch_to_api` | exception/error detail | `Error during dispatch to API` (498) |
| `analyze` | raw/processed/extracted input | `Extracted text content` (565) |
| `analyze` | response/output content | `Failed to summarize chunk` (721); `Summarization failed` (773); `Final Summary` (779) |
| `analyze.consume_generator` | exception/error detail | `Error consuming generator` (586) |
| `analyze` | exception/error detail | `Critical error in summarize function` (791) |
| `summarize_with_openai` | prompt content | `Custom prompt` (839); `System Message` (840) |
| `summarize_with_openai` | credential fragment | `Using API key from config` (819); `OpenAI API Key` (849) |
| `summarize_with_openai` | private endpoint/path | `Posting request to` (896) |
| `summarize_with_openai` | response/output content | `Summary not found in response` (958) |
| `summarize_with_openai` | exception/error detail | `API request failed` (964); `Unexpected error` (967) |
| `summarize_with_openai.stream_generator` | response/output content | `Error decoding JSON` (926); `Unexpected structure` (931) |
| `summarize_with_openai.stream_generator` | exception/error detail | `Error during streaming` (936) |
| `summarize_with_anthropic` | prompt content | `Prompt is` (1041) |
| `summarize_with_anthropic` | credential fragment | `Using API Key` (1001) |
| `summarize_with_anthropic` | private endpoint/path | `File not found` (1179); `Invalid JSON format in file` (1182) |
| `summarize_with_anthropic` | response/output content | `Summary` (1143); `Unexpected response format` (1149); `Failed to summarize` (1162); `Failed to process summary` (1165) |
| `summarize_with_anthropic` | exception/error detail | `Network error during attempt` (1171); `Error in processing` (1185) |
| `summarize_with_anthropic.stream_generator` | response/output content | `Error decoding JSON from line` (1122) |
| `summarize_with_cohere` | prompt content | `Prompt being sent` (1263) |
| `summarize_with_cohere` | credential fragment | `Using API Key` (1224) |
| `summarize_with_cohere` | response/output content | `request failed ... response.text` (1312, 1431); `API Response Data` (1414) |
| `summarize_with_cohere` | exception/error detail | `Error in processing` (1437) |
| `summarize_with_cohere._stream_events` | response/output content | `Skipping non-JSON stream line` (1348); `Error decoding JSON from line` (1357); `Skipping non-object stream event` (1364); `Unhandled streaming event type` (1380) |
| `summarize_with_groq` | raw/processed/extracted input | `Loaded data` (1477) |
| `summarize_with_groq` | prompt content | `Prompt being sent` (1512) |
| `summarize_with_groq` | credential fragment | `Using API Key` (1470) |
| `summarize_with_groq` | response/output content | `API Response Data` (1614); `request failed ... response.text` (1625) |
| `summarize_with_groq` | exception/error detail | `Error in processing` (1631) |
| `summarize_with_groq.stream_generator` | response/output content | `Error decoding JSON from line` (1578) |
| `summarize_with_openrouter` | credential fragment | `Using API Key` (1675) |
| `summarize_with_openrouter` | response/output content | unlabelled streamed `content` (1777); response-text `error_msg` (1786); `API Response Data` (1838); `request failed ... response.text` (1854) |
| `summarize_with_openrouter` | exception/error detail | `Error occurred while processing stream` (1791); `Error in processing` (1859) |
| `summarize_with_huggingface` | prompt content | `Prompt being sent` (1924) |
| `summarize_with_huggingface` | credential fragment | `Using API Key` (1891) |
| `summarize_with_huggingface` | response/output content | `Response JSON` (2014); `failed ... response.text` (2033) |
| `summarize_with_huggingface` | exception/error detail | `Error in processing` (2039) |
| `summarize_with_huggingface.stream_generator` | response/output content | `Unhandled streaming data` (1977); `Error decoding JSON from line` (1981) |
| `summarize_with_deepseek` | credential fragment | `Using API Key` (2071); `DeepSeek API Key` (2110) |
| `summarize_with_deepseek` | response/output content | `Error response` (2225) |
| `summarize_with_deepseek` | exception/error detail | `Error in processing` (2228) |
| `summarize_with_deepseek.stream_generator` | response/output content | `Error decoding JSON from line` (2174); `Key error ... in line` (2179) |
| `summarize_with_mistral` | credential fragment | `Using API Key` (2260); `Mistral API Key` (2299) |
| `summarize_with_mistral` | response/output content | `Error response` (2431) |
| `summarize_with_mistral` | exception/error detail | `Error in processing` (2434) |
| `summarize_with_mistral.stream_generator` | response/output content | `Unexpected data format` (2371); `Error decoding JSON from line` (2379); `Key error ... in line` (2384) |
| `summarize_with_google` | raw/processed/extracted input | `Raw input data` (2463); `Processed data` (2485); `Extracted text` (2503) |
| `summarize_with_google` | prompt content | `Custom prompt` (2504) |
| `summarize_with_google` | credential fragment | `Using API Key` (2459); `Google API Key` (2514) |
| `summarize_with_google` | response/output content | `Summary` (2628); `Error response` (2639) |
| `summarize_with_google` | exception/error detail | `Error parsing JSON string` (2476); `Error decoding JSON` (2642); `Error making API request` (2645); `Unexpected error` (2648) |
| `summarize_with_google.stream_generator` | response/output content | `Error decoding JSON from line` (2584); `Key error ... in line` (2589) |
| `summarize_with_mock_llm` | prompt content | `Custom prompt` (2672); `System Message` (2673) |
| `summarize_with_mock_llm` | exception/error detail | `Unexpected error` (2710) |
| `summarize_chunk` | response/output content | `Streaming error` (2739); `Summarization ... failed` (2749) |
| `summarize_chunk` | exception/error detail | `Error in summarize_chunk` (2760) |

Category totals: 8 raw/processed/extracted input, 9 prompt content, 13 credential fragments, 5 private endpoint/path values, 43 response/output-content sites, and 22 exception/error-detail sites.

## Final Review Audit Correction (2026-08-10)

The original implementation record classified 199 calls as private and 324 as reviewed-safe. Independent final review found that stable site `general-2efc909241862caf` in `summarize_with_cohere._stream_events` renders provider-controlled `event.get("type")`; it is response/output content, not bounded status metadata. The approved misclassification procedure therefore corrects the authoritative starting arithmetic to `200 private + 323 reviewed-safe = 523`, with General `100 private + 181 reviewed-safe`, `general_mid = 24`, and response/output content `72` overall. Earlier commits and verification transcripts that report `199/324` remain historical evidence of the audit state before this correction; they are not the final inventory.

Final verification also reproduced an unrelated 17-owner persistent-diagnostic
inventory drift on detached exact latest dev. That baseline incident has separate
backlog ownership and is not accepted into TASK-3796's manifest. This task names
the ownership generically because repository task hygiene prohibits a lower task
from forward-referencing a later, higher-numbered task.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every verified raw/processed/extracted input, prompt, response/output, credential-fragment, private endpoint/path, and exception/error-detail diagnostic in the inventory emits metadata only
- [x] #2 Sentinel tests capture representative real paths for every inventory category in both modules and prove distinctive private strings never reach diagnostics
- [x] #3 A fresh all-call sweep of both summarization modules finds no equivalent direct private diagnostic outside an explicit metadata-only helper
- [x] #4 The production diagnostic inventory is reconciled without changing unrelated owners, reasons, counts, or sink topology
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: ADR-029 already owns the metadata-only persistent-log boundary; this task applies a narrower call-site containment rule without changing sink admission or another system contract.

Detailed plan: [TASK-3796 implementation plan](../../Docs/superpowers/plans/2026-08-10-task-3796-summarization-diagnostic-privacy.md)

1. Rebase onto current `origin/dev`, recheck in-flight ownership, and reproduce the focused behavioral, inventory, lint, and formatter baselines.
2. Add a test-only stable-identity ledger covering all 523 starting calls: 200 private sites pending repair and 323 exact reviewed-safe calls.
3. Repair the 100 Local-module sites in four test-first provider batches, with direct-function canaries and per-batch reconciliation.
4. Repair the 100 General-module sites in four test-first provider batches, with direct-function canaries and per-batch reconciliation.
5. Run independent category/module mutations, prove restoration, and reconcile the exhaustive all-call boundary.
6. Regenerate only the two owned diagnostic-inventory entries, run touched-functionality verification, complete self-review, and close TASK-3796.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Reconciled all 523 starting calls: `200 private + 323 reviewed-safe` across
  Local `242 = 100 private + 142 safe` and General
  `281 = 100 private + 181 safe`. Final outcomes are `177 metadata + 23
  deleted + 323 frozen`; 500 calls remain in source and no site is pending or
  unclassified.
- The private category matrix is Local/General/total: input `13/8/21`, prompt
  `8/9/17`, credential `8/13/21`, endpoint/path `6/5/11`, response/output
  `29/43/72`, and exception/error detail `36/22/58`. The eight implementation
  batches reconcile as Local `24/23/22/31` and General `36/24/20/20`.
- Final review corrected Cohere site `general-2efc909241862caf` from the
  historical `199 private / 324 safe` audit to `200/323`: the real, fully
  consumed `summarize_with_cohere()` unknown-event sentinel proved that
  provider-controlled `event.get("type")` reached diagnostics. The corrected
  immutable starting-projection SHA-256 is
  `85a5c6b74f0cd4eb15f8ca0f8abfa5e18ca7f26f749d97fc7b781090cabd7733`.
- Direct real-function sentinels cover every category in both modules. Twelve
  independent runtime privacy mutations, two traceback-capture mutations, and
  three stable-guard mutations (unclassified call, changed frozen expression,
  and exception capture) each failed their owning assertion before inverse
  restoration. Final production blob hashes, including the PR review
  correction below, are Local `6f71f80cd94129f478844ac0bba6842c794c4f55`
  and General `b3595d2cd85b79d985b42a962afd807e60b500bd`.
- PR review added bounded `line_length` context to the Local LLM and Llama
  malformed-stream diagnostics and a sanitized provider token to the
  `analyze()` failure diagnostic. Direct real-function tests first failed on
  the absent context, then passed after the metadata-only corrections; no
  response content or provider-controlled value is logged.
- Only diagnostic arguments changed. Provider selection, request payloads and
  transport calls, response handling, retry/error paths, return strings,
  streaming laziness/chunks, and response-close behavior remain under direct
  tests and unchanged contracts.
- The governed manifest changes only Local `242 -> 229` (digest
  `6e78b604a2504fca4b07`) and General `281 -> 271` (digest
  `d37486940059e7af2679`), plus the derived TASK-492 total `1,167 -> 1,144`.
  Owners, reasons, unrelated checked entries, and six-file sink topology are
  unchanged. Exact latest dev independently carries an unrelated 17-owner
  baseline drift (`44` additions / `30` deletions; Git-patch SHA-256
  `b77bd95ccc84d3bac066e0971a8bc24e20fdb58bef9b762d5ba77aa6399db4dd`),
  which remains unblessed with separate To Do ownership. This lower-numbered
  task intentionally omits the later task ID under the repository's
  no-forward-reference rule.
- Fresh focused evidence: the privacy file passed `257` tests; the complete
  touched-functionality command passed `307` tests with only the approved
  current-dev baseline failure
  `test_production_diagnostic_inventory_and_sink_topology_are_unchanged`.
  Detached exact base `6d72f15f8332b6469a5d644d409b80914634a8dd` passed the
  other 13 architecture tests and failed that same sole node. Ruff lint,
  Local/helper/test formatting, Python compilation, and diff checks passed;
  General's full-file format check retained only the known line-317 alias hunk.
- ADR decision: no new ADR. ADR-029 already governs the metadata-only
  persistent-log boundary. The approved baseline-red deviation and the new
  provenance/projection incidents are documented in
  `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
