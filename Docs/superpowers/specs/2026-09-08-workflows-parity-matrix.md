# Workflow parity matrix

- Status: Proposed implementation scope; source inventory, not verified runtime parity.
- Date: 2026-09-08
- Task: [TASK-32077](../../../backlog/tasks/task-32077%20-%20Design-portable-Workflows-editor-and-local-execution-parity.md)
- Design: [Local execution and portable authoring](2026-09-08-workflows-local-first-parity-design.md)
- Server: dev at `6cd2745f696af04668a61c20b84ab8a9e69ca5e4`.
- Source: [server registry](https://github.com/rmusser01/tldw_server/blob/6cd2745f696af04668a61c20b84ab8a9e69ca5e4/tldw_Server_API/app/core/Workflows/registry.py).

## How to read this matrix

All 130 registered server step names appear exactly once below. No Chatbook workflow adapter was implemented or executed during this design task. An existing module is a reuse candidate, not proof of matching input/output, cancellation, permission, or failure semantics.

- **V1 minimum (21):** retained user-approved target, including dependency-aware optional capabilities; detailed adapter subsets are still proposed. A machine missing a model/package/tool reports Setup required; it does not silently switch to a server.
- **Sequential parity candidate (99):** compatible with the v1 orchestration model, but local support must be assessed and implemented per adapter. These can join v1; this label is not a release promise or a claim that a local equivalent exists.
- **Separate lifecycle design (7):** nesting, scheduling, process execution, or retained state needs an explicit ownership/limits contract before admission. These are not automatically assigned to the branching release.
- **V2 (1):** branching.
- **V3 (2):** map/fan-out and parallel orchestration. A map with concurrency 1 is still outside the initial editor/runtime profile.

Domain actions may internally use their established processing mechanisms; the v1 restriction is one authored workflow step at a time. Composite actions that would introduce unbounded children or hidden workflow orchestration need explicit admission review.

## Admission evidence and delivery sequence

The first local end-to-end milestone uses operation subsets of `media_ingest`, `prompt`, `llm`, `wait_for_human`, and `notes`: a local text file becomes a reviewed summary saved as a note. The inspected server file-ingestion path exposes extracted `text`; its error and extraction-disabled cases must also be covered, rather than inventing a new result shape. See the [pinned file-ingestion path](https://github.com/rmusser01/tldw_server/blob/6cd2745f696af04668a61c20b84ab8a9e69ca5e4/tldw_Server_API/app/core/Workflows/adapters/media/ingest.py#L96). This five-type slice is not a replacement for the 21-type v1 target and does not claim complete operation coverage of any type. Reviewed file/server exchange and paired-server workflow sync remain explicit v1 milestones in the design.

| Admission gate | Required evidence before a type/config is labeled locally runnable |
| --- | --- |
| Config and results | Exact field placement, defaults, operation subset, typed references, documented outputs/errors, and preserved unsupported fields. Discovery schemas are inventory, not a generic form/runtime implementation contract. |
| Binding and credentials | Explicit destination materialization, resource-ID authority, supported secret-reference path or destination-managed credentials, and fixed non-secret selections per run. A runtime secrets API alone does not prove adapter support. |
| Effects and permissions | Whole-definition effect inventory, final resolved target checks, normal tool/path permission, and local refusal of completion hooks. Imported fields never supply grants. |
| Attempt lifetime | Distinct execution/request/human deadlines, replay/idempotency evidence, no overlap with live timed-out workers, durable wait/decision identity, and restart credential recovery. |
| Bounds and privacy | Persistent run/attempt/token/output limits, correct artifact ownership, and explicit safety/privacy deltas. Payload-bearing log messages stay in private run data, not ordinary application diagnostics. |

No new adapter was implemented or verified by this revision. The two-destination JSON example in the design is a source-shaped contract fixture, not a passing runtime test. These additional gates leave all 130 catalog dispositions and the v2/v3 control-flow assignments unchanged.

## Inspected local building blocks

| Family | Existing source | What still needs work |
| --- | --- | --- |
| Templates | [prompt_template_manager.py](../../../tldw_chatbook/Chat/prompt_template_manager.py) | Match server typed-expression behavior, escaping, missing values, and error handling; do not assume the current string renderer is equivalent. |
| LLM calls | [Chat_Functions.py](../../../tldw_chatbook/Chat/Chat_Functions.py) (`chat_api_call`) | Map canonical config, provider/model resolution, streaming consumption, result fields, usage, and errors. Generated-content steps need their own contract tests. |
| Local RAG | [rag_service.py](../../../tldw_chatbook/RAG_Search/simplified/rag_service.py), [enhanced_rag_service.py](../../../tldw_chatbook/RAG_Search/simplified/enhanced_rag_service.py) | Validate per-field support. The existing agent-facing RAG tool has different argument names/ranges from the server workflow schema. |
| Notes/prompts | [ChaChaNotes_DB.py](../../../tldw_chatbook/DB/ChaChaNotes_DB.py), [local_prompt_service.py](../../../tldw_chatbook/Prompt_Management/local_prompt_service.py), [note_management_tools.py](../../../tldw_chatbook/Tools/note_management_tools.py) | Resolve the active profile/resource authority, map operations/results, and preserve ordinary write permission and version checks. |
| Ingestion and PDF | [local_file_ingestion.py](../../../tldw_chatbook/Local_Ingestion/local_file_ingestion.py), [ingest_capabilities.py](../../../tldw_chatbook/Library/ingest_capabilities.py), [PDF_Processing_Lib.py](../../../tldw_chatbook/Local_Ingestion/PDF_Processing_Lib.py) | Distinguish parse-only work from media persistence, apply path admission, honor optional dependencies, and map artifact/resource outputs. |
| MCP/tools | [mcp_tool_provider.py](../../../tldw_chatbook/Agents/mcp_tool_provider.py), [tool_catalog.py](../../../tldw_chatbook/Agents/tool_catalog.py) | Reuse permission-gated invocation and destination tool lookup. Registry lookup alone must not bypass normal review. |
| TTS | [TTS_Generation.py](../../../tldw_chatbook/TTS/TTS_Generation.py), [request_admission.py](../../../tldw_chatbook/TTS/request_admission.py) | Enter through admitted synthesis, consume streams, preserve model/voice compatibility, materialize output artifacts, and support cancellation. |
| STT | [coordinator.py](../../../tldw_chatbook/STT/coordinator.py), [contracts.py](../../../tldw_chatbook/STT/contracts.py) | Translate canonical request/output shapes through the existing coordinator and report actual installed backend support. |
| Image generation | [worker.py](../../../tldw_chatbook/Image_Generation/worker.py) | Reuse the validation entry point and artifact handling; a matching feature name does not establish server adapter parity. |
| Research | [local_research_service.py](../../../tldw_chatbook/Research_Interop/local_research_service.py), [local_research_engine.py](../../../tldw_chatbook/Research_Interop/local_research_engine.py) | Match launch/wait/bundle semantics and retained-run ownership. Keep launch and wait distinct. |
| Web search | [web_search_tool.py](../../../tldw_chatbook/Tools/web_search_tool.py), [WebSearch_APIs.py](../../../tldw_chatbook/Web_Scraping/WebSearch_APIs.py) | Map provider options, response shapes, deadlines, and network requirements. Local scheduling does not make web search offline. |
| Text/data utilities | Standard library where sufficient; template module above | Implement small canonical adapters with server fixtures; avoid introducing a dependency for a standard-library operation. |
| Run controls and approvals | New workflow run service/engine over existing private-storage and permission conventions | Durable waiting, local recipient binding, pause/cancel/retry, restart recovery, and uncertain-effect handling are new workflow functionality. |

Only these building blocks were traced in this design pass. Other specialized catalog entries below are inventory, not individual feasibility assessments.

## Complete server catalog disposition

| Server step type | Family | Proposed disposition |
| --- | --- | --- |
| `media_ingest` | Media and documents | V1 minimum |
| `prompt` | Templates and text/data | V1 minimum |
| `llm` | LLM and generated content | V1 minimum |
| `rag_search` | Search and RAG | V1 minimum |
| `kanban` | Knowledge and domain records | Sequential parity candidate |
| `mcp_tool` | Tools and external integrations | V1 minimum |
| `acp_stage` | Tools and external integrations | Separate lifecycle design |
| `tts` | Audio/video | V1 minimum |
| `webhook` | Tools and external integrations | Sequential parity candidate |
| `delay` | Runtime and orchestration | V1 minimum |
| `log` | Runtime and orchestration | V1 minimum |
| `wait_for_human` | Runtime and orchestration | V1 minimum |
| `wait_for_approval` | Runtime and orchestration | V1 minimum |
| `branch` | Runtime and orchestration | V2 |
| `map` | Runtime and orchestration | V3 |
| `process_media` | Media and documents | Sequential parity candidate |
| `policy_check` | LLM and generated content | Sequential parity candidate |
| `rss_fetch` | Tools and external integrations | Sequential parity candidate |
| `atom_fetch` | Tools and external integrations | Sequential parity candidate |
| `embed` | Search and RAG | Sequential parity candidate |
| `translate` | LLM and generated content | Sequential parity candidate |
| `stt_transcribe` | Audio/video | V1 minimum |
| `notify` | Tools and external integrations | Sequential parity candidate |
| `diff_change_detector` | Templates and text/data | Sequential parity candidate |
| `notes` | Knowledge and domain records | V1 minimum |
| `prompts` | Knowledge and domain records | V1 minimum |
| `chunking` | Search and RAG | V1 minimum |
| `web_search` | Search and RAG | Sequential parity candidate |
| `collections` | Knowledge and domain records | Sequential parity candidate |
| `chatbooks` | Knowledge and domain records | Sequential parity candidate |
| `evaluations` | Knowledge and domain records | Sequential parity candidate |
| `claims_extract` | Knowledge and domain records | Sequential parity candidate |
| `character_chat` | LLM and generated content | Sequential parity candidate |
| `moderation` | LLM and generated content | Sequential parity candidate |
| `sandbox_exec` | Tools and external integrations | Separate lifecycle design |
| `image_gen` | LLM and generated content | Sequential parity candidate |
| `summarize` | LLM and generated content | Sequential parity candidate |
| `query_expand` | Search and RAG | Sequential parity candidate |
| `rerank` | Search and RAG | Sequential parity candidate |
| `citations` | Search and RAG | Sequential parity candidate |
| `ocr` | Media and documents | Sequential parity candidate |
| `pdf_extract` | Media and documents | V1 minimum |
| `voice_intent` | Audio/video | Sequential parity candidate |
| `query_rewrite` | Search and RAG | Sequential parity candidate |
| `hyde_generate` | Search and RAG | Sequential parity candidate |
| `semantic_cache_check` | Search and RAG | Sequential parity candidate |
| `search_aggregate` | Search and RAG | Sequential parity candidate |
| `entity_extract` | Search and RAG | Sequential parity candidate |
| `bibliography_generate` | Search and RAG | Sequential parity candidate |
| `document_table_extract` | Media and documents | Sequential parity candidate |
| `audio_diarize` | Audio/video | Sequential parity candidate |
| `flashcard_generate` | LLM and generated content | Sequential parity candidate |
| `quiz_generate` | LLM and generated content | Sequential parity candidate |
| `quiz_evaluate` | LLM and generated content | Sequential parity candidate |
| `outline_generate` | LLM and generated content | Sequential parity candidate |
| `glossary_extract` | LLM and generated content | Sequential parity candidate |
| `mindmap_generate` | LLM and generated content | Sequential parity candidate |
| `eval_readability` | LLM and generated content | Sequential parity candidate |
| `json_transform` | Templates and text/data | Sequential parity candidate |
| `json_validate` | Templates and text/data | V1 minimum |
| `csv_to_json` | Templates and text/data | V1 minimum |
| `json_to_csv` | Templates and text/data | V1 minimum |
| `regex_extract` | Templates and text/data | V1 minimum |
| `text_clean` | Templates and text/data | V1 minimum |
| `xml_transform` | Templates and text/data | Sequential parity candidate |
| `template_render` | Templates and text/data | V1 minimum |
| `batch` | Templates and text/data | Sequential parity candidate |
| `workflow_call` | Runtime and orchestration | Separate lifecycle design |
| `parallel` | Runtime and orchestration | V3 |
| `cache_result` | Runtime and orchestration | Separate lifecycle design |
| `retry` | Runtime and orchestration | Separate lifecycle design |
| `checkpoint` | Runtime and orchestration | Separate lifecycle design |
| `s3_upload` | Tools and external integrations | Sequential parity candidate |
| `s3_download` | Tools and external integrations | Sequential parity candidate |
| `github_create_issue` | Tools and external integrations | Sequential parity candidate |
| `podcast_rss_publish` | Tools and external integrations | Sequential parity candidate |
| `llm_with_tools` | LLM and generated content | Sequential parity candidate |
| `llm_critique` | LLM and generated content | Sequential parity candidate |
| `context_build` | LLM and generated content | Sequential parity candidate |
| `document_merge` | Templates and text/data | Sequential parity candidate |
| `document_diff` | Templates and text/data | Sequential parity candidate |
| `markdown_to_html` | Templates and text/data | Sequential parity candidate |
| `html_to_markdown` | Templates and text/data | Sequential parity candidate |
| `keyword_extract` | Templates and text/data | Sequential parity candidate |
| `sentiment_analyze` | Templates and text/data | Sequential parity candidate |
| `language_detect` | Templates and text/data | Sequential parity candidate |
| `topic_model` | Templates and text/data | Sequential parity candidate |
| `token_count` | Templates and text/data | Sequential parity candidate |
| `context_window_check` | Templates and text/data | Sequential parity candidate |
| `llm_compare` | LLM and generated content | Sequential parity candidate |
| `image_describe` | LLM and generated content | Sequential parity candidate |
| `report_generate` | LLM and generated content | Sequential parity candidate |
| `newsletter_generate` | LLM and generated content | Sequential parity candidate |
| `audio_briefing_compose` | LLM and generated content | Sequential parity candidate |
| `slides_generate` | LLM and generated content | Sequential parity candidate |
| `diagram_generate` | LLM and generated content | Sequential parity candidate |
| `email_send` | Tools and external integrations | Sequential parity candidate |
| `screenshot_capture` | Tools and external integrations | Sequential parity candidate |
| `schedule_workflow` | Runtime and orchestration | Separate lifecycle design |
| `timing_start` | Runtime and orchestration | Sequential parity candidate |
| `timing_stop` | Runtime and orchestration | Sequential parity candidate |
| `multi_voice_tts` | Audio/video | Sequential parity candidate |
| `audio_normalize` | Audio/video | Sequential parity candidate |
| `audio_concat` | Audio/video | Sequential parity candidate |
| `audio_trim` | Audio/video | Sequential parity candidate |
| `audio_convert` | Audio/video | Sequential parity candidate |
| `audio_extract` | Audio/video | Sequential parity candidate |
| `audio_mix` | Audio/video | Sequential parity candidate |
| `video_thumbnail` | Audio/video | Sequential parity candidate |
| `video_trim` | Audio/video | Sequential parity candidate |
| `video_concat` | Audio/video | Sequential parity candidate |
| `video_convert` | Audio/video | Sequential parity candidate |
| `video_extract_frames` | Audio/video | Sequential parity candidate |
| `subtitle_generate` | Audio/video | Sequential parity candidate |
| `subtitle_translate` | Audio/video | Sequential parity candidate |
| `subtitle_burn` | Audio/video | Sequential parity candidate |
| `arxiv_search` | Research | Sequential parity candidate |
| `arxiv_download` | Research | Sequential parity candidate |
| `pubmed_search` | Research | Sequential parity candidate |
| `semantic_scholar_search` | Research | Sequential parity candidate |
| `google_scholar_search` | Research | Sequential parity candidate |
| `patent_search` | Research | Sequential parity candidate |
| `doi_resolve` | Research | Sequential parity candidate |
| `reference_parse` | Research | Sequential parity candidate |
| `bibtex_generate` | Research | Sequential parity candidate |
| `literature_review` | Research | Sequential parity candidate |
| `deep_research` | Research | Sequential parity candidate |
| `deep_research_wait` | Research | Sequential parity candidate |
| `deep_research_load_bundle` | Research | Sequential parity candidate |
| `deep_research_select_bundle_fields` | Research | Sequential parity candidate |

## Per-adapter release evidence

A local adapter is supported only when its accepted config, result shape, errors, resource bindings, permissions, timeout/retry behavior, and side effects have targeted conformance evidence. Unsupported options are named at preflight and never dropped. Execution tests must enter through the workflow adapter and real local service boundary, not a fake shaped to match the proposed implementation.

At least one complete offline workflow must run with tldw_server unavailable and cloud/internet routes disallowed, using installed local dependencies. Real server exchange and negotiated sync require separate end-to-end evidence; passing local fixtures cannot establish either.
