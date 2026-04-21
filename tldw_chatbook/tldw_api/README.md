# tldw_api

Files in this directory are not licensed under AGPL but rather the Apache License 2.0.
Specifically:
- `tldw_api/client.py`
- `tldw_api/exceptions.py`
- `tldw_api/prompt_chatbook_schemas.py`
- `tldw_api/schemas.py`
- `tldw_api/utils.py`

This keeps the shared client and schema layer reusable in commercial applications without forcing the rest of `tldw_chatbook` under the same terms.

## Prompt And Chatbook Parity Surface

The prompt/chatbook parity vertical extends the shared client with the server-backed operations used by the local Textual UI.

### Prompt methods

- `list_prompts(include_deleted=False)`
- `preview_prompt(request_data)`
- `create_prompt(request_data)`
- `list_prompt_versions(prompt_identifier)`
- `restore_prompt_version(prompt_identifier, version)`

### Chatbook methods

- `export_chatbook(request_data)`
- `preview_chatbook(chatbook_file_path)`
- `import_chatbook(chatbook_file_path, request_data)`
- `get_chatbook_export_job(job_id)`
- `get_chatbook_import_job(job_id)`

### Request models

`prompt_chatbook_schemas.py` contains the request contracts used by these endpoints:

- `PromptPreviewRequest`
- `PromptCreateRequest`
- `ChatbookExportRequest`
- `ChatbookImportRequest`

These contracts are intentionally thin mirrors of the server request shapes so `tldw_chatbook` can round-trip prompt and chatbook operations without introducing a second client abstraction layer.
