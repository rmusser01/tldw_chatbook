---
id: task-2150
title: RAG Answer live smoke with a real provider
status: To Do
assignee: []
labels:
  - library
  - rag
  - verification
dependencies: []
priority: high
---

## Description

PR-3 (feat/rag-v2-honest-answering) shipped grounded RAG answering under the owner ruling "answer honestly, accuracy over assumption". Every code path is test-pinned with injected fakes, and the failure/gate paths were live-verified — but the ruling's CORE case (a real model abstaining on an unsupported query rather than confabulating, and citing correctly on a supported one) is a model-behavior property no fake can pin. It has never run against a real provider: the verification environment permits no API keys and no local provider was running.

Do not present RAG Answer as live-verified until this smoke runs.

## Acceptance Criteria

- [ ] With the user's real provider config, a query the library genuinely lacks produces the abstention sentence ("Nothing in your library supports an answer to that."), not a confabulated answer
- [ ] A well-supported query produces an answer whose [S...] citations correspond to the visible evidence rows
- [ ] The citation-caution callout appears when the model answers without citing
- [ ] The "Generating answer..." in-flight line and Use-in-Console-stays-enabled behavior are observed with real provider latency (sub-second fakes could not exercise them)
- [ ] User Guide "Verified against" stamp updated for this page after the pass
