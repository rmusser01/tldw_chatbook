---
id: TASK-886
title: Fix the Ollama Start Service handler so it can spawn a process
status: To Do
assignee: []
created_date: '2026-07-27 13:31'
labels:
  - bug
  - llm-management
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
handle_ollama_start_service_button_pressed in Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py cannot work at all. It calls stream_worker_output_to_log one positional argument short of its signature, and separately its run_worker(...) call passes the command positionally into run_worker's own 'name' parameter and then also supplies name= as a keyword. The second fault reproduces directly as 'TypeError: got multiple values for argument name'. Pressing Start Ollama Service therefore never spawns a process and never assigns app.ollama_server_process, which is why Ollama had to be dropped from the Lab Models status list (PR #966) rather than reported as stopped forever. Found while verifying the Lab frame; pre-existing and unrelated to that branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pressing Start Ollama Service spawns the process without raising,app.ollama_server_process is assigned the running handle as its five sibling handlers do,stream_worker_output_to_log is called with its full required signature,Ollama is restored to LAB_SERVER_SOURCES in lab_server_status.py and reports running/stopped correctly,A test covers the handler dispatching without a TypeError
<!-- AC:END -->
