---
id: TASK-886
title: Fix the Ollama Start Service handler so it can spawn a process
status: Done
assignee: []
created_date: '2026-07-27 13:31'
updated_date: '2026-07-27 20:33'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved differently than filed. Dev's server_lifecycle refactor fixed the handler itself: SERVER_PROCESS_ATTRS includes ollama, handle_ollama_start_service_button_pressed reserves a claim, and run_server_subprocess publishes the handle via call_from_thread(publish_server_process, ...) like every sibling. Neither TypeError this task describes still exists.

What remained was the consequence: LAB_SERVER_SOURCES still excluded Ollama on the old reasoning, so the Models screen under-reported -- a user with Ollama running saw a count that ignored it. Ollama is restored, and a drift guard now asserts LAB_SERVER_SOURCES and SERVER_PROCESS_ATTRS agree in both directions so the next provider added to the lifecycle cannot go unreported the same way. Mutation-checked.
<!-- SECTION:NOTES:END -->
