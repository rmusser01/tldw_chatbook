---
id: TASK-40546
title: 'Windows user-configured .bat or .cmd binaries parse validated path arguments under cmd.exe'
status: To Do
assignee: []
created_date: '2026-11-12 09:00'
labels:
  - follow-up
  - security
dependencies: []
references:
  - tldw_chatbook/Utils/path_validation.py
  - tldw_chatbook/Model_Artifacts/gguf_admission.py
  - tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py
  - tldw_chatbook/TTS/audio_cpp_guided_launch.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
If a user configures a `.bat` or `.cmd` file as an executable, Windows CreateProcess runs it under cmd.exe, which parses the argument list again, and `subprocess.list2cmdline` does not escape `&`. Path arguments that were validated for the process (not for a shell) can then be read as shell syntax. The audit names the llama.cpp and llamafile server (`server_lifecycle.py`, fed by `gguf_admission.py`), sd.cpp, audio.cpp, transcribe.cpp, and ACP and MCP commands as user-configurable executables; only the llama.cpp path was traced to its call site (`gguf_admission.py`'s server `command` list). It exists on dev today (`4b90eb860b`) and depends on user configuration; it is not confirmed on Windows. No package code launches a batch file itself. Open PR #3472 (TASK-38227, which lets `validate_path_simple` accept legal file names with shell characters by default) widens this once merged, because `&` and similar are then valid in validated paths. Found by the TASK-38227 audit for PR #3472.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A `.bat` or `.cmd` executable configured for any of the listed launchers is either refused with a message that names the reason, or launched so that its path arguments are not parsed as cmd.exe syntax
- [ ] #2 A test on the argv or command-line construction shows a path with `&` stays one argument for such a binary, red on dev
- [ ] #3 The chosen behaviour is documented in the matching User Guide page
<!-- AC:END -->
