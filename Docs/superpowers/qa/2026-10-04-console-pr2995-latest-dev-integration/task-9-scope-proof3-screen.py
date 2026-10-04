from __future__ import annotations


class ChatScreen:
    def __init__(self):
        build_console_controllers(
            self,
            rag_source_types_accessor=(lambda: _console_library_rag_source_scope(self)),
            rag_top_k_accessor=lambda: _console_library_rag_profile_top_k(),
            read_trace_recovery_dispatch=lambda: dispatch_trace_call_recovery_action,
            read_trace_recovery_state=lambda: trace_call_recovery_state,
        )
