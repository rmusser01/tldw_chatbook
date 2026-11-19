import json
import sys
from pathlib import Path

import pytest

from Tests.Chat.test_console_archive_recovery_followup import (
    captured_followup,
    native_followup,
    recovery_stores,
    test_captured_native_turn_restores_original_second_row,
)
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory


@pytest.fixture(autouse=True)
def observe_surface(monkeypatch):
    original = ConsoleTraceBoundaryFactory.__call__
    observations = []

    def boundary(self, prepared, resolution, route):
        def trace(frame, event, arg):
            if (event == 'exception' and str(arg[1]) == 'unsupported_surface_change'
                    and frame.f_code.co_filename.endswith('console_trace_service.py')):
                names = ('prefix', 'suffix', 'incoming_changed', 'active_changed', 'compound',
                         'domains', 'active', 'physical_active', 'descriptors', 'values',
                         'completed_tool_turn', 'durable_values')
                observations.append({
                    'function': frame.f_code.co_name,
                    'line': frame.f_lineno,
                    'locals': {name: repr(frame.f_locals.get(name)) for name in names},
                    'request': repr(prepared.semantic),
                    'rows': [dict(row) for row in prepared.semantic.flattened_messages()],
                    'groups': [repr(group) for unit in prepared.semantic.compactable
                               for group in unit.thinking_groups],
                    'resolution': repr(resolution),
                })
            return trace
        prior = sys.gettrace()
        sys.settrace(trace)
        try:
            return original(self, prepared, resolution, route)
        finally:
            sys.settrace(prior)
    monkeypatch.setattr(ConsoleTraceBoundaryFactory, '__call__', boundary)
    yield
    Path('/private/tmp/archive-capture-task3-surface.json').write_text(
        json.dumps(observations, indent=2, default=list) + '\n')
