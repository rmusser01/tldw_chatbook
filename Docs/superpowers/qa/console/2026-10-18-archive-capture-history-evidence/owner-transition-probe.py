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

from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentTraceRequestFactory
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway


@pytest.fixture(autouse=True)
def observe_owner_transitions(monkeypatch):
    observations = []
    original_build = ConsoleAgentTraceRequestFactory.build
    original_prepare = ConsoleProviderGateway.prepare_chat_request

    def rows(value):
        return [{'role': row.get('role'), 'keys': sorted(row),
                 'owners': {key: item for key, item in row.items()
                            if key in {'_tldw_thinking_owner', '_tldw_call_thinking_owner',
                                       '_native_message_id', '_tldw_continuation_owner'}}}
                for row in value]

    def groups(request):
        return {'compactable': [
            {'owner': group.owner_message_id, 'blocks': len(group.blocks)}
            for unit in request.compactable for group in unit.thinking_groups
        ], 'active': [
            {'owner': group.owner_message_id, 'blocks': len(group.blocks)}
            for group in request.active_thinking_groups
        ]}

    def build(self, messages, **kwargs):
        result = original_build(self, messages, **kwargs)
        observations.append({
            'phase': 'factory', 'route': repr(kwargs.get('route')),
            'admitted': rows(self.admitted_request.flattened_messages()),
            'admitted_descriptors': [repr(v) for v in self.admitted_request.provenance.flattened_messages()],
            'admitted_groups': groups(self.admitted_request),
            'incoming': rows(messages),
            'result': rows(result.flattened_messages()),
            'result_descriptors': [repr(v) for v in result.provenance.flattened_messages()],
            'result_groups': groups(result),
        })
        return result

    def prepare(self, resolution, messages, **kwargs):
        result = original_prepare(self, resolution, messages, **kwargs)
        if hasattr(messages, 'flattened_messages'):
            observations.append({
                'phase': 'gateway', 'route': repr(kwargs.get('route')),
                'incoming': rows(messages.flattened_messages()),
                'incoming_groups': groups(messages),
                'sidecars': [{'owner': value.owner_message_id,
                              'blocks': [(b.source_format, b.status) for b in value.envelope.blocks]}
                             for value in kwargs.get('thinking_sidecar', ())],
                'owner_key': kwargs.get('thinking_owner_key'),
                'semantic': rows(result.semantic.flattened_messages()),
                'semantic_groups': groups(result.semantic),
                'provider_groups': [{'owner': group.owner_message_id, 'blocks': len(group.blocks)}
                                    for group in result.thinking_groups],
                'provider_rows': rows(result.messages_payload),
                'provider_descriptors': [repr(v) for v in result.provenance.messages_payload]
                                       if result.provenance else [],
            })
        return result

    monkeypatch.setattr(ConsoleAgentTraceRequestFactory, 'build', build)
    monkeypatch.setattr(ConsoleProviderGateway, 'prepare_chat_request', prepare)
    yield
    Path('/private/tmp/archive-capture-task3-owner-transitions.json').write_text(
        json.dumps(observations, indent=2) + '\n')
