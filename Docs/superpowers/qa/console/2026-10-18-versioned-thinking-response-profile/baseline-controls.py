import ast
import importlib
import pathlib
import subprocess
import sys

repo = pathlib.Path('/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook')
sys.path.insert(0, str(repo))
import Tests.conftest
base = '38cf68612508d23e9ae70a5deecf76a913fff3e0'
for name, function in [
    ('tldw_chatbook.Chat.console_semantic_revision', '_project_revision_response_envelope'),
    ('tldw_chatbook.Chat.console_trace_service', '_response_projection_profile'),
]:
    module = importlib.import_module(name)
    path = name.replace('.', '/') + '.py'
    source = subprocess.check_output(['git', 'show', base + ':' + path], cwd=repo)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == function)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(repo / path), 'exec'), module.__dict__)
module = importlib.import_module('Tests.Chat.test_console_trace_settlement')
path = 'Tests/Chat/test_console_trace_settlement.py'
source = subprocess.check_output(['git', 'show', base + ':' + path], cwd=repo)
exec(compile(source, str(repo / path), 'exec'), module.__dict__)
import pytest
raise SystemExit(pytest.main([
    path + '::test_oversized_response_is_replaced_by_one_bounded_labeled_artifact',
    path + '::test_queued_settlement_drops_oversized_response_and_usage_values',
    path + '::test_cold_restart_recovers_open_calls_monotonically_and_idempotently',
    '-q', '-p', 'no:cacheprovider', '--basetemp=/private/tmp/task4-baseline-controls',
]))
