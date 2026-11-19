import ast
import json
from pathlib import Path

root = Path.cwd()
evidence = Path('/private/tmp/mixins-97a-archive-fix')
original_flow = ast.parse((evidence / '97a-original-flow.py').read_text())
current_flow = ast.parse((root / 'Tests/UI/test_console_archive_recovery_flow.py').read_text())
changed = next(node for node in current_flow.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'test_pending_source_remount_expiry_and_detach_fail_closed')
removed_checks = 0
for node in ast.walk(changed):
    if not isinstance(node, ast.Lambda) or not isinstance(node.body, ast.BoolOp):
        continue
    prefix = node.body.values[0]
    expected = ast.parse('bool(rig.console.query("#recovery-confirm"))', mode='eval').body
    if ast.dump(prefix) != ast.dump(expected):
        continue
    remaining = node.body.values[1:]
    node.body = remaining[0] if len(remaining) == 1 else ast.BoolOp(op=ast.And(), values=remaining)
    removed_checks += 1
assert removed_checks == 2
assert ast.dump(original_flow) == ast.dump(current_flow)
original_tests = (evidence / '97a-original-tests.py').read_text()
current_tests = (root / 'Tests/UI/test_console_archive_recovery_cards.py').read_text()
anchor = '@pytest.mark.asyncio\n@pytest.mark.parametrize("keyboard", [False, True])'
assert original_tests[original_tests.index(anchor):] == current_tests[current_tests.index(anchor):]
original_card = (evidence / '97a-original-card.py').read_text()
current_card = (root / 'tldw_chatbook/Widgets/Chat_Widgets/archive_recovery_cards.py').read_text()
anchor = '    @on(Select.Changed, "#recovery-scope")'
assert original_card[original_card.index(anchor):] == current_card[current_card.index(anchor):]
print(json.dumps({'flow_only_two_presence_prefixes': True, 'existing_card_tests_byte_identical': True, 'scope_and_typed_decision_handlers_byte_identical': True}, indent=2))
