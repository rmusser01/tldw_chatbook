from pathlib import Path
import ast, hashlib, json
root=Path(__file__).resolve().parent
changes=[
('Tests/Chat/test_console_chat_create_integration.py', '    assert not controller._chat_creation_records and not controller.pending_chat_create_ids()\n', '    assert (\n        not controller._chat_creation_records\n        and not controller.pending_chat_create_ids()\n    )\n'),
('tldw_chatbook/Chat/console_agent_bridge.py', '            if tool == "new_chat" and not child_request and decision.get("remember", False):\n', '            if (\n                tool == "new_chat"\n                and not child_request\n                and decision.get("remember", False)\n            ):\n'),
('tldw_chatbook/Chat/console_chat_controller.py', '                and continuation.origin is not ConsoleSubmissionOrigin.AGENT_CHAT_START\n                and not (continuation.prepared and continuation.prepared.preserve_composer)\n', '                and continuation.origin is not ConsoleSubmissionOrigin.AGENT_CHAT_START\n                and not (\n                    continuation.prepared and continuation.prepared.preserve_composer\n                )\n'),
('tldw_chatbook/UI/Console_Modules/session.py', '        self._visible_agent_handoff_draft: tuple[str, str, int, ComposerDraftSnapshot] | None = None\n', '        self._visible_agent_handoff_draft: (\n            tuple[str, str, int, ComposerDraftSnapshot] | None\n        ) = None\n'),
]
records=[]
for path,old,new in changes:
 p=Path(path);before=p.read_text();assert before.count(old)==1,(path,before.count(old))
 after=before.replace(old,new);assert ast.dump(ast.parse(before))==ast.dump(ast.parse(after)),path
 p.write_text(after)
 records.append({'path':path,'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest(),'ast_equal':True,'old':old,'new':new})
(root/'finalfix-format-correction-detail.json').write_text(json.dumps(records,indent=2))
print('Four exact formatter statement corrections; Python AST equality verified for every path.')
