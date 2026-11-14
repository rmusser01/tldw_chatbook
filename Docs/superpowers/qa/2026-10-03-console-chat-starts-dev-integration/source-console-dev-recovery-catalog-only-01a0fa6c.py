from pathlib import Path
import json, hashlib, difflib, sqlite3, re
root=Path('.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration');capture=json.loads((root/'catalogs.json').read_text());delta=json.loads((root/'schema-object-deltas.json').read_text())
files={name:Path(name).read_text() for name in ('tldw_chatbook/DB/recovery_operations.py','tldw_chatbook/Backup_Recovery/sqlite_validation.py','Tests/Backup_Recovery/test_agent_runs_recovery_schema.py')};after=dict(files)
# Schema-data additions only: shared literal object deltas from four real installed constructors.
p='tldw_chatbook/DB/recovery_operations.py';text=after[p]
buffer='';statements=[]
for line in Path('tldw_chatbook/DB/migrations/agent_runs_v21_to_v22_chat_starts.sql').read_text().splitlines(True):
 if line.lstrip().startswith('--'):continue
 buffer+=line
 if sqlite3.complete_statement(buffer):
  statement=buffer.strip().rstrip(';');buffer=''
  if statement not in ('PRAGMA foreign_keys=ON','BEGIN IMMEDIATE','COMMIT'):statements.append(statement)
assert not buffer.strip()
append='\n\n# ADR-211: exact constructor-captured v22 object deltas; all v18/v21 variants remain.\n_AGENT_RUNS_V22_REMOVED = '+repr(tuple(delta['agent']['removed']))+'\n_AGENT_RUNS_V22_ADDED = '+repr(tuple(delta['agent']['added']))+'\n'
append+='''def _agent_runs_v22_catalog(schema):
    import re

    def catalog_key(sql):
        match = re.match(r'CREATE (?:UNIQUE |VIRTUAL )?(INDEX|TABLE|TRIGGER) (?:IF NOT EXISTS )?["`]?([^"` (]+)', sql)
        assert match is not None
        return match[1].lower(), match[2]

    unchanged = tuple(sql for sql in schema if sql not in _AGENT_RUNS_V22_REMOVED)
    return tuple(sorted(unchanged + _AGENT_RUNS_V22_ADDED, key=catalog_key))

_AGENT_RUNS_SCHEMA += tuple((22, _agent_runs_v22_catalog(schema)) for version, schema in _AGENT_RUNS_SCHEMA if version == 21)
'''
# Route stays unqualified and is omitted from this catalog-only candidate.
append+='_SUBSCRIPTIONS_V76_REPLACEMENTS = '+repr(dict(zip(delta['subscriptions']['removed'],delta['subscriptions']['added'])))+'\n_SUBSCRIPTIONS_V76_SCHEMA = tuple(_SUBSCRIPTIONS_V76_REPLACEMENTS.get(sql, sql) for sql in _SUBSCRIPTIONS_SCHEMA[1][1])\n_SUBSCRIPTIONS_SCHEMA += ((2, _SUBSCRIPTIONS_V76_SCHEMA),)\n'
# Verify the proposed pure catalog function before producing any diff.
ns={};exec(append.split('_AGENT_RUNS_SCHEMA +=')[0],ns)
import ast
parsed=ast.parse(text)
initial=ast.literal_eval(next(n.value for n in parsed.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_AGENT_RUNS_SCHEMA' for t in n.targets)))
# Capture proof already established identical deltas over all v21 predecessor variants.
for actual in capture['agent_migrated']:
 assert all(s in actual for s in delta['agent']['added'])
fresh = capture['agent_fresh']
closest = min(capture['agent_migrated'], key=lambda rows: len(set(rows) ^ set(fresh)))
removed = [sql for sql in closest if sql not in fresh]
added = [sql for sql in fresh if sql not in closest]
assert len(removed) == len(added) == 1
assert removed[0].startswith('CREATE TABLE automatic_work_chains ')
assert added[0].startswith('CREATE TABLE automatic_work_chains ')
append += '_AGENT_RUNS_V22_FRESH_CHAIN = ' + repr(added[0]) + '\n'
append += '_AGENT_RUNS_SCHEMA += ((22, tuple(_AGENT_RUNS_V22_FRESH_CHAIN if sql.startswith("CREATE TABLE automatic_work_chains ") else sql for sql in next(schema for version, schema in _AGENT_RUNS_SCHEMA if version == 22))),)\n'

text+=append
Path('/private/tmp/console-dev-recovery-data-01a0fa6c.patch').write_text(''.join(difflib.unified_diff(files[p].splitlines(True),text.splitlines(True),fromfile='a/'+p,tofile='b/'+p)))
text=text.replace('            (18, 21),','            (18, 21, 22),')
text=text.replace('            if stamp != (75,):','            actual = tuple(row[0] for row in connection.execute(\n                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"\n            ))\n            expected_stamp = 76 if actual == _SUBSCRIPTIONS_V76_SCHEMA else 75\n            if stamp != (expected_stamp,):')
after[p]=text
p='tldw_chatbook/Backup_Recovery/sqlite_validation.py';text=after[p]
old='''        if (
            owner.owner_id == "db.subscriptions"
            and any(row[1] == "db_schema_version" for row in actual)
            and connection.execute(
                "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
            ).fetchone()
            != (75,)
        ):
            return ("unsupported_schema_version",), None
'''
new='''        if owner.owner_id == "db.subscriptions" and any(
            row[1] == "db_schema_version" for row in actual
        ):
            from tldw_chatbook.DB.recovery_operations import _SUBSCRIPTIONS_V76_SCHEMA

            expected_stamp = 76 if tuple(row[3] for row in actual if row[3] is not None) == _SUBSCRIPTIONS_V76_SCHEMA else 75
            if connection.execute(
                "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
            ).fetchone() != (expected_stamp,):
                return ("unsupported_schema_version",), None
'''
assert text.count(old)==1;text=text.replace(old,new);after[p]=text
# Existing migration tests remain unchanged until route qualification.
Path('/private/tmp/console-dev-recovery-operations-proposal-01a0fa6c.py').write_text(after['tldw_chatbook/DB/recovery_operations.py'])
patch=''.join(''.join(difflib.unified_diff(files[p].splitlines(True),after[p].splitlines(True),fromfile='a/'+p,tofile='b/'+p)) for p in files)
Path('/private/tmp/console-dev-recovery-catalog-only-01a0fa6c.patch').write_text(patch)
Path('/private/tmp/console-dev-recovery-catalog-only-01a0fa6c.json').write_text(json.dumps({'before_sha256':{p:hashlib.sha256(s.encode()).hexdigest() for p,s in files.items()},'constructor_capture':str(root/'catalogs.json'),'capture_sha256':hashlib.sha256((root/'catalogs.json').read_bytes()).hexdigest(),'capture_check':str(root/'schema-object-deltas.json'),'deltas':delta,'notes':['No production files written. Schema additions preserve v18/v21 and standalone/v75 combined catalogs.','New exact v22 variants retain unchanged SQL from each v21 variant; eight added objects replace two old objects.','Combined subscriptions admits core76 only when full exact76catalog matches; old75catalog still requires stamp75.','21→22 literal installed migration route proposed. Its tightly scoped native authorizer actions are NOT yet added or qualified; route must remain unqualified until targeted RED/GREEN.','Core75 remains retained read-only with no new recovery migration route; pre-existing supported Prompts4→5/AgentRuns18→21 routes preserved.']},indent=2))
print('Read-only candidate /private/tmp/console-dev-recovery-catalog-only-01a0fa6c.patch and .json; data-only diff separately available; production unchanged.')
