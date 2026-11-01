from pathlib import Path
import json, sqlite3
import pytest
pytestmark = pytest.mark.bootstrap_profile

def test_capture(tmp_path):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA
    result = {}
    def capture(connection):
        return [r[0] for r in connection.execute('SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name')]
    db=CharactersRAGDB(tmp_path/'core.db','capture');result['core']=capture(db.get_connection());db.close()
    subs=SubscriptionsDB(tmp_path/'core.db');result['combined']=capture(subs._get_connection());subs.close()
    db=AgentRunsDB(tmp_path/'runs.db');result['agent_fresh']=capture(db._get_connection());db.close()
    variants=[]
    for i,(version,sqls) in enumerate(_AGENT_RUNS_SCHEMA):
        if version != 21:continue
        path=tmp_path/f'variant-{i}.db'
        with sqlite3.connect(path) as conn:
            for sql in sorted(sqls,key=lambda s:not s.startswith('CREATE TABLE')):
                if sql.startswith('CREATE TABLE sqlite_sequence'):continue
                conn.execute(sql)
            conn.execute('INSERT INTO schema_version(version) VALUES (21)')
        db=AgentRunsDB(path);variants.append(capture(db._get_connection()));db.close()
    result['agent_migrated']=variants
    Path('.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/catalogs.json').write_text(json.dumps(result,indent=2))
    from tldw_chatbook.DB.recovery_operations import _SUBSCRIPTIONS_SCHEMA
    old21=[sql for version,sql in _AGENT_RUNS_SCHEMA if version==21]
    deltas=[]
    for old,new in zip(old21,result['agent_migrated']):
        deltas.append({'removed':[s for s in old if s not in new],'added':[s for s in new if s not in old]})
    assert all(d==deltas[0] for d in deltas)
    oldcombined=_SUBSCRIPTIONS_SCHEMA[1][1]
    subs={'removed':[s for s in oldcombined if s not in result['combined']], 'added':[s for s in result['combined'] if s not in oldcombined]}
    assert len(subs['removed'])==len(subs['added'])==2
    Path('.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/schema-object-deltas.json').write_text(json.dumps({'agent':deltas[0],'subscriptions':subs},indent=2))
    print('Agent changed objects:',len(deltas[0]['removed']),len(deltas[0]['added']),'Combined subscriptions changed objects:2/2')


def test_proposed_catalogs_match_all_four_real_constructors():
    ns={'__name__':'candidate_catalog'}
    exec(compile(Path('/private/tmp/console-dev-recovery-operations-proposal-01a0fa6c.py').read_text(),'candidate','exec'),ns)
    capture=json.loads(Path('.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/catalogs.json').read_text())
    proposed=[schema for version,schema in ns['_AGENT_RUNS_SCHEMA'] if version==22]
    actual=[tuple(s) for s in capture['agent_migrated']]+[tuple(capture['agent_fresh'])]
    assert len(proposed)==4
    assert proposed==actual
    assert ns['_SUBSCRIPTIONS_SCHEMA'][-1][1]==tuple(capture['combined'])
    from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA, _SUBSCRIPTIONS_SCHEMA
    assert ns['_AGENT_RUNS_SCHEMA'][:4]==_AGENT_RUNS_SCHEMA
    assert ns['_SUBSCRIPTIONS_SCHEMA'][:2]==_SUBSCRIPTIONS_SCHEMA
    assert ns['_AgentRunsAdapter']('db.agent_runs',None,'runs.db',(18,21,22),ns['_AGENT_RUNS_SCHEMA']).schema_policy().migration_steps[0][:2]==(18,21)
