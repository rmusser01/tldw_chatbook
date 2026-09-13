"""Default RAG definition capture through the installed public/native path."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import json, os, subprocess, sys
from pathlib import Path
from threading import Event
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture, capture
from tldw_chatbook.Backup_Recovery.capture import CaptureReviewRequired
source = Path(os.environ['TLDW_CONFIG_PATH'])
data = Path(os.environ['XDG_DATA_HOME'])/'fixture'
source.write_text('[paths]\ndata_dir='+json.dumps(str(data))+'\n')
source.chmod(0o600)
profiles = data/'default_user'/'rag_profiles'
profiles.mkdir(parents=True, mode=0o700)
experiment = profiles/'experiments'/'retained'
experiment.mkdir(parents=True, mode=0o700)
profile = {'name':'retained','rag_config':{'embedding':{'api_key':'synthetic-rag-secret'}},'description':'enc:history prose'}
fixtures = {
 profiles/'profile.json': json.dumps(profile),
 profiles/'custom_profiles.json.migrated': json.dumps({'profiles':[profile]}),
 source.parent/'rag_pipelines.toml': '[pipelines.private]\nname="retained"\n[pipelines.private.parameters]\napi_key="synthetic-pipeline-secret"\n',
 experiment/'config.json': json.dumps(dict(experiment_id='retained',name='history',description='enc:description',control_profile='hybrid_basic',enable_ab_testing=True,track_metrics=True,save_results=True,test_profiles=[],metrics_to_track=[],traffic_split={},results_dir=str(experiment))),
 experiment/'results.json': json.dumps({'summary':{'experiment_id':'retained','name':'history','total_queries':1,'profiles':{}},'detailed_results':[{'timestamp':1,'profile':'hybrid_basic','query':'enc:private historical prose','metrics':{}}],'completed_at':'2026-09-10'}),
}
for path, text in fixtures.items():
 path.write_text(text)
 path.chmod(0o600)
route = sys.argv[1]
if route == 'shape':
 (profiles/'profile.json').write_text('{"unrecognized":true}')
elif route == 'unknown':
 (profiles/'unowned.bin').write_bytes(b'private unknown format')
elif route == 'oversize':
 with (profiles/'profile.json').open('wb') as stream:
  stream.truncate(16*1024**2+1)
original = {p:(p.read_bytes(),p.stat().st_mode,p.stat().st_mtime_ns) for p in fixtures}
options = {'allow_partial':True,'staging_parent':Path.home()}
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
# Establish the real local authority before reviewing first-use parent state.
admission_authority(bootstrap.default_bootstrap_root())
producer = '''
import os
from pathlib import Path
from Tests import network_guard
network_guard.install()
from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
_populate_required_dependencies(preview_capture((Path(os.environ["TLDW_CONFIG_PATH"]),),options={"allow_partial":True}))
'''
prepared = subprocess.run([sys.executable,'-c',producer],capture_output=True,text=True,timeout=30)
assert prepared.returncode==0, prepared.stderr[-2000:]
original[source]=(source.read_bytes(),source.stat().st_mode,source.stat().st_mtime_ns)
preview = preview_capture((source,), options=options)
expected = set(fixtures)
included = {i.path for i in preview.items if i.owner=='rag.definitions' and i.status=='included'}
assert included == expected, ('known definitions not included', [(str(i.path),i.status) for i in preview.items if i.owner=='rag.definitions'],preview.issues)
if route == 'unknown':
 unknown = [i for i in preview.items if i.path == profiles/'unowned.bin']
 assert len(unknown)==1 and unknown[0].status=='unsupported'
try:
 result = capture((source,),preview.scope_digest,Path.home()/'backup.tldw-backup.zip',options=options,cancel=Event())
except (CaptureReviewRequired, ValueError) as error:
 assert str(error)!='scope_changed'
 assert route in {'shape','oversize'}, ('recognized definitions refused',getattr(error,'issues',str(error)))
 if route=='shape':
  assert isinstance(error,CaptureReviewRequired)
  assert any(i.startswith('credential_rag_definition_format_unsupported:') for i in error.issues), error.issues
 else:
  assert str(error)=='capture_byte_limit', str(error)
else:
 assert route not in {'shape','oversize'}, 'invalid definitions accepted'
 manifest = json.loads(result.manifest_bytes)
 assert manifest['consistency']=='partial', 'fixture does not qualify unrelated owners'
 files = [f for f in manifest['files'] if f['owner_id']=='rag.definitions']
 assert len(files)==len(fixtures)
 payloads = {f['relative_path']:(result.root/f['payload']).read_text() for f in files}
 assert 'synthetic-rag-secret' not in ''.join(payloads.values())
 assert 'synthetic-pipeline-secret' not in ''.join(payloads.values())
 assert 'enc:private historical prose' in payloads['experiments/retained/results.json']
 assert 'enc:description' in payloads['experiments/retained/config.json']
 assert 'enc:history prose' in payloads['profile.json']
 producers = {f['logical_id']:f for f in manifest['producer_inventory']}
 excluded = [f for f in manifest['exclusions'] if producers[f['logical_id']]['owner_id']=='rag.definitions']
 assert len(excluded)==(1 if route=='unknown' else 0)
 assert {i.path for i in result.inventory.items if i.owner=='rag.definitions' and i.status=='included'}==expected
assert all((p.read_bytes(),p.stat().st_mode,p.stat().st_mtime_ns)==before for p,before in original.items())
assert 'tldw_chatbook.RAG_Search.config_profiles' not in sys.modules
assert 'chromadb' not in sys.modules
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["known", "shape", "unknown", "oversize"])
def test_public_default_definition_capture(tmp_path, route):
    _run(tmp_path, route, "success", script=_SCRIPT)
