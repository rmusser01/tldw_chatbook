from pathlib import Path
import gzip,hashlib,json,sys
WT=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook')
SOURCE=WT/'.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration'
QA=WT/'Docs/superpowers/qa/2026-10-03-console-chat-starts-dev-integration'
QA.mkdir(parents=True,exist_ok=True)
(QA/'source-archives').mkdir(exist_ok=True)
manifest={}
# Direct SDD files are the controller/worker evidence contract. Test caches and
# baseline extraction trees are deliberately outside this enumerated boundary.
for p in sorted(SOURCE.iterdir()):
 if not p.is_file():
  continue
 raw=p.read_bytes()
 compressed=gzip.compress(raw,mtime=0)
 command_receipt=False
 if p.suffix=='.json':
  try: record=json.loads(raw)
  except (ValueError,UnicodeDecodeError): record=None
  command_receipt=isinstance(record,dict) and 'argv' in record and 'returncode' in record and ('stdout' in record or 'output' in record)
 keep_receipt=p.name.startswith(('finalfix-','latest-final-','controller-','review1-final-','committed-'))
 archive=QA/'source-archives'/(p.name+'.gz')
 archive.write_bytes(compressed)
 if p.suffix=='.diff' or (len(raw)>262144 and p.suffix!='.md') or (command_receipt and not keep_receipt):
  readable=None
 elif p.suffix=='.log':
  readable=QA/'verification-logs'/p.name
 else:
  readable=QA/p.name
 if readable is not None:
  readable.parent.mkdir(parents=True,exist_ok=True)
  readable.write_bytes(raw)
 manifest[p.name]={'source_bytes':len(raw),'source_sha256':hashlib.sha256(raw).hexdigest(),'source_path':str(p.relative_to(WT)),'archive_path':str(archive.relative_to(QA)),'archive_sha256':hashlib.sha256(compressed).hexdigest(),'readable_path':str(readable.relative_to(QA)) if readable is not None else None,'readable_sha256':hashlib.sha256(raw).hexdigest() if readable is not None else None}
 assert gzip.decompress(archive.read_bytes())==raw
 if readable is not None: assert readable.read_bytes()==raw
(QA/'artifact-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'artifacts':len(manifest),'qa':str(QA)}))
