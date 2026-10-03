from pathlib import Path
import json,hashlib
ROOT=Path('/private/tmp/console-dev-live-01a0fa6c');WT=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook');QA=WT/'Docs/superpowers/qa/2026-10-03-console-chat-starts-dev-integration'
QA.mkdir(parents=True,exist_ok=True);manifest={}
sources=[p for p in sorted((ROOT/'evidence').iterdir()) if p.is_file()]
sources += [ROOT/name for name in ('launch.py','start_console.py','drive.py','snapshot.py','inspect_receipts.py','qualify_live.py','qualify_final_live.py','copy_live_qa.py','run_check.py','validate_finalfix_receipts.py','pack_qa.py','verify_qa_blobs.py','sweep_task_ids.py','sweep_task_ids_publication.py')]
for src in sources:
 raw=src.read_bytes();relative=src.relative_to(ROOT)
 target=QA/'live'/(relative.name if relative.parts[0]=='evidence' else Path('helpers')/relative)
 target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 assert target.read_bytes()==raw
 manifest[str(relative)]={'source_sha256':hashlib.sha256(raw).hexdigest(),'source_bytes':len(raw),'qa_path':str(target.relative_to(QA))}
(QA/'live-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'live_records':len(manifest),'copied_raw_databases':False,'qa':str(QA)}))
