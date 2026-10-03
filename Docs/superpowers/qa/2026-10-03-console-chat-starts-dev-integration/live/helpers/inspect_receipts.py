from pathlib import Path
import hashlib,json,sqlite3,sys
ROOT=Path('/private/tmp/console-dev-live-01a0fa6c')
label=sys.argv[1]
assert label.replace('-','').isalnum()
result={'label':label,'databases':{}}
for path in sorted((ROOT/'data').rglob('*.db')):
 db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
 db.row_factory=sqlite3.Row
 try:
  tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
  row={'path':str(path),'tables':sorted(tables)}
  if 'conversations' in tables:
   convs=[dict(x) for x in db.execute('SELECT * FROM conversations WHERE title LIKE ? ORDER BY created_at',('DEV_%',))]
   row['targets']=convs
   for conv in convs:
    cid=conv['id']
    conv['messages']=[dict(x) for x in db.execute('SELECT * FROM messages WHERE conversation_id=? ORDER BY timestamp, rowid',(cid,))]
    if 'console_dispatch_checkpoints' in tables:
     conv['checkpoints']=[dict(x) for x in db.execute('SELECT * FROM console_dispatch_checkpoints WHERE conversation_id=?',(cid,))]
  # Every identifier below is a fixed owned-schema name.
  for table in ('automatic_chat_start_attempts','automatic_work_chains','automatic_work_reservations','console_hook_continuation_receipts','schema_version'):
   if table in tables:
    row[table]=[dict(x) for x in db.execute('SELECT * FROM '+table)]
  result['databases'][path.name]=row
 finally:
  db.close()
real=Path('/Users/macbook-dev/.config/tldw_cli/config.toml')
result['real_config_sha256']=hashlib.sha256(real.read_bytes()).hexdigest()
p=ROOT/'evidence'/(label+'-receipts.json')
p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'receipt':str(p),'databases':len(result['databases']),'real_config_sha256':result['real_config_sha256']}))
