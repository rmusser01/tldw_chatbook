from pathlib import Path
import gzip,hashlib,json,subprocess
WT=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook');QA=WT/'Docs/superpowers/qa/2026-10-03-console-chat-starts-dev-integration';PREFIX=str(QA.relative_to(WT))
def blob(relative):return subprocess.check_output(['git','show','HEAD:'+PREFIX+'/'+str(relative)],cwd=WT)
manifest=json.loads(blob('artifact-source-manifest.json'));live=json.loads(blob('live-source-manifest.json'));checked=[]
paths=sorted({r['archive_path'] for r in manifest.values()}|{r['readable_path'] for r in manifest.values() if r['readable_path'] is not None}|{r['qa_path'] for r in live.values()})
requests=''.join('HEAD:'+PREFIX+'/'+relative+'\n' for relative in paths).encode()
stream=subprocess.check_output(['git','cat-file','--batch'],input=requests,cwd=WT);cursor=0;cache={}
for relative in paths:
 end=stream.index(b'\n',cursor);header=stream[cursor:end].decode().split();assert header[1]=='blob',header
 size=int(header[2]);cursor=end+1;cache[relative]=stream[cursor:cursor+size];cursor+=size;assert stream[cursor:cursor+1]==b'\n';cursor+=1
assert cursor==len(stream)
for name,r in manifest.items():
 archive=cache[r['archive_path']];assert hashlib.sha256(archive).hexdigest()==r['archive_sha256']
 raw=gzip.decompress(archive);assert len(raw)==r['source_bytes'] and hashlib.sha256(raw).hexdigest()==r['source_sha256']
 if r['readable_path'] is not None:assert cache[r['readable_path']]==raw
 checked.append(name)
for name,r in live.items():
 raw=cache[r['qa_path']];assert len(raw)==r['source_bytes'] and hashlib.sha256(raw).hexdigest()==r['source_sha256']
print(json.dumps({'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip(),'source_archive_records':len(checked),'live_records':len(live),'all_committed_blob_lengths_hashes_decompression_and_readable_copies_equal':True}))
