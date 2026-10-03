from pathlib import Path
import gzip,hashlib,json,subprocess
WT=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook');QA=WT/'Docs/superpowers/qa/2026-10-03-console-chat-starts-dev-integration';PREFIX=str(QA.relative_to(WT))
assert subprocess.check_output(['git','rev-parse','--show-object-format'],cwd=WT,text=True).strip()=='sha1'
entries=subprocess.check_output(['git','ls-tree','-r','-z','HEAD','--',PREFIX],cwd=WT).split(b'\0');cache={}
for entry in entries:
 if not entry:continue
 header,path=entry.split(b'\t',1);mode,kind,oid=header.split();assert kind==b'blob' and mode==b'100644'
 full=path.decode();relative=full[len(PREFIX)+1:];raw=(WT/full).read_bytes()
 digest=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest();assert digest==oid.decode(),relative
 cache[relative]=raw
manifest=json.loads(cache['artifact-source-manifest.json']);live=json.loads(cache['live-source-manifest.json'])
for name,r in manifest.items():
 archive=cache[r['archive_path']];assert hashlib.sha256(archive).hexdigest()==r['archive_sha256']
 raw=gzip.decompress(archive);assert len(raw)==r['source_bytes'] and hashlib.sha256(raw).hexdigest()==r['source_sha256']
 if r['readable_path'] is not None:assert cache[r['readable_path']]==raw
for name,r in live.items():
 raw=cache[r['qa_path']];assert len(raw)==r['source_bytes'] and hashlib.sha256(raw).hexdigest()==r['source_sha256']
print(json.dumps({'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip(),'source_archive_records':len(manifest),'live_records':len(live),'verified_git_tree_blobs':len(cache),'method':'Every committed QA tree blob ID equals its independently computed local Git blob digest; SHA256, length, gzip decompression and readable-copy equality are then checked against the committed manifests.','all_committed_blob_lengths_hashes_decompression_and_readable_copies_equal':True}))
