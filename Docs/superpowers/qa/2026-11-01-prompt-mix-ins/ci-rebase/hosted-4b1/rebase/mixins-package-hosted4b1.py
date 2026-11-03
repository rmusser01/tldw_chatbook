import gzip
import hashlib
import json
import re
from pathlib import Path

root = Path.cwd()
qa = root / "Docs/superpowers/qa/2026-11-01-prompt-mix-ins/ci-rebase/hosted-4b1"
qa.mkdir(parents=True, exist_ok=True)
mapping = {}
credential = re.compile(
    rb"\b(?:(?:sk|rk)_(?:live|test)_[A-Za-z0-9]+|"
    rb"sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|"
    rb"github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})\b"
)


def keep(source, group, relative_root=None, expected=None):
    source = Path(source)
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert expected is None or digest == expected, str(source)
    matches = credential.findall(data)
    assert not matches, (
        str(source),
        "credential-shaped content requires explicit review/redaction",
        [hashlib.sha256(value).hexdigest() for value in matches],
    )
    if str(source) in mapping:
        assert mapping[str(source)]["source_sha256"] == digest
        return
    relative = (
        source.relative_to(relative_root)
        if relative_root is not None and source.is_relative_to(relative_root)
        else Path(source.name)
    )
    target = qa / group / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = data
    trailing = data.endswith(b"\n\n") or any(
        line.endswith((b" ", b"\t")) for line in data.splitlines()
    )
    wrapped = source.suffix in {".log", ".xml", ".diff", ".patch", ".txt", ".jsonl"}
    wrapped |= source.suffix in {".py", ".md"} and trailing
    if wrapped:
        payload = (
            json.dumps({"sha256": digest, "raw_text": data.decode()}, indent=2).encode()
            + b"\n"
        )
        target = target.with_name(target.name + ".json")
    row = {
        "source_sha256": digest,
        "source_bytes": len(data),
        "retained_format": "raw_text_json" if wrapped else "raw",
    }
    if len(payload) > 200_000 or source.suffix == ".svg":
        row["uncompressed_retained_sha256"] = hashlib.sha256(payload).hexdigest()
        payload = gzip.compress(payload, mtime=0)
        target = target.with_name(target.name + ".gz")
        row["retained_encoding"] = "gzip"
    row["retained"] = target.relative_to(root).as_posix()
    assert all(old["retained"] != row["retained"] for old in mapping.values())
    target.write_bytes(payload)
    row["retained_sha256"] = hashlib.sha256(payload).hexdigest()
    mapping[str(source)] = row



fix=Path('/private/tmp/mixins-4b1-uat-fix')
for name,row in json.loads((fix/'artifact-inventory.json').read_text()).items():
    keep(fix/name,'uat',fix,row['sha256'])
keep(fix/'artifact-inventory.json','uat')
review=Path('/private/tmp/mixins-4b1-uat-review')
review_inventory=json.loads((review/'artifact-inventory.json').read_text())
for name,row in review_inventory.items():
    keep(review/name,'review',review,row['sha256'])
keep(review/'artifact-inventory.json','review')
rebase=Path('/private/tmp/mixins-4b1-rebase')
for name in ['protocol.json','source-before.json','source-after.json','rebase-proof.json','commands.json','size-guards.log','ci-contracts.log','test_module_size_ratchet.py']:
    keep(rebase/name,'rebase',rebase)
for source in ['/private/tmp/mixins-4b1-admission2.log','/private/tmp/mixins-4b1-uat-origin.json','/private/tmp/mixins-4b1-perf.log','/private/tmp/mixins-4b1-perf-origin.json','/private/tmp/mixins-4b1-latency-local.log','/private/tmp/mixins-4b1-latency-local-proof.json','/private/tmp/mixins-4b1-newdev-budget-proof.json']:
    keep(source,'hosted')
keep(__file__,'rebase')
for source,row in mapping.items():
    payload=(root/row['retained']).read_bytes()
    assert hashlib.sha256(payload).hexdigest()==row['retained_sha256']
    if row.get('retained_encoding')=='gzip':
        payload=gzip.decompress(payload)
        assert hashlib.sha256(payload).hexdigest()==row['uncompressed_retained_sha256']
    if row['retained_format']=='raw_text_json':
        wrapper=json.loads(payload)
        assert wrapper['sha256']==row['source_sha256']
        payload=wrapper['raw_text'].encode()
    assert payload==Path(source).read_bytes(),source
    assert len(payload)==row['source_bytes']
(qa/'artifact-map.json').write_text(json.dumps(mapping,indent=2)+'\n')
print(f'Retained and verified {len(mapping)} explicit artifact identities.')
