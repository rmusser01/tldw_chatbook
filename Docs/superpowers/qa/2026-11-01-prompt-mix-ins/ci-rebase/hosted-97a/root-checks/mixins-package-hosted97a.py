import gzip
import hashlib
import json
import re
from pathlib import Path

root = Path.cwd()
qa = root / "Docs/superpowers/qa/2026-11-01-prompt-mix-ins/ci-rebase/hosted-97a"
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




for inventory, group in [('/private/tmp/mixins-97a-archive-fix/artifact-inventory.json','repair'),('/private/tmp/mixins-97a-archive-review/artifact-inventory.json','review'),('/private/tmp/mixins-97a-final/artifact-inventory.json','root-checks')]:
    path=Path(inventory)
    inventory_data = json.loads(path.read_text())
    for row in inventory_data.get('files', inventory_data.get('artifacts', [])):
        keep(row['path'],group,path.parent,row['sha256'])
    keep(path,group)
for path in ['/private/tmp/mixins-97a-admission1.log','/private/tmp/mixins-97a-admission1-origin.json']:
    keep(path,'hosted')
for suffix in ['test_ui_ready_module_census-_boot_and_census.json','test_boot_worker_census-_boot_and_census.json','test_screen_preimport_payload_budget-_run_census.json','test_screen_preimport_payload_budget-warmup.json']:
    keep('/private/tmp/mixins-perf-pr3346-archive-final-'+suffix,'startup-results')
keep('/private/tmp/mixins-pr3346-frozen-final-five/probe.py','startup-results')
keep(__file__,'root-checks')
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
