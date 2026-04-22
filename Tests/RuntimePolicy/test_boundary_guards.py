from pathlib import Path


ALLOWLIST = {
    "tldw_chatbook/app.py",
    "tldw_chatbook/runtime_policy/bootstrap.py",
}


def test_raw_server_client_construction_is_confined_to_runtime_policy_boundaries():
    repo = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for path in repo.joinpath("tldw_chatbook", "runtime_policy").rglob("*.py"):
        rel = path.relative_to(repo).as_posix()
        if rel in ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        if "TLDWAPIClient(" in text or "ServerChatbookService(" in text:
            offenders.append(rel)

    assert offenders == []


def test_runtime_policy_snapshot_contract_stays_confined_to_authoritative_bootstrap_boundary():
    repo = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for path in repo.joinpath("tldw_chatbook").rglob("*.py"):
        rel = path.relative_to(repo).as_posix()
        if rel in ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        if "runtime_policy_snapshot" in text:
            offenders.append(rel)

    assert offenders == []
