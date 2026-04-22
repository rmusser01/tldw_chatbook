from pathlib import Path


DIRECT_SERVER_CLIENT_ALLOWLIST = {
    "tldw_chatbook/Chatbooks/server_chatbook_service.py",
    "tldw_chatbook/Event_Handlers/tldw_api_events.py",
    "tldw_chatbook/Study_Interop/server_quiz_service.py",
    "tldw_chatbook/UI/MediaIngestWindowRebuilt.py",
    "tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py",
    "tldw_chatbook/UI/Wizards/ChatbookImportWizard.py",
    "tldw_chatbook/runtime_policy/bootstrap.py",
}

RUNTIME_POLICY_SNAPSHOT_ALLOWLIST = {
    "tldw_chatbook/app.py",
    "tldw_chatbook/runtime_policy/bootstrap.py",
}


def test_raw_server_client_construction_is_confined_to_explicit_allowlist():
    repo = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for path in repo.joinpath("tldw_chatbook").rglob("*.py"):
        rel = path.relative_to(repo).as_posix()
        if rel in DIRECT_SERVER_CLIENT_ALLOWLIST:
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
        if rel in RUNTIME_POLICY_SNAPSHOT_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        if "runtime_policy_snapshot" in text:
            offenders.append(rel)

    assert offenders == []
