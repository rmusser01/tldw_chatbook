import os
import pytest

# --- Ported (chunking-engine-parity Task 4) ---------------------------------
# Upstream file: tldw_Server_API/tests/Chunking/test_chunking_templates_validate_schema.py
# Skipped: FastAPI TestClient endpoint fixture (spec §10.1 endpoint class); lands with #2. Remove this block when the module is vendored in
# its own sub-project and re-sync the test from upstream.
pytest.importorskip("tldw_chatbook.NoSuchDeferredModule",
                    reason="skipped: FastAPI TestClient endpoint fixture (spec §10.1 endpoint class); lands with #2")
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_validate_template_schema_classifier_errors():
    """TemplateConfig pydantic validation errors are returned as 200 with errors listed.

    Ensures we don't emit 422 for schema errors in /validate; instead, we surface them in the payload.
    """
    # Minimal app mounting the router directly
    os.environ.setdefault("AUTH_MODE", "single_user")
    from fastapi import FastAPI
    from tldw_Server_API.app.api.v1.endpoints.chunking_templates import router as tmpl_router
    from tldw_chatbook.Chunking._shims.AuthNZ.settings import get_settings

    app = FastAPI()
    app.include_router(tmpl_router, prefix="/api/v1")
    client = TestClient(app)
    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}

    payload = {
        "chunking": {"method": "sentences", "config": {}},
        "classifier": {"bogus_field": 1},  # not allowed per TemplateConfig
    }
    r = client.post("/api/v1/chunking/templates/validate", json=payload, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["valid"] is False
    # At least one error mentions classifier
    fields = [e.get("field", "") for e in (body.get("errors") or [])]
    msgs = [e.get("message", "") for e in (body.get("errors") or [])]
    assert any("classifier" in f for f in fields) or any("classifier" in m for m in msgs)
