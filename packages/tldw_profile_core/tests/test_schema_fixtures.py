import json
from pathlib import Path

from tldw_profile_core import export_json_schema


ROOT = Path(__file__).parents[1]


def test_schema_is_checked_in_and_deterministic(tmp_path):
    checked = json.loads((ROOT / "schemas" / "personal-context-v1.json").read_text())
    target = tmp_path / "schema.json"
    export_json_schema(target)
    assert json.loads(target.read_text()) == checked


def test_v1_fixtures_are_json_and_have_expected_validity_labels():
    fixtures = list((ROOT / "fixtures" / "v1").glob("*.json"))
    assert fixtures
    for fixture in fixtures:
        data = json.loads(fixture.read_text())
        assert data["valid"] in (True, False)
