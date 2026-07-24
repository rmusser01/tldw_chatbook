import pytest

@pytest.fixture
def rv():
    from tldw_chatbook.Image_Generation import request_validation as m
    return m

def _codes(issues):
    return {i.path for i in issues}

def test_valid_request_has_no_issues(rv):
    ok = {"backend": "swarmui", "prompt": "cat", "width": 512, "height": 512, "steps": 20, "cfg_scale": 7.0, "extra_params": {}}
    assert rv.validate_image_generation_request(ok) == []

def test_oversize_dimensions_flagged(rv):
    bad = {"backend": "swarmui", "prompt": "cat", "width": 9000, "height": 9000, "extra_params": {}}
    issues = rv.validate_image_generation_request(bad)
    assert any("width" in p for p in _codes(issues))

def test_negative_cfg_scale_flagged(rv):
    bad = {"backend": "swarmui", "prompt": "cat", "cfg_scale": -1.0, "extra_params": {}}
    assert any("cfg_scale" in p for p in _codes(rv.validate_image_generation_request(bad)))

def test_extra_params_not_in_allowlist_flagged(rv):
    bad = {"backend": "swarmui", "prompt": "cat", "extra_params": {"totally_unknown": 1}}
    issues = rv.validate_image_generation_request(bad)
    assert any("extra_params" in p for p in _codes(issues))
