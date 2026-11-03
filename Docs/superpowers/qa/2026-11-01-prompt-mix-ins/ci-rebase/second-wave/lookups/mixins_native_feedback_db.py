"""Qualify native feedback's missing real persistence precondition, assertions unchanged."""
import pytest

@pytest.fixture(autouse=True)
def native_feedback_db(request, monkeypatch):
    if request.node.path.name != 'test_console_configuration_worker_feedback.py':
        return
    from Tests.UI.app_factory import attach_chachanotes_db
    module = request.module
    original = module._build_test_app
    def build(*args, **kwargs):
        app = original(*args, **kwargs)
        attach_chachanotes_db(app)
        return app
    monkeypatch.setattr(module, '_build_test_app', build)
