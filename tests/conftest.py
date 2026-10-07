"""Legacy endpoint contract tests inject an authenticated owner explicitly.

Security tests in test_auth.py do not use these overrides; they exercise real
JWT/password verification, SQL ownership and scope checks with SQLite.
"""
import pytest


@pytest.fixture(autouse=True)
def legacy_endpoint_identity(request, monkeypatch):
    if request.node.get_closest_marker("integration"):
        yield
        return
    modules = {"test_api_case.py", "test_api_session.py", "test_api_evaluation.py",
               "test_api_user_scores.py", "test_api_knowledge.py", "test_case_fact_cards.py"}
    if request.path.name not in modules and request.node.name != "test_write_failure_returns_redacted_503_at_http_boundary":
        yield
        return
    from app.main import app
    from app.api.deps import get_current_user, require_owned_session, require_score_scope
    from app.services.auth_service import Principal
    overrides = app.dependency_overrides.copy()
    if request.node.name == "test_write_failure_returns_redacted_503_at_http_boundary":
        monkeypatch.setattr("app.db.runtime_schema.verify_runtime_schema", lambda engine: [])
    if request.path.name == "test_api_knowledge.py":
        from app.api.v1.pending import get_pending_service
        class ContractPending:
            def register(self, *args, **kwargs):
                return {"external_id":"esc_contract"}
        app.dependency_overrides[get_pending_service] = ContractPending
    owner = lambda: Principal(1001, "contract_test", "admin")
    app.dependency_overrides[get_current_user] = owner
    app.dependency_overrides[require_owned_session] = owner
    app.dependency_overrides[require_score_scope] = owner
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides.update(overrides)
