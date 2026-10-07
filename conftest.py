"""Unit runs isolate local credentials and forbid real outbound connections."""
import os
import socket
from pathlib import Path

import pytest


def pytest_configure(config):
    # Before test modules import app.main/get_settings, isolate the explicitly
    # requested offline suite (and collection). Integration runs keep real config.
    if config.option.collectonly or config.option.markexpr.strip() == "not integration":
        import app.config as settings_module

        patch = pytest.MonkeyPatch()
        config._offline_settings_patch = patch
        for key in list(os.environ):
            if key.startswith("APP_") or key in {"DEEPSEEK_API_KEY", "ALIYUN_API_KEY"}:
                patch.delenv(key)
        patch.setenv("APP_AUTH__JWT_SECRET", "offline-tests-only-secret-32-bytes-minimum")
        patch.setattr(settings_module, "ENV_FILE", Path(__file__).parent / ".pytest-no-env")
        patch.setitem(settings_module.Settings.model_config, "env_file", None)
        # PyMilvus imports call python-dotenv.load_dotenv() independently of
        # Settings. Disable that discovery too, before any SDK imports occur.
        import dotenv
        patch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
        settings_module.get_settings.cache_clear()


def pytest_unconfigure(config):
    patch = getattr(config, "_offline_settings_patch", None)
    if patch is not None:
        patch.undo()
        from app.config import get_settings
        get_settings.cache_clear()


@pytest.fixture(autouse=True)
def isolate_unit_knowledge_revision(request, tmp_path, monkeypatch):
    """替身入库也会刷新修订号；离线测试不得改本机知识发布标识。"""
    if request.node.get_closest_marker("integration"):
        return
    from app.rag import revision
    monkeypatch.setattr(revision, "PATH", tmp_path / "knowledge_revision")


@pytest.fixture(autouse=True)
def forbid_unit_network(request, monkeypatch):
    if request.node.get_closest_marker("integration"):
        return

    def blocked(*args, **kwargs):
        raise AssertionError("Non-integration test attempted networking; inject a fake/MockTransport")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
