"""前端静态托管冒烟测试（frontend/ 由 FastAPI 托管）。"""
from fastapi.testclient import TestClient

from app.main import app


class TestStaticServe:

    def test_index_served(self):
        resp = TestClient(app).get("/")
        assert resp.status_code == 200
        # v3 工作区已经把 CDN 路径改成仓库内的静态 Vue3 文件。
        assert 'src="/static/vue.global.prod.js"' in resp.text
        runtime = TestClient(app).get("/static/vue.global.prod.js")
        assert runtime.status_code == 200
        assert "vue v3." in runtime.text
        assert "保理营销教练" in resp.text

    def test_app_js_served(self):
        resp = TestClient(app).get("/static/app.js")
        assert resp.status_code == 200
        assert "API_BASE" in resp.text
