"""Static API docs must describe the real apps without starting their runtimes."""
import json

from backend.app import app as backend_app
from ml.app import app as ml_app
from scripts.build_docs import export_api


def test_exported_openapi_matches_both_applications_without_lifespan(tmp_path):
    counts = export_api(tmp_path)
    for name, app, required in (
        ("backend", backend_app, {"/api/v1/snapshot", "/api/v1/replay/start", "/api/v1/live/start"}),
        ("ml", ml_app, {"/v1/predict", "/v1/profiles"}),
    ):
        schema = json.loads((tmp_path / "api" / name / "openapi.json").read_text())
        assert schema == app.openapi()
        assert required <= schema["paths"].keys()
        assert counts[name] == len(schema["paths"])
    assert not list(tmp_path.rglob("*.db"))
    assert (tmp_path / ".nojekyll").is_file()


def test_pages_uses_relative_schema_and_disables_requests_to_static_host(tmp_path):
    export_api(tmp_path)
    for name in ("backend", "ml"):
        html = (tmp_path / "api" / name / "index.html").read_text()
        assert "url: './openapi.json'" in html
        assert '"supportedSubmitMethods": []' in html
        assert '"tryItOutEnabled": false' in html
