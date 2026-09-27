"""Build Sphinx and static Swagger/OpenAPI without starting application services."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from fastapi.openapi.docs import get_swagger_ui_html

ROOT = Path(__file__).resolve().parents[1]


def export_api(output: Path) -> dict[str, int]:
    """Export real FastAPI schemas without entering application lifespans."""
    from backend.app import app as backend_app
    from ml.app import app as ml_app

    counts = {}
    for name, app in (("backend", backend_app), ("ml", ml_app)):
        folder = output / "api" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "openapi.json").write_text(
            json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        page = get_swagger_ui_html(
            openapi_url="./openapi.json", title=f"{app.title} — API reference",
            swagger_ui_parameters={"supportedSubmitMethods": [], "tryItOutEnabled": False,
                                   "validatorUrl": None, "persistAuthorization": False})
        (folder / "index.html").write_bytes(page.body)
        counts[name] = len(app.openapi()["paths"])
    (output / ".nojekyll").write_text("", encoding="utf-8")
    return counts


def build(output: Path) -> dict:
    """Compile documentation strictly and attach schemas of the same sources."""
    output = output.resolve()
    subprocess.run([sys.executable, "-m", "sphinx", "-W", "--keep-going", "-b", "html",
                    str(ROOT / "docs"), str(output)], check=True, cwd=ROOT)
    paths = export_api(output)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    result = {"commit": revision.stdout.strip() if revision.returncode == 0 else None,
              "openapi_paths": paths, "services_started": False}
    (output / "build-info.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/_build/html")
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
