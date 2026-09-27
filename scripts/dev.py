"""Launch the local CPU stack; Ctrl-C stops only these child processes."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from dotenv import dotenv_values


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / "artifacts/models/manifest.json").exists():
        print("Train the model first: make train", file=sys.stderr)
        raise SystemExit(1)
    env = {**{k: v for k, v in dotenv_values(root / ".env").items() if v is not None}, **os.environ}
    env.setdefault("DATABASE_URL", "sqlite:///artifacts/runtime/dev.db")
    env.setdefault("ML_URL", "http://127.0.0.1:8001")
    env.setdefault("DATA_DIR", str(root / "dataset"))
    env.setdefault("MODEL_DIR", str(root / "artifacts/models"))
    env.setdefault("OMP_NUM_THREADS", "2")
    backend_port = env.get("BACKEND_PORT", "8000")
    env["BACKEND_PROXY_URL"] = f"http://127.0.0.1:{backend_port}"
    commands = [
        ([sys.executable, "-m", "uvicorn", "ml.app:app", "--host", "127.0.0.1", "--port", "8001"], root),
        ([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", backend_port], root),
        (["npm", "run", "dev", "--", "--host", "127.0.0.1"], root / "frontend"),
    ]
    children = []

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for command, cwd in commands:
            children.append(subprocess.Popen(command, cwd=cwd, env=env))
        print(f"Dashboard: http://127.0.0.1:5173 | API: http://127.0.0.1:{backend_port}/docs", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
