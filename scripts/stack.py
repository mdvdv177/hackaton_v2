"""Portable Compose entry point; never changes global Docker configuration."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]


@lru_cache
def compose_command() -> list[str]:
    docker = shutil.which("docker")
    if docker and subprocess.run([docker, "compose", "version"], capture_output=True).returncode == 0:
        return [docker, "compose"]
    candidates = [shutil.which("docker-compose"), "/Applications/Docker.app/Contents/Resources/cli-plugins/docker-compose"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            if subprocess.run([candidate, "version"], capture_output=True).returncode == 0:
                return [candidate]
    raise RuntimeError("Docker Compose is unavailable. Install/enable the Compose plugin.")


def compose(*args: str, project: str | None = None, env: dict | None = None, capture: bool = False):
    command = compose_command() + (["-p", project] if project else []) + list(args)
    return subprocess.run(command, cwd=ROOT, env={**os.environ, **(env or {})}, check=True,
                          capture_output=capture, text=True)


def doctor() -> dict:
    command = compose_command()
    subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], check=True, capture_output=True)
    for suffix in ("", "stream/"):
        if not (ROOT / "artifacts/models" / suffix / "manifest.json").exists():
            raise RuntimeError(f"Missing model profile: {suffix or 'official'}; run make train-stream")
    return {"docker": "ready", "compose": command, "model_profiles": ["official", "stream"]}


def available_port() -> int:
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return stream.getsockname()[1]


def isolated_env() -> dict[str, str]:
    return {key: str(available_port()) for key in ("BACKEND_PORT", "DASHBOARD_PORT", "NDTP_PORT", "EMULATOR_PORT")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["doctor", "up", "down", "ps"])
    parser.add_argument("--project")
    args = parser.parse_args()
    if args.action == "doctor":
        print(json.dumps(doctor(), ensure_ascii=False, indent=2))
    elif args.action == "up":
        doctor()
        compose("up", "--build", "-d", "--wait", project=args.project)
    else:
        compose(args.action, project=args.project)


if __name__ == "__main__":
    main()
