# Repository Guidelines

## Project Structure & Module Organization

This repository currently contains data and specifications; application source, tests, and build manifests are not yet present.

- `dataset/README.md`: prediction rules, schemas, and submission requirements.
- `dataset/train/`, `dataset/test/`: telemetry and schedules; `dataset/labels/`: training and evaluation targets.
- `dataset/validate/`: prediction inputs; `dataset/sample_submission.csv`: submission template and baseline.
- `dataset/docs/Emulator-and-Telematic-Packets-Specification.md`: NDTP protocol and emulator API.
- `dataset/ndtp-telemetry-emulator.tar`: Docker image; the root PDF contains product requirements.

For new implementation, keep Backend, ML, and dashboard independently runnable. The planned MVP uses CPU inference and historical replay. Share feature-building logic between training and inference.

## Build, Test, and Development Commands

Run these documented emulator commands from the repository root:

```bash
# Import the supplied image.
docker load -i dataset/ndtp-telemetry-emulator.tar
# Start its configuration API.
docker run --rm -p 18080:18080 \
  --add-host=host.docker.internal:host-gateway \
  --name ndtp-emu ndtp-telemetry-emulator:1.0
# Inspect supported telemetry cells in another terminal.
curl -fsS http://localhost:18080/api/cells
```

Sending telemetry additionally requires an NDTP receiver and `POST /api/config`; follow the specification. No application build or test command exists yet. Document commands when adding tooling.

## Coding Style & Naming Conventions

For new Python code, use four-space indentation, type hints, `snake_case` functions/modules, and `PascalCase` classes. For TypeScript, use two-space indentation, `camelCase` functions, and `PascalCase` components. Preserve dataset field names at ingestion boundaries; name derived durations with `_s`. No formatter or linter is currently configured.

## Testing Guidelines

For new Python modules, use pytest with `tests/test_*.py`; add dependencies before documenting `python -m pytest` as runnable. No coverage threshold exists. Prioritize time-cutoff invariance, missing telemetry, NDTP fragmentation/CRC/reconnect, offline/online feature parity, and submission validation. Report MAE against zero and `cur_dev_s` baselines.

## Data Integrity

Preserve supplied files; write generated artifacts separately. Features must use only telemetry with `event_time <= T`; targets belong in `(T+10 min, T+15 min]`. Exclude target columns and `time_fact_begin` from features. Never recover validate answers from overlapping train/test schedules. Test and validate telemetry overlap; do not claim independent-period evaluation.

Submissions require UTF-8, semicolon-separated `sample_id;prediction`, every validate ID exactly once, and finite signed predictions in seconds.

## Commit & Pull Request Guidelines

No Git history is available. Adopt concise imperative commits, such as `feat: add NDTP parser`. PRs should describe behavior, relevant requirements/issues, validation commands/results, and ML metric changes. Include screenshots for dashboard changes. Exclude secrets and generated artifacts unless explicitly required.
