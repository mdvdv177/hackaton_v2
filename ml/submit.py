"""Generate all validate predictions, preserving the organizer template order."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.model import DelayModel
from predictor.data import load_plan, load_points, load_traffic
from predictor.features import FeatureBuilder


def validate_submission(submission: pd.DataFrame, expected_ids: list[str]) -> None:
    if list(submission.columns) != ["sample_id", "prediction"]:
        raise ValueError("Submission columns must be exactly sample_id;prediction")
    if submission["sample_id"].duplicated().any() or len(submission) != len(expected_ids):
        raise ValueError("Submission must include every sample exactly once")
    if list(submission["sample_id"].astype(str)) != expected_ids:
        raise ValueError("Submission IDs/order must match the template")
    if not np.isfinite(pd.to_numeric(submission["prediction"], errors="coerce")).all():
        raise ValueError("Submission predictions must be finite signed seconds")


def generate(data_dir: Path, model_dir: Path, output: Path) -> pd.DataFrame:
    points = load_points(data_dir / "validate/points.csv")
    builder = FeatureBuilder(load_traffic(data_dir / "validate/traffic.csv"), load_plan(data_dir / "validate/schedule_plan.csv"))
    results = DelayModel(model_dir).predict(builder.build_many(points))
    predictions = {sample_id: result["prediction_delay_s"] for sample_id, result in zip(points["sample_id"], results, strict=True)}
    template = pd.read_csv(data_dir / "sample_submission.csv", sep=";", dtype={"sample_id": str})
    expected_ids = template["sample_id"].tolist()
    if set(expected_ids) != set(points["sample_id"]) or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Template and validate points do not contain the same unique sample IDs")
    submission = pd.DataFrame({"sample_id": expected_ids, "prediction": [predictions[key] for key in expected_ids]})
    validate_submission(submission, expected_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, sep=";", index=False, encoding="utf-8")
    print(f"Wrote {len(submission)} validated predictions to {output}", flush=True)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/submission.csv"))
    args = parser.parse_args()
    generate(args.data_dir, args.model_dir, args.output)


if __name__ == "__main__":
    main()
