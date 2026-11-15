import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path("/home/workspace/P04")
MODEL_PATH = ROOT / "models/compact_resource_predictor.joblib"
PROFILE_PATH = ROOT / "results/resnet50_layer_profiles.csv"
OUT = ROOT / "results/compact_resource_prediction"

def main():
    artifact = joblib.load(MODEL_PATH)
    metadata = json.loads((OUT / "model_metadata.json").read_text())

    assert metadata["evaluation_targets_used_for_training_or_selection"] is False

    calibration_ids = artifact["calibration_layer_ids"]
    evaluation_ids = artifact["evaluation_layer_ids"]
    targets = artifact["target_columns"]

    profiles = pd.read_csv(PROFILE_PATH).set_index("layer_id")

    calibration_targets = profiles.loc[calibration_ids, targets].astype(float)
    evaluation_targets = profiles.loc[evaluation_ids, targets].astype(float)

    predictions = pd.DataFrame(
        [artifact["model"]["target_medians"]] * len(evaluation_ids),
        index=evaluation_ids,
        columns=targets,
    )

    rows = []
    for target in targets:
        actual = evaluation_targets[target].to_numpy()
        predicted = predictions[target].to_numpy()
        mae = mean_absolute_error(actual, predicted)
        mean_actual = float(np.mean(actual))

        rows.append({
            "target": target,
            "mae": float(mae),
            "mean_actual": mean_actual,
            "normalized_mae": float(mae / max(mean_actual, 1e-12)),
            "calibration_median": float(calibration_targets[target].median()),
            "evaluation_mean": mean_actual,
        })

    metrics = pd.DataFrame(rows)
    predictions.to_csv(OUT / "heldout_predictions.csv", index_label="layer_id")
    metrics.to_csv(OUT / "heldout_metrics.csv", index=False)

    print("Held-out layers:", len(evaluation_ids))
    print("Model:", artifact["model"]["model_name"])
    print("\nHELD-OUT METRICS")
    print(metrics.to_string(index=False))
    print("\nMean normalized MAE:", metrics["normalized_mae"].mean())
    print("\nSaved:", OUT / "heldout_predictions.csv")
    print("Saved:", OUT / "heldout_metrics.csv")

if __name__ == "__main__":
    main()
