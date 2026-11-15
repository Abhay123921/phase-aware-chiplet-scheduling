import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_resource_prediction_v2 import (
    extract_features,
    TARGETS,
    MANIFEST,
    PROFILE_CSV,
)

OUTPUT_DIR = ROOT / "results/compact_resource_prediction_v3"

FEATURE_GROUPS = {
    "workload_only": [
        "workload_C",
        "workload_M",
        "workload_P",
        "workload_Q",
        "workload_R",
        "workload_S",
        "derived_output_elements",
        "derived_reduction_size",
    ],
    "mapping_only": [
        "inter_PE_spatial_spatial_C",
        "inter_PE_spatial_spatial_M",
        "inter_PE_spatial_spatial_P",
        "inter_PE_spatial_spatial_Q",
        "DRAM_temporal_C",
        "DRAM_temporal_M",
        "DRAM_temporal_P",
        "DRAM_temporal_Q",
    ],
}

FEATURE_GROUPS["combined"] = (
    FEATURE_GROUPS["workload_only"]
    + FEATURE_GROUPS["mapping_only"]
)


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text())
    calibration_ids = manifest["calibration_layer_ids"]

    profiles = pd.read_csv(PROFILE_CSV).set_index("layer_id")
    Y = profiles.loc[calibration_ids, TARGETS].astype(float)

    records = []
    for layer_id in calibration_ids:
        row = extract_features(layer_id)
        row["layer_id"] = layer_id
        records.append(row)

    X = pd.DataFrame(records).set_index("layer_id")
    X = X.replace([np.inf, -np.inf], np.nan)

    loo = LeaveOneOut()
    rows = []

    # Fold-specific median baseline.
    median_predictions = np.zeros_like(Y.values, dtype=float)

    for train_idx, test_idx in loo.split(Y):
        median_predictions[test_idx[0]] = (
            Y.iloc[train_idx].median(axis=0).values
        )

    for group_name, requested_features in FEATURE_GROUPS.items():
        missing = set(requested_features) - set(X.columns)
        if missing:
            raise ValueError(
                f"Missing features for {group_name}: {sorted(missing)}"
            )

        X_group = X[requested_features].copy()
        predictions = np.zeros_like(Y.values, dtype=float)

        for train_idx, test_idx in loo.split(X_group):
            X_train = X_group.iloc[train_idx]
            X_test = X_group.iloc[test_idx]
            Y_train = Y.iloc[train_idx]

            # Feature filtering uses training rows only.
            usable = [
                col for col in X_train.columns
                if X_train[col].notna().any()
                and X_train[col].nunique(dropna=True) > 1
            ]

            if not usable:
                predictions[test_idx[0]] = (
                    Y_train.median(axis=0).values
                )
                continue

            model = clone(make_model())
            model.fit(X_train[usable], Y_train)

            predictions[test_idx[0]] = model.predict(
                X_test[usable]
            )[0]

        for j, target in enumerate(TARGETS):
            mae = mean_absolute_error(
                Y.iloc[:, j].values,
                predictions[:, j],
            )
            baseline_mae = mean_absolute_error(
                Y.iloc[:, j].values,
                median_predictions[:, j],
            )

            rows.append({
                "feature_group": group_name,
                "target": target,
                "loocv_mae": float(mae),
                "median_baseline_mae": float(baseline_mae),
                "normalized_mae": float(
                    mae / max(Y.iloc[:, j].mean(), 1e-12)
                ),
                "improvement_vs_median_pct": float(
                    100.0 * (baseline_mae - mae)
                    / max(baseline_mae, 1e-12)
                ),
            })

    for j, target in enumerate(TARGETS):
        baseline_mae = mean_absolute_error(
            Y.iloc[:, j].values,
            median_predictions[:, j],
        )
        rows.append({
            "feature_group": "median_baseline",
            "target": target,
            "loocv_mae": float(baseline_mae),
            "median_baseline_mae": float(baseline_mae),
            "normalized_mae": float(
                baseline_mae / max(Y.iloc[:, j].mean(), 1e-12)
            ),
            "improvement_vs_median_pct": 0.0,
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "loocv_results.csv", index=False)

    summary = (
        result.groupby("feature_group")["normalized_mae"]
        .mean()
        .sort_values()
    )

    print("Calibration layers:", len(calibration_ids))
    print("Targets:", len(TARGETS))
    print("\n=== MEAN NORMALIZED MAE (LOWER IS BETTER) ===")
    print(summary.round(4).to_string())

    print("\n=== PER-TARGET RESULTS ===")
    print(
        result.pivot(
            index="target",
            columns="feature_group",
            values="normalized_mae",
        ).round(4).to_string()
    )

    print("\n=== IMPROVEMENT VS MEDIAN (%) ===")
    print(
        result[result["feature_group"] != "median_baseline"]
        .pivot(
            index="target",
            columns="feature_group",
            values="improvement_vs_median_pct",
        ).round(2).to_string()
    )

    print("\nSaved:", OUTPUT_DIR / "loocv_results.csv")


if __name__ == "__main__":
    main()
