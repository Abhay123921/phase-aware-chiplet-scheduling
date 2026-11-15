import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "configs/compact_calibration_split_v2.json"
PROFILE_CSV = ROOT / "results/resnet50_layer_profiles.csv"
ARTIFACT_ROOT = ROOT / "timeloop/outputs/resnet50"
OUTPUT_DIR = ROOT / "results/compact_resource_prediction_v2"
MODEL_DIR = ROOT / "models"

TARGETS = [
    "inputs_shared_bw",
    "weights_shared_bw",
    "outputs_shared_bw",
    "dram_inputs_shared_bw",
    "dram_weights_shared_bw",
    "dram_outputs_shared_bw",
]


def extract_features(layer_id):
    directory = ARTIFACT_ROOT / layer_id

    problem = yaml.safe_load(
        (directory / "parsed-processed-input.yaml").read_text()
    )["problem"]

    mapping = yaml.safe_load(
        (directory / "timeloop-mapper.map.yaml").read_text()
    )["mapping"]

    instance = problem["instance"]
    row = {}

    for key, value in instance.items():
        if isinstance(value, (int, float)):
            row[f"workload_{key}"] = float(value)

    dimensions = {
        key: float(instance[key])
        for key in ("C", "M", "R", "S", "N", "P", "Q", "G")
    }

    row["derived_output_elements"] = (
        dimensions["N"] * dimensions["M"] *
        dimensions["P"] * dimensions["Q"] *
        dimensions["G"]
    )

    row["derived_reduction_size"] = (
        dimensions["C"] * dimensions["R"] *
        dimensions["S"]
    )

    row["derived_logical_operations"] = (
        row["derived_output_elements"] *
        row["derived_reduction_size"]
    )

    for entry in mapping:
        if entry.get("type") not in ("spatial", "temporal"):
            continue

        prefix = f"{entry['target']}_{entry['type']}"

        for token in entry["factors"].split():
            dimension = token.rstrip("0123456789")
            factor = int(token[len(dimension):])
            row[f"{prefix}_{dimension}"] = float(factor)

    return row


def make_models():
    return {
        "median": None,
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=100,
                max_depth=2,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text())
    calibration_ids = manifest["calibration_layer_ids"]
    evaluation_ids = manifest["evaluation_layer_ids"]

    profiles = pd.read_csv(PROFILE_CSV).set_index("layer_id")

    records = []
    for layer_id in calibration_ids + evaluation_ids:
        features = extract_features(layer_id)
        features["layer_id"] = layer_id
        records.append(features)

    feature_frame = pd.DataFrame(records).set_index("layer_id")
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)

    X_cal = feature_frame.loc[calibration_ids].copy()
    X_eval = feature_frame.loc[evaluation_ids].copy()

    # Remove features constant or entirely missing in calibration only.
    usable = [
        col for col in X_cal.columns
        if X_cal[col].notna().any() and X_cal[col].nunique(dropna=True) > 1
    ]


    if len(usable) == 0:
        raise ValueError("No varying calibration features")

    Y_cal = profiles.loc[calibration_ids, TARGETS].astype(float)

    print("Calibration layers:", len(X_cal))
    print("Evaluation layers:", len(X_eval))
    print("Selected varying features:", len(usable))
    print("Features:", usable)

    loo = LeaveOneOut()
    cv_results = []
    models = make_models()

    # Generate the proper fold-specific median baseline.
    median_cv_predictions = np.zeros_like(Y_cal.values, dtype=float)
    fold_features = []

    for train_idx, test_idx in loo.split(X_cal):
        X_train = X_cal.iloc[train_idx]
        Y_train = Y_cal.iloc[train_idx]

        # Feature filtering uses training rows only.
        fold_usable = [
            col for col in X_train.columns
            if X_train[col].notna().any()
            and X_train[col].nunique(dropna=True) > 1
        ]
        if not fold_usable:
            raise ValueError("No varying features in a LOOCV training fold")

        fold_features.append(fold_usable)
        median_cv_predictions[test_idx[0]] = Y_train.median(axis=0).values

    for model_name, model in models.items():
        predictions = np.zeros_like(Y_cal.values, dtype=float)

        if model_name == "median":
            predictions[:] = median_cv_predictions
        else:
            for fold_number, (train_idx, test_idx) in enumerate(loo.split(X_cal)):
                X_train = X_cal.iloc[train_idx]
                X_test = X_cal.iloc[test_idx]
                Y_train = Y_cal.iloc[train_idx]

                fold_usable = fold_features[fold_number]
                fitted = clone(model)
                fitted.fit(X_train[fold_usable], Y_train)
                predictions[test_idx[0]] = fitted.predict(
                    X_test[fold_usable]
                )[0]

        for j, target in enumerate(TARGETS):
            mae = mean_absolute_error(
                Y_cal.iloc[:, j].values,
                predictions[:, j],
            )
            baseline_mae = mean_absolute_error(
                Y_cal.iloc[:, j].values,
                median_cv_predictions[:, j],
            )
            cv_results.append({
                "model": model_name,
                "target": target,
                "loocv_mae": float(mae),
                "median_baseline_mae": float(baseline_mae),
                "normalized_mae": float(
                    mae / max(Y_cal.iloc[:, j].mean(), 1e-12)
                ),
            })

    cv_frame = pd.DataFrame(cv_results)
    cv_frame.to_csv(OUTPUT_DIR / "loocv_results.csv", index=False)

    summary = (
        cv_frame.groupby("model")["normalized_mae"]
        .mean()
        .sort_values()
    )

    print("\nLOOCV NORMALIZED MAE (LOWER IS BETTER)")
    print(summary.to_string())

    selected_name = summary.index[0]
    selected_model = models[selected_name]

    # Fit selected model on all calibration layers.
    if selected_name == "median":
        fitted_model = {
            "model_name": "median",
            "target_medians": Y_cal.median().to_dict(),
        }
    else:
        fitted_model = clone(selected_model)
        fitted_model.fit(X_cal, Y_cal)
        fitted_model = {
            "model_name": selected_name,
            "pipeline": fitted_model,
        }

    artifact = {
        "model": fitted_model,
        "feature_columns": usable,
        "target_columns": TARGETS,
        "calibration_layer_ids": calibration_ids,
        "evaluation_layer_ids": evaluation_ids,
        "selection_metric": "mean LOOCV normalized MAE",
        "selected_model": selected_name,
        "limitations": [
            "Only 17 calibration layers are available.",
            "LOOCV estimates are high variance.",
            "Targets are layer-level Timeloop measurements.",
            "No direct compact-group measurements are available.",
        ],
    }

    joblib.dump(artifact, MODEL_DIR / "compact_resource_predictor_v2.joblib")

    metadata = {
        "selected_model": selected_name,
        "feature_columns": usable,
        "target_columns": TARGETS,
        "calibration_layer_ids": calibration_ids,
        "evaluation_layer_ids": evaluation_ids,
        "selection_metric": "mean LOOCV normalized MAE",
        "mean_normalized_mae_by_model": summary.to_dict(),
        "evaluation_targets_used_for_training_or_selection": False,
    }

    (OUTPUT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("\nSelected model:", selected_name)
    print("LOOCV results:", OUTPUT_DIR / "loocv_results.csv")
    print("Saved model:", MODEL_DIR / "compact_resource_predictor_v2.joblib")
    print("Saved metadata:", OUTPUT_DIR / "model_metadata.json")


if __name__ == "__main__":
    main()
