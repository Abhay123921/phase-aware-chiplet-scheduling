import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator.profiling.mapping_parser import parse_mapping_file
from simulator.profiling.compact_execution_groups import (
    build_compact_convolution_groups,
)
from simulator.profiling.compact_resource_estimator import (
    estimate_compact_group_resources,
)
from simulator.profiling.causal_compact_resource_estimator import (
    estimate_causal_compact_group_resources,
)
from simulator.scheduler.compact_group_scheduler import (
    simulate_compact_groups,
)
from simulator.hardware.model import load_hardware


PROFILE_CSV = ROOT / "results/resnet50_layer_profiles.csv"
ARTIFACT_ROOT = ROOT / "timeloop/outputs/resnet50"
MODEL_PATH = ROOT / "models/compact_resource_predictor_v2.joblib"
SCALE_PATH = ROOT / "configs/compact_calibration_scales_v2.json"
SPLIT_PATH = ROOT / "configs/compact_calibration_split_v2.json"
OUTPUT_DIR = ROOT / "results/causal_scheduler_comparison"

LAYER_IDS = [
    "46_layer4_0_conv3",
    "45_layer4_0_conv2",
]

POLICIES = [
    ("static", "static"),
    ("static_rr", "static_rr"),
    ("reactive", "reactive"),
    ("oracle_phase_aware", "phase_aware"),
]


def load_layer(layer_id, profiles, evaluation_ids):
    if layer_id not in evaluation_ids:
        raise ValueError(f"{layer_id} is not in the held-out evaluation split")

    directory = ARTIFACT_ROOT / layer_id
    mapping_path = directory / "timeloop-mapper.map.yaml"
    problem_path = directory / "parsed-processed-input.yaml"

    mapping = parse_mapping_file(mapping_path)
    problem_data = yaml.safe_load(problem_path.read_text())
    problem = problem_data["problem"]

    group_set = build_compact_convolution_groups(
        mapping=mapping,
        problem=problem,
        layer_id=layer_id,
    )

    if not group_set.validate():
        raise ValueError(f"Group validation failed for {layer_id}")

    layer_profile = profiles.loc[layer_id]

    return group_set, layer_profile


def summarize_result(estimator_name, policy_label, result):
    return {
        "estimator": estimator_name,
        "policy": policy_label,
        "makespan": result.makespan,
        "completed_groups": result.completed_groups,
        "remaps": result.remaps,
        "average_group_duration": result.average_group_duration,
        "chiplet_utilization": result.chiplet_utilization,
        "chiplet_group_counts": result.chiplet_group_counts,
        "workload_chiplet_counts": result.workload_chiplet_counts,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    profiles = pd.read_csv(PROFILE_CSV).set_index("layer_id")

    model_artifact = joblib.load(MODEL_PATH)
    calibration_baselines = model_artifact["model"]["target_medians"]

    scales_data = json.loads(SCALE_PATH.read_text())
    calibration_scales = scales_data["scales"]

    split = json.loads(SPLIT_PATH.read_text())
    calibration_ids = set(split["calibration_layer_ids"])
    evaluation_ids = set(split["evaluation_layer_ids"])

    if calibration_ids.intersection(evaluation_ids):
        raise ValueError("Calibration/evaluation split is not disjoint")

    if not set(LAYER_IDS).issubset(evaluation_ids):
        raise ValueError("Requested workloads are not both held out")

    if set(LAYER_IDS).intersection(calibration_ids):
        raise ValueError("Evaluation workload appears in calibration set")

    hardware = load_hardware()

    print("Held-out workload verification: PASSED")
    print("Calibration layers:", len(calibration_ids))
    print("Evaluation layers:", len(evaluation_ids))
    print("Selected layers:", LAYER_IDS)
    print("Selected model:", model_artifact["selected_model"])
    print("Calibration scales:", calibration_scales)

    layer_data = {}
    for layer_id in LAYER_IDS:
        group_set, layer_profile = load_layer(
            layer_id, profiles, evaluation_ids
        )

        offline_profiles = estimate_compact_group_resources(
            group_set, layer_profile
        )

        causal_profiles = estimate_causal_compact_group_resources(
            group_set, calibration_baselines
        )

        layer_data[layer_id] = {
            "group_set": group_set,
            "offline": offline_profiles,
            "causal": causal_profiles,
        }

        print(
            f"{layer_id}: groups={len(group_set.groups)}, "
            f"logical_work={group_set.total_logical_slots:,}, "
            f"valid={group_set.total_valid_contributions:,}, "
            f"padding={group_set.total_padding_slots:,}"
        )

    estimator_profiles = {
        "offline_timeloop_reference": "offline",
        "metadata_calibrated": "causal",
    }

    rows = []
    detailed_results = []

    for estimator_name, profile_key in estimator_profiles.items():
        workload_layers = {}

        for layer_id in LAYER_IDS:
            workload_layers[layer_id] = [
                (
                    layer_data[layer_id]["group_set"],
                    layer_data[layer_id][profile_key],
                )
            ]

        for policy_label, policy_name in POLICIES:
            print(
                f"\nRunning estimator={estimator_name}, "
                f"policy={policy_label}"
            )

            result = simulate_compact_groups(
                workload_layers=workload_layers,
                policy=policy_name,
                hardware=hardware,
                calibration_scales=calibration_scales,
                lookahead_groups=4,
                resource_aggregation="max",
            )

            summary = summarize_result(
                estimator_name, policy_label, result
            )

            rows.append({
                "estimator": estimator_name,
                "policy": policy_label,
                "makespan": result.makespan,
                "completed_groups": result.completed_groups,
                "remaps": result.remaps,
                "average_group_duration": result.average_group_duration,
                "chiplet_utilization": json.dumps(
                    result.chiplet_utilization, sort_keys=True
                ),
                "chiplet_group_counts": json.dumps(
                    result.chiplet_group_counts, sort_keys=True
                ),
                "workload_chiplet_counts": json.dumps(
                    result.workload_chiplet_counts, sort_keys=True
                ),
            })

            detailed_results.append(summary)

            print(
                f"  makespan={result.makespan:,.3f}, "
                f"groups={result.completed_groups:,}, "
                f"remaps={result.remaps}"
            )
            print("  utilization:", result.chiplet_utilization)

    csv_path = OUTPUT_DIR / "scheduler_comparison.csv"
    json_path = OUTPUT_DIR / "scheduler_comparison.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    payload = {
        "experiment": "offline_reference_vs_metadata_calibrated_scheduler",
        "selected_layers": LAYER_IDS,
        "calibration_layer_count": len(calibration_ids),
        "evaluation_layer_count": len(evaluation_ids),
        "selected_model": model_artifact["selected_model"],
        "calibration_scales": calibration_scales,
        "policies": [label for label, _ in POLICIES],
        "results": detailed_results,
        "limitations": [
            "Offline profiles allocate held-out Timeloop layer measurements "
            "to groups using logical access proxies.",
            "Metadata profiles use global calibration medians and a nominal "
            "16 logical-operations-per-cycle rate.",
            "The comparison changes both bandwidth and cycle estimates; "
            "it is not a bandwidth-only accuracy comparison.",
            "The phase-aware policy uses future group profiles and is "
            "therefore labelled oracle_phase_aware.",
            "The simulator is an abstract scheduling model, not a "
            "cycle-accurate hardware implementation.",
        ],
    }

    json_path.write_text(json.dumps(payload, indent=2))

    print("\nExperiment completed.")
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("Total runs:", len(rows))


if __name__ == "__main__":
    main()
