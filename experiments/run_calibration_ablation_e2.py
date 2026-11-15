import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

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
OFFLINE_SCALE_PATH = ROOT / "configs/compact_calibration_scales_v2.json"
CAUSAL_SCALE_PATH = ROOT / "configs/compact_calibration_scales_causal_v2.json"
SPLIT_PATH = ROOT / "configs/compact_calibration_split_v2.json"
OUTPUT_DIR = ROOT / "results/calibration_ablation_e2"

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

REQUIRED_SCALE_KEYS = {
    "input_bw",
    "weight_bw",
    "output_bw",
    "dram_input_bw",
    "dram_weight_bw",
    "dram_output_bw",
}


def load_scales(path):
    data = json.loads(path.read_text())
    scales = data["scales"]

    missing = REQUIRED_SCALE_KEYS - set(scales)
    if missing:
        raise ValueError(f"Missing scale keys in {path}: {sorted(missing)}")

    for key in REQUIRED_SCALE_KEYS:
        value = float(scales[key])
        if value <= 0:
            raise ValueError(f"Invalid scale {key} in {path}: {value}")
        scales[key] = value

    return data


def load_layer(layer_id, profiles, evaluation_ids):
    if layer_id not in evaluation_ids:
        raise ValueError(f"{layer_id} is not in the held-out evaluation split")

    directory = ARTIFACT_ROOT / layer_id
    mapping = parse_mapping_file(directory / "timeloop-mapper.map.yaml")
    problem_data = yaml.safe_load(
        (directory / "parsed-processed-input.yaml").read_text()
    )

    group_set = build_compact_convolution_groups(
        mapping=mapping,
        problem=problem_data["problem"],
        layer_id=layer_id,
    )

    if not group_set.validate():
        raise ValueError(f"Group validation failed for {layer_id}")

    return group_set, profiles.loc[layer_id]


def summarize_result(profile_name, scale_name, policy_label, result):
    return {
        "profile_source": profile_name,
        "scale_source": scale_name,
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

    offline_scale_data = load_scales(OFFLINE_SCALE_PATH)
    causal_scale_data = load_scales(CAUSAL_SCALE_PATH)

    scale_conditions = {
        "offline_v2": offline_scale_data["scales"],
        "causal_v2": causal_scale_data["scales"],
    }

    split = json.loads(SPLIT_PATH.read_text())
    calibration_ids = set(split["calibration_layer_ids"])
    evaluation_ids = set(split["evaluation_layer_ids"])

    if calibration_ids.intersection(evaluation_ids):
        raise ValueError("Calibration/evaluation split is not disjoint")

    if not set(LAYER_IDS).issubset(evaluation_ids):
        raise ValueError("Selected workloads are not held out")

    if set(LAYER_IDS).intersection(calibration_ids):
        raise ValueError("Evaluation workload appears in calibration set")

    hardware = load_hardware()

    print("E2 held-out verification: PASSED")
    print("Calibration layers:", len(calibration_ids))
    print("Evaluation layers:", len(evaluation_ids))
    print("Selected layers:", LAYER_IDS)
    print("Selected predictor:", model_artifact["selected_model"])

    layer_data = {}

    for layer_id in LAYER_IDS:
        group_set, layer_profile = load_layer(
            layer_id, profiles, evaluation_ids
        )

        layer_data[layer_id] = {
            "group_set": group_set,
            "offline": estimate_compact_group_resources(
                group_set, layer_profile
            ),
            "metadata_causal": estimate_causal_compact_group_resources(
                group_set, calibration_baselines
            ),
        }

        print(
            f"{layer_id}: groups={len(group_set.groups)}, "
            f"logical_work={group_set.total_logical_slots:,}"
        )

    profile_conditions = [
        ("offline_timeloop_reference", "offline"),
        ("metadata_causal", "metadata_causal"),
    ]

    rows = []
    detailed_results = []

    for profile_name, profile_key in profile_conditions:
        workload_layers = {
            layer_id: [
                (
                    layer_data[layer_id]["group_set"],
                    layer_data[layer_id][profile_key],
                )
            ]
            for layer_id in LAYER_IDS
        }

        for scale_name, scales in scale_conditions.items():
            for policy_label, policy_name in POLICIES:
                print(
                    f"\nRunning profile={profile_name}, "
                    f"scales={scale_name}, policy={policy_label}"
                )

                result = simulate_compact_groups(
                    workload_layers=workload_layers,
                    policy=policy_name,
                    hardware=hardware,
                    calibration_scales=scales,
                    lookahead_groups=4,
                    resource_aggregation="max",
                )

                summary = summarize_result(
                    profile_name, scale_name, policy_label, result
                )
                detailed_results.append(summary)

                rows.append({
                    "profile_source": profile_name,
                    "scale_source": scale_name,
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

                print(
                    f"  makespan={result.makespan:,.3f}, "
                    f"groups={result.completed_groups:,}, "
                    f"remaps={result.remaps}"
                )

    csv_path = OUTPUT_DIR / "calibration_ablation_e2.csv"
    json_path = OUTPUT_DIR / "calibration_ablation_e2.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    payload = {
        "experiment": "E2_controlled_calibration_ablation",
        "selected_layers": LAYER_IDS,
        "calibration_layer_count": len(calibration_ids),
        "evaluation_layer_count": len(evaluation_ids),
        "selected_model": model_artifact["selected_model"],
        "profile_conditions": [
            name for name, _ in profile_conditions
        ],
        "scale_conditions": {
            name: {
                "source_file": str(
                    OFFLINE_SCALE_PATH
                    if name == "offline_v2"
                    else CAUSAL_SCALE_PATH
                ),
                "scales": values,
            }
            for name, values in scale_conditions.items()
        },
        "policies": [label for label, _ in POLICIES],
        "run_count": len(detailed_results),
        "results": detailed_results,
        "limitations": [
            "Only two held-out ResNet-50 layers are evaluated concurrently.",
            "Offline group profiles allocate layer-level Timeloop measurements "
            "using logical access proxies.",
            "Metadata group profiles use calibration medians and a nominal "
            "16 logical-operations-per-cycle rate.",
            "Cross-combinations of profile source and scale source are "
            "normalization ablations, not independently optimized calibrations.",
            "The comparison changes both bandwidth and reference-cycle estimates.",
            "Oracle phase-aware scheduling accesses future group profiles.",
            "The simulator is an abstract scheduling model, not a "
            "cycle-accurate hardware implementation.",
            "No layer transition occurs in these single-layer workloads, "
            "so remapping overhead is not exercised.",
        ],
    }

    json_path.write_text(json.dumps(payload, indent=2))

    print("\nE2 experiment completed.")
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("Total runs:", len(detailed_results))


if __name__ == "__main__":
    main()
