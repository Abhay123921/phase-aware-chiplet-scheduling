"""E6: Sequential phase-varying ResNet layers concurrent with GEMM."""

import json
import runpy
import sys
from pathlib import Path

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
from simulator.scheduler.compact_group_scheduler import simulate_compact_groups
from simulator.hardware.model import load_hardware


CNN_LAYER_IDS = (
    "45_layer4_0_conv2",
    "46_layer4_0_conv3",
)
CNN_WORKLOAD_ID = "resnet50_sequential"
GEMM_WORKLOAD_ID = "gemm_12x10x18"

PROFILE_CSV = ROOT / "results/resnet50_layer_profiles.csv"
ARTIFACT_ROOT = ROOT / "timeloop/outputs/resnet50"
SPLIT_PATH = ROOT / "configs/compact_calibration_split_v2.json"
SCALE_PATH = ROOT / "configs/compact_calibration_scales_v2.json"
E5_SCRIPT = ROOT / "experiments/run_mixed_cnn_gemm_e5.py"
OUTPUT_DIR = ROOT / "results/phase_varying_mixed_e6_layer_diagnostics"

POLICIES = (
    ("static", "static"),
    ("static_rr", "static_rr"),
    ("reactive", "reactive"),
    ("causal_phase_aware", "causal_phase_aware"),
)
REQUIRED_SCALE_KEYS = {
    "input_bw", "weight_bw", "output_bw",
    "dram_input_bw", "dram_weight_bw", "dram_output_bw",
}


def load_cnn_layer(layer_id, profiles, evaluation_ids):
    if layer_id not in evaluation_ids:
        raise ValueError(f"{layer_id} is not in the held-out evaluation split")

    directory = ARTIFACT_ROOT / layer_id
    mapping = parse_mapping_file(
        directory / "timeloop-mapper.map.yaml"
    )
    problem_data = yaml.safe_load(
        (directory / "parsed-processed-input.yaml").read_text()
    )
    group_set = build_compact_convolution_groups(
        mapping=mapping,
        problem=problem_data["problem"],
        layer_id=layer_id,
    )
    if not group_set.validate():
        raise ValueError(f"Compact group validation failed: {layer_id}")

    group_profiles = estimate_compact_group_resources(
        group_set,
        profiles.loc[layer_id],
    )
    if len(group_set.groups) != len(group_profiles):
        raise ValueError(f"Group/profile mismatch: {layer_id}")

    return group_set, group_profiles


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT_DIR}"
        )

    if not E5_SCRIPT.is_file():
        raise FileNotFoundError(E5_SCRIPT)

    split = json.loads(SPLIT_PATH.read_text())
    calibration_ids = set(split["calibration_layer_ids"])
    evaluation_ids = set(split["evaluation_layer_ids"])

    if calibration_ids.intersection(evaluation_ids):
        raise ValueError("Calibration/evaluation split is not disjoint")
    if not set(CNN_LAYER_IDS).issubset(evaluation_ids):
        raise ValueError("A selected CNN layer is not held out")
    if set(CNN_LAYER_IDS).intersection(calibration_ids):
        raise ValueError("A selected CNN layer appears in calibration data")

    scale_data = json.loads(SCALE_PATH.read_text())
    scales = scale_data["scales"]
    missing = REQUIRED_SCALE_KEYS - set(scales)
    if missing:
        raise ValueError(f"Missing calibration scales: {sorted(missing)}")
    scales = {key: float(scales[key]) for key in REQUIRED_SCALE_KEYS}
    if any(value <= 0 for value in scales.values()):
        raise ValueError("Calibration scales must all be positive")

    profiles = pd.read_csv(PROFILE_CSV).set_index("layer_id")

    # Keep the two CNN layers sequential under one workload key.
    cnn_layers = []
    cnn_group_counts = {}
    for layer_id in CNN_LAYER_IDS:
        group_set, group_profiles = load_cnn_layer(
            layer_id, profiles, evaluation_ids
        )
        cnn_layers.append((group_set, group_profiles))
        cnn_group_counts[layer_id] = len(group_set.groups)
        print(
            f"{layer_id}: groups={len(group_set.groups):,}, "
            f"profiles={len(group_profiles):,}"
        )

    # Reuse the previously tested GEMM construction and estimator.
    e5 = runpy.run_path(str(E5_SCRIPT))
    gemm_group_set, gemm_profiles = e5["build_gemm_workload"]()

    workload_layers = {
        CNN_WORKLOAD_ID: cnn_layers,
        GEMM_WORKLOAD_ID: [(gemm_group_set, gemm_profiles)],
    }

    expected_groups = (
        sum(cnn_group_counts.values()) + len(gemm_group_set.groups)
    )
    hardware = load_hardware()

    print("\nE6 workload construction complete")
    print("CNN layer order:", list(CNN_LAYER_IDS))
    print("CNN groups:", sum(cnn_group_counts.values()))
    print("GEMM groups:", len(gemm_group_set.groups))
    print("Combined groups:", expected_groups)
    print("Held-out split validation: PASSED")
    print("Policies:", [label for label, _ in POLICIES])

    rows = []
    detailed_results = []

    for policy_label, policy_name in POLICIES:
        print(f"\nRunning policy: {policy_label}")
        result = simulate_compact_groups(
            workload_layers=workload_layers,
            policy=policy_name,
            hardware=hardware,
            calibration_scales=scales,
            lookahead_groups=4,
            resource_aggregation="max",
        )

        if result.completed_groups != expected_groups:
            raise RuntimeError(
                f"{policy_label}: completed {result.completed_groups}, "
                f"expected {expected_groups}"
            )
        if result.makespan <= 0:
            raise RuntimeError(f"{policy_label}: invalid makespan")

        record = {
            "policy": policy_label,
            "makespan": result.makespan,
            "completed_groups": result.completed_groups,
            "remaps": result.remaps,
            "average_group_duration": result.average_group_duration,
            "chiplet_group_counts": result.chiplet_group_counts,
            "chiplet_busy_time": result.chiplet_busy_time,
            "chiplet_utilization": result.chiplet_utilization,
            "workload_chiplet_counts": result.workload_chiplet_counts,
            "layer_placement_counts": result.layer_placement_counts,
        }
        detailed_results.append(record)

        rows.append({
            "policy": policy_label,
            "makespan": result.makespan,
            "completed_groups": result.completed_groups,
            "remaps": result.remaps,
            "cnn_conv2_groups": cnn_group_counts[CNN_LAYER_IDS[0]],
            "cnn_conv3_groups": cnn_group_counts[CNN_LAYER_IDS[1]],
            "gemm_groups": len(gemm_group_set.groups),
            "cnn_chiplet_counts": json.dumps(
                result.workload_chiplet_counts[CNN_WORKLOAD_ID],
                sort_keys=True,
            ),
            "cnn_layer_placement_counts": json.dumps(
                result.layer_placement_counts[CNN_WORKLOAD_ID], sort_keys=True,
            ),
            "gemm_chiplet_counts": json.dumps(
                result.workload_chiplet_counts[GEMM_WORKLOAD_ID],
                sort_keys=True,
            ),
            "chiplet_group_counts": json.dumps(
                result.chiplet_group_counts,
                sort_keys=True,
            ),
            "chiplet_utilization": json.dumps(
                result.chiplet_utilization,
                sort_keys=True,
            ),
        })

        print(
            f"  makespan={result.makespan:,.3f}, "
            f"groups={result.completed_groups:,}, "
            f"remaps={result.remaps}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    csv_path = OUTPUT_DIR / "phase_varying_mixed_e6.csv"
    json_path = OUTPUT_DIR / "phase_varying_mixed_e6.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    payload = {
        "experiment": "E6_sequential_phase_varying_cnn_with_concurrent_gemm",
        "cnn_workload_id": CNN_WORKLOAD_ID,
        "cnn_layer_order": list(CNN_LAYER_IDS),
        "cnn_group_counts": cnn_group_counts,
        "gemm_workload_id": GEMM_WORKLOAD_ID,
        "gemm_group_count": len(gemm_group_set.groups),
        "combined_group_count": expected_groups,
        "profile_source": (
            "layer-level Timeloop measurements allocated to logical groups"
        ),
        "scale_source": str(SCALE_PATH.relative_to(ROOT)),
        "held_out_layers": list(CNN_LAYER_IDS),
        "calibration_layer_count": len(calibration_ids),
        "evaluation_layer_count": len(evaluation_ids),
        "policies": [label for label, _ in POLICIES],
        "hardware": hardware.name,
        "results": detailed_results,
        "limitations": [
            "Abstract event-driven scheduling, not cycle-accurate simulation.",
            "No NoP contention or physical data migration is modeled.",
            "CNN group metrics are allocated from layer-level Timeloop measurements.",
            "GEMM group metrics are allocated from layer-level measurements using logical access shares.",
            "GEMM groups are candidate logical scheduling units.",
            "The causal phase-aware policy classifies the current ready group; it does not predict future layer phases.",
            "The CNN layer barrier is preserved, while GEMM progresses as an independent concurrent workload.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2))

    print("\nE6 completed.")
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
