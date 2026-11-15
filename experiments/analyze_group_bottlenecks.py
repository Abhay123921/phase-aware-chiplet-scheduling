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
from simulator.hardware.model import load_hardware


PROFILE_CSV = ROOT / "results/resnet50_layer_profiles.csv"
ARTIFACT_ROOT = ROOT / "timeloop/outputs/resnet50"
MODEL_PATH = ROOT / "models/compact_resource_predictor_v2.joblib"
SCALE_PATH = ROOT / "configs/compact_calibration_scales_v2.json"

LAYER_IDS = [
    "46_layer4_0_conv3",
    "45_layer4_0_conv2",
]

LOCAL = ("input_bw", "weight_bw", "output_bw")
DRAM = ("dram_input_bw", "dram_weight_bw", "dram_output_bw")


def load_layer(layer_id, profiles):
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
        raise ValueError(f"Invalid group set: {layer_id}")

    return group_set, profiles.loc[layer_id]


def analyze(estimator, layer_id, profiles, hardware, scales, baselines):
    group_set, layer_profile = load_layer(layer_id, profiles)

    if estimator == "offline_reference":
        group_profiles = estimate_compact_group_resources(
            group_set, layer_profile
        )
    else:
        group_profiles = estimate_causal_compact_group_resources(
            group_set, baselines
        )

    rows = []

    for chiplet_name, chiplet in hardware.chiplets.items():
        counts = {
            "compute": 0,
            "local_memory": 0,
            "dram": 0,
        }

        sums = {
            "compute": 0.0,
            "local_memory": 0.0,
            "dram": 0.0,
        }

        for profile in group_profiles:
            cycles = float(profile.reference_cycles)

            local_pressure = max(
                getattr(profile, key) / scales[key]
                for key in LOCAL
            ) / chiplet.memory_bandwidth

            dram_pressure = max(
                getattr(profile, key) / scales[key]
                for key in DRAM
            )

            components = {
                "compute": cycles / chiplet.compute_capacity,
                "local_memory": cycles * local_pressure,
                "dram": cycles * dram_pressure,
            }

            bottleneck = max(
                components,
                key=components.get,
            )

            counts[bottleneck] += 1

            for key, value in components.items():
                sums[key] += value

        total = len(group_profiles)

        row = {
            "estimator": estimator,
            "layer": layer_id,
            "chiplet": chiplet_name,
            "groups": total,
        }

        for key in counts:
            row[f"{key}_dominance_pct"] = (
                100.0 * counts[key] / total
            )
            row[f"mean_{key}_cycles"] = sums[key] / total

        rows.append(row)

    return rows


def main():
    profiles = pd.read_csv(PROFILE_CSV).set_index("layer_id")

    artifact = joblib.load(MODEL_PATH)
    baselines = artifact["model"]["target_medians"]

    scales_data = json.loads(SCALE_PATH.read_text())
    scales = scales_data["scales"]

    hardware = load_hardware()

    rows = []

    for estimator in (
        "offline_reference",
        "metadata_calibrated",
    ):
        for layer_id in LAYER_IDS:
            print(
                f"Analyzing {estimator}: {layer_id}",
                flush=True,
            )

            rows.extend(
                analyze(
                    estimator,
                    layer_id,
                    profiles,
                    hardware,
                    scales,
                    baselines,
                )
            )

    output_dir = ROOT / "results/group_bottleneck_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame(rows)

    output_path = output_dir / "bottleneck_summary.csv"
    result.to_csv(output_path, index=False)

    columns = [
        "estimator",
        "layer",
        "chiplet",
        "compute_dominance_pct",
        "local_memory_dominance_pct",
        "dram_dominance_pct",
    ]

    print("\n=== BOTTLENECK DOMINANCE (%) ===")
    print(result[columns].round(2).to_string(index=False))

    print("\n=== MEAN COMPONENT DURATIONS ===")
    duration_columns = [
        "estimator",
        "layer",
        "chiplet",
        "mean_compute_cycles",
        "mean_local_memory_cycles",
        "mean_dram_cycles",
    ]

    print(
        result[duration_columns]
        .round(3)
        .to_string(index=False)
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
