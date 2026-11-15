import json
import sys
from pathlib import Path

import joblib
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator.profiling.mapping_parser import parse_mapping_file
from simulator.profiling.compact_execution_groups import (
    build_compact_convolution_groups,
)
from simulator.profiling.causal_compact_resource_estimator import (
    estimate_causal_compact_group_resources,
)

SPLIT_PATH = ROOT / "configs/compact_calibration_split_v2.json"
MODEL_PATH = ROOT / "models/compact_resource_predictor_v2.joblib"
ARTIFACT_ROOT = ROOT / "timeloop/outputs/resnet50"
OUTPUT_PATH = ROOT / "configs/compact_calibration_scales_causal_v2.json"

PROFILE_KEYS = (
    "input_bw",
    "weight_bw",
    "output_bw",
    "dram_input_bw",
    "dram_weight_bw",
    "dram_output_bw",
)


def main():
    split = json.loads(SPLIT_PATH.read_text())
    artifact = joblib.load(MODEL_PATH)

    calibration_ids = split["calibration_layer_ids"]
    if set(calibration_ids) != set(artifact["calibration_layer_ids"]):
        raise ValueError("Calibration IDs differ between split and model artifact")

    baselines = artifact["model"]["target_medians"]
    maxima = {key: 0.0 for key in PROFILE_KEYS}
    total_groups = 0

    print("Calibration layers:", len(calibration_ids))
    print("Estimator:", artifact["selected_model"])

    for index, layer_id in enumerate(calibration_ids, 1):
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
            raise ValueError(f"Group validation failed: {layer_id}")

        profiles = estimate_causal_compact_group_resources(
            group_set,
            baselines,
        )

        for key in PROFILE_KEYS:
            maxima[key] = max(
                maxima[key],
                max(float(getattr(profile, key)) for profile in profiles),
            )

        total_groups += len(profiles)
        print(
            f"[{index}/{len(calibration_ids)}] "
            f"{layer_id}: {len(profiles):,} groups"
        )

        del profiles
        del group_set

    output = {
        "experiment": "causal_compact_group_calibration",
        "model": "ResNet-50",
        "split_manifest": "configs/compact_calibration_split_v2.json",
        "calibration_layer_count": len(calibration_ids),
        "total_groups": total_groups,
        "baseline_model": artifact["selected_model"],
        "baseline_source": "models/compact_resource_predictor_v2.joblib",
        "baseline_values": baselines,
        "scales": maxima,
        "method": "maximum causal-estimated compact-group bandwidth per resource",
        "provenance": (
            "Derived exclusively from causal metadata-based group estimates "
            "on the 17 calibration layers."
        ),
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")
    print("\nSaved:", OUTPUT_PATH)
    print("Causal scales:")
    for key, value in maxima.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
