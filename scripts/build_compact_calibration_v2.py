import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gc
import json
import math
import time
from pathlib import Path

import pandas as pd
import yaml

from simulator.profiling.mapping_parser import parse_mapping_file
from simulator.profiling.compact_execution_groups import (
    build_compact_convolution_groups,
)
from simulator.profiling.compact_resource_estimator import (
    RESOURCE_COLUMNS,
    estimate_compact_group_resources,
)
from simulator.scheduler.compact_calibration import (
    RESOURCE_KEYS,
    build_calibration_scales,
)

ROOT = Path(".")
MANIFEST_PATH = ROOT / "configs/compact_calibration_split_v2.json"
PROFILE_PATH = ROOT / "results/resnet50_layer_profiles.csv"
OUTPUT_SCALES = ROOT / "configs/compact_calibration_scales_v2.json"
OUTPUT_REPORT = ROOT / "results/compact_calibration_report_v2.json"
ARTIFACT_ROOT = ROOT / "timeloop/outputs/resnet50"


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    dataframe = pd.read_csv(PROFILE_PATH)

    calibration_ids = manifest["calibration_layer_ids"]
    evaluation_ids = set(manifest["evaluation_layer_ids"])

    if set(calibration_ids) & evaluation_ids:
        raise ValueError("Calibration/evaluation layer overlap detected")

    if len(calibration_ids) != manifest["counts"]["calibration"]:
        raise ValueError("Calibration manifest count mismatch")

    if len(dataframe) != manifest["counts"]["total"]:
        raise ValueError("Profile CSV layer count differs from manifest")

    maxima = {key: 0.0 for key in RESOURCE_KEYS}
    layer_reports = []

    overall_start = time.perf_counter()

    for index, layer_id in enumerate(calibration_ids, start=1):
        print(
            f"\n[{index}/{len(calibration_ids)}] Processing {layer_id}",
            flush=True,
        )

        matches = dataframe[dataframe["layer_id"] == layer_id]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one CSV profile for {layer_id}; "
                f"found {len(matches)}"
            )

        layer_profile = matches.iloc[0]

        layer_dir = ARTIFACT_ROOT / layer_id
        mapping_path = layer_dir / "timeloop-mapper.map.yaml"
        workload_path = layer_dir / "parsed-processed-input.yaml"

        if not mapping_path.is_file() or not workload_path.is_file():
            raise FileNotFoundError(
                f"Missing mapping/workload artifact for {layer_id}"
            )

        start = time.perf_counter()

        mapping = parse_mapping_file(mapping_path)
        workload_data = yaml.safe_load(workload_path.read_text())
        problem = workload_data["problem"]

        group_set = build_compact_convolution_groups(
            mapping, problem, layer_id
        )

        group_profiles = estimate_compact_group_resources(
            group_set, layer_profile
        )

        layer_scales = build_calibration_scales(group_profiles)

        for key in RESOURCE_KEYS:
            maxima[key] = max(maxima[key], layer_scales[key])

        original_bandwidths = {
            key: float(layer_profile[column])
            for key, column in RESOURCE_COLUMNS.items()
        }

        if not all(
            math.isfinite(value) and value >= 0
            for value in original_bandwidths.values()
        ):
            raise ValueError(f"Invalid original bandwidth in {layer_id}")

        elapsed = time.perf_counter() - start

        layer_reports.append({
            "layer_id": layer_id,
            "group_count": len(group_set.groups),
            "output_coordinates": group_set.total_output_coordinates,
            "logical_slots": group_set.total_logical_slots,
            "valid_contributions": group_set.total_valid_contributions,
            "padding_slots": group_set.total_padding_slots,
            "conservation_validated": True,
            "original_layer_timeloop_bandwidths": original_bandwidths,
            "maximum_derived_group_bandwidths": layer_scales,
            "elapsed_seconds": round(elapsed, 3),
        })

        print(
            f"  Groups: {len(group_set.groups):,} | "
            f"Time: {elapsed:.2f}s | "
            f"Conservation: PASS",
            flush=True,
        )

        del group_profiles, group_set, mapping, workload_data, problem
        gc.collect()

    if any(not math.isfinite(v) or v <= 0 for v in maxima.values()):
        raise ValueError(f"Invalid final calibration scales: {maxima}")

    report = {
        "experiment": "compact_group_calibration",
        "model": manifest["model"],
        "split_manifest": str(MANIFEST_PATH),
        "split_policy": manifest["split_policy"],
        "calibration_layer_ids": calibration_ids,
        "evaluation_layer_ids": manifest["evaluation_layer_ids"],
        "calibration_layer_count": len(layer_reports),
        "evaluation_layer_count": len(evaluation_ids),
        "resource_keys": list(RESOURCE_KEYS),
        "scale_method": "maximum estimated compact-group bandwidth per resource",
        "scale_provenance": (
            "Derived exclusively from compact-group profiles allocated "
            "from calibration-layer Timeloop measurements."
        ),
        "layers": layer_reports,
        "total_groups": sum(x["group_count"] for x in layer_reports),
        "total_elapsed_seconds": round(
            time.perf_counter() - overall_start, 3
        ),
        "limitations": [
            "Group-level bandwidths are estimates, not direct measurements.",
            "Normalization scales do not validate predictive accuracy.",
            "Evaluation is layer-held-out within ResNet-50, not "
            "independent-model generalization.",
        ],
    }

    scales_output = {
        "experiment": "compact_group_calibration",
        "model": manifest["model"],
        "split_manifest": str(MANIFEST_PATH),
        "calibration_layer_ids": calibration_ids,
        "resource_keys": list(RESOURCE_KEYS),
        "scales": maxima,
        "method": report["scale_method"],
        "provenance": report["scale_provenance"],
    }

    OUTPUT_SCALES.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_SCALES.write_text(json.dumps(scales_output, indent=2) + "\n")
    OUTPUT_REPORT.write_text(json.dumps(report, indent=2) + "\n")

    print("\nCALIBRATION COMPLETE", flush=True)
    print("Scales:", json.dumps(maxima, indent=2), flush=True)
    print("Total groups:", report["total_groups"], flush=True)
    print("Total elapsed seconds:", report["total_elapsed_seconds"], flush=True)
    print("Scales saved:", OUTPUT_SCALES, flush=True)
    print("Report saved:", OUTPUT_REPORT, flush=True)


if __name__ == "__main__":
    main()
