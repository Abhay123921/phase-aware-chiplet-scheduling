"""E4: Locality-aware scheduling comparison on held-out ResNet-50 layers."""

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
from simulator.profiling.convolution_adapter import (
    ConvolutionProjectionAdapter,
)
from simulator.profiling.group_locality import analyze_group_locality
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
OUTPUT_DIR = ROOT / "results/locality_aware_e4"

LAYER_IDS = [
    "46_layer4_0_conv3",
    "45_layer4_0_conv2",
]

POLICIES = [
    ("static", "static"),
    ("static_rr", "static_rr"),
    ("reactive", "reactive"),
    ("causal_phase_aware", "causal_phase_aware"),
    ("oracle_phase_aware", "phase_aware"),
    ("locality_aware", "locality_aware"),
]

REQUIRED_SCALE_KEYS = {
    "input_bw",
    "weight_bw",
    "output_bw",
    "dram_input_bw",
    "dram_weight_bw",
    "dram_output_bw",
}

# Both selected Timeloop workloads use 16-bit data, i.e. 2 bytes/element.
ELEMENT_BYTES = 2

LOCALITY_WINDOW_SIZE = 16
LOCALITY_DECAY = 0.8
LOCALITY_TOLERANCE = 0.05


def load_scales(path):
    data = json.loads(path.read_text())
    scales = data["scales"]
    missing = REQUIRED_SCALE_KEYS - set(scales)
    if missing:
        raise ValueError(
            f"Missing scale keys in {path}: {sorted(missing)}"
        )
    for key in REQUIRED_SCALE_KEYS:
        value = float(scales[key])
        if value <= 0:
            raise ValueError(f"Invalid scale {key} in {path}: {value}")
        scales[key] = value
    return data


def load_layer(layer_id, profiles, evaluation_ids):
    if layer_id not in evaluation_ids:
        raise ValueError(
            f"{layer_id} is not in the held-out evaluation split"
        )

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
        raise ValueError(f"Group validation failed for {layer_id}")

    return group_set, profiles.loc[layer_id], problem_data["problem"]


def make_locality_feature_provider(adapters):
    """Build exact group features only when the scheduler requests them."""

    calls = {"count": 0}

    def provider(workload_name, layer_index, group_index, group_set, group):
        adapter = adapters[group_set.layer_id]
        feature = analyze_group_locality(
            adapter=adapter,
            group_set=group_set,
            group=group,
            element_bytes=ELEMENT_BYTES,
        )
        calls["count"] += 1
        if calls["count"] % 5000 == 0:
            print(
                f"  Lazy locality features computed: {calls['count']:,}"
            )
        return feature

    return provider, calls


def summarize_result(
    profile_name, scale_name, policy_label, result
):
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
        "locality_window_size": LOCALITY_WINDOW_SIZE,
        "locality_decay": LOCALITY_DECAY,
        "locality_tolerance": LOCALITY_TOLERANCE,
    }


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing E4 output directory: "
            f"{OUTPUT_DIR}"
        )

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
        raise ValueError(
            "Evaluation workload appears in calibration set"
        )

    hardware = load_hardware()

    print("E4 held-out verification: PASSED")
    print("Calibration layers:", len(calibration_ids))
    print("Evaluation layers:", len(evaluation_ids))
    print("Selected layers:", LAYER_IDS)
    print("Selected predictor:", model_artifact["selected_model"])
    print("Locality element size:", ELEMENT_BYTES, "bytes")
    print(
        "Locality settings:",
        {
            "window_size": LOCALITY_WINDOW_SIZE,
            "decay": LOCALITY_DECAY,
            "tolerance": LOCALITY_TOLERANCE,
        },
    )

    layer_data = {}
    locality_adapters = {}

    for layer_id in LAYER_IDS:
        group_set, layer_profile, problem = load_layer(
            layer_id, profiles, evaluation_ids
        )

        locality_adapters[layer_id] = (
            ConvolutionProjectionAdapter.from_problem(problem)
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
            f"{layer_id}: groups={len(group_set.groups):,}, "
            f"logical_work={group_set.total_logical_slots:,}, "
            "projection adapter ready"
        )

    locality_provider, locality_call_count = (
        make_locality_feature_provider(locality_adapters)
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

                kwargs = {
                    "workload_layers": workload_layers,
                    "policy": policy_name,
                    "hardware": hardware,
                    "calibration_scales": scales,
                    "lookahead_groups": 4,
                    "resource_aggregation": "max",
                }

                if policy_name == "locality_aware":
                    kwargs.update({
                        "locality_feature_provider": locality_provider,
                        "locality_window_size": LOCALITY_WINDOW_SIZE,
                        "locality_decay": LOCALITY_DECAY,
                        "locality_tolerance": LOCALITY_TOLERANCE,
                    })

                result = simulate_compact_groups(**kwargs)

                summary = summarize_result(
                    profile_name,
                    scale_name,
                    policy_label,
                    result,
                )
                detailed_results.append(summary)

                rows.append({
                    "profile_source": profile_name,
                    "scale_source": scale_name,
                    "policy": policy_label,
                    "makespan": result.makespan,
                    "completed_groups": result.completed_groups,
                    "remaps": result.remaps,
                    "average_group_duration": (
                        result.average_group_duration
                    ),
                    "chiplet_utilization": json.dumps(
                        result.chiplet_utilization,
                        sort_keys=True,
                    ),
                    "chiplet_group_counts": json.dumps(
                        result.chiplet_group_counts,
                        sort_keys=True,
                    ),
                    "workload_chiplet_counts": json.dumps(
                        result.workload_chiplet_counts,
                        sort_keys=True,
                    ),
                    "locality_window_size": LOCALITY_WINDOW_SIZE,
                    "locality_decay": LOCALITY_DECAY,
                    "locality_tolerance": LOCALITY_TOLERANCE,
                })

                print(
                    f"  makespan={result.makespan:,.3f}, "
                    f"groups={result.completed_groups:,}, "
                    f"remaps={result.remaps}"
                )

    expected_runs = (
        len(profile_conditions)
        * len(scale_conditions)
        * len(POLICIES)
    )
    if len(detailed_results) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} runs, "
            f"completed {len(detailed_results)}"
        )

    # Create the output directory only after all runs have completed.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    csv_path = OUTPUT_DIR / "locality_aware_e4.csv"
    json_path = OUTPUT_DIR / "locality_aware_e4.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    payload = {
        "experiment": "E4_locality_aware_scheduling_comparison",
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
        "locality_configuration": {
            "element_bytes": ELEMENT_BYTES,
            "window_size": LOCALITY_WINDOW_SIZE,
            "decay": LOCALITY_DECAY,
            "tolerance": LOCALITY_TOLERANCE,
            "selection_rule": (
                "Locality is used as a tie-breaker among candidates "
                "whose duration-plus-remap score is within the "
                "configured tolerance of the best candidate."
            ),
        },
        "run_count": len(detailed_results),
        "locality_feature_computations": locality_call_count["count"],
        "results": detailed_results,
        "limitations": [
            "Only two held-out ResNet-50 convolution layers are "
            "evaluated concurrently.",
            "The locality extractor uses exact projected input and "
            "weight coordinates for compact convolution groups.",
            "The selected layers use 16-bit data; element_bytes is "
            "therefore set to 2 for these experiments.",
            "Geometric overlap represents potential reuse, not "
            "measured cache hits or guaranteed data movement savings.",
            "The locality policy uses overlap only as a placement "
            "tie-breaker; it does not subtract execution time or "
            "model actual cache residency, eviction, or SRAM capacity.",
            "Locality history is bounded and workload-specific; "
            "comparisons are restricted to matching layer identities.",
            "Offline group profiles allocate layer-level Timeloop "
            "measurements using logical access proxies.",
            "Metadata group profiles use calibration medians and a "
            "nominal 16 logical-operations-per-cycle rate.",
            "Cross-combinations of profile source and scale source "
            "are normalization ablations, not independently "
            "optimized calibrations.",
            "The simulator is an abstract scheduling model, not a "
            "cycle-accurate hardware implementation.",
            "No layer transition occurs in these single-layer "
            "workloads, so remapping overhead is not exercised.",
        ],
    }

    json_path.write_text(json.dumps(payload, indent=2))

    print("\nE4 experiment completed.")
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("Total runs:", len(detailed_results))


if __name__ == "__main__":
    main()
