"""E5: Concurrent real ResNet-50 CNN and mapped GEMM workloads."""

import json
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
from simulator.profiling.operator_graph import (
    OperatorNode,
    TensorNode,
    build_operator_dependency_graph,
)
from simulator.profiling.gemm_spatial_execution_groups import (
    build_gemm_spatial_execution_groups,
)
from simulator.profiling.execution_group_profiler import (
    build_execution_group_profiles,
)
from simulator.profiling.execution_group_tensor_access import (
    build_execution_group_tensor_accesses,
)
from simulator.profiling.gemm_compact_resource_estimator import (
    estimate_gemm_execution_group_resources,
)
from simulator.profiling.gemm_scheduler_adapter import (
    adapt_gemm_execution_groups_for_scheduler,
)
from simulator.scheduler.compact_group_scheduler import simulate_compact_groups
from simulator.hardware.model import load_hardware


CNN_LAYER = "46_layer4_0_conv3"
GEMM_ID = "gemm_12x10x18"
GEMM_DIMS = {"M": 12, "N": 10, "K": 18}
POLICIES = ("static", "static_rr", "reactive")
PROFILE_CSV = ROOT / "results/resnet50_layer_profiles.csv"
CNN_ROOT = ROOT / "timeloop/outputs/resnet50" / CNN_LAYER
GEMM_ROOT = ROOT / "timeloop/outputs/gemm" / GEMM_ID
SCALE_PATH = ROOT / "configs/compact_calibration_scales_v2.json"
OUTPUT_DIR = ROOT / "results/mixed_cnn_gemm_e5"


def build_gemm_workload():
    mapping = parse_mapping_file(
        GEMM_ROOT / "timeloop-mapper.map.yaml"
    )
    operator_id = "gemm0"
    operator_graph = build_operator_dependency_graph(
        workload_id=GEMM_ID,
        operators=(
            OperatorNode(
                operator_id=operator_id,
                operator_type="gemm",
                input_tensors=("Inputs", "Weights"),
                output_tensors=("C",),
                attributes={"workload_dimensions": GEMM_DIMS},
            ),
        ),
        tensors=(
            TensorNode(
                tensor_id="Inputs",
                shape=(12, 18),
                is_external=True,
            ),
            TensorNode(
                tensor_id="Weights",
                shape=(18, 10),
                is_external=True,
            ),
            TensorNode(
                tensor_id="C",
                shape=(12, 10),
                producer_operator_id=operator_id,
            ),
        ),
    )
    graph, ldgs = build_gemm_spatial_execution_groups(
        mapping=mapping,
        workload_dimensions=GEMM_DIMS,
        workload_id=GEMM_ID,
        operator_graph=operator_graph,
        operator_id=operator_id,
    )
    tile_profiles = build_execution_group_profiles(
        execution_group_graph=graph,
        operator_graph=operator_graph,
        ldg_graphs=ldgs,
    )
    accesses = build_execution_group_tensor_accesses(
        execution_group_graph=graph,
        operator_graph=operator_graph,
        ldg_graphs=ldgs,
    )
    layer_profile = {
        "reference_compute_work": 2160,
        "reference_execution_cycles": 144,
        "buffer_input_shared_bw": 4.50,
        "buffer_weight_shared_bw": 6.25,
        "buffer_output_shared_bw": 2.50,
        "dram_input_transfers": 216,
        "dram_weight_transfers": 180,
        "dram_output_transfers": 120,
    }
    profiles = estimate_gemm_execution_group_resources(
        execution_group_graph=graph,
        group_profiles=tile_profiles,
        group_accesses=accesses,
        layer_profile=layer_profile,
    )
    return adapt_gemm_execution_groups_for_scheduler(
        execution_group_graph=graph,
        gemm_profiles=profiles,
    )


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT_DIR}"
        )

    profile_table = pd.read_csv(PROFILE_CSV).set_index("layer_id")
    cnn_layer_profile = profile_table.loc[CNN_LAYER]
    cnn_mapping = parse_mapping_file(
        CNN_ROOT / "timeloop-mapper.map.yaml"
    )
    cnn_problem = yaml.safe_load(
        (CNN_ROOT / "parsed-processed-input.yaml").read_text()
    )["problem"]
    cnn_group_set = build_compact_convolution_groups(
        mapping=cnn_mapping,
        problem=cnn_problem,
        layer_id=CNN_LAYER,
    )
    if not cnn_group_set.validate():
        raise ValueError("CNN compact execution groups failed validation")
    cnn_profiles = estimate_compact_group_resources(
        cnn_group_set,
        cnn_layer_profile,
    )

    gemm_group_set, gemm_profiles = build_gemm_workload()

    workload_layers = {
        CNN_LAYER: [(cnn_group_set, cnn_profiles)],
        GEMM_ID: [(gemm_group_set, gemm_profiles)],
    }
    expected_groups = (
        len(cnn_group_set.groups) + len(gemm_group_set.groups)
    )
    scales = json.loads(SCALE_PATH.read_text())["scales"]
    hardware = load_hardware()

    print("E5 workload construction complete")
    print(f"CNN groups: {len(cnn_group_set.groups):,}")
    print(f"GEMM groups: {len(gemm_group_set.groups):,}")
    print(f"Combined groups: {expected_groups:,}")
    print("Policies:", POLICIES)

    rows = []
    detailed = []
    for policy in POLICIES:
        print(f"Running policy: {policy}")
        result = simulate_compact_groups(
            workload_layers=workload_layers,
            policy=policy,
            hardware=hardware,
            calibration_scales=scales,
            lookahead_groups=4,
            resource_aggregation="max",
        )
        if result.completed_groups != expected_groups:
            raise RuntimeError(
                f"{policy}: expected {expected_groups} groups, "
                f"completed {result.completed_groups}"
            )
        if result.makespan <= 0:
            raise RuntimeError(f"{policy}: non-positive makespan")

        record = {
            "policy": policy,
            "makespan": result.makespan,
            "completed_groups": result.completed_groups,
            "remaps": result.remaps,
            "average_group_duration": result.average_group_duration,
            "chiplet_group_counts": result.chiplet_group_counts,
            "chiplet_busy_time": result.chiplet_busy_time,
            "workload_chiplet_counts": result.workload_chiplet_counts,
            "chiplet_utilization": result.chiplet_utilization,
        }
        detailed.append(record)
        rows.append({
            "policy": policy,
            "makespan": result.makespan,
            "completed_groups": result.completed_groups,
            "remaps": result.remaps,
            "cnn_groups": len(cnn_group_set.groups),
            "gemm_groups": len(gemm_group_set.groups),
            "cnn_chiplet_counts": json.dumps(
                result.workload_chiplet_counts[CNN_LAYER],
                sort_keys=True,
            ),
            "gemm_chiplet_counts": json.dumps(
                result.workload_chiplet_counts[GEMM_ID],
                sort_keys=True,
            ),
            "chiplet_group_counts": json.dumps(
                result.chiplet_group_counts,
                sort_keys=True,
            ),
        })
        print(
            f"  makespan={result.makespan:,.3f}, "
            f"groups={result.completed_groups:,}, "
            f"remaps={result.remaps}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    csv_path = OUTPUT_DIR / "mixed_cnn_gemm_e5.csv"
    json_path = OUTPUT_DIR / "mixed_cnn_gemm_e5.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    payload = {
        "experiment": "E5_mixed_real_cnn_gemm_concurrent_scheduling",
        "cnn_layer": CNN_LAYER,
        "cnn_group_count": len(cnn_group_set.groups),
        "gemm_workload": GEMM_ID,
        "gemm_dimensions": GEMM_DIMS,
        "gemm_group_count": len(gemm_group_set.groups),
        "combined_group_count": expected_groups,
        "profile_source": "layer-level Timeloop measurements allocated to logical groups",
        "scale_source": str(SCALE_PATH.relative_to(ROOT)),
        "policies": list(POLICIES),
        "hardware": "P04_4Chiplet_Baseline",
        "results": detailed,
        "limitations": [
            "Abstract event-driven scheduling, not cycle-accurate simulation.",
            "No NoP contention or physical data migration is modeled.",
            "CNN group metrics are allocated from layer-level measurements.",
            "GEMM group metrics are allocated from layer-level Timeloop measurements using logical access shares.",
            "GEMM groups are candidate logical scheduling units, not claims of physical execution order.",
            "CNN and GEMM use different workload shapes and group granularities.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print("E5 completed.")
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
