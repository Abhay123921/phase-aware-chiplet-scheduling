import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator.profiling.compact_execution_groups import (
    CompactExecutionGroup,
    CompactExecutionGroupSet,
)
from simulator.profiling.compact_resource_estimator import (
    CompactGroupResourceProfile,
)
from simulator.scheduler.compact_group_scheduler import simulate_compact_groups

OUT = ROOT / "results/controlled_phase_e7"
SCALES = {k: 1.0 for k in (
    "input_bw", "weight_bw", "output_bw",
    "dram_input_bw", "dram_weight_bw", "dram_output_bw"
)}
N = 32


def build_workload(name, local):
    groups = tuple(
        CompactExecutionGroup(
            group_id=f"{name}_{i}",
            output_origin=(0, 0, 0, 0, 0),
            output_shape=(1, 1, 1, 1, 1),
            spatial_factors=(1, 1, 1, 1, 1),
            logical_slots=1,
            valid_contributions=1,
            padding_slots=0,
        )
        for i in range(N)
    )
    group_set = CompactExecutionGroupSet(
        layer_id=name,
        output_axes=("N", "M", "Q", "P", "G"),
        dimensions=(1, 1, 1, 1, 1, 1, 1, 1),
        spatial_factors=(1, 1, 1, 1, 1),
        groups=groups,
        total_output_coordinates=N,
        total_logical_slots=N,
        total_valid_contributions=N,
        total_padding_slots=0,
    )
    profiles = tuple(
        CompactGroupResourceProfile(
            group_id=f"{name}_{i}",
            layer_id=name,
            estimated_compute_work=100.0,
            reference_cycles=100.0,
            estimated_input_accesses=1.0,
            estimated_weight_accesses=1.0,
            estimated_output_accesses=1.0,
            input_bw=local,
            weight_bw=local,
            output_bw=local,
            dram_input_bw=0.1,
            dram_weight_bw=0.1,
            dram_output_bw=0.1,
        )
        for i in range(N)
    )
    return [(group_set, profiles)]


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUT}")

    workloads = {
        "compute_workload": build_workload("compute_phase", 0.1),
        "memory_workload": build_workload("memory_phase", 2.0),
    }
    rows = []
    details = []

    for policy in ("static_rr", "reactive", "causal_phase_aware"):
        result = simulate_compact_groups(
            workload_layers=workloads,
            policy=policy,
            calibration_scales=SCALES,
        )
        record = {
            "policy": policy,
            "makespan": result.makespan,
            "completed_groups": result.completed_groups,
            "remaps": result.remaps,
            "chiplet_group_counts": result.chiplet_group_counts,
            "workload_chiplet_counts": result.workload_chiplet_counts,
            "layer_placement_counts": result.layer_placement_counts,
        }
        details.append(record)
        rows.append({
            "policy": policy,
            "makespan": result.makespan,
            "completed_groups": result.completed_groups,
            "remaps": result.remaps,
            "compute_chiplet_counts": json.dumps(
                result.workload_chiplet_counts["compute_workload"],
                sort_keys=True,
            ),
            "memory_chiplet_counts": json.dumps(
                result.workload_chiplet_counts["memory_workload"],
                sort_keys=True,
            ),
        })
        print(
            f"{policy}: makespan={result.makespan:.3f}, "
            f"groups={result.completed_groups}, remaps={result.remaps}"
        )
        print("  compute:", result.workload_chiplet_counts["compute_workload"])
        print("  memory: ", result.workload_chiplet_counts["memory_workload"])

    OUT.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(OUT / "controlled_phase_e7.csv", index=False)
    (OUT / "controlled_phase_e7.json").write_text(
        json.dumps({"experiment": "E7_controlled_compute_memory_phases",
                    "groups_per_workload": N, "results": details}, indent=2)
    )
    print("E7 completed:", OUT)


if __name__ == "__main__":
    main()
