from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "results/resnet50_layer_profiles.csv"
OUTPUT_PATH = ROOT / "results/temporal_overlap_experiment.csv"

RESOURCE_COLUMNS = {
    "input_bw": "inputs_shared_bw",
    "weight_bw": "weights_shared_bw",
    "output_bw": "outputs_shared_bw",
    "dram_input_bw": "dram_inputs_shared_bw",
    "dram_weight_bw": "dram_weights_shared_bw",
    "dram_output_bw": "dram_outputs_shared_bw",
}

CAPACITY_FACTORS = [1.0, 1.5, 2.0]
ARRIVAL_OFFSETS = [0.0, 0.25, 0.50]


def make_trace(df, workload_name, arrival_time):
    """Create sequential layer intervals using reference cycle counts."""
    trace = []
    clock = float(arrival_time)

    for row in df.to_dict("records"):
        duration = float(row["summary_cycles"])

        item = {
            "workload": workload_name,
            "layer_id": row["layer_id"],
            "regime": int(row["resource_regime"]),
            "start": clock,
            "end": clock + duration,
        }

        for resource, column in RESOURCE_COLUMNS.items():
            item[resource] = float(row[column])

        trace.append(item)
        clock += duration

    return trace


def analyze_overlap(trace_a, trace_b, capacities):
    """Measure pressure over intervals; do not model slowdown."""
    traces = trace_a + trace_b
    boundaries = sorted(
        {item["start"] for item in traces}
        | {item["end"] for item in traces}
    )

    total_time = 0.0
    overloaded_time = 0.0
    resource_overloaded_time = {
        resource: 0.0 for resource in RESOURCE_COLUMNS
    }
    max_pressure = {resource: 0.0 for resource in RESOURCE_COLUMNS}
    max_any_pressure = 0.0
    simultaneous_workload_time = 0.0

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        duration = end - start
        if duration <= 0:
            continue

        active = [
            item for item in traces
            if item["start"] < end and item["end"] > start
        ]

        if not active:
            continue

        total_time += duration

        active_workloads = {item["workload"] for item in active}
        if len(active_workloads) == 2:
            simultaneous_workload_time += duration

        pressures = {}
        for resource in RESOURCE_COLUMNS:
            demand = sum(item[resource] for item in active)
            pressure = demand / capacities[resource]
            pressures[resource] = pressure
            max_pressure[resource] = max(
                max_pressure[resource], pressure
            )

            if pressure > 1.0:
                resource_overloaded_time[resource] += duration

        interval_max = max(pressures.values())
        max_any_pressure = max(max_any_pressure, interval_max)

        if interval_max > 1.0:
            overloaded_time += duration

    return {
        "simulated_union_time_cycles": total_time,
        "both_workloads_active_time_cycles": simultaneous_workload_time,
        "any_resource_overloaded_time_cycles": overloaded_time,
        "any_resource_overloaded_time_fraction": (
            overloaded_time / total_time if total_time else 0.0
        ),
        "max_any_resource_pressure": max_any_pressure,
        **{
            f"max_pressure_{r}": v
            for r, v in max_pressure.items()
        },
        **{
            f"overloaded_time_fraction_{r}": (
                v / total_time if total_time else 0.0
            )
            for r, v in resource_overloaded_time.items()
        },
    }


def main():
    df = pd.read_csv(PROFILE_PATH).sort_values("layer_index")
    if len(df) != 53:
        raise ValueError(f"Expected 53 layers, found {len(df)}")

    # Use observed single-layer peak demand only as a normalization scale.
    # It is NOT asserted to be physical maximum sustainable capacity.
    peak = {
        resource: float(df[column].max())
        for resource, column in RESOURCE_COLUMNS.items()
    }

    reference_total_cycles = float(df["summary_cycles"].sum())
    results = []

    for offset_fraction in ARRIVAL_OFFSETS:
        trace_a = make_trace(df, "A", 0.0)
        trace_b = make_trace(
            df,
            "B",
            offset_fraction * reference_total_cycles,
        )

        for factor in CAPACITY_FACTORS:
            capacities = {
                resource: peak[resource] * factor
                for resource in RESOURCE_COLUMNS
            }

            metrics = analyze_overlap(
                trace_a, trace_b, capacities
            )

            results.append({
                "arrival_offset_fraction": offset_fraction,
                "capacity_factor": factor,
                **metrics,
            })

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)

    print("===== TEMPORAL OVERLAP EXPERIMENT =====")
    print("Workloads: 2 copies of the same ResNet-50 layer sequence")
    print(f"Layers per workload: {len(df)}")
    print(f"Reference cycles per workload: {reference_total_cycles:.0f}")
    print(f"Arrival offsets: {ARRIVAL_OFFSETS}")
    print(f"Capacity factors: {CAPACITY_FACTORS}")
    print(f"Output rows: {len(out)}")
    print(f"Missing cells: {int(out.isna().sum().sum())}")
    print(f"Saved: {OUTPUT_PATH}")
    print()

    print("===== OVERLOADED TIME FRACTION =====")
    print(
        out[
            [
                "arrival_offset_fraction",
                "capacity_factor",
                "both_workloads_active_time_cycles",
                "any_resource_overloaded_time_fraction",
                "max_any_resource_pressure",
            ]
        ].round(4).to_string(index=False)
    )
    print()

    print("===== RESOURCE-SPECIFIC OVERLOADED TIME =====")
    resource_columns = [
        f"overloaded_time_fraction_{r}"
        for r in RESOURCE_COLUMNS
    ]
    print(
        out[
            [
                "arrival_offset_fraction",
                "capacity_factor",
                *resource_columns,
            ]
        ].round(4).to_string(index=False)
    )

    assert len(out) == len(ARRIVAL_OFFSETS) * len(CAPACITY_FACTORS)
    assert not out.isna().any().any()
    assert (
        out["any_resource_overloaded_time_fraction"].between(0, 1).all()
    )

    print("\nPASS: temporal overlap trace analysis completed.")


if __name__ == "__main__":
    main()
