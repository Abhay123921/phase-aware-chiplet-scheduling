from itertools import combinations
from pathlib import Path

import pandas as pd

from simulator.interference.resource_contention import (
    WorkloadDemand,
    evaluate_contention,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "results/resnet50_layer_profiles.csv"
OUTPUT_PATH = ROOT / "results/profile_contention_experiment.csv"

RESOURCE_COLUMNS = {
    "input_bw": "inputs_shared_bw",
    "weight_bw": "weights_shared_bw",
    "output_bw": "outputs_shared_bw",
    "dram_input_bw": "dram_inputs_shared_bw",
    "dram_weight_bw": "dram_weights_shared_bw",
    "dram_output_bw": "dram_outputs_shared_bw",
}


def main():
    df = pd.read_csv(PROFILE_PATH)

    required = [
        "layer_id",
        "resource_regime",
        *RESOURCE_COLUMNS.values(),
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required profile columns: {missing}")

    # Reference single-layer peak demand per resource.
    # This is a normalization point for stress testing, NOT a measured
    # maximum sustainable hardware capacity.
    peak_demand = {
        resource: float(df[column].max())
        for resource, column in RESOURCE_COLUMNS.items()
    }

    # Explicit hypothetical capacity scenarios relative to observed
    # single-layer peaks. These do not represent measured hardware.
    capacity_factors = [1.0, 1.5, 2.0]

    rows = []

    records = df.to_dict("records")
    for a, b in combinations(records, 2):
        demand_a = {
            resource: float(a[column])
            for resource, column in RESOURCE_COLUMNS.items()
        }
        demand_b = {
            resource: float(b[column])
            for resource, column in RESOURCE_COLUMNS.items()
        }

        wa = WorkloadDemand(a["layer_id"], demand_a)
        wb = WorkloadDemand(b["layer_id"], demand_b)

        for factor in capacity_factors:
            capacities = {
                resource: peak_demand[resource] * factor
                for resource in RESOURCE_COLUMNS
            }

            result = evaluate_contention([wa, wb], capacities)
            pressure = result["pressure"]

            row = {
                "layer_a": a["layer_id"],
                "layer_b": b["layer_id"],
                "regime_a": int(a["resource_regime"]),
                "regime_b": int(b["resource_regime"]),
                "capacity_factor": factor,
                "max_resource_pressure": result["max_pressure"],
                "overloaded_resource_count": len(
                    result["overloaded_resources"]
                ),
                "overloaded_resources": ",".join(
                    result["overloaded_resources"]
                ),
            }

            for resource, value in pressure.items():
                row[f"pressure_{resource}"] = value

            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)

    print("===== PROFILE-BASED CONTENTION EXPERIMENT =====")
    print(f"Layers: {len(df)}")
    print(f"Distinct layer pairs: {len(list(combinations(range(len(df)), 2)))}")
    print(f"Capacity scenarios: {capacity_factors}")
    print(f"Output rows: {len(out)}")
    print(f"Missing cells: {int(out.isna().sum().sum())}")
    print(f"Saved: {OUTPUT_PATH}")
    print()

    print("===== NORMALIZATION PEAKS (WORDS/CYCLE) =====")
    for resource, peak in peak_demand.items():
        print(f"{resource:20s}: {peak:.3f}")
    print()

    print("===== MEAN MAXIMUM RESOURCE PRESSURE BY SCENARIO =====")
    print(
        out.groupby("capacity_factor")["max_resource_pressure"]
        .agg(["mean", "median", "max"])
        .round(3)
        .to_string()
    )
    print()

    print("===== OVERLOAD FREQUENCY BY SCENARIO =====")
    print(
        out.groupby("capacity_factor")["overloaded_resource_count"]
        .agg(
            pairs_with_any_overload=lambda s: int((s > 0).sum()),
            total_pairs="count",
        )
        .to_string()
    )
    print()

    print("===== MOST PRESSURED PAIRS AT CAPACITY FACTOR 1.0 =====")
    sample = out[out["capacity_factor"] == 1.0].nlargest(
        10, "max_resource_pressure"
    )
    print(
        sample[
            [
                "layer_a",
                "regime_a",
                "layer_b",
                "regime_b",
                "max_resource_pressure",
                "overloaded_resources",
            ]
        ].to_string(index=False)
    )

    assert len(out) == 53 * 52 // 2 * len(capacity_factors)
    assert not out.isna().any().any()
    print("\nPASS: real-profile pairwise stress test completed.")


if __name__ == "__main__":
    main()
