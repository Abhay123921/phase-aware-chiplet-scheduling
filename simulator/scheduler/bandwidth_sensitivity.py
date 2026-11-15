from pathlib import Path
import pandas as pd

from simulator.hardware.model import load_hardware

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "results/resnet50_layer_profiles.csv"
OUTPUT_PATH = ROOT / "results/bandwidth_sensitivity.csv"


def main():
    profiles = pd.read_csv(PROFILE_PATH)
    hardware = load_hardware()

    # Sensitivity assumptions, NOT measured reference bandwidth capacities.
    # Each scenario asks how results change if the effective bandwidth
    # capacity were calibrated to these values.
    reference_bandwidth_scenarios = [16.0, 24.0, 32.0]

    rows = []

    for _, layer in profiles.iterrows():
        ref_cycles = float(layer["summary_cycles"])
        demand = float(layer["buffer_bandwidth"])

        for reference_bw in reference_bandwidth_scenarios:
            for chiplet in hardware.chiplets.values():
                compute_cycles = (
                    ref_cycles / chiplet.compute_capacity
                )

                # Hypothetical effective bandwidth capacity, relative
                # to the scenario's assumed reference capacity.
                chiplet_bw = (
                    reference_bw * chiplet.memory_bandwidth
                )

                memory_cycles = (
                    ref_cycles * demand / chiplet_bw
                )

                estimated_cycles = max(
                    compute_cycles,
                    memory_cycles,
                )

                rows.append({
                    "layer_id": layer["layer_id"],
                    "resource_regime": int(layer["resource_regime"]),
                    "chiplet": chiplet.name,
                    "reference_bandwidth_assumption": reference_bw,
                    "buffer_bandwidth_demand_feature": demand,
                    "compute_cycles": compute_cycles,
                    "hypothetical_memory_cycles": memory_cycles,
                    "sensitivity_estimated_cycles": estimated_cycles,
                    "is_memory_limited": memory_cycles > compute_cycles,
                })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("===== BANDWIDTH SENSITIVITY =====")
    print(f"Rows: {len(result)}")
    print(f"Missing cells: {int(result.isna().sum().sum())}")
    print(f"Saved: {OUTPUT_PATH}")
    print()

    print("===== MEAN ESTIMATED CYCLES BY SCENARIO =====")
    summary = (
        result.groupby(
            ["reference_bandwidth_assumption", "chiplet"]
        )["sensitivity_estimated_cycles"]
        .mean()
        .unstack("chiplet")
        .round(0)
    )
    print(summary.to_string())
    print()

    print("===== MEMORY-LIMITED LAYER COUNTS =====")
    limited = (
        result.groupby(
            ["reference_bandwidth_assumption", "chiplet"]
        )["is_memory_limited"]
        .sum()
        .unstack("chiplet")
    )
    print(limited.to_string())

    assert len(result) == (
        len(profiles)
        * len(hardware.chiplets)
        * len(reference_bandwidth_scenarios)
    )
    assert not result.isna().any().any()

    print("\nPASS: bandwidth sensitivity sweep completed.")


if __name__ == "__main__":
    main()
