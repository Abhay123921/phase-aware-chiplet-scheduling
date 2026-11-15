from pathlib import Path

import pandas as pd

import simulator.scheduler.phase_aware_scheduler as scheduler
from simulator.hardware.model import load_hardware


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "results/scheduler_sensitivity.csv"

REMAP_COSTS = [0.00, 0.01, 0.05, 0.10, 0.20]
LOOKAHEADS = [1, 2, 3, 4, 6]


def main():
    profiles = scheduler.load_profiles()
    hardware = load_hardware()
    chiplets = hardware.chiplets

    resource_scales = scheduler.build_resource_scales(profiles)
    costs = scheduler.precompute_costs(
        profiles, chiplets, resource_scales
    )

    static_chiplet, _ = scheduler.choose_static_mapping(
        profiles, chiplets, costs
    )

    rows = []

    # Static mapping is invariant to remapping cost.
    static_trace = scheduler.run_policy(
        "static",
        profiles,
        chiplets,
        costs,
        static_chiplet=static_chiplet,
    )
    static_total = float(
        static_trace["modeled_total_cycles"].sum()
    )

    rows.append({
        "policy": "static",
        "remap_cost_fraction": 0.0,
        "lookahead_layers": 1,
        "execution_cycles": float(
            static_trace["estimated_execution_cycles"].sum()
        ),
        "remapping_count": int(
            static_trace["remapped"].sum()
        ),
        "remap_overhead_cycles": float(
            static_trace["remap_overhead_cycles"].sum()
        ),
        "modeled_total_cycles": static_total,
        "saving_vs_static_percent": 0.0,
    })

    for remap_fraction in REMAP_COSTS:
        scheduler.REMAP_COST_FRACTION = remap_fraction

        reactive_trace = scheduler.run_policy(
            "reactive",
            profiles,
            chiplets,
            costs,
        )
        reactive_total = float(
            reactive_trace["modeled_total_cycles"].sum()
        )

        rows.append({
            "policy": "reactive",
            "remap_cost_fraction": remap_fraction,
            "lookahead_layers": 1,
            "execution_cycles": float(
                reactive_trace["estimated_execution_cycles"].sum()
            ),
            "remapping_count": int(
                reactive_trace["remapped"].sum()
            ),
            "remap_overhead_cycles": float(
                reactive_trace["remap_overhead_cycles"].sum()
            ),
            "modeled_total_cycles": reactive_total,
            "saving_vs_static_percent": (
                100.0 * (static_total - reactive_total) / static_total
            ),
        })

        for horizon in LOOKAHEADS:
            phase_trace = scheduler.run_policy(
                "phase_aware",
                profiles,
                chiplets,
                costs,
                lookahead_layers=horizon,
            )
            phase_total = float(
                phase_trace["modeled_total_cycles"].sum()
            )

            rows.append({
                "policy": "phase_aware",
                "remap_cost_fraction": remap_fraction,
                "lookahead_layers": horizon,
                "execution_cycles": float(
                    phase_trace["estimated_execution_cycles"].sum()
                ),
                "remapping_count": int(
                    phase_trace["remapped"].sum()
                ),
                "remap_overhead_cycles": float(
                    phase_trace["remap_overhead_cycles"].sum()
                ),
                "modeled_total_cycles": phase_total,
                "saving_vs_static_percent": (
                    100.0 * (static_total - phase_total) / static_total
                ),
            })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("===== SCHEDULER SENSITIVITY EXPERIMENT =====")
    print(f"Layers: {len(profiles)}")
    print(f"Static mapping: {static_chiplet}")
    print(f"Remapping cost fractions: {REMAP_COSTS}")
    print(f"Lookahead horizons: {LOOKAHEADS}")
    print("Costs are model estimates, not measured hardware latency.")
    print()

    print("===== PHASE-AWARE RESULTS =====")
    phase = result[result["policy"] == "phase_aware"]
    print(
        phase[
            [
                "remap_cost_fraction",
                "lookahead_layers",
                "remapping_count",
                "remap_overhead_cycles",
                "modeled_total_cycles",
                "saving_vs_static_percent",
            ]
        ].round(4).to_string(index=False)
    )

    print()
    print("===== BEST CONFIGURATION AT EACH REMAPPING COST =====")
    dynamic = result[result["policy"].isin(["reactive", "phase_aware"])]
    best_indices = dynamic.groupby(
        "remap_cost_fraction"
    )["modeled_total_cycles"].idxmin()

    print(
        dynamic.loc[best_indices, [
            "policy",
            "remap_cost_fraction",
            "lookahead_layers",
            "remapping_count",
            "modeled_total_cycles",
            "saving_vs_static_percent",
        ]].sort_values("remap_cost_fraction").round(4).to_string(index=False)
    )

    assert not result.isna().any().any()
    assert len(result) == 1 + len(REMAP_COSTS) * (1 + len(LOOKAHEADS))
    assert (result["modeled_total_cycles"] > 0).all()
    assert (result["remapping_count"] >= 0).all()

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print("PASS: sensitivity sweep and consistency checks.")


if __name__ == "__main__":
    main()
