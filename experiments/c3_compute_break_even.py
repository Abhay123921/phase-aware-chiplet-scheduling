from pathlib import Path
from dataclasses import replace

import pandas as pd

import simulator.scheduler.phase_aware_scheduler as scheduler
from simulator.hardware.model import load_hardware


ROOT = Path(__file__).resolve().parents[1]

SUMMARY_PATH = ROOT / "results/c3_break_even_summary.csv"
TRACE_PATH = ROOT / "results/c3_break_even_trace.csv"
CHOICES_PATH = ROOT / "results/c3_break_even_layer_choices.csv"

C3_COMPUTE_VALUES = [0.75, 1.00, 1.10, 1.25]

REMAP_COST = 0.05
LOOKAHEAD = 2


def main():
    profiles = scheduler.load_profiles()
    hardware = load_hardware()

    original_chiplets = hardware.chiplets
    resource_scales = scheduler.build_resource_scales(profiles)

    scheduler.REMAP_COST_FRACTION = REMAP_COST

    summary_rows = []
    trace_frames = []
    choice_frames = []

    print("===== C3 COMPUTE BREAK-EVEN EXPERIMENT =====")
    print(f"Layers: {len(profiles)}")
    print(f"Chiplets: {list(original_chiplets)}")
    print(f"C3 compute values: {C3_COMPUTE_VALUES}")
    print(f"Remapping penalty: {REMAP_COST}")
    print(f"Lookahead: {LOOKAHEAD}")
    print()

    for compute_value in C3_COMPUTE_VALUES:

        config_name = f"C3_compute_{compute_value:.2f}"

        # Modify only C3 compute capacity.
        chiplets = dict(original_chiplets)

        chiplets["C3"] = replace(
            original_chiplets["C3"],
            compute_capacity=compute_value,
        )

        # Recompute all placement costs for this configuration.
        costs = scheduler.precompute_costs(
            profiles,
            chiplets,
            resource_scales,
        )

        static_chiplet, static_totals = (
            scheduler.choose_static_mapping(
                profiles, chiplets, costs
            )
        )

        policy_traces = {}

        # Run all three policies.
        policy_traces["static"] = scheduler.run_policy(
            "static",
            profiles,
            chiplets,
            costs,
            static_chiplet=static_chiplet,
        )

        policy_traces["reactive"] = scheduler.run_policy(
            "reactive",
            profiles,
            chiplets,
            costs,
        )

        policy_traces["phase_aware"] = scheduler.run_policy(
            "phase_aware",
            profiles,
            chiplets,
            costs,
            lookahead_layers=LOOKAHEAD,
        )

        # Find the locally optimal chiplet for every layer.
        local_rows = []

        for i, layer in profiles.iterrows():

            feasible = [
                name for name in chiplets
                if (i, name) in costs
            ]

            local_best = min(
                feasible,
                key=lambda name:
                    costs[(i, name)]["estimated_cycles"],
            )

            row = {
                "configuration": config_name,
                "c3_compute_capacity": compute_value,
                "layer_index": int(layer["layer_index"]),
                "layer_id": layer["layer_id"],
                "local_best_chiplet": local_best,
                "local_best_cost": costs[
                    (i, local_best)
                ]["estimated_cycles"],
                "c3_feasible": (i, "C3") in costs,
            }

            for name in chiplets:
                row[f"{name}_cost"] = (
                    costs[(i, name)]["estimated_cycles"]
                    if (i, name) in costs
                    else None
                )

            for policy, trace in policy_traces.items():
                row[f"{policy}_chiplet"] = (
                    trace.iloc[i]["chiplet"]
                )

            local_rows.append(row)

        choices = pd.DataFrame(local_rows)
        choice_frames.append(choices)

        # Save policy traces and calculate summary statistics.
        for policy, trace in policy_traces.items():

            trace = trace.copy()
            trace["configuration"] = config_name
            trace["c3_compute_capacity"] = compute_value

            trace_frames.append(trace)

            total_execution = float(
                trace["estimated_execution_cycles"].sum()
            )

            remap_count = int(trace["remapped"].sum())

            remap_overhead = float(
                trace["remap_overhead_cycles"].sum()
            )

            total_cost = float(
                trace["modeled_total_cycles"].sum()
            )

            selected_c3 = int(
                (trace["chiplet"] == "C3").sum()
            )

            policy_placements = trace["chiplet"].tolist()
            local_placements = choices[
                "local_best_chiplet"
            ].tolist()

            locally_inferior = sum(
                p != local
                for p, local in zip(
                    policy_placements,
                    local_placements,
                )
            )

            local_regret = sum(
                costs[(i, policy_placements[i])][
                    "estimated_cycles"
                ]
                - costs[(i, local_placements[i])][
                    "estimated_cycles"
                ]
                for i in range(len(profiles))
            )

            summary_rows.append({
                "configuration": config_name,
                "c3_compute_capacity": compute_value,
                "policy": policy,
                "static_chiplet": static_chiplet,
                "layers": len(trace),
                "c3_selected_layers": selected_c3,
                "c3_local_optimal_layers": int(
                    (choices["local_best_chiplet"] == "C3").sum()
                ),
                "remapping_count": remap_count,
                "execution_cycles": total_execution,
                "remap_overhead_cycles": remap_overhead,
                "modeled_total_cycles": total_cost,
                "locally_inferior_decisions": locally_inferior,
                "local_regret_cycles": local_regret,
                "unique_chiplets_used": trace["chiplet"].nunique(),
                "remap_cost_fraction": REMAP_COST,
                "lookahead_layers": (
                    LOOKAHEAD if policy == "phase_aware" else 1
                ),
            })

    summary = pd.DataFrame(summary_rows)
    traces = pd.concat(trace_frames, ignore_index=True)
    choices = pd.concat(choice_frames, ignore_index=True)

    # Relative improvement of phase-aware over reactive.
    summary["improvement_vs_reactive_percent"] = 0.0

    for config_name in summary["configuration"].unique():

        mask = summary["configuration"] == config_name
        subset = summary[mask]

        reactive_total = float(
            subset.loc[
                subset["policy"] == "reactive",
                "modeled_total_cycles",
            ].iloc[0]
        )

        phase_total = float(
            subset.loc[
                subset["policy"] == "phase_aware",
                "modeled_total_cycles",
            ].iloc[0]
        )

        improvement = (
            100.0 * (reactive_total - phase_total)
            / reactive_total
        )

        summary.loc[
            mask & (summary["policy"] == "phase_aware"),
            "improvement_vs_reactive_percent",
        ] = improvement

    # Save independent experiment outputs.
    summary.to_csv(SUMMARY_PATH, index=False)
    traces.to_csv(TRACE_PATH, index=False)
    choices.to_csv(CHOICES_PATH, index=False)

    # Display concise results.
    print("===== SUMMARY =====")

    columns = [
        "c3_compute_capacity",
        "policy",
        "c3_selected_layers",
        "c3_local_optimal_layers",
        "remapping_count",
        "execution_cycles",
        "remap_overhead_cycles",
        "modeled_total_cycles",
        "locally_inferior_decisions",
        "improvement_vs_reactive_percent",
    ]

    print(
        summary[columns].round(4).to_string(index=False)
    )

    print()
    print("===== LOCAL CHIPLET OPTIMALITY =====")

    local_counts = (
        choices.groupby(
            ["c3_compute_capacity", "local_best_chiplet"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print(local_counts.to_string())

    print()
    print("===== OUTPUT FILES =====")
    print(SUMMARY_PATH)
    print(TRACE_PATH)
    print(CHOICES_PATH)

    # Consistency checks.
    assert len(summary) == len(C3_COMPUTE_VALUES) * 3
    assert len(traces) == len(C3_COMPUTE_VALUES) * 3 * len(profiles)
    assert len(choices) == len(C3_COMPUTE_VALUES) * len(profiles)

    assert not summary.isna().any().any()
    assert (summary["modeled_total_cycles"] > 0).all()

    print()
    print("PASS: C3 compute break-even experiment completed.")


if __name__ == "__main__":
    main()
