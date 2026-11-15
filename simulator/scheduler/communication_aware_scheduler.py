from pathlib import Path

import pandas as pd

from simulator.hardware.model import load_hardware
from simulator.scheduler import phase_aware_scheduler as base
from simulator.scheduler.communication_cost import (
    COORDINATES,
    REFERENCE_HOP_LATENCY_CYCLES,
    REFERENCE_NOP_WORDS_PER_CYCLE,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "results/communication_aware_scheduler_trace.csv"
SUMMARY_PATH = ROOT / "results/communication_aware_scheduler_summary.csv"

LOOKAHEAD = 3
WORD_BYTES = 2


def hops(source, destination):
    x1, y1 = COORDINATES[source]
    x2, y2 = COORDINATES[destination]
    return abs(x1 - x2) + abs(y1 - y2)


def communication_cost(previous_layer, source, destination, hardware):
    """Estimated transfer of the previous layer's output on a placement change."""
    if source == destination:
        return 0.0

    words = float(previous_layer["outputs_utilized_capacity"])
    bandwidth = (
        REFERENCE_NOP_WORDS_PER_CYCLE * hardware.nop_bandwidth
    )
    if bandwidth <= 0:
        raise ValueError("Effective NoP bandwidth must be positive")

    return (
        words / bandwidth
        + hops(source, destination)
        * REFERENCE_HOP_LATENCY_CYCLES
        * hardware.nop_hop_latency
    )


def run_policy(policy, profiles, chiplets, costs, hardware, static_chiplet=None):
    rows = []
    previous_placement = None

    for i, layer in profiles.iterrows():
        candidates = [
            name for name in chiplets
            if (i, name) in costs
        ]

        if policy == "static":
            chosen = static_chiplet

        elif policy == "reactive":
            def score(name):
                execution = costs[(i, name)]["estimated_cycles"]
                transfer = 0.0

                if previous_placement is not None and i > 0:
                    transfer = communication_cost(
                        profiles.iloc[i - 1],
                        previous_placement,
                        name,
                        hardware,
                    )

                return execution + transfer

            chosen = min(candidates, key=score)

        elif policy == "phase_aware":
            end = min(len(profiles), i + LOOKAHEAD)
            horizon = list(range(i, end))

            # Dynamic programming: minimum cost for each ending chiplet.
            dp = {}
            first_choice = {}

            for name in candidates:
                execution = costs[(i, name)]["estimated_cycles"]
                transfer = 0.0

                if previous_placement is not None and i > 0:
                    transfer = communication_cost(
                        profiles.iloc[i - 1],
                        previous_placement,
                        name,
                        hardware,
                    )

                dp[name] = execution + transfer
                first_choice[name] = name

            for j in horizon[1:]:
                next_dp = {}
                next_first = {}
                previous_layer = profiles.iloc[j - 1]

                destinations = [
                    name for name in chiplets
                    if (j, name) in costs
                ]

                for destination in destinations:
                    execution = costs[(j, destination)]["estimated_cycles"]
                    best_cost = float("inf")
                    best_first = None

                    for source, partial_cost in dp.items():
                        transfer = communication_cost(
                            previous_layer,
                            source,
                            destination,
                            hardware,
                        )
                        candidate_cost = (
                            partial_cost + transfer + execution
                        )

                        if candidate_cost < best_cost:
                            best_cost = candidate_cost
                            best_first = first_choice[source]

                    next_dp[destination] = best_cost
                    next_first[destination] = best_first

                dp = next_dp
                first_choice = next_first

            best_final = min(dp, key=dp.get)
            chosen = first_choice[best_final]

        else:
            raise ValueError(f"Unknown policy: {policy}")

        estimate = costs[(i, chosen)]
        transfer = 0.0
        changed = (
            previous_placement is not None
            and chosen != previous_placement
        )

        if changed and i > 0:
            transfer = communication_cost(
                profiles.iloc[i - 1],
                previous_placement,
                chosen,
                hardware,
            )

        rows.append({
            "policy": policy,
            "layer_index": int(layer["layer_index"]),
            "layer_id": layer["layer_id"],
            "resource_regime": (
                int(layer["resource_regime"])
                if "resource_regime" in profiles.columns
                and pd.notna(layer["resource_regime"])
                else -1
            ),
            "chiplet": chosen,
            "estimated_execution_cycles": estimate["estimated_cycles"],
            "communication_cycles_proxy": transfer,
            "remapped": changed,
            "modeled_total_cycles":
                estimate["estimated_cycles"] + transfer,
        })

        previous_placement = chosen

    return pd.DataFrame(rows)


def main():
    profiles = base.load_profiles()
    hardware = load_hardware()
    chiplets = hardware.chiplets

    scales = base.build_resource_scales(profiles)
    costs = base.precompute_costs(profiles, chiplets, scales)
    static_chiplet, _ = base.choose_static_mapping(
        profiles, chiplets, costs
    )

    traces = [
        run_policy(
            policy,
            profiles,
            chiplets,
            costs,
            hardware,
            static_chiplet=static_chiplet,
        )
        for policy in ["static", "reactive", "phase_aware"]
    ]

    trace = pd.concat(traces, ignore_index=True)
    trace.to_csv(OUTPUT_PATH, index=False)

    summary = (
        trace.groupby("policy")
        .agg(
            layers=("layer_id", "count"),
            execution_cycles=("estimated_execution_cycles", "sum"),
            remapping_count=("remapped", "sum"),
            communication_cycles_proxy=("communication_cycles_proxy", "sum"),
            modeled_total_cycles=("modeled_total_cycles", "sum"),
            unique_chiplets_used=("chiplet", "nunique"),
        )
        .reset_index()
    )
    summary["communication_fraction_percent"] = (
        100.0
        * summary["communication_cycles_proxy"]
        / summary["modeled_total_cycles"]
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    print("===== COMMUNICATION-AWARE SCHEDULER =====")
    print(f"Layers: {len(profiles)}")
    print(f"Static mapping: {static_chiplet}")
    print(f"Phase-aware lookahead: {LOOKAHEAD} layers")
    print(f"Assumed reference NoP bandwidth: {REFERENCE_NOP_WORDS_PER_CYCLE} words/cycle")
    print(f"Assumed word size: {WORD_BYTES} bytes")
    print("Communication costs are uncalibrated proxies.")
    print()

    print("===== POLICY SUMMARY =====")
    print(summary.round(4).to_string(index=False))

    print()
    print("===== FIRST 12 PHASE-AWARE DECISIONS =====")
    print(
        trace[trace["policy"] == "phase_aware"][
            [
                "layer_id",
                "resource_regime",
                "chiplet",
                "estimated_execution_cycles",
                "remapped",
                "communication_cycles_proxy",
            ]
        ].head(12).to_string(index=False)
    )

    assert len(trace) == 3 * len(profiles)
    assert not trace.isna().any().any()
    assert (trace["communication_cycles_proxy"] >= 0).all()
    assert (trace["modeled_total_cycles"] > 0).all()

    static = trace[trace["policy"] == "static"]
    assert static["remapped"].sum() == 0
    assert static["chiplet"].nunique() == 1

    print()
    print(f"Saved trace: {OUTPUT_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print("PASS: communication-aware scheduler checks.")


if __name__ == "__main__":
    main()
