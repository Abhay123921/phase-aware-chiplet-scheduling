from dataclasses import replace
from pathlib import Path

import pandas as pd

from simulator.hardware.model import load_hardware
from simulator.scheduler import phase_aware_scheduler as base
from simulator.scheduler.communication_aware_scheduler import run_policy


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "results/nop_sensitivity.csv"

BANDWIDTH_SCALES = [1.0, 0.1, 0.01, 0.001, 0.0001]
HOP_LATENCY_MULTIPLIERS = [1, 10, 100]
LOOKAHEAD = 3


def main():
    profiles = base.load_profiles()
    hardware = load_hardware()
    chiplets = hardware.chiplets

    scales = base.build_resource_scales(profiles)
    costs = base.precompute_costs(profiles, chiplets, scales)
    static_chiplet, _ = base.choose_static_mapping(
        profiles, chiplets, costs
    )

    rows = []

    for bandwidth_scale in BANDWIDTH_SCALES:
        for latency_multiplier in HOP_LATENCY_MULTIPLIERS:
            scenario_hardware = replace(
                hardware,
                nop_bandwidth=(
                    hardware.nop_bandwidth * bandwidth_scale
                ),
                nop_hop_latency=(
                    hardware.nop_hop_latency * latency_multiplier
                ),
            )

            for policy in ["static", "reactive", "phase_aware"]:
                trace = run_policy(
                    policy,
                    profiles,
                    chiplets,
                    costs,
                    scenario_hardware,
                    static_chiplet=static_chiplet,
                )

                execution = float(
                    trace["estimated_execution_cycles"].sum()
                )
                communication = float(
                    trace["communication_cycles_proxy"].sum()
                )
                total = float(
                    trace["modeled_total_cycles"].sum()
                )

                rows.append({
                    "policy": policy,
                    "bandwidth_scale": bandwidth_scale,
                    "hop_latency_multiplier": latency_multiplier,
                    "lookahead_layers": (
                        LOOKAHEAD if policy == "phase_aware" else 1
                    ),
                    "remapping_count": int(
                        trace["remapped"].sum()
                    ),
                    "execution_cycles_proxy": execution,
                    "communication_cycles_proxy": communication,
                    "modeled_total_cycles": total,
                    "communication_fraction_percent": (
                        100.0 * communication / total
                        if total else 0.0
                    ),
                })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("===== NoP SENSITIVITY EXPERIMENT =====")
    print(f"Layers: {len(profiles)}")
    print(f"Bandwidth scales: {BANDWIDTH_SCALES}")
    print(f"Hop-latency multipliers: {HOP_LATENCY_MULTIPLIERS}")
    print("All NoP scenarios are assumptions for sensitivity testing.")
    print()

    print("===== POLICY RESULTS =====")
    print(
        result[
            [
                "policy",
                "bandwidth_scale",
                "hop_latency_multiplier",
                "remapping_count",
                "communication_cycles_proxy",
                "modeled_total_cycles",
                "communication_fraction_percent",
            ]
        ].round(4).to_string(index=False)
    )

    print()
    print("===== BEST POLICY PER NOP SCENARIO =====")
    idx = result.groupby(
        ["bandwidth_scale", "hop_latency_multiplier"]
    )["modeled_total_cycles"].idxmin()

    print(
        result.loc[idx, [
            "bandwidth_scale",
            "hop_latency_multiplier",
            "policy",
            "remapping_count",
            "modeled_total_cycles",
            "communication_fraction_percent",
        ]].sort_values(
            ["bandwidth_scale", "hop_latency_multiplier"]
        ).round(4).to_string(index=False)
    )

    assert len(result) == (
        len(BANDWIDTH_SCALES)
        * len(HOP_LATENCY_MULTIPLIERS)
        * 3
    )
    assert not result.isna().any().any()
    assert (result["communication_cycles_proxy"] >= 0).all()
    assert (result["modeled_total_cycles"] > 0).all()

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print("PASS: NoP sensitivity checks.")


if __name__ == "__main__":
    main()
