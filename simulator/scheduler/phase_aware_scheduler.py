from pathlib import Path

import pandas as pd

from simulator.hardware.model import load_hardware


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "results/resnet50_layer_profiles.csv"
OUTPUT_PATH = ROOT / "results/phase_aware_scheduler_trace.csv"
SUMMARY_PATH = ROOT / "results/phase_aware_scheduler_summary.csv"

# Explicit experimental assumptions, not measured hardware parameters.
REMAP_COST_FRACTION = 0.05
LOOKAHEAD_LAYERS = 2

RESOURCE_COLUMNS = {
    "input_bw": "inputs_shared_bw",
    "weight_bw": "weights_shared_bw",
    "output_bw": "outputs_shared_bw",
    "dram_input_bw": "dram_inputs_shared_bw",
    "dram_weight_bw": "dram_weights_shared_bw",
    "dram_output_bw": "dram_outputs_shared_bw",
}


def load_profiles():
    profiles = pd.read_csv(PROFILE_PATH)

    required = {
        "layer_id",
        "layer_index",
        "summary_cycles",
        "local_capacity_peak",
        *RESOURCE_COLUMNS.values(),
    }
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"Missing profile columns: {sorted(missing)}")

    profiles = profiles.sort_values("layer_index").reset_index(drop=True)

    if profiles.empty:
        raise ValueError("No layer profiles found")
    if profiles[list(required - {"layer_id"})].isna().any().any():
        raise ValueError("Required profile columns contain missing values")
    if (profiles["summary_cycles"] <= 0).any():
        raise ValueError("Layer reference cycles must be positive")

    return profiles


def build_resource_scales(profiles):
    """Normalize each resource by its maximum observed layer demand."""
    scales = {}
    for resource, column in RESOURCE_COLUMNS.items():
        maximum = float(profiles[column].max())
        if maximum <= 0:
            raise ValueError(f"Invalid maximum demand for {resource}")
        scales[resource] = maximum
    return scales


def estimate_layer_cost(layer, chiplet, resource_scales):
    """
    First-order normalized execution-cost proxy.

    Compute estimate:
        reference cycles / compute-capacity scale

    Memory estimate:
        reference cycles * maximum normalized resource pressure

    The memory pressure is adjusted by the chiplet's bandwidth scale.
    The final estimate is the maximum of compute and memory estimates.

    This is a scheduling proxy, not a calibrated latency predictor.
    """
    reference_cycles = float(layer["summary_cycles"])

    compute_cycles = (
        reference_cycles / chiplet.compute_capacity
    )

    normalized_demands = []
    for resource, column in RESOURCE_COLUMNS.items():
        demand = float(layer[column])
        normalized_demands.append(
            demand / resource_scales[resource]
        )

    max_normalized_demand = max(normalized_demands)

    memory_pressure = (
        max_normalized_demand / chiplet.memory_bandwidth
    )

    # Relative-cost proxy: do not impose a reference-cycle floor on
    # memory time. This permits faster chiplets to benefit from phases
    # whose normalized memory demand is sufficiently low.
    #
    # This remains an uncalibrated sensitivity model: normalized layer
    # demand is not a measured physical bandwidth capacity.
    memory_cycles = reference_cycles * memory_pressure

    estimated_cycles = max(compute_cycles, memory_cycles)

    fits_sram = (
        float(layer["local_capacity_peak"])
        <= chiplet.sram_capacity_words
    )

    if not fits_sram:
        return None

    return {
        "estimated_cycles": estimated_cycles,
        "compute_cycles": compute_cycles,
        "memory_cycles": memory_cycles,
        "normalized_max_resource_demand": max_normalized_demand,
        "memory_pressure_proxy": memory_pressure,
    }


def precompute_costs(profiles, chiplets, resource_scales):
    costs = {}

    for i, layer in profiles.iterrows():
        for name, chiplet in chiplets.items():
            estimate = estimate_layer_cost(
                layer, chiplet, resource_scales
            )
            if estimate is not None:
                costs[(i, name)] = estimate

    for i in range(len(profiles)):
        feasible = [
            name for name in chiplets
            if (i, name) in costs
        ]
        if not feasible:
            raise RuntimeError(
                f"No SRAM-feasible chiplet for "
                f"{profiles.iloc[i]['layer_id']}"
            )

    return costs


def choose_static_mapping(profiles, chiplets, costs):
    """Choose one chiplet minimizing total estimated execution cost."""
    totals = {}

    for name in chiplets:
        layer_costs = [
            costs[(i, name)]["estimated_cycles"]
            for i in range(len(profiles))
            if (i, name) in costs
        ]

        if len(layer_costs) == len(profiles):
            totals[name] = sum(layer_costs)

    if not totals:
        raise RuntimeError(
            "No single chiplet can host every layer under the SRAM screen"
        )

    return min(totals, key=totals.get), totals


def run_policy(
    policy,
    profiles,
    chiplets,
    costs,
    static_chiplet=None,
    lookahead_layers=LOOKAHEAD_LAYERS,
):
    trace = []
    current_chiplet = None

    for i, layer in profiles.iterrows():
        candidates = [
            name for name in chiplets
            if (i, name) in costs
        ]

        if policy == "static":
            chosen = static_chiplet

        elif policy == "reactive":
            # One-layer horizon: current execution cost plus the cost
            # of switching from the previous placement.
            def reactive_score(name):
                execution = costs[(i, name)]["estimated_cycles"]
                remap = (
                    REMAP_COST_FRACTION * float(layer["summary_cycles"])
                    if current_chiplet is not None
                    and name != current_chiplet
                    else 0.0
                )
                return execution + remap

            chosen = min(candidates, key=reactive_score)

        elif policy == "phase_aware":
            # Dynamic programming over a short sequence of upcoming
            # layers. Each transition includes remapping overhead.
            end = min(
                len(profiles),
                i + max(1, lookahead_layers),
            )
            horizon = list(range(i, end))

            # dp[k][chiplet] = minimum cost through horizon[k],
            # ending on that chiplet.
            dp = {}
            first_step = {}

            for name in candidates:
                execution = costs[(i, name)]["estimated_cycles"]
                remap = (
                    REMAP_COST_FRACTION * float(layer["summary_cycles"])
                    if current_chiplet is not None
                    and name != current_chiplet
                    else 0.0
                )
                dp[name] = execution + remap
                first_step[name] = name

            for j in horizon[1:]:
                next_dp = {}
                next_first_step = {}

                future_layer = profiles.iloc[j]
                future_candidates = [
                    name for name in chiplets
                    if (j, name) in costs
                ]

                for destination in future_candidates:
                    execution = costs[
                        (j, destination)
                    ]["estimated_cycles"]

                    best_cost = float("inf")
                    best_first = None

                    for previous, previous_cost in dp.items():
                        switch_cost = (
                            REMAP_COST_FRACTION
                            * float(future_layer["summary_cycles"])
                            if previous != destination
                            else 0.0
                        )

                        candidate_cost = (
                            previous_cost + execution + switch_cost
                        )

                        if candidate_cost < best_cost:
                            best_cost = candidate_cost
                            best_first = first_step[previous]

                    next_dp[destination] = best_cost
                    next_first_step[destination] = best_first

                dp = next_dp
                first_step = next_first_step

            best_final = min(dp, key=dp.get)
            chosen = first_step[best_final]

        else:
            raise ValueError(f"Unknown policy: {policy}")

        estimate = costs[(i, chosen)]
        changed = (
            current_chiplet is not None
            and chosen != current_chiplet
        )
        remap_overhead = (
            REMAP_COST_FRACTION * float(layer["summary_cycles"])
            if changed else 0.0
        )

        trace.append({
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
            "reference_cycles": float(layer["summary_cycles"]),
            "estimated_execution_cycles": estimate["estimated_cycles"],
            "compute_cycles_proxy": estimate["compute_cycles"],
            "memory_cycles_proxy": estimate["memory_cycles"],
            "normalized_max_resource_demand":
                estimate["normalized_max_resource_demand"],
            "memory_pressure_proxy": estimate["memory_pressure_proxy"],
            "remapped": changed,
            "remap_overhead_cycles": remap_overhead,
            "modeled_total_cycles":
                estimate["estimated_cycles"] + remap_overhead,
        })

        current_chiplet = chosen

    return pd.DataFrame(trace)


def summarize(trace):
    rows = []

    for policy, group in trace.groupby("policy"):
        rows.append({
            "policy": policy,
            "layers": len(group),
            "estimated_execution_cycles":
                group["estimated_execution_cycles"].sum(),
            "remapping_count": int(group["remapped"].sum()),
            "remap_overhead_cycles":
                group["remap_overhead_cycles"].sum(),
            "modeled_total_cycles":
                group["modeled_total_cycles"].sum(),
            "unique_chiplets_used": group["chiplet"].nunique(),
            "assumption_remap_cost_fraction": REMAP_COST_FRACTION,
            "assumption_lookahead_layers": (
                LOOKAHEAD_LAYERS if policy == "phase_aware" else 1
            ),
        })

    return pd.DataFrame(rows).sort_values("policy")


def main():
    profiles = load_profiles()
    hardware = load_hardware()
    chiplets = hardware.chiplets
    resource_scales = build_resource_scales(profiles)

    costs = precompute_costs(
        profiles, chiplets, resource_scales
    )

    static_chiplet, static_totals = choose_static_mapping(
        profiles, chiplets, costs
    )

    traces = [
        run_policy(
            "static",
            profiles,
            chiplets,
            costs,
            static_chiplet=static_chiplet,
        ),
        run_policy(
            "reactive",
            profiles,
            chiplets,
            costs,
        ),
        run_policy(
            "phase_aware",
            profiles,
            chiplets,
            costs,
        ),
    ]

    trace = pd.concat(traces, ignore_index=True)
    summary = summarize(trace)

    trace.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("===== PHASE-AWARE SCHEDULER EXPERIMENT =====")
    print(f"Layers: {len(profiles)}")
    print(f"Candidate chiplets: {len(chiplets)}")
    print(f"Feasible layer-chiplet placements: {len(costs)}")
    print(f"Static mapping choice: {static_chiplet}")
    print(f"Remapping overhead assumption: {REMAP_COST_FRACTION:.1%}")
    print(f"Phase-aware lookahead: {LOOKAHEAD_LAYERS} layers")
    print("Execution estimates are normalized proxies, not measured latency.")
    print()

    print("===== STATIC MAPPING COST BY CHIPLET =====")
    for name, total in sorted(static_totals.items()):
        print(f"{name}: {total:.2f} estimated cycles")
    print()

    print("===== POLICY SUMMARY =====")
    print(summary.to_string(index=False))
    print()

    print("===== FIRST 12 PHASE-AWARE DECISIONS =====")
    phase_trace = trace[trace["policy"] == "phase_aware"]
    print(
        phase_trace[
            [
                "layer_id",
                "resource_regime",
                "chiplet",
                "estimated_execution_cycles",
                "remapped",
                "remap_overhead_cycles",
            ]
        ].head(12).to_string(index=False)
    )

    assert len(trace) == 3 * len(profiles)
    assert not trace.isna().any().any()
    assert (trace["estimated_execution_cycles"] > 0).all()
    assert (trace["remap_overhead_cycles"] >= 0).all()
    assert (summary["remapping_count"] >= 0).all()
    assert summary["layers"].eq(len(profiles)).all()

    static_trace = trace[trace["policy"] == "static"]
    assert static_trace["remapped"].sum() == 0
    assert static_trace["chiplet"].nunique() == 1

    print()
    print(f"Saved trace: {OUTPUT_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print("PASS: scheduler output and consistency checks.")


if __name__ == "__main__":
    main()
