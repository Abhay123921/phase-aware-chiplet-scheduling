from pathlib import Path

import pandas as pd

from simulator.hardware.model import load_hardware
from simulator.scheduler import phase_aware_scheduler as base


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "results/online_contention_scheduler.csv"

CAPACITY_FACTORS = [1.0, 1.5, 2.0]
ARRIVAL_OFFSETS = [0.0, 0.25]

RESOURCE_COLUMNS = {
    "input_bw": "inputs_shared_bw",
    "weight_bw": "weights_shared_bw",
    "output_bw": "outputs_shared_bw",
    "dram_input_bw": "dram_inputs_shared_bw",
    "dram_weight_bw": "dram_weights_shared_bw",
    "dram_output_bw": "dram_outputs_shared_bw",
}

LOCAL_RESOURCES = ("input_bw", "weight_bw", "output_bw")
DRAM_RESOURCES = ("dram_input_bw", "dram_weight_bw", "dram_output_bw")
REMAP_COST_FRACTION = 0.05


def layer_demand(profiles, index):
    row = profiles.iloc[index]
    return {
        resource: float(row[column])
        for resource, column in RESOURCE_COLUMNS.items()
    }


def resource_pressures(active, profiles, chiplets, capacities):
    local = {
        name: {r: 0.0 for r in LOCAL_RESOURCES}
        for name in chiplets
    }
    dram = {r: 0.0 for r in DRAM_RESOURCES}

    for w in active:
        idx = w["layer"]
        chiplet_name = w["chiplet"]
        demand = layer_demand(profiles, idx)

        for resource in LOCAL_RESOURCES:
            local[chiplet_name][resource] += demand[resource]

        for resource in DRAM_RESOURCES:
            dram[resource] += demand[resource]

    pressures = {}

    for chiplet_name, demands in local.items():
        bw_scale = float(chiplets[chiplet_name].memory_bandwidth)
        for resource, amount in demands.items():
            capacity = capacities[resource] * bw_scale
            pressures[f"{chiplet_name}_{resource}"] = amount / capacity

    for resource, amount in dram.items():
        pressures[resource] = amount / capacities[resource]

    return pressures


def choose_online_chiplet(
    workload, active, profiles, chiplets, costs, capacities
):
    """Greedy current-layer choice using current competing demand."""
    idx = workload["layer"]
    candidates = [
        name for name in chiplets
        if (idx, name) in costs
    ]

    scored = []

    for candidate in candidates:
        previous = workload["last_chiplet"]
        remap = (
            REMAP_COST_FRACTION * float(profiles.iloc[idx]["summary_cycles"])
            if previous is not None and candidate != previous
            else 0.0
        )

        # Evaluate candidate with this workload assigned to it and
        # other currently active workloads held at their current placements.
        trial_active = []
        for other in active:
            if other is workload:
                trial_active.append({**other, "chiplet": candidate})
            else:
                trial_active.append(other)

        if not any(other is workload for other in active):
            trial_active.append({
                **workload,
                "chiplet": candidate,
            })

        pressures = resource_pressures(
            trial_active, profiles, chiplets, capacities
        )
        contention = max(1.0, max(pressures.values()))

        execution = costs[(idx, candidate)]["estimated_cycles"]
        score = execution * contention + remap
        scored.append((score, candidate))

    return min(scored)[1]



def choose_phase_aware_chiplet(
    workload, active, profiles, chiplets, costs, capacities,
    horizon=3,
):
    """
    Short-horizon dynamic programming using the same contention score
    as reactive scheduling. Future competitor demand is held constant,
    so this is an optimistic, simplified lookahead proxy.
    """
    start = workload["layer"]
    stop = min(len(profiles), start + max(1, horizon))
    previous_chiplet = workload["last_chiplet"]

    def score_layer(index, candidate, previous):
        trial = [
            {**other}
            for other in active
            if other is not workload
        ]
        trial.append({
            **workload,
            "layer": index,
            "chiplet": candidate,
        })

        pressures = resource_pressures(
            trial, profiles, chiplets, capacities
        )
        contention = max(1.0, max(pressures.values()))
        execution = costs[(index, candidate)]["estimated_cycles"]

        remap = (
            REMAP_COST_FRACTION
            * float(profiles.iloc[index]["summary_cycles"])
            if previous is not None and candidate != previous
            else 0.0
        )
        return execution * contention + remap

    # dp maps the current candidate chiplet to (total cost, first choice).
    first_candidates = [
        name for name in chiplets
        if (start, name) in costs
    ]
    dp = {
        candidate: (
            score_layer(start, candidate, previous_chiplet),
            candidate,
        )
        for candidate in first_candidates
    }

    for index in range(start + 1, stop):
        next_dp = {}
        candidates = [
            name for name in chiplets
            if (index, name) in costs
        ]

        for candidate in candidates:
            best = None
            for previous, (past_cost, first_choice) in dp.items():
                future_cost = score_layer(index, candidate, previous)
                proposal = (past_cost + future_cost, first_choice)
                if best is None or proposal[0] < best[0]:
                    best = proposal

            if best is not None:
                next_dp[candidate] = best

        dp = next_dp

    if not dp:
        raise RuntimeError("No feasible phase-aware placement found.")

    return min(dp.values(), key=lambda item: item[0])[1]

def simulate(policy, profiles, chiplets, costs, static_chiplet,
             capacity_factor, arrival_offset):
    reference_total = float(profiles["summary_cycles"].sum())
    arrivals = [0.0, arrival_offset * reference_total]

    capacities = {
        resource: float(profiles[column].max()) * capacity_factor
        for resource, column in RESOURCE_COLUMNS.items()
    }

    workloads = [
        {
            "name": "A",
            "arrival": arrivals[0],
            "layer": 0,
            "remaining": None,
            "completion": None,
            "chiplet": None,
            "last_chiplet": None,
            "remaps": 0,
        },
        {
            "name": "B",
            "arrival": arrivals[1],
            "layer": 0,
            "remaining": None,
            "completion": None,
            "chiplet": None,
            "last_chiplet": None,
            "remaps": 0,
        },
    ]

    now = 0.0
    max_pressure_seen = 0.0
    overload_time = 0.0
    active_time = 0.0
    events = 0

    while any(w["completion"] is None for w in workloads):
        events += 1
        if events > 20000:
            raise RuntimeError("Event limit exceeded")

        # Launch each newly ready layer. Ready workloads are considered
        # in stable order A then B for reproducibility.
        for w in workloads:
            if (
                w["completion"] is not None
                or w["remaining"] is not None
                or w["arrival"] > now
            ):
                continue

            active = [
                other for other in workloads
                if other["completion"] is None
                and other["remaining"] is not None
            ]

            if policy == "static":
                chosen = static_chiplet
            elif policy == "reactive":
                chosen = choose_online_chiplet(
                    w, active, profiles, chiplets, costs, capacities
                )
            elif policy == "phase_aware":
                chosen = choose_phase_aware_chiplet(
                    w, active, profiles, chiplets, costs, capacities,
                    horizon=3,
                )
            else:
                raise ValueError(f"Unknown policy: {policy}")

            if (
                w["last_chiplet"] is not None
                and chosen != w["last_chiplet"]
            ):
                w["remaps"] += 1

            w["last_chiplet"] = chosen
            w["chiplet"] = chosen
            w["remaining"] = costs[(w["layer"], chosen)]["estimated_cycles"]

        active = [
            w for w in workloads
            if w["completion"] is None and w["remaining"] is not None
        ]

        if not active:
            future = [
                w["arrival"] for w in workloads
                if w["completion"] is None and w["remaining"] is None
            ]
            if not future:
                break
            now = min(future)
            continue

        pressures = resource_pressures(
            active, profiles, chiplets, capacities
        )
        max_pressure = max(pressures.values())
        max_pressure_seen = max(max_pressure_seen, max_pressure)
        slowdown = max(1.0, max_pressure)

        next_event = min(w["remaining"] for w in active) * slowdown
        future_arrivals = [
            w["arrival"] - now for w in workloads
            if w["completion"] is None
            and w["remaining"] is None
            and w["arrival"] > now
        ]
        if future_arrivals:
            next_event = min(next_event, min(future_arrivals))

        if max_pressure > 1.0:
            overload_time += next_event
        active_time += next_event

        progress = next_event / slowdown
        now += next_event

        for w in active:
            w["remaining"] -= progress

        for w in active:
            if w["remaining"] <= 1e-7:
                w["layer"] += 1
                w["remaining"] = None
                w["chiplet"] = None

                if w["layer"] >= len(profiles):
                    w["completion"] = now

    completion_a = workloads[0]["completion"]
    completion_b = workloads[1]["completion"]

    return {
        "makespan_cycles_proxy": max(completion_a, completion_b),
        "completion_a_cycles_proxy": completion_a,
        "completion_b_cycles_proxy": completion_b,
        "max_resource_pressure": max_pressure_seen,
        "overload_time_fraction_proxy": (
            overload_time / active_time if active_time else 0.0
        ),
        "remapping_count": sum(w["remaps"] for w in workloads),
    }


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
    for factor in CAPACITY_FACTORS:
        for offset in ARRIVAL_OFFSETS:
            for policy in ("static", "reactive", "phase_aware"):
                metrics = simulate(
                    policy, profiles, chiplets, costs,
                    static_chiplet, factor, offset
                )
                rows.append({
                    "policy": policy,
                    "capacity_factor": factor,
                    "arrival_offset_fraction": offset,
                    **metrics,
                })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("===== ONLINE CONTENTION-AWARE SCHEDULER =====")
    print("This is an uncalibrated normalized stress model.")
    print(result.to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")
    print("PASS: online scheduler experiment completed.")


if __name__ == "__main__":
    main()
