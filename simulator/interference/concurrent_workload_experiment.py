from pathlib import Path

import pandas as pd

from simulator.hardware.model import load_hardware
from simulator.scheduler import phase_aware_scheduler as base
from simulator.scheduler.communication_aware_scheduler import run_policy


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "results/concurrent_workload_experiment.csv"

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


def build_assignments(
    policy,
    profiles,
    chiplets,
    costs,
    hardware,
    static_chiplet,
):
    trace = run_policy(
        policy,
        profiles,
        chiplets,
        costs,
        hardware,
        static_chiplet=static_chiplet,
    )
    return trace["chiplet"].tolist()


def simulate_pair(
    profiles,
    assignments_a,
    assignments_b,
    costs,
    chiplets,
    capacity_factor,
    arrival_offset_fraction,
):
    """
    Discrete-event simulation of two sequential layer traces.

    The slowdown is a shared-resource pressure proxy:
        slowdown = max(1, max_resource(sum demands / capacity))

    The same slowdown applies to every active workload. This is a
    deliberately simple model, not a calibrated contention predictor.
    """
    reference_total = float(profiles["summary_cycles"].sum())
    arrival_b = arrival_offset_fraction * reference_total

    resource_capacity = {
        resource: (
            float(profiles[column].max()) * capacity_factor
        )
        for resource, column in RESOURCE_COLUMNS.items()
    }

    workloads = [
        {
            "name": "A",
            "arrival": 0.0,
            "layer": 0,
            "remaining": None,
            "completion": None,
            "assignments": assignments_a,
            "last_chiplet": None,
            "remaps": 0,
        },
        {
            "name": "B",
            "arrival": arrival_b,
            "layer": 0,
            "remaining": None,
            "completion": None,
            "assignments": assignments_b,
            "last_chiplet": None,
            "remaps": 0,
        },
    ]

    now = 0.0
    overload_time = 0.0
    total_active_time = 0.0
    max_pressure_seen = 0.0
    event_count = 0

    while any(w["completion"] is None for w in workloads):
        event_count += 1
        if event_count > 10000:
            raise RuntimeError("Event limit exceeded")

        # Start any workload whose arrival time has been reached.
        for w in workloads:
            if (
                w["completion"] is None
                and w["remaining"] is None
                and w["arrival"] <= now
            ):
                idx = w["layer"]
                chiplet_name = w["assignments"][idx]
                chiplet = chiplets[chiplet_name]
                reference_cycles = float(
                    profiles.iloc[idx]["summary_cycles"]
                )
                w["remaining"] = costs[(idx, chiplet_name)][
                    "estimated_cycles"
                ]

                if (
                    w["last_chiplet"] is not None
                    and chiplet_name != w["last_chiplet"]
                ):
                    w["remaps"] += 1

                w["last_chiplet"] = chiplet_name

        active = [
            w for w in workloads
            if w["completion"] is None
            and w["remaining"] is not None
        ]

        if not active:
            next_arrivals = [
                w["arrival"] for w in workloads
                if w["completion"] is None
                and w["remaining"] is None
            ]
            if not next_arrivals:
                break
            now = min(next_arrivals)
            continue

        # SRAM bandwidth is modeled per chiplet; DRAM bandwidth
        # remains a shared global resource.
        local_resources = ("input_bw", "weight_bw", "output_bw")
        dram_resources = (
            "dram_input_bw", "dram_weight_bw", "dram_output_bw"
        )

        local_demand = {
            name: {r: 0.0 for r in local_resources}
            for name in chiplets
        }
        dram_demand = {r: 0.0 for r in dram_resources}

        for w in active:
            idx = w["layer"]
            layer = profiles.iloc[idx]
            assigned_chiplet = w["assignments"][idx]

            for resource, column in RESOURCE_COLUMNS.items():
                demand = float(layer[column])

                if resource in local_resources:
                    local_demand[assigned_chiplet][resource] += demand
                else:
                    dram_demand[resource] += demand

        pressures = {}

        # Chiplet memory_bandwidth values are relative configuration
        # scales. Capacity factors remain explicit stress assumptions.
        for chiplet_name, demands in local_demand.items():
            bandwidth_scale = float(
                chiplets[chiplet_name].memory_bandwidth
            )
            for resource, demand in demands.items():
                capacity = resource_capacity[resource] * bandwidth_scale
                pressures[f"{chiplet_name}_{resource}"] = demand / capacity

        for resource, demand in dram_demand.items():
            pressures[resource] = demand / resource_capacity[resource]

        max_pressure = max(pressures.values())
        max_pressure_seen = max(max_pressure_seen, max_pressure)
        slowdown = max(1.0, max_pressure)

        # Advance only to the next layer completion OR the next
        # workload arrival, whichever happens first. This ensures that
        # a newly arriving workload begins competing at its actual
        # scheduled arrival time rather than at the next layer boundary.
        next_layer_event = (
            min(w["remaining"] for w in active) * slowdown
        )

        future_arrivals = [
            w["arrival"]
            for w in workloads
            if w["completion"] is None
            and w["remaining"] is None
            and w["arrival"] > now
        ]

        dt = next_layer_event
        if future_arrivals:
            time_to_arrival = min(future_arrivals) - now
            dt = min(dt, time_to_arrival)

        if max_pressure > 1.0:
            overload_time += dt
        total_active_time += dt

        # All active workloads progress at the same modeled rate.
        progress = dt / slowdown
        now += dt

        for w in active:
            w["remaining"] -= progress

        # Complete finished layers and prepare their next layer.
        for w in active:
            if w["remaining"] <= 1e-7:
                w["layer"] += 1
                w["remaining"] = None

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
            overload_time / total_active_time
            if total_active_time else 0.0
        ),
        "remapping_count": (
            workloads[0]["remaps"] + workloads[1]["remaps"]
        ),
    }


def main():
    profiles = base.load_profiles()
    hardware = load_hardware()
    chiplets = hardware.chiplets

    resource_scales = base.build_resource_scales(profiles)
    costs = base.precompute_costs(
        profiles, chiplets, resource_scales
    )
    static_chiplet, _ = base.choose_static_mapping(
        profiles, chiplets, costs
    )

    policies = ["static", "reactive", "phase_aware"]
    assignments = {
        policy: build_assignments(
            policy,
            profiles,
            chiplets,
            costs,
            hardware,
            static_chiplet,
        )
        for policy in policies
    }

    rows = []

    for capacity_factor in CAPACITY_FACTORS:
        for arrival_offset in ARRIVAL_OFFSETS:
            for policy in policies:
                metrics = simulate_pair(
                    profiles,
                    assignments[policy],
                    assignments[policy],
                    costs,
                    chiplets,
                    capacity_factor,
                    arrival_offset,
                )

                rows.append({
                    "policy": policy,
                    "capacity_factor": capacity_factor,
                    "arrival_offset_fraction": arrival_offset,
                    **metrics,
                })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("===== CONCURRENT WORKLOAD EXPERIMENT =====")
    print("Workloads: two copies of the same ResNet-50 layer sequence")
    print(f"Layers per workload: {len(profiles)}")
    print(f"Static mapping: {static_chiplet}")
    print(f"Capacity factors: {CAPACITY_FACTORS}")
    print(f"Arrival offsets: {ARRIVAL_OFFSETS}")
    print("All resource capacities are normalized assumptions.")
    print("Latency and overload metrics are model proxies.")
    print()

    print("===== MAKESPAN BY POLICY =====")
    print(
        result[
            [
                "policy",
                "capacity_factor",
                "arrival_offset_fraction",
                "makespan_cycles_proxy",
                "completion_a_cycles_proxy",
                "completion_b_cycles_proxy",
                "remapping_count",
                "max_resource_pressure",
                "overload_time_fraction_proxy",
            ]
        ].round(4).to_string(index=False)
    )

    print()
    print("===== BEST POLICY PER SCENARIO =====")
    idx = result.groupby(
        ["capacity_factor", "arrival_offset_fraction"]
    )["makespan_cycles_proxy"].idxmin()

    print(
        result.loc[idx, [
            "capacity_factor",
            "arrival_offset_fraction",
            "policy",
            "makespan_cycles_proxy",
            "remapping_count",
            "overload_time_fraction_proxy",
        ]].sort_values(
            ["capacity_factor", "arrival_offset_fraction"]
        ).round(4).to_string(index=False)
    )

    assert len(result) == (
        len(CAPACITY_FACTORS)
        * len(ARRIVAL_OFFSETS)
        * len(policies)
    )
    assert not result.isna().any().any()
    assert (result["makespan_cycles_proxy"] > 0).all()
    assert (
        result["overload_time_fraction_proxy"].between(0, 1)
    ).all()

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print("PASS: concurrent workload simulation checks.")


if __name__ == "__main__":
    main()
