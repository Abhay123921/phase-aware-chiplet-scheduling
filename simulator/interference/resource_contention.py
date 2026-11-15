from dataclasses import dataclass


RESOURCES = (
    "input_bw",
    "weight_bw",
    "output_bw",
    "dram_input_bw",
    "dram_weight_bw",
    "dram_output_bw",
)


@dataclass(frozen=True)
class WorkloadDemand:
    name: str
    demand: dict[str, float]


def evaluate_contention(workloads, capacities):
    """
    Calculate normalized resource pressure.

    Capacities must use the same units as the supplied demands.
    They are explicit model inputs, not automatically inferred
    physical limits.

    Provisional slowdown proxy:
        max(1, maximum resource pressure)

    This proxy is for sensitivity experiments only. It is not a
    calibrated latency or execution-time prediction.
    """
    if not workloads:
        raise ValueError("At least one workload is required")

    pressure = {}

    for resource in RESOURCES:
        capacity = float(capacities[resource])

        if capacity <= 0:
            raise ValueError(
                f"Capacity for {resource} must be positive"
            )

        total_demand = sum(
            float(workload.demand[resource])
            for workload in workloads
        )

        if total_demand < 0:
            raise ValueError(
                f"Demand for {resource} cannot be negative"
            )

        pressure[resource] = total_demand / capacity

    max_pressure = max(pressure.values())

    return {
        "pressure": pressure,
        "max_pressure": max_pressure,
        "overloaded_resources": [
            resource
            for resource, value in pressure.items()
            if value > 1.0
        ],
        "provisional_slowdown_proxy": max(1.0, max_pressure),
    }


def main():
    # Synthetic example to validate the calculation.
    # These are illustrative values, not measurements of two real layers.
    workload_a = WorkloadDemand(
        "workload_A",
        {
            "input_bw": 8.0,
            "weight_bw": 4.0,
            "output_bw": 2.0,
            "dram_input_bw": 0.2,
            "dram_weight_bw": 0.1,
            "dram_output_bw": 0.1,
        },
    )

    workload_b = WorkloadDemand(
        "workload_B",
        {
            "input_bw": 6.0,
            "weight_bw": 5.0,
            "output_bw": 3.0,
            "dram_input_bw": 0.3,
            "dram_weight_bw": 0.2,
            "dram_output_bw": 0.1,
        },
    )

    # Illustrative capacities, deliberately explicit.
    capacities = {
        "input_bw": 20.0,
        "weight_bw": 12.0,
        "output_bw": 8.0,
        "dram_input_bw": 1.0,
        "dram_weight_bw": 1.0,
        "dram_output_bw": 1.0,
    }

    result = evaluate_contention(
        [workload_a, workload_b],
        capacities,
    )

    print("===== SYNTHETIC CONTENTION TEST =====")
    print("These values are illustrative, not hardware measurements.")
    print()

    for resource, pressure in result["pressure"].items():
        print(f"{resource:20s}: pressure={pressure:.3f}")

    print()
    print("Maximum pressure:", round(result["max_pressure"], 3))
    print("Overloaded resources:", result["overloaded_resources"])
    print(
        "Provisional slowdown proxy:",
        round(result["provisional_slowdown_proxy"], 3),
    )

    assert abs(result["pressure"]["input_bw"] - 0.7) < 1e-9
    assert abs(result["pressure"]["weight_bw"] - 0.75) < 1e-9
    assert abs(result["pressure"]["output_bw"] - 0.625) < 1e-9
    assert result["overloaded_resources"] == []
    assert result["provisional_slowdown_proxy"] == 1.0

    print("\nPASS: contention arithmetic checks.")


if __name__ == "__main__":
    main()
