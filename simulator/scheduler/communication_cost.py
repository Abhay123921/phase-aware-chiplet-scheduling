from pathlib import Path

import pandas as pd

from simulator.hardware.model import load_hardware


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "results/resnet50_layer_profiles.csv"
OUTPUT_PATH = ROOT / "results/chiplet_communication_costs.csv"

# Explicit experimental assumptions, NOT measured hardware parameters.
WORD_BYTES = 2
REFERENCE_NOP_WORDS_PER_CYCLE = 16.0
REFERENCE_HOP_LATENCY_CYCLES = 1.0

# Assumed placement on the configured 2x2 mesh.
# This is a modeling convention, not a verified physical floorplan.
COORDINATES = {
    "C1": (0, 0),
    "C2": (0, 1),
    "C3": (1, 0),
    "C4": (1, 1),
}


def manhattan_hops(source, destination):
    x1, y1 = COORDINATES[source]
    x2, y2 = COORDINATES[destination]
    return abs(x1 - x2) + abs(y1 - y2)


def estimate_transfer(
    words,
    hops,
    nop_bandwidth_scale,
    hop_latency_scale,
):
    if words < 0:
        raise ValueError("Transfer words cannot be negative")
    if nop_bandwidth_scale <= 0:
        raise ValueError("NoP bandwidth scale must be positive")

    effective_words_per_cycle = (
        REFERENCE_NOP_WORDS_PER_CYCLE * nop_bandwidth_scale
    )
    transfer_cycles = words / effective_words_per_cycle
    hop_cycles = (
        hops
        * REFERENCE_HOP_LATENCY_CYCLES
        * hop_latency_scale
    )

    return {
        "transfer_words": words,
        "transfer_bytes": words * WORD_BYTES,
        "data_transfer_cycles_proxy": transfer_cycles,
        "hop_latency_cycles_proxy": hop_cycles,
        "total_communication_cycles_proxy":
            transfer_cycles + hop_cycles,
    }


def main():
    profiles = pd.read_csv(PROFILE_PATH)
    hardware = load_hardware()

    required = {
        "layer_index",
        "layer_id",
        "outputs_utilized_capacity",
    }
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    profiles = profiles.sort_values("layer_index").reset_index(drop=True)
    rows = []

    # Estimate movement of the previous layer's output whenever the
    # next layer is assigned to a different chiplet.
    for i in range(1, len(profiles)):
        previous = profiles.iloc[i - 1]
        current = profiles.iloc[i]

        words = float(previous["outputs_utilized_capacity"])

        for source in hardware.chiplets:
            for destination in hardware.chiplets:
                if source == destination:
                    hops = 0
                else:
                    hops = manhattan_hops(source, destination)

                if source == destination:
                    estimate = {
                        "transfer_words": 0.0,
                        "transfer_bytes": 0.0,
                        "data_transfer_cycles_proxy": 0.0,
                        "hop_latency_cycles_proxy": 0.0,
                        "total_communication_cycles_proxy": 0.0,
                    }
                else:
                    estimate = estimate_transfer(
                        words=words,
                        hops=hops,
                        nop_bandwidth_scale=hardware.nop_bandwidth,
                        hop_latency_scale=hardware.nop_hop_latency,
                    )

                rows.append({
                    "previous_layer": previous["layer_id"],
                    "next_layer": current["layer_id"],
                    "source_chiplet": source,
                    "destination_chiplet": destination,
                    "remap_required": source != destination,
                    "previous_output_words_proxy": words,
                    "previous_output_bytes_proxy": words * WORD_BYTES,
                    "mesh_hops_assumed": hops,
                    **estimate,
                })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("===== CHIPLET COMMUNICATION-COST MODEL =====")
    print(f"Layer transitions evaluated: {len(profiles) - 1}")
    print(f"Chiplets: {len(hardware.chiplets)}")
    print(f"Transition rows: {len(result)}")
    print(f"Word size assumption: {WORD_BYTES} bytes")
    print(
        "Reference NoP bandwidth assumption:",
        f"{REFERENCE_NOP_WORDS_PER_CYCLE} words/cycle",
    )
    print(
        "Reference hop-latency assumption:",
        f"{REFERENCE_HOP_LATENCY_CYCLES} cycles/hop",
    )
    print("These assumptions are not calibrated hardware values.")
    print()

    print("===== EXAMPLE: FIRST LAYER TRANSITION =====")
    first_transition = profiles.iloc[0]["layer_id"]
    next_transition = profiles.iloc[1]["layer_id"]

    example = result[
        (result["previous_layer"] == first_transition)
        & (result["next_layer"] == next_transition)
    ]

    print(
        example[
            [
                "source_chiplet",
                "destination_chiplet",
                "mesh_hops_assumed",
                "previous_output_words_proxy",
                "previous_output_bytes_proxy",
                "total_communication_cycles_proxy",
            ]
        ].to_string(index=False)
    )

    print()
    print("===== COMMUNICATION-COST RANGE =====")
    moved = result[result["remap_required"]]
    print(
        moved["total_communication_cycles_proxy"]
        .describe()
        .round(3)
        .to_string()
    )

    assert len(result) == (len(profiles) - 1) * 16
    assert not result.isna().any().any()
    assert (result["transfer_words"] >= 0).all()
    assert (result["total_communication_cycles_proxy"] >= 0).all()
    assert (
        result.loc[
            ~result["remap_required"],
            "total_communication_cycles_proxy",
        ] == 0
    ).all()

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print("PASS: communication-cost consistency checks.")


if __name__ == "__main__":
    main()
