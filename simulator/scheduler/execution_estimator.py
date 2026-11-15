from pathlib import Path
import pandas as pd

from simulator.hardware.model import load_hardware


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "results/resnet50_layer_profiles.csv"


def estimate_compute_cycles(reference_cycles, compute_capacity):
    """First-order compute-only estimate in reference cycle units."""
    if reference_cycles < 0:
        raise ValueError("reference_cycles cannot be negative")
    if compute_capacity <= 0:
        raise ValueError("compute_capacity must be positive")

    return reference_cycles / compute_capacity


def main():
    profiles = pd.read_csv(PROFILE_PATH)
    hardware = load_hardware()

    # All chiplets are evaluated against the same reference layer profile.
    layer = profiles.loc[profiles["layer_id"] == "01_conv1"].iloc[0]
    reference_cycles = float(layer["summary_cycles"])

    print("===== SINGLE-LAYER COMPUTE ESTIMATOR =====")
    print(f"Layer: {layer['layer_id']}")
    print(f"Reference cycles: {reference_cycles:.0f}")
    print(
        "Reference compute rate:",
        f"{layer['summary_computes'] / reference_cycles:.3f}",
        "operations per reference cycle",
    )
    print()
    print(
        f"{'Chiplet':<10}{'Compute scale':>15}"
        f"{'Estimated cycles':>22}{'Speedup':>12}"
    )

    for chiplet in hardware.chiplets.values():
        estimated = estimate_compute_cycles(
            reference_cycles,
            chiplet.compute_capacity,
        )
        speedup = reference_cycles / estimated if estimated else 0.0

        print(
            f"{chiplet.name:<10}"
            f"{chiplet.compute_capacity:>15.2f}"
            f"{estimated:>22.2f}"
            f"{speedup:>12.2f}x"
        )

    # Verify that the normalized reference gives the measured cycle count.
    reference_estimate = estimate_compute_cycles(reference_cycles, 1.0)
    assert abs(reference_estimate - reference_cycles) < 1e-9

    # Verify the expected relative compute-only scaling.
    for chiplet in hardware.chiplets.values():
        estimate = estimate_compute_cycles(
            reference_cycles,
            chiplet.compute_capacity,
        )
        assert abs(
            reference_cycles / estimate - chiplet.compute_capacity
        ) < 1e-9

    print("\nPASS: reference-cycle and relative-scaling checks.")


if __name__ == "__main__":
    main()
