from pathlib import Path
import pandas as pd

from simulator.hardware.model import load_hardware

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "results/resnet50_layer_profiles.csv"
OUTPUT_PATH = ROOT / "results/layer_mapping_evaluation.csv"


def main():
    profiles = pd.read_csv(PROFILE_PATH)
    hardware = load_hardware()

    rows = []

    for _, layer in profiles.iterrows():
        reference_cycles = float(layer["summary_cycles"])

        for chiplet in hardware.chiplets.values():
            estimated_cycles = (
                reference_cycles / chiplet.compute_capacity
            )

            rows.append({
                "layer_id": layer["layer_id"],
                "resource_regime": int(layer["resource_regime"]),
                "chiplet": chiplet.name,
                "reference_cycles": reference_cycles,
                "compute_only_estimated_cycles": estimated_cycles,
                "compute_capacity_scale": chiplet.compute_capacity,
                "buffer_bandwidth_demand_feature": float(
                    layer["buffer_bandwidth"]
                ),
                "dram_bandwidth_demand_feature": float(
                    layer["dram_bandwidth"]
                ),
                "dram_energy_fraction": float(
                    layer["dram_energy_fraction"]
                ),
                "local_capacity_peak_words": float(
                    layer["local_capacity_peak"]
                ),
                "local_sram_capacity_words": (
                    chiplet.sram_capacity_words
                ),
                "sram_peak_screen_pass": (
                    float(layer["local_capacity_peak"])
                    <= chiplet.sram_capacity_words
                ),
                "energy_reference_uJ": float(
                    layer["total_energy_uJ"]
                ),
            })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("===== LAYER-MAPPING EVALUATION =====")
    print(f"Layers: {profiles['layer_id'].nunique()}")
    print(f"Chiplets: {len(hardware.chiplets)}")
    print(f"Evaluation rows: {len(result)}")
    print(f"Missing cells: {int(result.isna().sum().sum())}")
    print(f"Saved: {OUTPUT_PATH}")
    print()

    print("===== MEAN COMPUTE-ONLY ESTIMATED CYCLES =====")
    print(
        result.groupby("chiplet")[
            "compute_only_estimated_cycles"
        ].mean().round(2).to_string()
    )
    print()

    print("===== LAYER COUNTS BY RESOURCE REGIME =====")
    print(
        profiles["resource_regime"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()

    print("===== SRAM PEAK-FOOTPRINT SCREEN =====")
    print(
        result.groupby("chiplet")["sram_peak_screen_pass"]
        .agg(["sum", "count"])
        .to_string()
    )
    print()

    print("===== FIRST EIGHT MAPPING ROWS =====")
    print(
        result[
            [
                "layer_id",
                "resource_regime",
                "chiplet",
                "compute_only_estimated_cycles",
                "buffer_bandwidth_demand_feature",
                "dram_bandwidth_demand_feature",
            ]
        ].head(8).to_string(index=False)
    )

    assert len(result) == len(profiles) * len(hardware.chiplets)
    assert result["layer_id"].nunique() == 53
    assert result["chiplet"].nunique() == 4
    assert not result.isna().any().any()

    print("\nPASS: all layer-chiplet combinations evaluated.")


if __name__ == "__main__":
    main()
