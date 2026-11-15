from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

stats = pd.read_csv(RESULTS / "resnet50_timeloop_stats.csv")
signatures = pd.read_csv(RESULTS / "resnet50_resource_signatures.csv")
memory = pd.read_csv(RESULTS / "resnet50_memory_profiles.csv")
regimes = pd.read_csv(RESULTS / "resnet50_resource_regimes.csv")

# Retain the raw Timeloop measurements as the foundation of each profile.
df = stats.copy()

# Add derived resource signatures without duplicating columns already in stats.
signature_columns = [
    "layer_id",
    "layer_index",
    "compute_intensity",
    "memory_access_per_op",
    "buffer_bandwidth",
    "dram_bandwidth",
    "dram_energy_fraction",
    "cycle_intensity",
    "mac_energy_fraction",
]
df = df.merge(
    signatures[signature_columns],
    on="layer_id",
    how="left",
    validate="one_to_one",
)

# Add per-dataspace and aggregate memory-footprint features.
memory_columns = [
    "layer_id",
    "local_capacity_peak",
    "local_capacity_sum",
    "dram_capacity_peak",
    "dram_capacity_sum",
]
df = df.merge(
    memory[memory_columns],
    on="layer_id",
    how="left",
    validate="one_to_one",
)

# Keep the discovered regime label; do not recluster the layers here.
df = df.merge(
    regimes[["layer_id", "regime"]].rename(
        columns={"regime": "resource_regime"}
    ),
    on="layer_id",
    how="left",
    validate="one_to_one",
)

# Explicit reference-architecture assumptions.
REFERENCE_MAC_INSTANCES = 16
REFERENCE_SRAM_WORDS = 4096

df["reference_compute_work"] = df["summary_computes"]
df["reference_execution_cycles"] = df["summary_cycles"]
df["reference_sram_capacity_words"] = REFERENCE_SRAM_WORDS
df["local_capacity_peak_fraction"] = (
    df["local_capacity_peak"] / REFERENCE_SRAM_WORDS
)
df["local_capacity_sum_fraction"] = (
    df["local_capacity_sum"] / REFERENCE_SRAM_WORDS
)
df["reference_mac_instances"] = REFERENCE_MAC_INSTANCES

# This is a conservative screening flag, not a hardware guarantee.
# The summed footprint is compared with the reference SRAM capacity.
df["sram_sum_fits_reference_screen"] = (
    df["local_capacity_sum"] <= REFERENCE_SRAM_WORDS
)

out = RESULTS / "resnet50_layer_profiles.csv"
df.to_csv(out, index=False)

print(f"Saved: {out}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Missing cells: {int(df.isna().sum().sum())}")
print(
    "Unique layer IDs:",
    df["layer_id"].nunique(),
)
print(
    "Resource regimes:",
    df["resource_regime"].value_counts().sort_index().to_dict(),
)
print(
    "Local capacity sum range:",
    int(df["local_capacity_sum"].min()),
    "to",
    int(df["local_capacity_sum"].max()),
    "words",
)
print(
    "Layers passing conservative reference SRAM screen:",
    int(df["sram_sum_fits_reference_screen"].sum()),
    "/",
    len(df),
)
print("\nFirst five profiles:")
print(
    df[
        [
            "layer_id",
            "summary_computes",
            "summary_cycles",
            "compute_intensity",
            "buffer_bandwidth",
            "dram_bandwidth",
            "local_capacity_sum",
            "resource_regime",
        ]
    ].head().to_string(index=False)
)
